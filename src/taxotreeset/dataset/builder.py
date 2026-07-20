"""Dataset builder for materializing taxonomic training shards on disk.

This module provides two cooperating components:

1. **The worker function** ``extract_parent_node_worker``: runs in a
   spawned multiprocessing process, reads sequences from the LMDB
   vault, samples subsequences via the sliding-window strategy in
   ``sequence_utils``, and writes them to Parquet (or CSV) files
   organized by split (train/val/test).

2. **The DatasetBuilder class**: handles the upstream logic of
   stratified splitting and orchestrates the worker pool when the
   build runs in parallel mode.

The worker is intentionally module-level (not a method) because the
spawn-based multiprocessing protocol requires the target callable to
be picklable and importable by name from the child process. A method
on a class instance would carry the instance state along with it,
which is wasteful and slow.

The builder uses spawn rather than fork for the worker pool. Spawn
costs ~1-2 seconds of import overhead per worker, but it isolates
each worker from the parent's address space. This prevents two
common pitfalls when forking:

- The parent has loaded the full taxonomic tree into memory; fork
  would duplicate that footprint per worker via COW pages, then
  cause page-out pressure under heavy mutation.
- The parent may hold an open LMDB write handle from the discovery
  phase; forking would share that handle, leading to undefined
  behavior at the mmap layer.

Workers use a bounded buffer to flush rows to Parquet in chunks
rather than accumulating the entire dataset in memory. This keeps
worker memory roughly constant even when producing millions of rows
per head.

Typical usage::

    from taxotreeset.dataset.builder import DatasetBuilder

    builder = DatasetBuilder(
        output_dir="data/datasets",
        max_subseq_len=2000,
        seed=42,
        output_format="parquet",
    )
    builder.build_node_dataset(extraction_jobs, parallel=True)
"""

import hashlib
import logging
import math
import multiprocessing
import os
from typing import Any

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from taxotreeset.dataset.sequence_utils import extract_subseqs
from taxotreeset.dataset.utils import (
    _pool_worker_initializer,
    _read_single_sequence,
)

logger = logging.getLogger("TaxoTreeSet.Dataset.Builder")

_BUFFER_SIZE_ROWS = 10_000
_DEFAULT_MIN_SUBSEQ_LEN = 100
_PARQUET_COMPRESSION = "snappy"
_SPLITS = ("train", "val", "test")
_LOW_MEMORY_THRESHOLD_GB = 12
_LOW_MEMORY_WORKER_COUNT = 2
_WORKERS_RESERVED_FOR_PARENT = 2

# Target rows per extraction shard. A head's per-split tasks are partitioned into
# shards of roughly this many sampled rows (balanced by each task's ``n``), so a
# head with many source genomes is spread across workers instead of pinning one
# core. Heads whose whole split is smaller than this stay a single shard (no
# fan-out overhead). Chosen well above _BUFFER_SIZE_ROWS so each shard amortizes
# its per-task pickling/IPC while still yielding shards >> worker count on big heads.
_SHARD_ROWS_TARGET = 50_000


def extract_parent_node_worker(
    job: tuple,
    min_subseq_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
) -> bool:
    """Build train/val/test Parquet files for a single parent node.

    The non-sharded (serial) extraction path. Iterates over the job's split
    tasks, reads each source sequence from LMDB, slices it to the assigned
    fraction, samples subsequences, and streams rows to Parquet using a bounded
    buffer that flushes every ``_BUFFER_SIZE_ROWS`` rows.

    Memory is bounded by ``_BUFFER_SIZE_ROWS`` rather than the total
    output volume, so workers operate in constant memory regardless
    of how large the head's training shard turns out to be.

    Each split is written to a ``.tmp`` sibling and atomically renamed, so an
    interrupted run never leaves a partial ``<split>.<fmt>`` that a resume would
    trust as complete.

    Args:
        job: Tuple of (parent_taxid, target_dir, parent_tasks,
            max_subseq_len, seed, output_format). The ``parent_tasks``
            element is a dict with keys 'train', 'val', 'test'; each
            value is a list of task dicts with keys 'fasta_path',
            'header_id', 'start_pct', 'end_pct', 'n', and 'class_idx'.
        min_subseq_len: Lower bound on each subsequence length, in bp.

    Returns:
        True on completion. Failures within a single task are logged
        and skipped, not raised, so the worker always returns True
        unless an unrecoverable I/O error occurs.
    """
    _parent_taxid, target_dir, parent_tasks, max_subseq_len, _seed, output_format = job

    # Skip heads whose every non-empty split file already exists on disk.
    # This makes Stage 4 resumable across job restarts: completed heads are
    # detected by the presence of all expected output files and not re-run.
    expected_splits = [
        s for s in _SPLITS if parent_tasks.get(s)
    ]
    if expected_splits and all(
        os.path.exists(os.path.join(target_dir, f"{s}.{output_format}"))
        for s in expected_splits
    ):
        return True

    for split in _SPLITS:
        tasks = parent_tasks.get(split, [])
        if not tasks:
            continue

        output_path = os.path.join(target_dir, f"{split}.{output_format}")
        tmp_path = f"{output_path}.tmp"
        _write_split_parquet(
            tasks=tasks,
            output_path=tmp_path,
            max_subseq_len=max_subseq_len,
            min_subseq_len=min_subseq_len,
        )
        if os.path.exists(tmp_path):
            os.replace(tmp_path, output_path)

    return True


def _write_split_parquet(
    tasks: list[dict[str, Any]],
    output_path: str,
    max_subseq_len: int,
    min_subseq_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
) -> None:
    """Write subsequences from a list of tasks to a Parquet file.

    Uses a bounded row buffer that flushes every
    ``_BUFFER_SIZE_ROWS`` accumulated rows, keeping memory constant.
    The Parquet writer is opened lazily on the first flush so empty
    splits do not produce empty files.

    Args:
        tasks: List of task dictionaries describing the source
            sequences, their slicing fractions, the number of samples
            to draw, and the class index to assign.
        output_path: Destination path for the Parquet file.
        max_subseq_len: Upper bound on each subsequence length, in
            base pairs.
        min_subseq_len: Lower bound on each subsequence length, in bp.
    """
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal writer
        if not buffer:
            return
        table = _buffer_to_arrow_table(buffer)
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                table.schema,
                compression=_PARQUET_COMPRESSION,
            )
        writer.write_table(table)
        buffer.clear()

    try:
        for task in tasks:
            extracted_rows = _extract_subseqs_for_task(
                task, max_subseq_len, min_subseq_len
            )
            buffer.extend(extracted_rows)
            if len(buffer) >= _BUFFER_SIZE_ROWS:
                flush_buffer()
        flush_buffer()
    finally:
        if writer is not None:
            writer.close()


def _extract_subseqs_for_task(
    task: dict[str, Any],
    max_subseq_len: int,
    min_subseq_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
) -> list[dict[str, Any]]:
    """Read, slice, and sample subsequences for a single task.

    Args:
        task: Task dictionary with the keys documented in
            ``extract_parent_node_worker``.
        max_subseq_len: Upper bound on each subsequence length.
        min_subseq_len: Lower bound on each subsequence length, in bp — the same
            ``--min-subseq-len`` used for capacity and n-distribution, threaded
            through so extraction honours the configured floor instead of a fixed
            default.

    Returns:
        List of row dictionaries ready to append to the buffer.
        Empty list if the source sequence could not be read.
    """
    full_sequence = _read_single_sequence(task["fasta_path"], task["header_id"])
    if not full_sequence:
        return []

    start_index = int(len(full_sequence) * task["start_pct"])
    end_index = int(len(full_sequence) * task["end_pct"])
    sliced_sequence = full_sequence[start_index:end_index]

    sampled_subsequences = extract_subseqs(
        seq=sliced_sequence,
        n=task["n"],
        min_len=min_subseq_len,
        max_len=max_subseq_len,
    )

    class_index = int(task["class_idx"])
    return [
        {"seq": subseq, "class_idx": class_index} for subseq in sampled_subsequences
    ]


def _buffer_to_arrow_table(buffer: list[dict[str, Any]]) -> pa.Table:
    """Convert a row buffer to an Arrow Table with normalized dtypes.

    Args:
        buffer: List of row dictionaries from
            ``_extract_subseqs_for_task``.

    Returns:
        Arrow Table with class_idx as int32 (compact) and seq as
        string. Index is not preserved.
    """
    dataframe = pd.DataFrame(buffer)
    dataframe["class_idx"] = dataframe["class_idx"].astype("int32")
    return pa.Table.from_pandas(dataframe, preserve_index=False)


def _partition_tasks(
    tasks: list[dict[str, Any]],
    shard_rows_target: int,
) -> list[list[dict[str, Any]]]:
    """Split a split's tasks into shards balanced by estimated rows (``n``).

    The number of shards is ``ceil(sum(n) / shard_rows_target)``, capped at the
    task count (a shard can hold at least one task). Tasks are greedily assigned
    to the least-loaded shard in descending-``n`` order, so shard row-counts stay
    close to uniform. A single-task or below-target split returns one shard, i.e.
    no fan-out for small heads.

    Args:
        tasks: Worker-ready task dicts for one split of one head.
        shard_rows_target: Approximate rows per shard.

    Returns:
        A list of non-empty task sublists (the shards), in shard-index order.
    """
    total = sum(int(t.get("n", 0)) for t in tasks)
    n_shards = max(1, min(len(tasks), math.ceil(total / shard_rows_target)))
    if n_shards <= 1:
        return [list(tasks)]

    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
    loads = [0] * n_shards
    for task in sorted(tasks, key=lambda t: int(t.get("n", 0)), reverse=True):
        i = min(range(n_shards), key=lambda k: loads[k])
        shards[i].append(task)
        loads[i] += int(task.get("n", 0))
    return [s for s in shards if s]


def _shard_hash(shard_tasks: list[dict[str, Any]]) -> str:
    """Return a short content hash of a shard's ordered task list.

    Embedded in the part-file name so a resumed run only reuses a part when its
    task set is byte-identical. If the upstream schedule changed (different tasks,
    counts, or partition), the hash changes, the part is recomputed, and the stale
    part — no longer referenced by any plan — is cleaned up by ``_plan_shards``.
    """
    digest = hashlib.blake2s(digest_size=4)
    for t in shard_tasks:
        digest.update(
            repr((
                t.get("fasta_path"), t.get("header_id"),
                t.get("start_pct"), t.get("end_pct"),
                t.get("n"), t.get("class_idx"),
            )).encode()
        )
    return digest.hexdigest()


def _plan_shards(
    jobs: list[tuple],
    shard_rows_target: int,
    min_subseq_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
) -> tuple[list[tuple], list[tuple]]:
    """Fan head-jobs out into per-(head, split, shard) shard-jobs + merge-jobs.

    Splits whose final merged file already exists are skipped (resume), so no
    shard or merge work is scheduled for them. Part-file names are content-hashed
    (``<split>.part{idx:05d}.{hash}.<fmt>``), so an interrupted run reuses a part
    only when its tasks are identical; parts left by a superseded schedule (a
    stale hash, or a higher index from a larger prior partition) are deleted
    before dispatch so they are neither reused nor orphaned on disk.

    Args:
        jobs: Head-jobs ``(taxid, target_dir, parent_tasks, max_subseq_len, seed,
            output_format)`` as built by the orchestrator.
        shard_rows_target: Approximate rows per shard (see ``_partition_tasks``).
        min_subseq_len: Lower bound on each subsequence length, in bp; carried in
            each shard-job so the worker extracts to the configured floor.

    Returns:
        ``(shard_jobs, merge_jobs)``. A shard-job is ``(target_dir, split,
        part_path, shard_tasks, max_subseq_len, min_subseq_len)``; a merge-job is
        ``(target_dir, split, part_paths, final_path)``.
    """
    import glob

    shard_jobs: list[tuple] = []
    merge_jobs: list[tuple] = []
    for job in jobs:
        _taxid, target_dir, parent_tasks, max_subseq_len, _seed, output_format = job
        for split in _SPLITS:
            tasks = parent_tasks.get(split, [])
            if not tasks:
                continue
            final_path = os.path.join(target_dir, f"{split}.{output_format}")
            if os.path.exists(final_path):
                continue  # resume: this split is already built
            part_paths: list[str] = []
            for idx, shard_tasks in enumerate(
                _partition_tasks(tasks, shard_rows_target)
            ):
                part_path = os.path.join(
                    target_dir,
                    f"{split}.part{idx:05d}.{_shard_hash(shard_tasks)}."
                    f"{output_format}",
                )
                part_paths.append(part_path)
                shard_jobs.append((
                    target_dir, split, part_path, shard_tasks,
                    max_subseq_len, min_subseq_len,
                ))
            # Drop any parts of this split not in the current plan (stale hash or
            # a higher index from a bigger prior partition), so a changed
            # schedule never reuses them and they do not accumulate on disk.
            keep = set(part_paths)
            for stale in glob.glob(
                os.path.join(target_dir, f"{split}.part*.{output_format}")
            ):
                if stale not in keep:
                    try:
                        os.remove(stale)
                    except OSError:
                        pass
            merge_jobs.append((target_dir, split, part_paths, final_path))
    return shard_jobs, merge_jobs


def _shard_worker(shard_job: tuple) -> bool:
    """Extract one shard's tasks to a part file (spawn-pool target).

    Writes to a ``.tmp`` sibling and atomically renames, so an interrupted write
    never leaves a corrupt part that a resume would trust. A shard whose tasks
    yield no rows produces no file (``_write_split_parquet`` opens lazily).

    Args:
        shard_job: ``(target_dir, split, part_path, shard_tasks, max_subseq_len,
            min_subseq_len)``.

    Returns:
        True on completion.
    """
    (
        target_dir, _split, part_path, shard_tasks, max_subseq_len, min_subseq_len
    ) = shard_job
    if os.path.exists(part_path):
        return True  # resume: this shard is already built
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = f"{part_path}.tmp"
    _write_split_parquet(shard_tasks, tmp_path, max_subseq_len, min_subseq_len)
    if os.path.exists(tmp_path):
        os.replace(tmp_path, part_path)
    return True


def _merge_worker(merge_job: tuple) -> bool:
    """Concatenate a split's part files into its final Parquet (spawn-pool target).

    Copies each part's row groups into one writer (no re-sampling, constant
    memory), writes to a ``.tmp`` sibling, atomically renames to the final path,
    then deletes the parts. The atomic rename makes the merge crash-safe: an
    interrupted merge leaves the parts intact and no final file, so a resume
    redoes it. An empty split (no part produced any rows) yields no final file,
    matching the un-sharded behavior.

    Args:
        merge_job: ``(target_dir, split, part_paths, final_path)``.

    Returns:
        True on completion.
    """
    _target_dir, _split, part_paths, final_path = merge_job
    if os.path.exists(final_path):
        for part in part_paths:  # tidy any leftover parts from a prior run
            if os.path.exists(part):
                os.remove(part)
        return True

    existing = [p for p in part_paths if os.path.exists(p)]
    if not existing:
        return True  # empty split -> no file, as before

    tmp_path = f"{final_path}.tmp"
    writer: pq.ParquetWriter | None = None
    try:
        for part in existing:
            parquet_file = pq.ParquetFile(part)
            for batch in parquet_file.iter_batches():
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(
                        tmp_path, table.schema, compression=_PARQUET_COMPRESSION
                    )
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp_path, final_path)
    for part in existing:
        os.remove(part)
    return True


class DatasetBuilder:
    """Materialize the train/val/test Parquet shards for the cascade.

    Acts as the dispatch layer between the generation orchestrator
    (which schedules extraction jobs, already split into train/val/test) and
    the worker pool that writes Parquet files. Its single service,
    ``build_node_dataset``, fans each head's per-split tasks into balanced
    shards, runs them on a spawn pool sized to the host's RAM, and merges each
    split's shards back into one file. The stratified split itself lives
    upstream in the generation orchestrator.

    Attributes:
        output_dir: Root directory for the generated training shards.
        max_subseq_len: Upper bound on each subsequence length, in bp.
        min_subseq_len: Lower bound on each subsequence length, in bp.
        seed: Random seed recorded for the run (carried in each job tuple).
        output_format: Either 'parquet' (production) or 'csv' (debug).
    """

    def __init__(
        self,
        output_dir: str,
        max_subseq_len: int,
        seed: int,
        output_format: str,
        min_subseq_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
    ) -> None:
        """Initialize the dataset builder.

        Args:
            output_dir: Root directory for the generated training
                shards.
            max_subseq_len: Upper bound on each subsequence length,
                in base pairs.
            seed: Random seed for the deterministic shuffle in the
                stratified split.
            output_format: 'parquet' or 'csv'. Parquet is recommended
                for production; CSV may be useful for debugging.
            min_subseq_len: Lower bound on each subsequence length, in bp —
                the ``--min-subseq-len`` used upstream for capacity and
                n-distribution, so extraction samples to the same floor.
        """
        self.output_dir: str = output_dir
        self.max_subseq_len: int = max_subseq_len
        self.seed: int = seed
        self.output_format: str = output_format
        self.min_subseq_len: int = min_subseq_len

    def build_node_dataset(
        self,
        jobs: list[tuple],
        parallel: bool = False,
    ) -> list[bool]:
        """Run the extraction workers, optionally in parallel.

        In serial mode, jobs are dispatched one at a time in the
        current process; useful for debugging. In parallel mode, a
        spawn-based pool is used with worker count adjusted for the
        host's available RAM.

        The spawn context is preferred over fork to prevent workers
        from inheriting the parent's full address space (the
        taxonomic tree, LMDB handles, etc.).

        Args:
            jobs: List of job tuples to dispatch.
            parallel: When True, use a worker pool. When False,
                execute jobs sequentially in the current process.

        Returns:
            List of worker return values (one per job).
        """
        if not parallel:
            return [
                extract_parent_node_worker(job, self.min_subseq_len)
                for job in jobs
            ]

        # Task-level sharding: fan each head's per-split tasks out into balanced
        # shards so a head with many source genomes is spread across workers
        # instead of pinning one core (the straggler that idles cores at the end
        # of a batch). Shards write part files; a merge pass concatenates each
        # split's parts back into the single <split>.<fmt> file downstream expects.
        shard_jobs, merge_jobs = _plan_shards(
            jobs, _SHARD_ROWS_TARGET, self.min_subseq_len
        )
        if not shard_jobs and not merge_jobs:
            return [True] * len(jobs)  # every split already built (resume)

        worker_count = self._compute_worker_count()
        logger.info(
            f"[BUILDER] Worker pool: {worker_count} processes; "
            f"{len(shard_jobs)} shard(s) over {len(jobs)} head(s)"
        )

        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=worker_count,
            initializer=_pool_worker_initializer,
        ) as pool:
            with tqdm(
                total=len(shard_jobs), desc="Extracting shards", unit="shard"
            ) as progress_bar:
                for _ in pool.imap_unordered(_shard_worker, shard_jobs, chunksize=1):
                    progress_bar.update(1)
            with tqdm(
                total=len(merge_jobs), desc="Merging shards", unit="file"
            ) as progress_bar:
                for _ in pool.imap_unordered(_merge_worker, merge_jobs, chunksize=1):
                    progress_bar.update(1)
        return [True] * len(jobs)

    @staticmethod
    def _compute_worker_count() -> int:
        """Choose a worker count based on available system memory.

        Returns ``_LOW_MEMORY_WORKER_COUNT`` (2) when the host has
        less than ``_LOW_MEMORY_THRESHOLD_GB`` of RAM, which is the
        typical WSL or laptop configuration. Otherwise reserves
        ``_WORKERS_RESERVED_FOR_PARENT`` (2) cores for the parent
        process and uses the remainder.

        Returns:
            Number of worker processes to spawn.
        """
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        if total_memory_gb < _LOW_MEMORY_THRESHOLD_GB:
            return _LOW_MEMORY_WORKER_COUNT

        cpu_count = multiprocessing.cpu_count()
        return max(1, cpu_count - _WORKERS_RESERVED_FOR_PARENT)
