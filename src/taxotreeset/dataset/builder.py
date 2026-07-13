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
    splits = builder.prepare_stratified_split(sequence_leaf_nodes)
    builder.build_node_dataset(extraction_jobs, parallel=True)
"""

import logging
import math
import multiprocessing
import os
from typing import Any

import numpy as np
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
_STRATIFIED_SPLIT_RATIOS = (0.70, 0.85)
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


def extract_parent_node_worker(job: tuple) -> bool:
    """Build train/val/test Parquet files for a single parent node.

    Designed as the target callable of a spawned multiprocessing pool.
    Iterates over the job's split tasks, reads each source sequence
    from LMDB, slices it to the assigned fraction, samples
    subsequences, and streams rows to Parquet using a bounded buffer
    that flushes every ``_BUFFER_SIZE_ROWS`` rows.

    Memory is bounded by ``_BUFFER_SIZE_ROWS`` rather than the total
    output volume, so workers operate in constant memory regardless
    of how large the head's training shard turns out to be.

    Args:
        job: Tuple of (parent_taxid, target_dir, parent_tasks,
            max_subseq_len, seed, output_format). The ``parent_tasks``
            element is a dict with keys 'train', 'val', 'test'; each
            value is a list of task dicts with keys 'fasta_path',
            'header_id', 'start_pct', 'end_pct', 'n', and 'class_idx'.

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
        _write_split_parquet(
            tasks=tasks,
            output_path=output_path,
            max_subseq_len=max_subseq_len,
        )

    return True


def _write_split_parquet(
    tasks: list[dict[str, Any]],
    output_path: str,
    max_subseq_len: int,
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
            extracted_rows = _extract_subseqs_for_task(task, max_subseq_len)
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
) -> list[dict[str, Any]]:
    """Read, slice, and sample subsequences for a single task.

    Args:
        task: Task dictionary with the keys documented in
            ``extract_parent_node_worker``.
        max_subseq_len: Upper bound on each subsequence length.

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
        min_len=_DEFAULT_MIN_SUBSEQ_LEN,
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


def _plan_shards(
    jobs: list[tuple],
    shard_rows_target: int,
) -> tuple[list[tuple], list[tuple]]:
    """Fan head-jobs out into per-(head, split, shard) shard-jobs + merge-jobs.

    Splits whose final merged file already exists are skipped (resume), so no
    shard or merge work is scheduled for them. Part-file names are deterministic
    (``<split>.part{idx:05d}.<fmt>``) given the task order, so an interrupted run
    reuses completed parts on restart.

    Args:
        jobs: Head-jobs ``(taxid, target_dir, parent_tasks, max_subseq_len, seed,
            output_format)`` as built by the orchestrator.
        shard_rows_target: Approximate rows per shard (see ``_partition_tasks``).

    Returns:
        ``(shard_jobs, merge_jobs)``. A shard-job is ``(target_dir, split,
        part_path, shard_tasks, max_subseq_len)``; a merge-job is ``(target_dir,
        split, part_paths, final_path)``.
    """
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
                    target_dir, f"{split}.part{idx:05d}.{output_format}"
                )
                part_paths.append(part_path)
                shard_jobs.append(
                    (target_dir, split, part_path, shard_tasks, max_subseq_len)
                )
            merge_jobs.append((target_dir, split, part_paths, final_path))
    return shard_jobs, merge_jobs


def _shard_worker(shard_job: tuple) -> bool:
    """Extract one shard's tasks to a part file (spawn-pool target).

    Writes to a ``.tmp`` sibling and atomically renames, so an interrupted write
    never leaves a corrupt part that a resume would trust. A shard whose tasks
    yield no rows produces no file (``_write_split_parquet`` opens lazily).

    Args:
        shard_job: ``(target_dir, split, part_path, shard_tasks, max_subseq_len)``.

    Returns:
        True on completion.
    """
    target_dir, _split, part_path, shard_tasks, max_subseq_len = shard_job
    if os.path.exists(part_path):
        return True  # resume: this shard is already built
    os.makedirs(target_dir, exist_ok=True)
    tmp_path = f"{part_path}.tmp"
    _write_split_parquet(shard_tasks, tmp_path, max_subseq_len)
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
    (which schedules extraction jobs) and the worker pool (which
    writes Parquet files). Provides two services:

    - **Stratified splitting** via ``prepare_stratified_split``,
      which partitions sequence leaves into train/val/test sets
      with deterministic shuffling controlled by the seed.

    - **Parallel build dispatch** via ``build_node_dataset``, which
      configures a multiprocessing pool sized to the host's RAM and
      delegates each parent-node job to a worker.

    Attributes:
        output_dir: Root directory for the generated training shards.
        max_subseq_len: Upper bound on each subsequence length, in bp.
        seed: Random seed for the deterministic shuffle.
        output_format: Either 'parquet' (production) or 'csv' (debug).
    """

    def __init__(
        self,
        output_dir: str,
        max_subseq_len: int,
        seed: int,
        output_format: str,
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
        """
        self.output_dir: str = output_dir
        self.max_subseq_len: int = max_subseq_len
        self.seed: int = seed
        self.output_format: str = output_format

    def prepare_stratified_split(self, nodes: list) -> dict[str, list[tuple]]:
        """Partition sequence leaves into train/val/test sets.

        Two scenarios are supported based on the number of available
        sequence leaves:

        1. **Sufficient diversity** (>= 3 leaves): leaves are shuffled
           deterministically and partitioned by index. Each leaf is
           fully assigned to a single split, with no intra-sequence
           leakage between splits.

        2. **Extreme scarcity** (< 3 leaves): the same sequence is
           sliced into the three splits by fraction (70/15/15).
           Intra-sequence leakage is accepted as the cost of having
           any training data at all for these low-data classes.

        Args:
            nodes: List of taxon nodes whose sequence leaves will be
                collected and partitioned.

        Returns:
            Dictionary with three keys ('train', 'val', 'test'); each
            value is a list of (fasta_path, header_id, start_pct,
            end_pct) tuples describing what each worker should read.
        """
        splits: dict[str, list[tuple]] = {key: [] for key in _SPLITS}

        all_leaves = self._collect_sequence_leaves(nodes)
        if not all_leaves:
            return splits

        np.random.seed(self.seed)
        np.random.shuffle(all_leaves)

        if len(all_leaves) >= 3:
            return self._split_by_distinct_leaves(all_leaves, splits)
        return self._split_by_sequence_fractions(all_leaves, splits)

    @staticmethod
    def _collect_sequence_leaves(nodes: list) -> list:
        """Gather all sequence leaves under the given nodes.

        Args:
            nodes: List of parent nodes to scan.

        Returns:
            Flat list of leaf nodes whose rank is 'sequence'.
        """
        leaves: list = []
        for node in nodes:
            leaves.extend(
                leaf for leaf in node.leaves if getattr(leaf, "rank", "") == "sequence"
            )
        return leaves

    def _split_by_distinct_leaves(
        self,
        all_leaves: list,
        splits: dict[str, list[tuple]],
    ) -> dict[str, list[tuple]]:
        """Assign whole leaves to train/val/test by index ranges.

        Uses the global ratios ``_STRATIFIED_SPLIT_RATIOS`` to compute
        train and validation cut indices. Each leaf is assigned to a
        single split with start_pct=0.0 and end_pct=1.0, meaning the
        worker will read the entire sequence into the target split.

        Args:
            all_leaves: Shuffled list of leaf nodes.
            splits: Pre-initialized splits dictionary to populate.

        Returns:
            The populated splits dictionary.
        """
        leaf_count = len(all_leaves)
        train_ratio, val_ratio = _STRATIFIED_SPLIT_RATIOS

        train_cut = max(1, int(leaf_count * train_ratio))
        val_cut = train_cut + max(1, int(leaf_count * (val_ratio - train_ratio)))
        # Guarantee test receives at least one leaf: the two max(1, ...) floors
        # otherwise consume all three leaves at leaf_count == 3 (train=2, val=1,
        # test=0), producing a class with zero test support.
        if val_cut >= leaf_count:
            val_cut = leaf_count - 1
            if train_cut >= val_cut:
                train_cut = val_cut - 1

        for index, leaf in enumerate(all_leaves):
            task = (
                getattr(leaf, "fasta_path", ""),
                getattr(leaf, "header_id", ""),
                0.0,
                1.0,
            )
            if index < train_cut:
                splits["train"].append(task)
            elif index < val_cut:
                splits["val"].append(task)
            else:
                splits["test"].append(task)

        return splits

    @staticmethod
    def _split_by_sequence_fractions(
        all_leaves: list,
        splits: dict[str, list[tuple]],
    ) -> dict[str, list[tuple]]:
        """Slice each leaf's sequence across all three splits.

        Used when the leaf count is too low for distinct assignment
        (< 3 leaves). Each sequence is read in three fractions: the
        first 70% goes to train, the next 15% to val, the final 15%
        to test. This produces some data in every split at the cost
        of accepting intra-sequence leakage.

        Args:
            all_leaves: Shuffled list of leaf nodes.
            splits: Pre-initialized splits dictionary to populate.

        Returns:
            The populated splits dictionary.
        """
        for leaf in all_leaves:
            fasta_path = getattr(leaf, "fasta_path", "")
            header_id = getattr(leaf, "header_id", "")
            splits["train"].append((fasta_path, header_id, 0.0, 0.70))
            splits["val"].append((fasta_path, header_id, 0.70, 0.85))
            splits["test"].append((fasta_path, header_id, 0.85, 1.0))
        return splits

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
            return [extract_parent_node_worker(job) for job in jobs]

        # Task-level sharding: fan each head's per-split tasks out into balanced
        # shards so a head with many source genomes is spread across workers
        # instead of pinning one core (the straggler that idles cores at the end
        # of a batch). Shards write part files; a merge pass concatenates each
        # split's parts back into the single <split>.<fmt> file downstream expects.
        shard_jobs, merge_jobs = _plan_shards(jobs, _SHARD_ROWS_TARGET)
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
