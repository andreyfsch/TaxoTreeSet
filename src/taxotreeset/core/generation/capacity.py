"""Capacity computation for taxonomic nodes via exact union or Bloom filter.

The "capacity" of a taxonomic node is the number of unique
subsequences of length ``min_len`` extractable via sliding window
from all sequence leaves descending from it. This count is the input
that drives the per-class balancing layer: when balancing siblings,
each child's n_per_class is bounded by the minimum capacity across
the group.

This module provides two computation strategies and the dispatcher
that selects between them:

1. **Exact** (``_capacity_exact``): counts unique sliding-window
   subseqs without loss. Pure-ACGT windows are packed into 2 bits per
   base (4 bases per byte) and deduplicated; the rare windows holding
   IUPAC ambiguity codes are tracked in an exact string set. The two
   groups are disjoint, so their unique counts sum exactly. Memory is
   adaptive: mid-size clades deduplicate in memory with ``np.unique``,
   while supernodes whose key count would risk exhausting RAM switch to
   prefix-bucketed deduplication on disk (256 buckets by the first
   packed byte), bounding peak memory regardless of clade size. This
   replaced an earlier string-set implementation that could reach tens
   of gigabytes of RAM on viral heads like Caudoviricetes or the
   Viruses root.

2. **Approximate** (``_capacity_approximate``): uses a Bloom filter
   sized for ``BLOOM_EXPECTED_INSERTIONS`` distinct items at a
   ``BLOOM_FALSE_POSITIVE_RATE`` target false-positive rate. Memory
   is constant at ~12 MB regardless of clade size, making this the
   recommended mode on memory-constrained hosts such as WSL or
   laptops.

Both strategies support **early termination**: when ``max_useful``
is provided, the function returns as soon as the unique count
exceeds 5 times that target. The multiplier compensates for the
hard cap that downstream balancing will apply, avoiding wasted
work scanning genomes whose contribution will be discarded.

The module also exposes a process-local cache
(``_read_sequence_cached``) that wraps the LMDB reader to memoize
recently accessed sequences. The cache is bounded by
``_SEQUENCE_CACHE_MAX_ENTRIES`` with a simple FIFO eviction policy.

Typical usage::

    from taxotreeset.core.generation.capacity import compute_node_capacity

    capacity = compute_node_capacity(
        node=some_node,
        min_len=100,
        leaf_cache={},
        mode="approximate",
        max_useful=20_000,
    )
"""

import contextlib
import logging

from tqdm import tqdm

from taxotreeset.core.generation._bloom import (  # re-exported for callers/tests
    _bloom_get_bit as _bloom_get_bit,
    _bloom_set_bit as _bloom_set_bit,
    _build_bloom_filter,
    _consume_sequence_into_bloom as _consume_sequence_into_bloom,
    _consume_sequence_into_bloom_vectorized,
    _generate_bloom_hashes as _generate_bloom_hashes,
)
from taxotreeset.core.generation._encoding import (
    _BASES_PER_BYTE,
    _HASHED_PREFIX_BUCKETS,
    _encode_windows_2bit,
)
from taxotreeset.core.generation._gpu import (
    _detect_cuda_device_count,
    _gpu_encode_unique,
)
from taxotreeset.core.generation.constants import (
    BLOOM_EXPECTED_INSERTIONS,
    BLOOM_FALSE_POSITIVE_RATE,
)
from taxotreeset.dataset.utils import _read_single_sequence

logger = logging.getLogger("TaxoTreeSet.Core.Generation.Capacity")

_SEQUENCE_CACHE: dict[tuple[str, str], str] = {}
_SEQUENCE_CACHE_MAX_ENTRIES: int = 30_000

_EARLY_STOP_SAFETY_MULTIPLIER: int = 5
_PROGRESS_LOG_INTERVAL: int = 200

# Raw pure-ACGT keys accumulate until this many are pending, then a single
# np.unique compacts them. A large threshold keeps the number of (costly)
# unique passes tiny while bounding the peak by the pending-buffer size.
_HASHED_FLUSH_THRESHOLD: int = 8_000_000

# Above this many accumulated pure-ACGT keys, in-memory np.unique would risk
# exhausting RAM (the sort allocates a full copy). Such supernodes switch to
# prefix-bucketed deduplication on disk: keys are partitioned into 256 files
# by their first packed byte (the first four 2-bit bases), then each bucket is
# uniqued independently and the unique counts summed. Keys in different buckets
# can never be equal, so the sum is exact.
_HASHED_DISK_THRESHOLD: int = 30_000_000
# _HASHED_PREFIX_BUCKETS (256, by first packed byte) moved to ._encoding and
# imported above (shared with _gpu, cycle-free).

# The bottom-up capacity pass keeps several nodes' key sets alive at once
# (the recursion frontier), unlike the single-node _capacity_exact, so its
# spill-to-disk threshold is derived from available RAM rather than fixed.
# The budget is a fraction of available memory, divided by the per-key cost
# of a merge (the np.unique sort allocates copies) and an estimate of how
# many limit-size sets coexist on the frontier.
_BOTTOM_UP_RAM_FRACTION: float = 0.5
_BOTTOM_UP_MERGE_OVERHEAD: int = 3
_BOTTOM_UP_LIVE_SETS_ESTIMATE: int = 4

# Filename for the leaf-phase checkpoint written to spill_dir after Phase 1
# completes.  A subsequent run with the same spill_dir detects this file and
# skips already-computed leaves, making the leaf phase resumable across SLURM
# job boundaries.
_LEAF_CHECKPOINT_FNAME: str = "capacity_leaf_checkpoint.json"

# Minimum sequence length (bases) for the GPU path to be profitable.
# Below this the per-leaf overhead of H2D transfer and CUDA kernel launch
# exceeds the bandwidth gain.  Chosen conservatively so that even a low-end
# GPU (GTX 1650, 128 GB/s) breaks even vs a modern Xeon core.
_GPU_MIN_BASES: int = 500_000

# Process-local CUDA device index, set once by _leaf_pool_initializer when
# the worker process starts.  -1 means this worker has no GPU assignment.
_WORKER_GPU_DEVICE_ID: int = -1


def _resolve_bottom_up_threshold(key_bytes: int) -> int:
    """Derive the in-memory key ceiling for the bottom-up pass from RAM.

    Computes how many packed keys may be held in memory before a node spills
    to bucket files, sizing the ceiling to a fraction of currently available
    RAM divided by the per-key merge overhead and the number of limit-size
    sets expected to coexist on the recursion frontier. Falls back to the
    fixed supernode threshold when available memory cannot be measured.

    Args:
        key_bytes: Width of one packed key in bytes.

    Returns:
        The maximum number of in-memory keys before spilling to disk.
    """
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:
        return _HASHED_DISK_THRESHOLD
    budget = available * _BOTTOM_UP_RAM_FRACTION
    per_key_cost = (
        key_bytes * _BOTTOM_UP_MERGE_OVERHEAD * _BOTTOM_UP_LIVE_SETS_ESTIMATE
    )
    threshold = int(budget // per_key_cost)
    return max(threshold, 1)



def _read_sequence_cached(fasta_path: str, header_id: str) -> str:
    """Read a sequence from LMDB with a per-process in-memory cache.

    Wraps ``_read_single_sequence`` with a FIFO cache that bounds
    memory usage by ``_SEQUENCE_CACHE_MAX_ENTRIES``. When the cache
    reaches its ceiling, the oldest half of the entries are evicted
    in a single pass.

    Args:
        fasta_path: Path to the LMDB vault directory.
        header_id: Sequence header identifier (LMDB key).

    Returns:
        The decoded sequence string, or an empty string when the
        underlying read fails.
    """
    cache_key = (fasta_path, header_id)
    if cache_key in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[cache_key]

    sequence = _read_single_sequence(fasta_path, header_id) or ""

    if len(_SEQUENCE_CACHE) >= _SEQUENCE_CACHE_MAX_ENTRIES:
        existing_keys = list(_SEQUENCE_CACHE.keys())
        eviction_count = len(existing_keys) // 2
        for old_key in existing_keys[:eviction_count]:
            del _SEQUENCE_CACHE[old_key]

    _SEQUENCE_CACHE[cache_key] = sequence
    return sequence


class _NodeCapacityKeys:
    """Accumulator of a node's unique capacity keys.

    A node's capacity is the number of unique fixed-length subsequences
    found across all sequence leaves beneath it. This class accumulates
    those subsequences as deduplicated keys so that a parent node can be
    resolved by merging its children's accumulators, instead of rescanning
    every descendant leaf from scratch.

    Pure-ACGT subsequences are stored as packed 2-bit keys; subsequences
    containing IUPAC ambiguity codes are kept as exact strings. The two
    groups are disjoint by construction, so their counts add up without
    double counting.

    The packed keys are held in one of two interchangeable representations:

    * In memory, as a deduplicated array. This is the fast common path for
      clades whose keys fit in RAM.
    * On disk, partitioned into prefix-bucket files by each key's first
      byte, once a clade's keys would risk exhausting RAM. Keys in
      different buckets can never be equal, so each bucket is deduplicated
      independently and the counts summed, bounding peak memory regardless
      of clade size.

    The ambiguous subsequences, always few, stay in memory in both modes.
    """

    def __init__(
        self,
        pure_keys,
        ambiguous_count: int,
        key_bytes: int,
        bucket_paths=None,
        tmp_dir=None,
    ):
        """Store the deduplicated key groups in memory or on disk.

        Args:
            pure_keys: Deduplicated array of packed 2-bit keys (memory
                mode), or None when the keys live on disk.
            ambiguous_count: Count of unique ambiguous subsequences in
                this accumulator. Stored as an integer — not the strings
                themselves — so it does not accumulate unbounded memory
                when propagated up through tens of thousands of leaves.
            key_bytes: Width of one packed key in bytes.
            bucket_paths: The 256 prefix-bucket file paths (disk mode), or
                None when the keys live in memory.
            tmp_dir: Temporary directory holding the bucket files (disk
                mode), removed on release, or None in memory mode.
        """
        self._pure_keys = pure_keys
        self._ambiguous_count = ambiguous_count
        self._key_bytes = key_bytes
        self._bucket_paths = bucket_paths
        self._tmp_dir = tmp_dir

    @property
    def _on_disk(self) -> bool:
        """Return True when the packed keys are stored on disk."""
        return self._bucket_paths is not None

    @classmethod
    def from_sequence_leaf(
        cls,
        leaf,
        min_len: int,
        void_dtype,
        disk_threshold: int,
        use_cache: bool = True,
        spill_dir: str | None = None,
    ):
        """Build an accumulator from a single sequence leaf.

        Reads the leaf's sequence, enumerates its sliding windows, and
        deduplicates them into the two key groups. A leaf whose unique
        keys exceed the disk threshold spills to bucket files immediately,
        which matters for very large single genomes.

        Args:
            leaf: A sequence-rank leaf node with ``fasta_path`` and
                ``header_id`` attributes.
            min_len: Sliding window size in base pairs.
            void_dtype: The numpy void dtype sized to one packed key.
            disk_threshold: Maximum in-memory keys before spilling the
                leaf's keys to bucket files.
            use_cache: When False, bypasses the module-level sequence
                cache. Pass False for the bottom-up pass, where each leaf
                is read exactly once and caching only inflates RSS.

        Returns:
            A populated accumulator, empty when the leaf has no readable
            sequence or one shorter than ``min_len``.
        """
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        key_bytes = void_dtype.itemsize
        empty = np.empty((0,), dtype=void_dtype)
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            return cls(empty, 0, key_bytes)
        if use_cache:
            sequence = _read_sequence_cached(fasta_path, header_id)
        else:
            sequence = _read_single_sequence(fasta_path, header_id) or ""
        if not sequence or len(sequence) < min_len:
            return cls(empty, 0, key_bytes)
        seq_arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        n_windows = max(0, seq_arr.shape[0] - min_len + 1)
        if n_windows == 0:
            return cls(empty, 0, key_bytes)
        # _encode_windows_2bit allocates a codes array of shape
        # (n_windows, min_len) uint8. For large scaffolds this can reach
        # several GB and OOM before any disk-spillover threshold fires.
        # Cap individual batches to ~512 MiB of codes.
        _encode_batch = max(100_000, (512 << 20) // min_len)
        if n_windows > _encode_batch:
            # Delete the Python string before entering the chunked path.
            # seq_arr's backing bytes object (from sequence.encode) is kept
            # alive by seq_arr.base; only the string itself is freed here,
            # saving sequence_length bytes for the duration of the call.
            del sequence
            return cls._from_chunked_sequence(
                seq_arr, min_len, _encode_batch, void_dtype, key_bytes,
                spill_dir=spill_dir,
            )
        windows = sliding_window_view(seq_arr, min_len)
        keys, pure_mask = _encode_windows_2bit(windows, min_len)
        pure_keys = np.unique(keys) if keys.shape[0] else empty
        ambiguous: set = set()
        if not pure_mask.all():
            ambiguous = {
                sequence[i : i + min_len]
                for i in np.flatnonzero(~pure_mask).tolist()
            }
        if pure_keys.shape[0] >= disk_threshold:
            return cls._spilled_from_arrays(
                [pure_keys], len(ambiguous), key_bytes, spill_dir=spill_dir,
            )
        return cls(pure_keys, len(ambiguous), key_bytes)

    @classmethod
    def _from_chunked_sequence(
        cls,
        seq_arr,
        min_len: int,
        chunk_size: int,
        void_dtype,
        key_bytes: int,
        spill_dir: str | None = None,
    ):
        """Encode a large sequence in batches to avoid peak-memory OOM.

        ``_encode_windows_2bit`` creates a ``codes`` array of shape
        ``(n_windows, min_len)`` uint8. For scaffolds larger than ~5 MB
        (at min_len=100) this intermediate allocation reaches several GB and
        triggers an OOM before the disk-spillover threshold can react. This
        method processes the sequence in ``chunk_size``-window batches,
        flushing each batch's keys into prefix-bucketed disk files and
        deduplicating bucket-by-bucket at the end.

        Ambiguous windows (containing N or IUPAC codes) are intentionally
        not tracked here. Building a per-sequence set of ambiguous strings
        across all chunks allocates O(n_ambiguous × min_len) bytes for a
        single scaffold, which can reach several GiB for large chromosomes
        with many N-runs. Because ambiguous subsequences are filtered out
        during dataset generation, omitting them from the capacity count
        does not affect the usable training-sample estimate.

        Args:
            seq_arr: The full sequence encoded as a uint8 numpy array.
                The caller must ensure ``seq_arr``'s backing buffer remains
                alive for the duration of this call.
            min_len: Sliding window size in bases.
            chunk_size: Number of windows to encode per batch.
            void_dtype: Numpy void dtype sized to one packed key.
            key_bytes: Width of one packed key in bytes.

        Returns:
            A disk-mode accumulator owning a fresh bucket directory.
        """
        import os
        import tempfile

        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        tmp_dir = tempfile.mkdtemp(prefix="tts_capacity_", dir=spill_dir)
        bucket_paths = _bucket_writer_paths(tmp_dir)
        n_windows = seq_arr.shape[0] - min_len + 1

        with contextlib.ExitStack() as stack:
            bucket_files = [stack.enter_context(open(p, "ab")) for p in bucket_paths]
            for chunk_start in range(0, n_windows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_windows)
                # Include the trailing bases needed to form complete windows.
                sub = seq_arr[chunk_start: chunk_end + min_len - 1]
                windows_chunk = sliding_window_view(sub, min_len)
                keys, pure_mask = _encode_windows_2bit(windows_chunk, min_len)
                if keys.shape[0]:
                    _flush_keys_to_buckets(keys, bucket_files, key_bytes)
                # Ambiguous windows are intentionally skipped — see docstring.

        for path in bucket_paths:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            raw = np.fromfile(path, dtype=np.uint8)
            unique_bucket = np.unique(raw.view(void_dtype))
            unique_bucket.tofile(path)

        return cls(None, 0, key_bytes, bucket_paths, tmp_dir)

    @classmethod
    def _spilled_from_arrays(
        cls, pure_arrays: list, ambiguous_count: int, key_bytes: int,
        spill_dir: str | None = None,
    ):
        """Create a disk-mode accumulator from in-memory key arrays.

        Args:
            pure_arrays: Arrays of packed keys to write to buckets.
            ambiguous_count: Count of unique ambiguous subsequences.
            key_bytes: Width of one packed key in bytes.

        Returns:
            A disk-mode accumulator owning a fresh bucket directory.
        """
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="tts_capacity_", dir=spill_dir)
        bucket_paths = _bucket_writer_paths(tmp_dir)
        with contextlib.ExitStack() as stack:
            bucket_files = [stack.enter_context(open(p, "wb")) for p in bucket_paths]
            for keys in pure_arrays:
                if keys is not None and keys.shape[0]:
                    _flush_keys_to_buckets(keys, bucket_files, key_bytes)
        return cls(None, ambiguous_count, key_bytes, bucket_paths, tmp_dir)

    @classmethod
    def merge(cls, parts: list, void_dtype, disk_threshold: int):
        """Merge several accumulators into one, deduplicating across them.

        The union is exact and captures subsequences shared between
        siblings (conserved regions), so a parent's count is the size of
        the union, never the naive sum. When the combined pure keys stay
        small they are unioned in memory with ``np.unique``; when they
        would exhaust RAM the merge spills to bucket files, appending each
        child's keys (already-bucketed children are concatenated bucket by
        bucket, since the partition is identical).

        Args:
            parts: Accumulators to merge (typically a node's children).
            void_dtype: The numpy void dtype sized to one packed key.
            disk_threshold: Maximum combined in-memory keys before the
                merge spills to bucket files.

        Returns:
            A single accumulator holding the deduplicated union.
        """
        import numpy as np

        key_bytes = void_dtype.itemsize
        ambiguous_count = sum(p._ambiguous_count for p in parts)

        # A single child (e.g. a passthrough node) needs no reprocessing:
        # its set already is the union, so adopt it directly rather than
        # re-deduplicating millions of keys. Ownership of the keys moves
        # to the new accumulator, leaving the child empty so its later
        # release does not drop the adopted storage.
        if len(parts) == 1:
            return parts[0]._transfer_ownership(ambiguous_count)

        any_on_disk = any(part._on_disk for part in parts)
        memory_arrays = [
            part._pure_keys
            for part in parts
            if not part._on_disk and part._pure_keys.shape[0]
        ]
        memory_total = sum(arr.shape[0] for arr in memory_arrays)

        if not any_on_disk and memory_total < disk_threshold:
            if memory_arrays:
                pure_keys = np.unique(np.concatenate(memory_arrays))
            else:
                pure_keys = np.empty((0,), dtype=void_dtype)
            return cls(pure_keys, ambiguous_count, key_bytes)

        return cls._spilled_merge(parts, ambiguous_count, key_bytes)

    @classmethod
    def _spilled_merge(
        cls, parts: list, ambiguous_count: int, key_bytes: int,
        spill_dir: str | None = None,
    ):
        """Merge accumulators into a fresh disk-mode accumulator.

        In-memory children are flushed into the new buckets; disk-mode
        children have each of their bucket files appended to the matching
        new bucket, which is exact because both use the same first-byte
        partition.

        Args:
            parts: Accumulators to merge.
            ambiguous_count: Sum of per-part ambiguous counts.
            key_bytes: Width of one packed key in bytes.

        Returns:
            A disk-mode accumulator owning a fresh bucket directory.
        """
        import os
        import shutil
        import tempfile

        import numpy as np

        void_dtype = np.dtype((np.void, key_bytes))
        tmp_dir = tempfile.mkdtemp(prefix="tts_capacity_", dir=spill_dir)
        bucket_paths = _bucket_writer_paths(tmp_dir)

        # Write every child's keys into the new buckets one child at a time
        # (streaming, never holding more than one child in memory), then
        # deduplicate one bucket at a time. A node thus propagates an
        # already-unique set so dedup work neither repeats nor accumulates
        # as sets rise through the tree, while peak memory stays bounded by
        # a single bucket (~1/256 of the keys) rather than the whole set.
        with contextlib.ExitStack() as stack:
            bucket_files = [stack.enter_context(open(p, "ab")) for p in bucket_paths]
            for part in parts:
                if part._on_disk:
                    for index, child_bucket in enumerate(part._bucket_paths):
                        if os.path.exists(child_bucket) and os.path.getsize(
                            child_bucket
                        ):
                            with open(child_bucket, "rb") as source:
                                shutil.copyfileobj(source, bucket_files[index])
                elif (
                    part._pure_keys is not None and part._pure_keys.shape[0]
                ):
                    _flush_keys_to_buckets(
                        part._pure_keys, bucket_files, key_bytes
                    )

        for path in bucket_paths:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            raw = np.fromfile(path, dtype=np.uint8)
            unique_bucket = np.unique(raw.view(void_dtype))
            unique_bucket.tofile(path)
        return cls(None, ambiguous_count, key_bytes, bucket_paths, tmp_dir)

    def _inplace_extend(self, child: "_NodeCapacityKeys", key_bytes: int) -> None:
        """Append a child's keys directly into this accumulator's bucket files.

        Requires self to be in disk mode. The child is not modified or
        released by this call. This avoids the O(self) copy that
        ``_spilled_merge`` would incur when updating a growing accumulator
        one child at a time: only the child's data is written, not a copy
        of the existing accumulator.

        Args:
            child: Accumulator whose keys are appended to self's buckets.
            key_bytes: Width of one packed key in bytes.
        """
        import os
        import shutil

        assert self._on_disk, "_inplace_extend requires a disk-mode accumulator"

        self._ambiguous_count += child._ambiguous_count
        with contextlib.ExitStack() as stack:
            bucket_files = [
                stack.enter_context(open(p, "ab")) for p in self._bucket_paths
            ]
            if child._on_disk:
                for index, child_bucket in enumerate(child._bucket_paths):
                    if os.path.exists(child_bucket) and os.path.getsize(child_bucket):
                        with open(child_bucket, "rb") as source:
                            shutil.copyfileobj(source, bucket_files[index])
            elif child._pure_keys is not None and child._pure_keys.shape[0]:
                _flush_keys_to_buckets(child._pure_keys, bucket_files, key_bytes)

    def _transfer_ownership(self, ambiguous_count: int):
        """Move this accumulator's key storage into a new accumulator.

        Used when a parent has a single child: the child's set already is
        the union, so the parent adopts its in-memory array or its bucket
        files directly. This object is left empty afterwards so that the
        traversal's later call to its release does not drop the storage now
        owned by the returned accumulator.

        Args:
            ambiguous_count: Ambiguous subsequence count for the new
                accumulator (already summed by the caller).

        Returns:
            A new accumulator owning this one's pure-key storage.
        """
        adopted = _NodeCapacityKeys(
            self._pure_keys,
            ambiguous_count,
            self._key_bytes,
            self._bucket_paths,
            self._tmp_dir,
        )
        self._pure_keys = None
        self._bucket_paths = None
        self._tmp_dir = None
        self._ambiguous_count = 0
        return adopted

    def cardinality(self) -> int:
        """Return the number of unique subsequences accumulated.

        Returns:
            The count of unique pure keys plus unique ambiguous
            subsequences.
        """
        if self._on_disk:
            pure_count = _count_unique_bucketed_on_disk(
                self._bucket_paths, self._key_bytes
            )
        else:
            pure_count = int(self._pure_keys.shape[0])
        return pure_count + self._ambiguous_count

    def release(self) -> None:
        """Free the accumulated keys once the node has been resolved.

        Drops the in-memory array, or closes and removes the bucket files
        and their temporary directory in disk mode.
        """
        import os

        self._pure_keys = None
        self._ambiguous_count = 0
        if self._bucket_paths is not None:
            for path in self._bucket_paths:
                if os.path.exists(path):
                    os.remove(path)
            if self._tmp_dir is not None and os.path.isdir(self._tmp_dir):
                os.rmdir(self._tmp_dir)
            self._bucket_paths = None
            self._tmp_dir = None


def _save_leaf_checkpoint(
    leaf_accumulators: dict,
    spill_dir: str,
    min_len: int,
    void_dtype,
) -> None:
    """Persist leaf accumulators to disk so a resumed run can skip Phase 1.

    Disk-mode accumulators already live in spill_dir as bucket files; only
    their paths need to be recorded.  Memory-mode accumulators are serialised
    to ``capacity_leaf_{i}.bin`` files in spill_dir so they survive the
    process boundary.

    Args:
        leaf_accumulators: Mapping of str(leaf_name) to accumulator.
        spill_dir: Directory where checkpoint and auxiliary files are written.
        min_len: Window size used to produce the accumulators; stored in the
            checkpoint so a resume with a different min_len is detected and
            rejected.
        void_dtype: Numpy void dtype for packed keys.
    """
    import json
    import os

    checkpoint: dict = {"min_len": min_len, "leaves": {}}
    for idx, (name, acc) in enumerate(leaf_accumulators.items()):
        if acc._on_disk:
            checkpoint["leaves"][name] = {
                "mode": "disk",
                "buckets": acc._bucket_paths,
                "tmp_dir": acc._tmp_dir,
                "amb": acc._ambiguous_count,
                "kb": acc._key_bytes,
            }
        else:
            fpath = os.path.join(spill_dir, f"capacity_leaf_{idx}.bin")
            if acc._pure_keys is not None and acc._pure_keys.shape[0]:
                acc._pure_keys.tofile(fpath)
            else:
                open(fpath, "wb").close()
            checkpoint["leaves"][name] = {
                "mode": "memory",
                "file": fpath,
                "amb": acc._ambiguous_count,
                "kb": acc._key_bytes,
            }

    checkpoint_path = os.path.join(spill_dir, _LEAF_CHECKPOINT_FNAME)
    with open(checkpoint_path, "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh)


def _load_leaf_checkpoint(
    spill_dir: str,
    min_len: int,
    void_dtype,
) -> dict | None:
    """Load a previously saved leaf checkpoint if valid.

    Returns None when no checkpoint exists, the checkpoint was produced
    with a different min_len, or any referenced file is missing (which
    indicates the spill_dir was partially cleaned between runs).

    Args:
        spill_dir: Directory that may contain the checkpoint file.
        min_len: Window size expected in the checkpoint.
        void_dtype: Numpy void dtype for packed keys.

    Returns:
        Dict mapping str(leaf_name) to a reconstructed _NodeCapacityKeys,
        or None when the checkpoint cannot be used.
    """
    import json
    import os

    import numpy as np

    checkpoint_path = os.path.join(spill_dir, _LEAF_CHECKPOINT_FNAME)
    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, encoding="utf-8") as fh:
        checkpoint = json.load(fh)

    if checkpoint.get("min_len") != min_len:
        return None

    result: dict = {}
    for name, info in checkpoint["leaves"].items():
        kb = info["kb"]
        amb = info["amb"]
        if info["mode"] == "disk":
            buckets = info["buckets"]
            tmp_dir = info["tmp_dir"]
            if not all(os.path.exists(b) for b in buckets):
                return None
            result[name] = _NodeCapacityKeys(None, amb, kb, buckets, tmp_dir)
        else:
            fpath = info["file"]
            if not os.path.exists(fpath):
                return None
            raw = np.fromfile(fpath, dtype=void_dtype)
            result[name] = _NodeCapacityKeys(
                raw.copy() if raw.size else np.empty((0,), dtype=void_dtype),
                amb,
                kb,
            )

    return result or None


def _delete_leaf_checkpoint(spill_dir: str) -> None:
    """Remove the leaf checkpoint and its auxiliary bin files.

    Called after a successful Phase 2 so stale checkpoints do not
    interfere with future runs in the same spill_dir.
    """
    import os

    checkpoint_path = os.path.join(spill_dir, _LEAF_CHECKPOINT_FNAME)
    if not os.path.exists(checkpoint_path):
        return

    try:
        import json
        with open(checkpoint_path, encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        for info in checkpoint.get("leaves", {}).values():
            if info.get("mode") == "memory":
                fpath = info.get("file", "")
                if fpath and os.path.exists(fpath):
                    os.remove(fpath)
        os.remove(checkpoint_path)
    except Exception:
        pass


def _cleanup_spill_dirs(spill_dir: str) -> None:
    """Remove all tts_capacity_* directories inside spill_dir.

    Called in two situations:
    - At the start of a fresh run (no valid checkpoint found) to evict
      directories left behind by previous failed runs.
    - After a successful Phase 2 to remove directories created by the
      current run once their data has been merged and is no longer needed.

    Errors are logged as warnings and never propagate — a failure to clean
    up is unfortunate but must not abort an otherwise successful pipeline.
    """
    import glob
    import os
    import shutil

    pattern = os.path.join(spill_dir, "tts_capacity_*")
    removed = 0
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                removed += 1
            else:
                logging.getLogger("TaxoTreeSet.Core.Generation.Capacity").warning(
                    "[bottom-up] Could not fully remove spill dir %s", path
                )
        except OSError as exc:
            logging.getLogger("TaxoTreeSet.Core.Generation.Capacity").warning(
                "[bottom-up] Could not remove spill dir %s: %s", path, exc
            )
    if removed:
        logging.getLogger("TaxoTreeSet.Core.Generation.Capacity").info(
            "[bottom-up] Removed %d spill dir(s) from %s.", removed, spill_dir
        )


def _leaf_worker_task(
    fasta_path: str,
    header_id: str,
    leaf_name,
    min_len: int,
    key_bytes: int,
    disk_threshold: int,
    spill_dir: str | None,
) -> tuple:
    """Process one sequence leaf and return a serialisable result tuple.

    Top-level (non-nested) so the multiprocessing machinery can pickle it
    for both fork and spawn start methods.  Returns:
        (leaf_name, on_disk, data, ambiguous_count, key_bytes, tmp_dir)
    where ``data`` is a list of bucket paths when ``on_disk`` is True, or
    the raw bytes of the packed-key array when False.
    """
    import numpy as np

    void_dtype = np.dtype((np.void, key_bytes))

    class _LeafProxy:
        rank = "sequence"

    proxy = _LeafProxy()
    proxy.fasta_path = fasta_path
    proxy.header_id = header_id
    proxy.name = leaf_name

    acc = _NodeCapacityKeys.from_sequence_leaf(
        proxy, min_len, void_dtype, disk_threshold,
        use_cache=False, spill_dir=spill_dir,
    )

    if acc._on_disk:
        bucket_paths = acc._bucket_paths
        tmp_dir = acc._tmp_dir
        amb = acc._ambiguous_count
        # Null storage refs so the accumulator's release() won't delete the
        # spill files when the worker process exits.
        acc._bucket_paths = None
        acc._tmp_dir = None
        return (leaf_name, True, bucket_paths, amb, key_bytes, tmp_dir)

    raw = (
        acc._pure_keys.tobytes()
        if acc._pure_keys is not None and acc._pure_keys.shape[0]
        else b""
    )
    return (leaf_name, False, raw, acc._ambiguous_count, key_bytes, None)


def _reconstruct_leaf_keys(result: tuple, void_dtype) -> "_NodeCapacityKeys":
    """Rebuild a _NodeCapacityKeys from a _leaf_worker_task result."""
    import numpy as np

    _, on_disk, data, amb, kb, tmp_dir = result
    if on_disk:
        return _NodeCapacityKeys(None, amb, kb, data, tmp_dir)
    arr = (
        np.frombuffer(data, dtype=void_dtype).copy()
        if data
        else np.empty((0,), dtype=void_dtype)
    )
    return _NodeCapacityKeys(arr, amb, kb)


class _BottomUpCapacityComputer:
    """Two-phase bottom-up capacity computation for a taxonomic tree.

    Implements :func:`compute_all_capacities`. Phase 1 processes sequence
    leaves in parallel into packed-key accumulators, evicting the oldest
    in-memory accumulators to a single flat-bin file when the RAM budget is
    exceeded. Phase 2 folds the accumulators bottom-up by set-union, recording
    each internal node's capacity. State shared across the phases (the leaf
    accumulators, the flat-bin index, the running counters and the progress
    bar) lives on the instance.
    """

    def __init__(
        self,
        min_len: int,
        spill_dir: str | None,
        n_workers: int | None,
        n_gpu_workers: int | None,
    ) -> None:
        import logging
        import os

        import numpy as np
        import psutil

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) - 1)
        if n_gpu_workers is None:
            n_gpu_workers = _detect_cuda_device_count()
        n_gpu_workers = max(0, n_gpu_workers)

        self.min_len = min_len
        self.spill_dir = spill_dir
        self.n_workers = n_workers
        self.n_gpu_workers = n_gpu_workers
        self.key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
        self.void_dtype = np.dtype((np.void, self.key_bytes))
        self.disk_threshold = _resolve_bottom_up_threshold(self.key_bytes)
        self.logger = logging.getLogger("TaxoTreeSet.Core.Generation.Capacity")

        self.capacities: dict[str, int] = {}
        self.accumulators: dict[str, _NodeCapacityKeys] = {}
        # name → (bin_path, byte_offset, n_keys, amb_count)
        self.flat_bins: dict[str, tuple[str, int, int, int]] = {}
        self.flat_bin_file: str | None = None
        self.in_memory_key_count = 0
        # Reserve 25 % of currently available RAM for leaf accumulation.
        self.ram_budget_keys = max(
            1, int(psutil.virtual_memory().available * 0.25) // self.key_bytes
        )
        self.leaves_done = 0
        self.total_leaves = 0
        self.pbar = None

    # ── orchestration ────────────────────────────────────────────────────

    def run(self, tree_root) -> dict:
        """Compute every node's capacity and return the name → capacity map."""
        all_leaves = [
            n for n in tree_root.leaves if getattr(n, "rank", "") == "sequence"
        ]
        self.total_leaves = len(all_leaves)
        self._log_start()

        self._leaf_phase(all_leaves)

        root_set = self._merge_subtree(tree_root)
        root_set.release()
        self.logger.info(
            "[bottom-up] Done: %d nodes resolved.", len(self.capacities)
        )

        self._cleanup()
        return self.capacities

    def _log_start(self) -> None:
        import psutil

        gpu_info = (
            f", gpu_workers={self.n_gpu_workers}"
            f" (devices 0-{self.n_gpu_workers - 1})"
            if self.n_gpu_workers > 0
            else " (CPU-only)"
        )
        self.logger.info(
            "[bottom-up] Starting: %d sequence leaves, disk_threshold=%s keys "
            "(%.2f GiB), sys_avail=%.2f GiB, cpu_workers=%d%s",
            self.total_leaves,
            f"{self.disk_threshold:,}",
            self.disk_threshold * self.key_bytes / 2**30,
            psutil.virtual_memory().available / 2**30,
            self.n_workers,
            gpu_info,
        )

    def _empty_acc(self) -> "_NodeCapacityKeys":
        import numpy as np

        return _NodeCapacityKeys(
            np.empty((0,), dtype=self.void_dtype), 0, self.key_bytes
        )

    # ── Phase 1: parallel leaf processing ────────────────────────────────

    def _leaf_phase(self, all_leaves: list) -> None:
        """Resume any checkpoint, process the remaining leaves, checkpoint."""
        self._resume_from_checkpoint()
        self.leaves_done = len(self.accumulators)

        # Progress bar visible on any TTY or file stream (nohup included).
        # Dynamic miniters keeps the bar from flooding the log on fast datasets
        # while still updating at least every 60 seconds on slow ones.
        self.pbar = tqdm(
            total=self.total_leaves,
            initial=self.leaves_done,
            desc="Unique k-mer analysis",
            unit="leaf",
            dynamic_ncols=True,
            miniters=1,
            smoothing=0.05,
        )

        valid_leaves = self._select_valid_leaves(all_leaves)
        self._process_valid_leaves(valid_leaves)
        self.pbar.close()
        self._save_leaf_checkpoint_maybe(valid_leaves)

    def _resume_from_checkpoint(self) -> None:
        """Load a valid leaf checkpoint, or clear stale spill dirs on a fresh run."""
        if not self.spill_dir:
            return
        restored = _load_leaf_checkpoint(
            self.spill_dir, self.min_len, self.void_dtype
        )
        if restored:
            self.accumulators.update(restored)
            self.logger.info(
                "[bottom-up] Resuming from checkpoint: %d/%d leaves already computed.",
                len(restored), self.total_leaves,
            )
        else:
            # Fresh run with no valid checkpoint — evict any tts_capacity_*
            # directories left by previous failed runs before starting Phase 1.
            _cleanup_spill_dirs(self.spill_dir)

    def _select_valid_leaves(self, all_leaves: list) -> list:
        """Mark unusable leaves empty; return the leaves still to process."""
        for leaf in all_leaves:
            if not getattr(leaf, "fasta_path", "") or not getattr(leaf, "header_id", ""):
                self.accumulators[str(leaf.name)] = self._empty_acc()

        # Exclude leaves whose accumulators were restored from the checkpoint.
        return [
            leaf for leaf in all_leaves
            if getattr(leaf, "fasta_path", "") and getattr(leaf, "header_id", "")
            and str(leaf.name) not in self.accumulators
        ]

    def _process_valid_leaves(self, valid_leaves: list) -> None:
        """Run the leaf worker pool (or a sequential fallback) and record results."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        total_pool = self.n_workers + self.n_gpu_workers
        task_fn = (
            _leaf_worker_task_auto if self.n_gpu_workers > 0 else _leaf_worker_task
        )

        if total_pool > 1 and valid_leaves:
            # When GPU workers are present we must use the spawn start method so
            # each worker process initialises its own CUDA context from scratch.
            # Forking a process that has already touched CUDA state is undefined
            # behaviour in the CUDA driver.  The spawn overhead (~300 ms per
            # worker) is negligible compared to leaf processing times.
            pool_kwargs: dict = {"max_workers": total_pool}
            if self.n_gpu_workers > 0:
                spawn_ctx = multiprocessing.get_context("spawn")
                pool_kwargs["mp_context"] = spawn_ctx
                # Value and Lock must be created in the same spawn context so
                # they can be safely passed as initializer args to spawn workers.
                gpu_counter = spawn_ctx.Value("i", 0)
                gpu_lock = spawn_ctx.Lock()
                pool_kwargs["initializer"] = _leaf_pool_initializer
                pool_kwargs["initargs"] = (gpu_counter, gpu_lock, self.n_gpu_workers)

            with ProcessPoolExecutor(**pool_kwargs) as executor:
                future_map = {
                    executor.submit(
                        task_fn,
                        leaf.fasta_path, leaf.header_id, leaf.name,
                        self.min_len, self.key_bytes, self.disk_threshold,
                        self.spill_dir,
                    ): leaf.name
                    for leaf in valid_leaves
                }
                for future in as_completed(future_map):
                    try:
                        result = future.result()
                        # Release the IPC bytes immediately so completed futures
                        # don't accumulate ~500 KB each across 18 000+ leaves.
                        # CPython holds the result in Future._result until the
                        # Future is GC'd; as_completed keeps all futures alive
                        # internally, so we must clear it ourselves.
                        future._result = None
                        self._record(result)
                    except Exception as exc:
                        leaf_name = future_map[future]
                        self.logger.error(
                            "[bottom-up] leaf %s failed: %s — using empty accumulator",
                            leaf_name, exc,
                        )
                        self.accumulators[str(leaf_name)] = self._empty_acc()
                        self.leaves_done += 1
                        self.pbar.update(1)
        else:
            for leaf in valid_leaves:
                try:
                    result = _leaf_worker_task(
                        leaf.fasta_path, leaf.header_id, leaf.name,
                        self.min_len, self.key_bytes, self.disk_threshold,
                        self.spill_dir,
                    )
                    self._record(result)
                except Exception as exc:
                    self.logger.error(
                        "[bottom-up] leaf %s failed: %s — using empty accumulator",
                        leaf.name, exc,
                    )
                    self.accumulators[str(leaf.name)] = self._empty_acc()
                    self.leaves_done += 1
                    self.pbar.update(1)

    def _record(self, result: tuple) -> None:
        """Store a leaf result, evict if over the RAM budget, update progress."""
        import psutil

        leaf_name = result[0]
        acc = _reconstruct_leaf_keys(result, self.void_dtype)
        self.accumulators[str(leaf_name)] = acc
        # Snapshot cardinality and on-disk flag BEFORE potential eviction, which
        # nulls acc._pure_keys to allow GC — accessing it afterwards would fail.
        n_keys = acc.cardinality()
        acc_on_disk = acc._on_disk
        if not acc_on_disk and acc._pure_keys is not None:
            self.in_memory_key_count += acc._pure_keys.shape[0]
        if self.in_memory_key_count > self.ram_budget_keys:
            self._evict_to_flat_bins()
        self.leaves_done += 1
        self.pbar.update(1)
        if self.leaves_done % 50 == 0 or self.leaves_done == self.total_leaves:
            avail = psutil.virtual_memory().available / 2**30
            self.logger.info(
                "[bottom-up] leaves %d/%d  sys_avail=%.2f GiB  leaf=%s  keys=%s%s",
                self.leaves_done, self.total_leaves, avail,
                leaf_name,
                f"{n_keys:,}",
                "  [on-disk]" if acc_on_disk else "",
            )

    def _evict_to_flat_bins(self) -> None:
        """Evict ~50 % of in-memory leaf accumulators to a single flat-bin file.

        All evicted leaves are appended sequentially to ONE file per run,
        recording each leaf's byte offset in flat_bins.  This issues a
        single file-create + one large sequential write per eviction event
        instead of N small file creates, which is critical on NTFS/VHDX
        where per-file overhead limits effective throughput to ~5 MB/s
        with thousands of small files vs ~100 MB/s for sequential writes.
        """
        import os
        import tempfile

        if self.flat_bin_file is None:
            fd, path = tempfile.mkstemp(
                prefix="tts_capacity_flatbins_", dir=self.spill_dir, suffix=".bin"
            )
            os.close(fd)
            self.flat_bin_file = path
        bin_path = self.flat_bin_file
        target = max(1, self.in_memory_key_count // 2)
        evicted_keys = 0
        to_evict = []
        for name, acc in self.accumulators.items():
            if acc._on_disk or acc._pure_keys is None or acc._pure_keys.shape[0] == 0:
                continue
            to_evict.append((name, acc))
            evicted_keys += acc._pure_keys.shape[0]
            if evicted_keys >= target:
                break
        # One sequential write per eviction event — all leaves appended to the
        # same file.  f.tell() gives the byte offset before each array is written.
        with open(bin_path, "ab") as f:
            for name, acc in to_evict:
                offset = f.tell()
                n_keys = acc._pure_keys.shape[0]
                acc._pure_keys.tofile(f)
                self.flat_bins[name] = (bin_path, offset, n_keys, acc._ambiguous_count)
                del self.accumulators[name]
                acc._pure_keys = None
                acc._ambiguous_count = 0
        self.in_memory_key_count -= evicted_keys
        self.logger.info(
            "[bottom-up] Evicted %d leaves to flat-bin file (%.2f GiB); "
            "in-memory keys remaining: %d",
            len(to_evict), evicted_keys * self.key_bytes / 2**30,
            self.in_memory_key_count,
        )

    def _save_leaf_checkpoint_maybe(self, valid_leaves: list) -> None:
        """Save a leaf checkpoint, unless flat-bin eviction made it incomplete.

        Skipped when flat-bin eviction occurred: evicted leaves are absent from
        accumulators, so the checkpoint would be incomplete.  A crash during
        Phase 2 in that case requires re-running Phase 1 from scratch.
        """
        if self.spill_dir and valid_leaves and not self.flat_bins:
            _save_leaf_checkpoint(
                self.accumulators, self.spill_dir, self.min_len, self.void_dtype
            )
            self.logger.info(
                "[bottom-up] Leaf checkpoint saved (%d leaves) to %s",
                len(self.accumulators), self.spill_dir,
            )
        elif self.flat_bins:
            self.logger.info(
                "[bottom-up] Checkpoint skipped: %d leaves are in flat-bin storage "
                "(Phase 2 crash requires re-running Phase 1).",
                len(self.flat_bins),
            )

    # ── Phase 2: sequential bottom-up merge ──────────────────────────────

    def _merge_subtree(self, node) -> "_NodeCapacityKeys":
        """Resolve a node's key set by bottom-up union of its children.

        Leaf nodes load their accumulator from storage; internal nodes fold
        their children one at a time (bounding peak memory to one child's keys
        alongside the running accumulator), record their capacity, and return
        the merged accumulator.
        """
        import psutil

        if getattr(node, "rank", "") == "sequence":
            return self._resolve_leaf(str(node.name))

        if not node.children:
            return self._empty_acc()

        # Progressive accumulation: process one child at a time so that
        # at most one child's key array is live simultaneously alongside
        # the running accumulator. This bounds peak memory to
        # O(disk_threshold) regardless of how many children a node has,
        # instead of O(n_children × disk_threshold) with a batch merge.
        running = self._merge_subtree(node.children[0])
        for child_node in node.children[1:]:
            child_set = self._merge_subtree(child_node)
            running = self._merge_pair(running, child_set)

        cap = running.cardinality()
        self.capacities[str(node.name)] = cap
        avail = psutil.virtual_memory().available / 2**30
        self.logger.info(
            "[bottom-up] node %s (%s)  cap=%s  sys_avail=%.2f GiB%s",
            node.name, getattr(node, "rank", ""), f"{cap:,}", avail,
            "  [on-disk]" if running._on_disk else "",
        )
        return running

    def _resolve_leaf(self, leaf_name: str) -> "_NodeCapacityKeys":
        """Pop and return a leaf accumulator from memory or flat-bin storage."""
        import os

        import numpy as np

        acc = self.accumulators.pop(leaf_name, None)
        if acc is not None:
            return acc
        if leaf_name in self.flat_bins:
            bin_path, offset, n_keys, amb = self.flat_bins.pop(leaf_name)
            if os.path.exists(bin_path):
                with open(bin_path, "rb") as f:
                    f.seek(offset)
                    raw = np.fromfile(f, dtype=self.void_dtype, count=n_keys)
            else:
                raw = np.empty((0,), dtype=self.void_dtype)
            # The file is shared; deletion happens in bulk after Phase 2.
            return _NodeCapacityKeys(raw, amb, self.key_bytes)
        return self._empty_acc()

    def _merge_pair(
        self,
        running: "_NodeCapacityKeys",
        child_set: "_NodeCapacityKeys",
    ) -> "_NodeCapacityKeys":
        """Fold ``child_set`` into ``running`` by set-union, spilling as needed.

        Returns the merged accumulator; both inputs are released appropriately.
        """
        import numpy as np

        if running._on_disk:
            running._inplace_extend(child_set, self.key_bytes)
            child_set.release()
            return running
        if child_set._on_disk:
            spilled = _NodeCapacityKeys._spilled_from_arrays(
                [running._pure_keys],
                running._ambiguous_count,
                self.key_bytes,
                spill_dir=self.spill_dir,
            )
            running.release()
            spilled._inplace_extend(child_set, self.key_bytes)
            child_set.release()
            return spilled
        r_count = (
            running._pure_keys.shape[0]
            if running._pure_keys is not None else 0
        )
        c_count = (
            child_set._pure_keys.shape[0]
            if child_set._pure_keys is not None else 0
        )
        if r_count + c_count < self.disk_threshold:
            arrays = [
                a for a in [running._pure_keys, child_set._pure_keys]
                if a is not None and a.shape[0]
            ]
            pure_keys = (
                np.unique(np.concatenate(arrays))
                if arrays
                else np.empty((0,), dtype=self.void_dtype)
            )
            new_running = _NodeCapacityKeys(
                pure_keys,
                running._ambiguous_count + child_set._ambiguous_count,
                self.key_bytes,
            )
            running.release()
            child_set.release()
            return new_running
        spilled = _NodeCapacityKeys._spilled_merge(
            [running, child_set],
            running._ambiguous_count + child_set._ambiguous_count,
            self.key_bytes,
            spill_dir=self.spill_dir,
        )
        running.release()
        child_set.release()
        return spilled

    # ── cleanup ──────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove the flat-bin file, leaf checkpoint and stale spill dirs."""
        import os

        # Phase 2 completed successfully — flat-bin file, checkpoint, and spill
        # dirs are no longer needed.  Clean up all so the spill_dir stays lean.
        if self.flat_bin_file is not None:
            try:
                os.remove(self.flat_bin_file)
            except OSError:
                pass
            self.flat_bin_file = None
        if self.spill_dir:
            _delete_leaf_checkpoint(self.spill_dir)
            _cleanup_spill_dirs(self.spill_dir)


def compute_all_capacities(
    tree_root,
    min_len: int,
    spill_dir: str | None = None,
    n_workers: int | None = None,
    n_gpu_workers: int | None = None,
) -> dict:
    """Compute every node's capacity in one bottom-up pass.

    The pass has two phases:

    1. **Parallel leaf phase**: every sequence leaf is processed
       concurrently by a pool of worker processes.  Workers read from
       LMDB (safe for concurrent readers), enumerate sliding-window
       k-mers, and write packed-key spill files.  On an n-core host the
       wall time of this phase approaches max(leaf_time) instead of
       sum(leaf_times).

    2. **Sequential merge phase**: the pre-computed leaf accumulators
       are merged bottom-up.  Each internal node folds its children's
       key sets via set-union (``np.unique`` in memory or bucket-file
       append on disk), so shared subsequences — including conserved
       regions such as telomeric repeats — are counted once at every
       ancestor level.

    Args:
        tree_root: Root of the taxonomic tree.
        min_len: Sliding-window size in base pairs.
        spill_dir: Directory for temporary bucket files.  Defaults to
            the OS temp dir when None.  Set to a path on a large drive
            to avoid inflating the system-disk VHDX.
        n_workers: CPU worker processes for the leaf phase.  Defaults to
            ``os.cpu_count() - 1`` when None.  Pass 1 to disable CPU
            parallelism (useful for debugging or single-core hosts).
        n_gpu_workers: GPU worker processes for large leaves.  Each
            worker is pinned to one CUDA device (round-robin).  Defaults
            to auto-detect: uses all available CUDA devices when CuPy is
            installed, or 0 when CuPy is absent.  Pass 0 to disable GPU.

    Returns:
        Dictionary mapping each node's name (TaxID string) to its
        capacity (count of unique subseqs of length ``min_len``).
    """
    return _BottomUpCapacityComputer(
        min_len, spill_dir, n_workers, n_gpu_workers
    ).run(tree_root)


def compute_node_capacity(
    node,
    min_len: int,
    leaf_cache: dict,
    mode: str = "exact",
    max_useful: int | None = None,
) -> int:
    """Compute the biological capacity of a taxonomic node.

    Dispatches to the exact or approximate computation strategy
    depending on the ``mode`` argument. The node's capacity is the
    number of unique subsequences of length ``min_len`` extractable
    via sliding window from all sequence leaves descending from it.

    Args:
        node: bigtree Node whose capacity to compute.
        min_len: Sliding window size in base pairs.
        leaf_cache: Pre-computed cache mapping node taxid (string)
            to its list of sequence leaves. When the cache misses,
            the function falls back to scanning ``node.leaves``.
        mode: 'exact' for set-union computation, 'approximate' for
            Bloom-filter estimation.
        max_useful: Optional ceiling on useful capacity. When the
            accumulated unique count exceeds 5 times this value, the
            function returns early.

    Returns:
        The node's capacity (count or estimate of unique subseqs).

    Raises:
        ValueError: If ``mode`` is not 'exact' or 'approximate'.
    """
    if mode not in ("exact", "approximate"):
        raise ValueError(
            f"Unknown capacity mode: {mode!r} (expected 'exact' or 'approximate')."
        )
    
    all_seq_leaves = leaf_cache.get(str(node.name), [])
    if not all_seq_leaves:
        all_seq_leaves = [
            leaf for leaf in node.leaves if getattr(leaf, "rank", "") == "sequence"
        ]

    if not all_seq_leaves:
        return 0

    if mode == "exact":
        return _capacity_exact(all_seq_leaves, min_len, max_useful=max_useful)
    if mode == "approximate":
        return _capacity_approximate(all_seq_leaves, min_len, max_useful=max_useful)


def _bucket_writer_paths(tmp_dir: str) -> list:
    """Return the 256 prefix-bucket file paths under a temp directory.

    Args:
        tmp_dir: Directory to hold the per-bucket key files.

    Returns:
        List of 256 file paths, indexed by the first packed byte of a key.
    """
    import os

    return [
        os.path.join(tmp_dir, f"bucket_{i:03d}.bin")
        for i in range(_HASHED_PREFIX_BUCKETS)
    ]


def _count_unique_bucketed_on_disk(bucket_paths: list, key_bytes: int) -> int:
    """Count unique keys across prefix-bucket files, one bucket at a time.

    Each file holds raw packed keys (fixed ``key_bytes`` per key) whose first
    byte equals the bucket index. Keys in different buckets cannot be equal,
    so de-duplicating each bucket independently and summing the per-bucket
    unique counts yields the exact global unique count, while never holding
    more than one bucket in memory at once.

    Args:
        bucket_paths: The 256 per-bucket file paths.
        key_bytes: Width of each packed key in bytes.

    Returns:
        Total number of unique keys across all buckets.
    """
    import os

    import numpy as np

    void_dtype = np.dtype((np.void, key_bytes))
    total_unique = 0
    for path in bucket_paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        raw = np.fromfile(path, dtype=np.uint8)
        keys = raw.reshape(-1, key_bytes).view(void_dtype).reshape(-1)
        total_unique += int(np.unique(keys).shape[0])
        del raw, keys
    return total_unique


def _flush_keys_to_buckets(keys, bucket_files, key_bytes: int) -> None:
    """Append a batch of packed keys to their prefix-bucket files.

    Keys are partitioned by their first byte (the first four 2-bit bases),
    which is the prefix that defines the bucket. Writing in first-byte order
    groups the file appends so each bucket file is touched at most once per
    flush.

    Args:
        keys: (N,) void-typed array of packed keys to distribute.
        bucket_files: List of 256 open binary file handles, indexed by byte.
        key_bytes: Width of each packed key in bytes.
    """
    import numpy as np

    as_bytes = keys.view(np.uint8).reshape(-1, key_bytes)
    first_byte = as_bytes[:, 0]
    order = np.argsort(first_byte, kind="stable")
    sorted_bytes = as_bytes[order]
    sorted_first = first_byte[order]
    # Boundaries between distinct first-byte values in the sorted array.
    split_points = np.flatnonzero(np.diff(sorted_first)) + 1
    starts = np.concatenate([[0], split_points])
    ends = np.concatenate([split_points, [len(sorted_first)]])
    for start, end in zip(starts, ends):
        bucket = int(sorted_first[start])
        bucket_files[bucket].write(sorted_bytes[start:end].tobytes())


# 2-bit window encoding (_get_acgt_lut, _encode_windows_2bit, _BASES_PER_BYTE)
# moved to ._encoding and imported at the top of this module.


# GPU-accelerated kernels (_detect_cuda_device_count, _gpu_sliding_window_view,
# _gpu_unique_rows, _gpu_encode_unique) moved to ._gpu and imported at the top of
# this module. The CPU/GPU leaf workers below call _gpu_encode_unique.


def _leaf_pool_initializer(counter, lock, n_gpus: int) -> None:
    """Assign each worker a GPU device ID or mark it as CPU-only.

    Called once when a worker process starts.  The first ``n_gpus``
    workers to call this initializer receive CUDA device indices
    0 … n_gpus-1; subsequent workers are marked CPU-only
    (``_WORKER_GPU_DEVICE_ID = -1``).

    A shared counter protected by a lock guarantees exactly one worker
    per device regardless of process-start timing.
    """
    global _WORKER_GPU_DEVICE_ID
    with lock:
        idx = counter.value
        counter.value += 1
    device_id = idx if idx < n_gpus else -1
    _WORKER_GPU_DEVICE_ID = device_id
    if device_id >= 0:
        try:
            import cupy as cp
            cp.cuda.Device(device_id).use()
        except Exception:
            _WORKER_GPU_DEVICE_ID = -1


def _leaf_worker_task_auto(
    fasta_path: str,
    header_id: str,
    leaf_name,
    min_len: int,
    key_bytes: int,
    disk_threshold: int,
    spill_dir: str | None,
) -> tuple:
    """Dispatch a leaf to the GPU path or the CPU path.

    GPU path: used when this worker was assigned a CUDA device
    (``_WORKER_GPU_DEVICE_ID >= 0``) and the sequence is at least
    ``_GPU_MIN_BASES`` long.

    CPU path: used for short sequences, CPU-only workers, or when the
    GPU path raises any exception (OOM, driver error, …).  The returned
    tuple is identical in format to ``_leaf_worker_task``.
    """
    import numpy as np
    from taxotreeset.dataset.utils import _read_single_sequence

    device_id = _WORKER_GPU_DEVICE_ID

    if device_id >= 0:
        seq = _read_single_sequence(fasta_path, header_id) or ""
        if len(seq) >= _GPU_MIN_BASES:
            try:
                seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
                keys_void = _gpu_encode_unique(seq_arr, min_len, device_id, key_bytes)
                # For very large unique key sets spill to disk so the IPC pipe
                # doesn't carry gigabytes of raw bytes.  The threshold mirrors
                # the CPU path's disk_threshold parameter.
                if keys_void.shape[0] > disk_threshold:
                    acc = _NodeCapacityKeys._spilled_from_arrays(
                        [keys_void], 0, key_bytes, spill_dir=spill_dir,
                    )
                    bpaths = acc._bucket_paths
                    tdir = acc._tmp_dir
                    acc._bucket_paths = None
                    acc._tmp_dir = None
                    return (leaf_name, True, bpaths, 0, key_bytes, tdir)
                return (leaf_name, False, keys_void.tobytes(), 0, key_bytes, None)
            except Exception as exc:
                logger.warning(
                    "[bottom-up-gpu] leaf %s GPU encode failed (%s) — retrying on CPU",
                    leaf_name, exc,
                )

    return _leaf_worker_task(
        fasta_path, header_id, leaf_name, min_len, key_bytes, disk_threshold, spill_dir,
    )


def _iter_leaf_keys(seq_leaves, min_len):
    """Yield ``(pure_keys, ambiguous_windows)`` for each usable sequence leaf.

    Reads each leaf's sequence, skips leaves missing a source or shorter than
    ``min_len``, slices it into ``min_len`` windows, and encodes them. Pure-ACGT
    windows are returned as packed 2-bit keys; IUPAC-ambiguous windows are
    returned as their raw strings (the caller keeps them in an exact set).

    Args:
        seq_leaves: Sequence-rank leaf nodes to scan.
        min_len: Sliding window size in base pairs.

    Yields:
        ``(keys, ambiguous_windows)`` per usable leaf: a numpy array of packed
        pure-ACGT keys and a list of ambiguous window strings.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    for leaf in seq_leaves:
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            continue
        sequence = _read_sequence_cached(fasta_path, header_id)
        if not sequence or len(sequence) < min_len:
            continue
        seq_arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        windows = sliding_window_view(seq_arr, min_len)
        keys, pure_mask = _encode_windows_2bit(windows, min_len)
        ambiguous_windows: list[str] = []
        if not pure_mask.all():
            ambig_idx = np.flatnonzero(~pure_mask)
            ambiguous_windows = [
                sequence[i : i + min_len] for i in ambig_idx.tolist()
            ]
        yield keys, ambiguous_windows


def _compact_pure_keys(unique_pure, pending):
    """Merge the pending key chunks into ``unique_pure`` and deduplicate.

    Returns the deduplicated array and a reset ``(pending, pending_count)``.
    """
    if not pending:
        return unique_pure, [], 0
    import numpy as np

    return np.unique(np.concatenate([unique_pure, *pending])), [], 0


def _open_key_buckets(unique_pure, pending, key_bytes):
    """Open the on-disk prefix buckets and spill the in-memory keys into them.

    Args:
        unique_pure: Already-compacted keys held in memory.
        pending: Not-yet-compacted key chunks.
        key_bytes: Packed key width in bytes.

    Returns:
        ``(tmp_dir, bucket_files, bucket_paths)`` for the freshly opened buckets.
    """
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="tts_exact_")
    bucket_paths = _bucket_writer_paths(tmp_dir)
    bucket_files: list = []
    try:
        for path in bucket_paths:
            bucket_files.append(open(path, "wb"))
    except OSError:
        for handle in bucket_files:
            handle.close()
        raise
    if unique_pure.shape[0]:
        _flush_keys_to_buckets(unique_pure, bucket_files, key_bytes)
    for chunk in pending:
        _flush_keys_to_buckets(chunk, bucket_files, key_bytes)
    return tmp_dir, bucket_files, bucket_paths


def _cleanup_key_buckets(tmp_dir, bucket_files, bucket_paths) -> None:
    """Close any open bucket files and delete the temporary bucket directory."""
    import os

    if tmp_dir is None:
        return
    for handle in bucket_files or []:
        if not handle.closed:
            handle.close()
    for path in bucket_paths or []:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(tmp_dir)


def _capacity_exact(
    seq_leaves: list,
    min_len: int,
    max_useful: int | None = None,
) -> int:
    """Compute exact capacity via 2-bit packing of unique subseqs.

    Returns the exact count of unique ``min_len``-length sliding-window
    subsequences, but stores pure-ACGT windows as packed 2-bit keys instead of
    full strings, cutting memory severalfold. Rare windows containing IUPAC
    ambiguity codes are kept in an exact string set; the two groups are disjoint
    by construction, so their unique counts add up without double counting and
    the result is exact.

    Deduplication is adaptive in scale:

    * Small and mid-size nodes accumulate keys in memory and compact them
      with ``np.unique``. This is the fast common path.
    * Supernodes whose accumulated key count would make an in-memory sort
      risk exhausting RAM (above ``_HASHED_DISK_THRESHOLD``) switch to
      prefix-bucketed deduplication on disk: keys are partitioned into
      ``_HASHED_PREFIX_BUCKETS`` files by their first packed byte, then each
      bucket is uniqued independently and the counts summed. No more than one
      bucket is held in memory at once, so peak RAM stays bounded regardless
      of clade size, at the cost of temporary disk I/O.

    The per-leaf read/encode, the in-memory compaction, the disk-bucket
    activation, and the temp-file cleanup are delegated to ``_iter_leaf_keys``,
    ``_compact_pure_keys``, ``_open_key_buckets`` and ``_cleanup_key_buckets``.

    Args:
        seq_leaves: Sequence-rank leaf nodes to scan.
        min_len: Sliding window size in base pairs.
        max_useful: Optional early-stop target; scanning stops once the
            unique count provably exceeds ``max_useful`` times the
            ``_EARLY_STOP_SAFETY_MULTIPLIER``. Early stop applies only on the
            in-memory path.

    Returns:
        The total number of unique sliding-window subsequences.
    """
    import numpy as np

    early_stop_threshold = (
        max_useful * _EARLY_STOP_SAFETY_MULTIPLIER if max_useful else None
    )
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
    void_dtype = np.dtype((np.void, key_bytes))

    unique_pure = np.empty((0,), dtype=void_dtype)
    pending: list = []
    pending_count = 0
    seen_keys_total = 0  # cumulative pure keys observed (pre-dedup)
    ambiguous: set[str] = set()
    disk_mode = False
    tmp_dir = bucket_files = bucket_paths = None

    try:
        for keys, ambiguous_windows in _iter_leaf_keys(seq_leaves, min_len):
            ambiguous.update(ambiguous_windows)
            if not keys.shape[0]:
                continue
            seen_keys_total += keys.shape[0]
            if disk_mode:
                _flush_keys_to_buckets(keys, bucket_files, key_bytes)
                continue
            pending.append(keys)
            pending_count += keys.shape[0]

            # Switch to disk mode once the node proves to be a supernode.
            if seen_keys_total >= _HASHED_DISK_THRESHOLD:
                tmp_dir, bucket_files, bucket_paths = _open_key_buckets(
                    unique_pure, pending, key_bytes
                )
                unique_pure = np.empty((0,), dtype=void_dtype)
                pending, pending_count, disk_mode = [], 0, True
                continue

            if pending_count >= _HASHED_FLUSH_THRESHOLD:
                unique_pure, pending, pending_count = _compact_pure_keys(
                    unique_pure, pending
                )
                if early_stop_threshold and (
                    unique_pure.shape[0] + len(ambiguous) >= early_stop_threshold
                ):
                    return int(unique_pure.shape[0]) + len(ambiguous)

        if disk_mode:
            for handle in bucket_files:
                handle.close()
            return _count_unique_bucketed_on_disk(bucket_paths, key_bytes) + len(
                ambiguous
            )

        unique_pure, pending, pending_count = _compact_pure_keys(
            unique_pure, pending
        )
        return int(unique_pure.shape[0]) + len(ambiguous)
    finally:
        _cleanup_key_buckets(tmp_dir, bucket_files, bucket_paths)


def _capacity_approximate(
    seq_leaves: list,
    min_len: int,
    max_useful: int | None = None,
) -> int:
    """Estimate capacity using a Bloom filter with ~1% false-positive rate.

    Sizes the Bloom filter via the classic Bloom-Floyd formula to
    absorb up to ``BLOOM_EXPECTED_INSERTIONS`` distinct items at the
    target rate ``BLOOM_FALSE_POSITIVE_RATE``. Each subseq is hashed
    via the double-hashing scheme (h1 + i * h2) mod m.

    Memory is constant at roughly 12 megabytes regardless of clade
    size. Recommended on memory-constrained hosts (WSL, laptops).

    Args:
        seq_leaves: List of sequence leaf nodes.
        min_len: Sliding window size.
        max_useful: Optional ceiling for early termination.

    Returns:
        Estimated count of unique subsequences, within ~1% of exact
        for clades smaller than the filter's expected-insertion
        budget. Larger clades will accumulate more false positives.
    """
    bit_array, hash_count = _build_bloom_filter(
        expected_insertions=BLOOM_EXPECTED_INSERTIONS,
        false_positive_rate=BLOOM_FALSE_POSITIVE_RATE,
    )
    bit_array_size = len(bit_array) * 8
    unique_count = 0
    early_stop_threshold = (
        max_useful * _EARLY_STOP_SAFETY_MULTIPLIER if max_useful else None
    )
    total_leaves = len(seq_leaves)

    for processed_count, leaf in enumerate(seq_leaves, start=1):
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            continue

        sequence = _read_sequence_cached(fasta_path, header_id)
        if not sequence or len(sequence) < min_len:
            continue

        unique_count += _consume_sequence_into_bloom_vectorized(
            sequence=sequence,
            min_len=min_len,
            bit_array=bit_array,
            bit_array_size=bit_array_size,
            hash_count=hash_count,
        )

        if processed_count % _PROGRESS_LOG_INTERVAL == 0:
            logger.info(
                f"  [CAPACITY-PROGRESS] {processed_count}/{total_leaves} "
                f"leaves, unique count: {unique_count:,}"
            )

        if early_stop_threshold and unique_count >= early_stop_threshold:
            logger.info(
                f"  [CAPACITY-EARLY-STOP] after {processed_count}/"
                f"{total_leaves} leaves: count={unique_count:,} >= "
                f"{early_stop_threshold:,} threshold (cap will apply)."
            )
            break

    return unique_count


# Bloom-filter primitives (_build_bloom_filter, _consume_sequence_into_bloom[_vectorized],
# _bloom_set_bit, _bloom_get_bit, _generate_bloom_hashes) moved to ._bloom and
# imported at the top of this module. _capacity_approximate (above) calls them.
