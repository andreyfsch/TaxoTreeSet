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

import logging
import math

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

# Exact-hashed encoding: each ACGT base packs into 2 bits, 4 bases per byte.
# The packed key length in bytes is derived from min_len at call time as
# ceil(min_len / _BASES_PER_BYTE), so the method holds for any window size.
_BASES_PER_BYTE: int = 4
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
_HASHED_PREFIX_BUCKETS: int = 256


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
    """Accumulator of a node's unique capacity keys, held in memory.

    A node's capacity is the number of unique fixed-length subsequences
    found across all sequence leaves beneath it. This class accumulates
    those subsequences as deduplicated keys so that a parent node can be
    resolved by merging its children's accumulators, instead of rescanning
    every descendant leaf from scratch.

    Pure-ACGT subsequences are stored as packed 2-bit keys; subsequences
    containing IUPAC ambiguity codes are kept as exact strings. The two
    groups are disjoint by construction, so their counts add up without
    double counting.

    This in-memory representation suits clades whose key count fits in
    RAM. Larger clades will spill to disk in a later change.
    """

    def __init__(self, pure_keys, ambiguous_subseqs: set):
        """Store the deduplicated key groups.

        Args:
            pure_keys: A deduplicated numpy array of packed 2-bit keys.
            ambiguous_subseqs: A set of exact ambiguous subsequences.
        """
        self._pure_keys = pure_keys
        self._ambiguous_subseqs = ambiguous_subseqs

    @classmethod
    def from_sequence_leaf(cls, leaf, min_len: int, void_dtype):
        """Build an accumulator from a single sequence leaf.

        Reads the leaf's sequence, enumerates its sliding windows, and
        deduplicates them into the two key groups.

        Args:
            leaf: A sequence-rank leaf node with ``fasta_path`` and
                ``header_id`` attributes.
            min_len: Sliding window size in base pairs.
            void_dtype: The numpy void dtype sized to one packed key.

        Returns:
            A populated accumulator, empty when the leaf has no readable
            sequence or one shorter than ``min_len``.
        """
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        empty = np.empty((0,), dtype=void_dtype)
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            return cls(empty, set())
        sequence = _read_sequence_cached(fasta_path, header_id)
        if not sequence or len(sequence) < min_len:
            return cls(empty, set())
        seq_arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        windows = sliding_window_view(seq_arr, min_len)
        keys, pure_mask = _encode_windows_2bit(windows, min_len)
        pure_keys = np.unique(keys) if keys.shape[0] else empty
        ambiguous: set = set()
        if not pure_mask.all():
            ambiguous = {
                sequence[i : i + min_len]
                for i in np.flatnonzero(~pure_mask).tolist()
            }
        return cls(pure_keys, ambiguous)

    @classmethod
    def merge(cls, parts: list, void_dtype):
        """Merge several accumulators into one, deduplicating across them.

        The union is exact: concatenated pure keys are reduced with
        ``np.unique`` and ambiguous subsequences are unioned as sets. This
        captures subsequences shared between siblings (conserved regions),
        so a parent's count is the size of the union, never the naive sum.

        Args:
            parts: Accumulators to merge (typically a node's children).
            void_dtype: The numpy void dtype sized to one packed key.

        Returns:
            A single accumulator holding the deduplicated union.
        """
        import numpy as np

        pure_arrays = [
            part._pure_keys for part in parts if part._pure_keys.shape[0]
        ]
        if pure_arrays:
            pure_keys = np.unique(np.concatenate(pure_arrays))
        else:
            pure_keys = np.empty((0,), dtype=void_dtype)
        ambiguous: set = set()
        for part in parts:
            ambiguous |= part._ambiguous_subseqs
        return cls(pure_keys, ambiguous)

    def cardinality(self) -> int:
        """Return the number of unique subsequences accumulated.

        Returns:
            The count of unique pure keys plus unique ambiguous
            subsequences.
        """
        return int(self._pure_keys.shape[0]) + len(self._ambiguous_subseqs)

    def release(self) -> None:
        """Drop the accumulated keys to free memory once no longer needed."""
        self._pure_keys = None
        self._ambiguous_subseqs = None


def compute_all_capacities(tree_root, min_len: int) -> dict:
    """Compute every node's capacity in one bottom-up pass.

    Walks the tree in post-order: each sequence leaf is scanned once to
    build its set of unique subsequence keys, and each internal node is
    resolved by merging its children's sets rather than rescanning their
    leaves. A node's capacity is the size of the merged set, which
    deduplicates subsequences shared between siblings.

    This replaces resolving capacities top-down and independently per node,
    where every leaf was rescanned once for each of its ancestors. The
    bottom-up pass scans each leaf exactly once.

    Args:
        tree_root: Root node of the taxonomic tree to resolve.
        min_len: Sliding window size in base pairs. Also the minimum
            subsequence length, so changing it changes every capacity.

    Returns:
        Dictionary mapping each node's name (TaxID string) to its capacity.
    """
    import numpy as np

    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
    void_dtype = np.dtype((np.void, key_bytes))
    capacities: dict[str, int] = {}

    def _resolve(node) -> _NodeCapacityKeys:
        if getattr(node, "rank", "") == "sequence":
            return _NodeCapacityKeys.from_sequence_leaf(
                node, min_len, void_dtype
            )
        child_sets = [_resolve(child) for child in node.children]
        merged = _NodeCapacityKeys.merge(child_sets, void_dtype)
        for child_set in child_sets:
            child_set.release()
        capacities[str(node.name)] = merged.cardinality()
        return merged

    root_set = _resolve(tree_root)
    root_set.release()
    return capacities


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


_ACGT_LUT_CACHE = None


def _get_acgt_lut():
    """Return the cached 256-entry ACGT-to-2-bit lookup table, building it once.

    Non-ACGT bytes map to the sentinel 255, which marks a window as
    ambiguous (containing IUPAC ambiguity codes or N) so it is routed to
    the exact string-set path rather than the 2-bit-packed path. The table
    is built lazily on first use because numpy is imported lazily within
    this module.

    Returns:
        numpy uint8 array of length 256; index by ASCII byte value.
    """
    global _ACGT_LUT_CACHE
    if _ACGT_LUT_CACHE is None:
        import numpy as np

        lut = np.full(256, 255, dtype=np.uint8)
        for code, base in enumerate(b"ACGT"):
            lut[base] = code
        _ACGT_LUT_CACHE = lut
    return _ACGT_LUT_CACHE


def _encode_windows_2bit(windows, min_len: int):
    """Pack pure-ACGT sliding windows into fixed-length 2-bit byte keys.

    Each base occupies 2 bits and four bases pack into one byte, so a
    window of ``min_len`` bases packs into ceil(min_len / 4) bytes. The
    key length is derived from ``min_len`` here, so the encoding is valid
    for any window size, not just the default of 100.

    Windows containing any non-ACGT symbol are not encoded; the returned
    boolean mask marks which windows were pure ACGT. Ambiguous windows are
    handled separately by the caller via an exact string set, keeping the
    two domains disjoint so their unique counts sum without double counting.

    Args:
        windows: (N, min_len) uint8 array of ASCII base values, typically
            a ``sliding_window_view`` over one sequence.
        min_len: Window size in bases; determines the packed key length.

    Returns:
        Two-tuple ``(packed_keys, pure_mask)``:
            - packed_keys: (M,) array of void-typed keys of width
              ceil(min_len / 4) bytes, one per pure-ACGT window (M <= N).
            - pure_mask: (N,) boolean array, True where the window was
              pure ACGT.
    """
    import numpy as np

    codes = _get_acgt_lut()[windows]
    pure_mask = np.all(codes != np.uint8(255), axis=1)
    pure = codes[pure_mask]
    n_pure = pure.shape[0]
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
    if n_pure == 0:
        empty = np.empty((0,), dtype=np.dtype((np.void, key_bytes)))
        return empty, pure_mask

    # Pad the base axis up to a multiple of 4 with zeros so it reshapes
    # cleanly into groups of four 2-bit codes. The padding is identical for
    # every window, so it cannot introduce a spurious collision.
    pad = (-min_len) % _BASES_PER_BYTE
    if pad:
        pure = np.concatenate(
            [pure, np.zeros((n_pure, pad), dtype=np.uint8)], axis=1
        )
    groups = pure.reshape(n_pure, key_bytes, _BASES_PER_BYTE)
    packed = (
        groups[:, :, 0]
        | (groups[:, :, 1] << np.uint8(2))
        | (groups[:, :, 2] << np.uint8(4))
        | (groups[:, :, 3] << np.uint8(6))
    ).astype(np.uint8)
    keys = np.ascontiguousarray(packed).view(
        np.dtype((np.void, key_bytes))
    ).reshape(n_pure)
    return keys, pure_mask


# noqa rationale: intrinsic complexity from the adaptive design --
# in-memory np.unique for mid-size clades vs. prefix-bucketed on-disk
# deduplication for supernodes, plus two disjoint encoding paths
# (pure-ACGT 2-bit packing and an exact string set for IUPAC-ambiguous
# windows). Validated bit-exact against the former string-set
# implementation. Refactoring is deferred until an automated test suite
# guarantees behavioral equivalence; see docs/TODOs/complexity_refactor.md.
def _capacity_exact(  # noqa: C901
    seq_leaves: list,
    min_len: int,
    max_useful: int | None = None,
) -> int:
    """Compute exact capacity via 2-bit packing of unique subseqs.

    Functionally identical to ``_capacity_exact`` -- it returns the exact
    count of unique ``min_len``-length sliding-window subsequences -- but
    stores pure-ACGT windows as packed 2-bit keys instead of full strings,
    cutting memory severalfold. Rare windows containing IUPAC ambiguity
    codes are kept in an exact string set; the two groups are disjoint by
    construction, so their unique counts add up without double counting and
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
    import os
    import tempfile

    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

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

    # Disk-mode state, lazily activated when the node proves too large.
    disk_mode = False
    tmp_dir = None
    bucket_files = None
    bucket_paths = None

    def _compact(unique_pure, pending):
        if not pending:
            return unique_pure, [], 0
        combined = np.concatenate([unique_pure, *pending])
        return np.unique(combined), [], 0

    def _activate_disk_mode(unique_pure, pending):
        nonlocal tmp_dir, bucket_files, bucket_paths
        tmp_dir = tempfile.mkdtemp(prefix="tts_exact_")
        bucket_paths = _bucket_writer_paths(tmp_dir)
        bucket_files = [open(p, "wb") for p in bucket_paths]
        # Spill whatever is already in memory to the buckets.
        if unique_pure.shape[0]:
            _flush_keys_to_buckets(unique_pure, bucket_files, key_bytes)
        for chunk in pending:
            _flush_keys_to_buckets(chunk, bucket_files, key_bytes)

    try:
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
            if keys.shape[0]:
                seen_keys_total += keys.shape[0]
                if disk_mode:
                    _flush_keys_to_buckets(keys, bucket_files, key_bytes)
                else:
                    pending.append(keys)
                    pending_count += keys.shape[0]
            if not pure_mask.all():
                ambig_idx = np.flatnonzero(~pure_mask)
                ambiguous.update(
                    sequence[i : i + min_len] for i in ambig_idx.tolist()
                )

            # Switch to disk mode once the node proves to be a supernode.
            if not disk_mode and seen_keys_total >= _HASHED_DISK_THRESHOLD:
                _activate_disk_mode(unique_pure, pending)
                unique_pure = np.empty((0,), dtype=void_dtype)
                pending = []
                pending_count = 0
                disk_mode = True
                continue

            if not disk_mode and pending_count >= _HASHED_FLUSH_THRESHOLD:
                unique_pure, pending, pending_count = _compact(
                    unique_pure, pending
                )
                if early_stop_threshold and (
                    unique_pure.shape[0] + len(ambiguous)
                    >= early_stop_threshold
                ):
                    return int(unique_pure.shape[0]) + len(ambiguous)

        if disk_mode:
            for handle in bucket_files:
                handle.close()
            unique_count = _count_unique_bucketed_on_disk(
                bucket_paths, key_bytes
            )
            return unique_count + len(ambiguous)

        unique_pure, pending, pending_count = _compact(unique_pure, pending)
        return int(unique_pure.shape[0]) + len(ambiguous)
    finally:
        if tmp_dir is not None:
            if bucket_files is not None:
                for handle in bucket_files:
                    if not handle.closed:
                        handle.close()
            for path in bucket_paths:
                if os.path.exists(path):
                    os.remove(path)
            os.rmdir(tmp_dir)


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


def _build_bloom_filter(
    expected_insertions: int,
    false_positive_rate: float,
) -> tuple[bytearray, int]:
    """Size and allocate a Bloom filter for given target parameters.

    Computes the optimal bit array size m and hash count k from the
    expected insertion count n and false-positive rate p::

        m = -n * ln(p) / (ln(2)^2)
        k = (m / n) * ln(2)

    Args:
        expected_insertions: Maximum number of distinct items.
        false_positive_rate: Target false-positive probability.

    Returns:
        Two-tuple ``(bit_array, hash_count)``:
            - bit_array: bytearray of size ceil(m/8) bytes.
            - hash_count: optimal number of hash functions k.
    """
    bit_count = int(
        -expected_insertions * math.log(false_positive_rate) / (math.log(2) ** 2)
    )
    hash_count = max(1, int((bit_count / expected_insertions) * math.log(2)))
    bit_array = bytearray((bit_count + 7) // 8)
    return bit_array, hash_count


def _consume_sequence_into_bloom(
    sequence: str,
    min_len: int,
    bit_array: bytearray,
    bit_array_size: int,
    hash_count: int,
) -> int:
    """Reference implementation of Bloom insertion, kept for debugging.

    No longer called by the production pipeline; ``_capacity_approximate``
    routes through ``_consume_sequence_into_bloom_vectorized``, which is
    7-10x faster on real viral sequences. This sequential implementation
    is retained because:

    1. It serves as the readable specification against which the
       vectorized implementation is validated.
    2. It produces a bit-identical bit_array for the same inputs, so
       any future regression in the vectorized path can be caught by
       comparing against this baseline.
    3. Its semantics are exact ("sequential snapshot"): each window
       sees the bit array updated by prior windows. The vectorized
       implementation processes chunks of 2048 windows, which can
       lead to a ~0.005% over-count when duplicate k-mers appear
       within the same chunk. The bit array remains identical
       regardless because bit-set is idempotent.

    Args:
        sequence: DNA sequence to scan.
        min_len: Sliding window size.
        bit_array: Bloom filter bit array (mutated in place).
        bit_array_size: Total bit count of the array.
        hash_count: Number of hash functions to apply per item.

    Returns:
        Exact count of items not already present in the filter when
        scanned (the increment in unique count).
    """
    new_items_count = 0
    sequence_length = len(sequence)

    for window_start in range(sequence_length - min_len + 1):
        subseq_bytes = sequence[window_start : window_start + min_len].encode("ascii")
        bit_positions = list(
            _generate_bloom_hashes(subseq_bytes, bit_array_size, hash_count)
        )

        already_present = all(
            _bloom_get_bit(bit_array, position) for position in bit_positions
        )
        if not already_present:
            new_items_count += 1
            for position in bit_positions:
                _bloom_set_bit(bit_array, position)

    return new_items_count



def _consume_sequence_into_bloom_vectorized(
    sequence: str,
    min_len: int,
    bit_array: bytearray,
    bit_array_size: int,
    hash_count: int,
    chunk_size: int = 2048,
) -> int:
    """Vectorized batch insertion of sliding-window subseqs into a Bloom filter.

    Functionally equivalent to ``_consume_sequence_into_bloom`` but ~20-50x
    faster on long sequences. Uses numpy to compute all hash positions in
    parallel, process bit reads/writes in batch, and avoid the Python loop.

    Operates on chunks of ``chunk_size`` windows at a time to bound the
    error introduced by snapshot semantics: within a chunk, bit reads
    happen before any writes, so duplicate windows in the same chunk
    each count as new (vs. sequential semantics where only the first
    counts). The chunk size keeps this drift small relative to the
    Bloom filter's intrinsic ~1% false-positive rate.

    Args:
        sequence: DNA sequence to scan.
        min_len: Sliding window size.
        bit_array: Bloom filter bit array (mutated in place via numpy
            buffer view).
        bit_array_size: Total bit count of the array.
        hash_count: Number of hash functions per item.
        chunk_size: Number of windows processed per batch. Smaller
            chunks reduce snapshot drift at the cost of marginal
            speed. The default 2048 yields drift < 0.1% in practice.

    Returns:
        Approximate count of items not already present in the filter.
        May slightly overestimate vs the sequential implementation when
        the sequence contains duplicate k-mers within the same chunk.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    seq_bytes = sequence.encode("ascii")
    seq_len = len(seq_bytes)
    if seq_len < min_len:
        return 0

    seq_arr = np.frombuffer(seq_bytes, dtype=np.uint8)
    windows = sliding_window_view(seq_arr, min_len)
    n_windows = windows.shape[0]

    # numpy-aliased view of the Bloom bit array (mutations reflect back)
    bit_view = np.frombuffer(bit_array, dtype=np.uint8)
    k_offsets = np.arange(hash_count, dtype=np.uint64)
    bit_array_size_u64 = np.uint64(bit_array_size)

    new_items_total = 0

    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_windows)
        chunk = windows[chunk_start:chunk_end]

        # Extract h1 (first 8 bytes) and h2 (last 8 bytes) of each window
        # If min_len < 8, pad windows with zeros to enable view as uint64
        if min_len >= 8:
            h1_bytes = np.ascontiguousarray(chunk[:, :8])
            h2_bytes = np.ascontiguousarray(chunk[:, -8:])
        else:
            padded = np.zeros((chunk.shape[0], 8), dtype=np.uint8)
            padded[:, :min_len] = chunk
            h1_bytes = padded
            h2_bytes = padded

        h1 = h1_bytes.view(np.uint64).reshape(-1) & np.uint64(0x7FFFFFFFFFFFFFFF)
        h2 = h2_bytes.view(np.uint64).reshape(-1) & np.uint64(0x7FFFFFFFFFFFFFFF)

        # positions[i, k] = (h1[i] + k * h2[i]) % bit_array_size
        # Apply mod m BEFORE the product to avoid uint64 overflow, which
        # would silently diverge from the sequential implementation's
        # arbitrary-precision Python integer semantics. Since
        # m < 2^27 in practice, h1_mod * hash_count + h2_mod stays
        # well below 2^32 with hash_count = 6.
        h1_mod = h1 % bit_array_size_u64
        h2_mod = h2 % bit_array_size_u64
        positions = (h1_mod[:, None] + k_offsets[None, :] * h2_mod[:, None]) % bit_array_size_u64

        byte_idx = (positions >> np.uint64(3)).astype(np.int64)  # // 8
        bit_offset = (positions & np.uint64(7)).astype(np.uint8)  # % 8

        # already_present[i] = all(bit_array[byte_idx[i, k]] bit bit_offset[i, k] set)
        bit_masks = np.uint8(1) << bit_offset
        existing = bit_view[byte_idx] & bit_masks
        already_present = (existing == bit_masks).all(axis=1)

        new_items_total += int((~already_present).sum())

        # Set all bits for this chunk (idempotent on already-set bits)
        np.bitwise_or.at(bit_view, byte_idx.ravel(), bit_masks.ravel())

    return new_items_total


def _bloom_set_bit(bit_array: bytearray, index: int) -> None:
    """Set the bit at the given index in the bit array.

    Args:
        bit_array: Backing bytearray.
        index: Zero-based bit position.
    """
    bit_array[index // 8] |= 1 << (index % 8)


def _bloom_get_bit(bit_array: bytearray, index: int) -> int:
    """Get the bit value at the given index in the bit array.

    Args:
        bit_array: Backing bytearray.
        index: Zero-based bit position.

    Returns:
        0 if the bit is unset, non-zero if it is set.
    """
    return (bit_array[index // 8] >> (index % 8)) & 1


def _generate_bloom_hashes(
    item_bytes: bytes,
    bit_array_size: int,
    hash_count: int,
):
    """Yield k hash positions for an item using double-hashing.

    Combines two 64-bit pseudo-random values extracted from the
    item's byte representation as ``(h1 + i * h2) mod m`` to obtain
    k positions cheaply. This is a standard Bloom filter optimization
    that approximates k independent hash functions.

    Args:
        item_bytes: Byte representation of the item.
        bit_array_size: Modulus for the position projection (total
            bit count of the filter).
        hash_count: Number of positions to yield.

    Yields:
        Sequence of integer positions in [0, bit_array_size).
    """
    h1 = int.from_bytes(item_bytes[:8].ljust(8, b"\x00"), "little") & 0x7FFFFFFFFFFFFFFF
    h2 = (
        int.from_bytes(item_bytes[-8:].ljust(8, b"\x00"), "little") & 0x7FFFFFFFFFFFFFFF
    )
    for index in range(hash_count):
        yield (h1 + index * h2) % bit_array_size
