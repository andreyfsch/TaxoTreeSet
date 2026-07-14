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
    _encode_windows_2bit,
)
from taxotreeset.core.generation._bottomup import (  # re-exported; public API below
    _BottomUpCapacityComputer,
)
from taxotreeset.core.generation._diskdedup import (  # re-exported for callers/tests
    _bucket_writer_paths,
    _cleanup_key_buckets,
    _compact_pure_keys,
    _count_unique_bucketed_on_disk,
    _flush_keys_to_buckets,
)
from taxotreeset.core.generation._keys import (  # re-exported for _spill lazy import + tests
    _NodeCapacityKeys as _NodeCapacityKeys,
)
from taxotreeset.core.generation._spill import (  # re-exported for tests
    _cleanup_spill_dirs as _cleanup_spill_dirs,
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


# The bottom-up computer + its pool workers moved to ._bottomup; the 2-bit encoding,
# Bloom, and GPU kernels live in ._encoding / ._bloom / ._gpu (imported at the top).


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


def _open_key_buckets(unique_pure, pending, key_bytes):
    """Open the on-disk prefix buckets and spill the in-memory keys into them.

    Stays in this module (rather than ``_diskdedup``) so its ``_flush_keys_to_buckets``
    calls resolve through capacity's namespace — ``_capacity_exact`` flushes both
    here and inline, and tests patch a single ``capacity._flush_keys_to_buckets``.

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
