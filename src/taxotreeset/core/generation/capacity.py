"""Capacity computation for taxonomic nodes via exact union or Bloom filter.

The "capacity" of a taxonomic node is the number of unique
subsequences of length ``min_len`` extractable via sliding window
from all sequence leaves descending from it. This count is the input
that drives the per-class balancing layer: when balancing siblings,
each child's n_per_class is bounded by the minimum capacity across
the group.

This module provides two computation strategies and the dispatcher
that selects between them:

1. **Exact** (``_capacity_exact``): accumulates all unique sliding-
   window subseqs in a Python set. Precise but memory-intensive on
   large clades; the set can reach hundreds of gigabytes on viral
   heads like Caudoviricetes.

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

    from src.taxotreeset.core.generation.capacity import compute_node_capacity

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

from src.taxotreeset.core.generation.constants import (
    BLOOM_EXPECTED_INSERTIONS,
    BLOOM_FALSE_POSITIVE_RATE,
)
from src.taxotreeset.dataset.utils import _read_single_sequence

logger = logging.getLogger("TaxoTreeSet.Core.Generation.Capacity")

_SEQUENCE_CACHE: dict[tuple[str, str], str] = {}
_SEQUENCE_CACHE_MAX_ENTRIES: int = 30_000

_EARLY_STOP_SAFETY_MULTIPLIER: int = 5
_PROGRESS_LOG_INTERVAL: int = 200


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


def _capacity_exact(
    seq_leaves: list,
    min_len: int,
    max_useful: int | None = None,
) -> int:
    """Compute exact capacity via set union of sliding-window subseqs.

    Iterates over every sequence leaf, applies a sliding window of
    length ``min_len``, and accumulates unique subseqs in a Python
    set. Memory grows proportionally with the unique subseq count;
    use ``_capacity_approximate`` in memory-constrained environments.

    Args:
        seq_leaves: List of sequence leaf nodes.
        min_len: Sliding window size in base pairs.
        max_useful: Optional ceiling for early termination.

    Returns:
        The total number of unique sliding-window subsequences.
    """
    union_set: set[str] = set()
    early_stop_threshold = (
        max_useful * _EARLY_STOP_SAFETY_MULTIPLIER if max_useful else None
    )

    for leaf in seq_leaves:
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            continue

        sequence = _read_sequence_cached(fasta_path, header_id)
        if not sequence or len(sequence) < min_len:
            continue

        sequence_length = len(sequence)
        for window_start in range(sequence_length - min_len + 1):
            union_set.add(sequence[window_start : window_start + min_len])

        if early_stop_threshold and len(union_set) >= early_stop_threshold:
            break

    return len(union_set)


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
