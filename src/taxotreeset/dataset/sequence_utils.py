"""Subsequence sampling helpers for the dataset generation pipeline.

This module provides primitives for extracting fixed-quantity samples of
subsequences from a parent DNA sequence. The strategy adapts to the
sequence length relative to the sampling budget, choosing between three
distinct sampling scenarios to maintain extraction efficiency and
diversity:

1. **Non-overlapping sampling** for sequences that are at least twice
   as long as the total sampling budget. Random positions are drawn
   until they pack into the sequence without overlap, using a sorted
   blacklist and binary search to keep collision detection in
   logarithmic time.

2. **Flanked block sampling** for sequences in the intermediate range.
   Half of the samples are drawn from the left margin, half from the
   right, with optional uniform spacing between them. An odd sample
   count produces an extra middle window centered on the sequence.

3. **Bounded random sampling with reverse complement** for sequences
   that cannot fit the requested budget without overlap. Subsequences
   are drawn at uniformly random positions and lengths; their reverse
   complements are added when budget remains, doubling the effective
   diversity of short sequences such as viroids.

In **all three** scenarios the length of each emitted subsequence is drawn
independently in ``[min_len, max_len]`` (see :func:`_draw_subseq_length`); the
scenarios differ only in *placement*. This keeps the emitted length distribution
identical regardless of which scenario runs — and the scenario is selected from
the per-leaf window budget ``n`` — so sequence length never becomes a spurious
signal for the class a window belongs to.

The module also exposes the IUPAC reverse-complement transformation
table as a precomputed translation lookup, which is faster than calling
``str.replace`` repeatedly. The table supports the complete IUPAC
ambiguity alphabet (Y, R, W, S, K, M, D, H, V, B, N) so the module is
safe to use with sequences containing degenerate bases.

Typical usage::

    from taxotreeset.dataset.sequence_utils import (
        extract_subseqs,
        get_complement,
    )

    samples = extract_subseqs(
        seq="ACGTACGTACGT...",
        n=100,
        min_len=200,
        max_len=2000,
    )
    rev_comp = get_complement("ATCGY")  # returns "RCGAT"

References:
    IUPAC nucleotide ambiguity codes: Cornish-Bowden, A. (1985).
    Nucleic Acids Research, 13(9), 3021-3030.
    https://doi.org/10.1093/nar/13.9.3021
"""

import bisect
import random


_IUPAC_COMPLEMENT_MAP: dict[str, str] = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "Y": "R",
    "R": "Y",
    "W": "W",
    "S": "S",
    "K": "M",
    "M": "K",
    "D": "H",
    "H": "D",
    "V": "B",
    "B": "V",
    "N": "N",
}


def _build_iupac_translation_table() -> dict[int, int]:
    """Build the byte-level translation table for IUPAC complementation.

    Generates a 256-entry table where each byte maps to its IUPAC
    complement. Unknown bytes default to 'N'. Both uppercase and
    lowercase bases are accepted; lowercase inputs are normalized to
    uppercase in the output.

    Returns:
        A translation table suitable for str.translate().
    """
    target_chars = ["N"] * 256
    for base, complement in _IUPAC_COMPLEMENT_MAP.items():
        target_chars[ord(base)] = complement
        target_chars[ord(base.lower())] = complement
    source_chars = "".join(chr(i) for i in range(256))
    return str.maketrans(source_chars, "".join(target_chars))


_IUPAC_TRANSLATE_TABLE: dict[int, int] = _build_iupac_translation_table()


def get_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence with IUPAC support.

    Performs reverse complementation using the precomputed IUPAC
    translation table. All ambiguity codes (Y, R, W, S, K, M, D, H, V,
    B, N) are mapped to their canonical complements. Unknown characters
    are mapped to 'N'.

    Args:
        sequence: DNA sequence to reverse-complement. Case-insensitive.

    Returns:
        The reverse complement of the input, in uppercase.

    Example:
        >>> get_complement("ATCG")
        'CGAT'
        >>> get_complement("ATCGY")  # Y is the pyrimidine ambiguity code
        'RCGAT'
    """
    return sequence[::-1].translate(_IUPAC_TRANSLATE_TABLE)


def extract_subseqs(
    seq: str,
    n: int,
    min_len: int,
    max_len: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample n subsequences from a DNA sequence with adaptive strategy.

    Selects one of three sampling scenarios based on the ratio between
    the sequence length and the requested sampling budget:

    - **Non-overlapping** when ``len(seq) >= 2 * n * max_len``.
    - **Flanked blocks** when ``n * max_len <= len(seq) < 2 * n * max_len``.
    - **Bounded random with reverse complement** when
      ``len(seq) < n * max_len`` and ``len(seq) >= min_len``.

    Sequences shorter than ``min_len`` return an empty list rather than
    raising, so callers can process possibly degenerate inputs without
    wrapping every call.

    Args:
        seq: Parent DNA sequence from which subsequences are drawn.
        n: Number of subsequences to return. Must be positive.
        min_len: Minimum length of each returned subsequence.
        max_len: Maximum length of each returned subsequence. Must be
            greater than or equal to ``min_len``.
        rng: Optional ``random.Random`` instance to drive sampling.
            When None, the module-level ``random`` is used. Pass an
            explicit RNG for deterministic, reproducible sampling.

    Returns:
        A list of at most n subsequences, each between min_len and
        max_len characters. May contain fewer than n elements when the
        sequence is shorter than min_len, or in the bounded random
        scenario when diversity is exhausted.

    Raises:
        ValueError: If n is non-positive, min_len is non-positive, or
            min_len exceeds max_len.

    Example:
        >>> import random
        >>> sequence = "ACGT" * 10000
        >>> samples = extract_subseqs(sequence, n=10, min_len=100, max_len=200,
        ...                            rng=random.Random(42))
        >>> len(samples)
        10
    """
    _validate_extraction_parameters(n=n, min_len=min_len, max_len=max_len)

    if rng is None:
        rng = random.Random()

    if len(seq) < min_len:
        return []

    if len(seq) >= 2 * n * max_len:
        return _sample_non_overlapping(seq, n, min_len, max_len, rng)

    if len(seq) >= n * max_len:
        return _sample_flanked_blocks(seq, n, min_len, max_len, rng)

    return _sample_bounded_random_with_complement(seq, n, min_len, max_len, rng)


def _draw_subseq_length(
    rng: random.Random, min_len: int, max_len: int, available: int
) -> int:
    """Draw a random subsequence length in ``[min_len, max_len]``.

    The result is clamped to the ``available`` base pairs remaining at the chosen
    position. Drawing the length *per window* — instead of always emitting a
    ``max_len`` block — is what keeps the emitted length distribution identical
    across every sampling strategy, so the strategy actually used (which is
    chosen from the per-leaf window budget ``n``) never leaks into the sequence
    length.

    Without this, leaves contributing few windows each (e.g. the externally
    sampled reject negatives, whose budget is spread across hundreds of leaves)
    would fall into the long-sequence tiling strategies and emit only ``max_len``
    windows, while leaves contributing many windows fall into the random branch
    and emit short ones — baking a spurious length-to-class signal into training.

    Args:
        rng: Random source.
        min_len: Lower bound on the length.
        max_len: Upper bound on the length.
        available: Base pairs available from the chosen start to the end of the
            sequence; the drawn length never exceeds this.

    Returns:
        A length in ``[min_len, min(max_len, available)]``. When ``available`` is
        at or below ``min_len`` the available count is returned as-is.
    """
    upper = min(max_len, available)
    if upper <= min_len:
        return upper
    return rng.randint(min_len, upper)


def _validate_extraction_parameters(n: int, min_len: int, max_len: int) -> None:
    """Validate sampling parameter constraints.

    Args:
        n: Requested sample count.
        min_len: Minimum sample length.
        max_len: Maximum sample length.

    Raises:
        ValueError: If any constraint is violated.
    """
    if n <= 0:
        raise ValueError(f"n must be positive (got {n})")
    if min_len <= 0:
        raise ValueError(f"min_len must be positive (got {min_len})")
    if min_len > max_len:
        raise ValueError(f"min_len must be <= max_len (got {min_len} > {max_len})")


def _sample_non_overlapping(
    seq: str,
    n: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> list[str]:
    """Sample n non-overlapping subsequences from a long parent sequence.

    Each window's length is drawn independently in ``[min_len, max_len]`` (see
    :func:`_draw_subseq_length`); only the start position is constrained so the
    reserved ``[start, start + length)`` intervals never overlap. Uses a sorted
    blacklist of intervals and binary search to detect collisions in O(log n) per
    candidate. The strategy is applicable when the parent sequence is at least
    twice the total sampling budget (``len(seq) >= 2 * n * max_len``), which
    leaves ample room even at the maximum length.

    Args:
        seq: Parent DNA sequence.
        n: Number of samples to draw.
        min_len: Minimum length of each sample.
        max_len: Maximum length of each sample.
        rng: Random number generator.

    Returns:
        A list of n non-overlapping subsequences of varying length.
    """
    samples: list[str] = []
    occupied_intervals: list[tuple[int, int]] = []

    while len(samples) < n:
        length = _draw_subseq_length(rng, min_len, max_len, len(seq))
        candidate_start = rng.randrange(0, len(seq) - length + 1)
        candidate_end = candidate_start + length
        candidate = (candidate_start, candidate_end)

        insertion_point = bisect.bisect_left(occupied_intervals, candidate)

        if insertion_point > 0:
            previous_end = occupied_intervals[insertion_point - 1][1]
            if candidate_start < previous_end:
                continue

        if insertion_point < len(occupied_intervals):
            next_start = occupied_intervals[insertion_point][0]
            if candidate_end > next_start:
                continue

        occupied_intervals.insert(insertion_point, candidate)
        samples.append(seq[candidate_start:candidate_end])

    return samples


def _sample_flanked_blocks(
    seq: str, n: int, min_len: int, max_len: int, rng: random.Random
) -> list[str]:
    """Sample n quasi-non-overlapping blocks from a moderate-length sequence.

    Distributes samples evenly between the left and right margins of the
    sequence, advancing each cursor by ``max_len`` plus a uniform spacing so the
    blocks stay quasi-non-overlapping even at the maximum length. The window
    *length* emitted at each cursor is drawn in ``[min_len, max_len]`` (see
    :func:`_draw_subseq_length`); a shorter draw simply leaves a wider gap, never
    an overlap. When ``n`` is odd, an extra block is centered on the sequence.

    Applicable when the parent sequence cannot accommodate strictly
    non-overlapping blocks but is still long enough to space them
    (``n * max_len <= len(seq) < 2 * n * max_len``).

    Args:
        seq: Parent DNA sequence.
        n: Number of samples to draw.
        min_len: Minimum length of each sample.
        max_len: Maximum length of each sample.
        rng: Random number generator (drives the per-window length).

    Returns:
        A list of n subsequences of varying length at spaced positions.
    """
    samples: list[str] = []
    remaining_budget = (len(seq) // max_len) - n

    if remaining_budget <= 0:
        return _sample_contiguous_blocks(seq, n, min_len, max_len, rng)

    spacing = max(0, (len(seq) - n * max_len) // (n + 1))
    left_cursor = 0
    right_cursor = len(seq) - max_len
    paired_blocks = n // 2

    for _ in range(paired_blocks):
        left_len = _draw_subseq_length(rng, min_len, max_len, len(seq) - left_cursor)
        samples.append(seq[left_cursor : left_cursor + left_len])
        left_cursor += max_len + spacing
        right_len = _draw_subseq_length(rng, min_len, max_len, len(seq) - right_cursor)
        samples.append(seq[right_cursor : right_cursor + right_len])
        right_cursor -= max_len + spacing

    if n % 2 != 0:
        center = len(seq) // 2
        center_len = _draw_subseq_length(rng, min_len, max_len, len(seq))
        start = min(max(0, center - center_len // 2), len(seq) - center_len)
        samples.append(seq[start : start + center_len])

    return samples


def _sample_contiguous_blocks(
    seq: str, n: int, min_len: int, max_len: int, rng: random.Random
) -> list[str]:
    """Sample n contiguous blocks starting from the left of the sequence.

    Used as a fallback when ``_sample_flanked_blocks`` lacks budget for its
    spacing strategy. Each cursor advances by ``max_len`` (keeping the blocks
    adjacent and within the sequence, since this path is reached only when
    ``len(seq) >= n * max_len``), while the emitted length is drawn in
    ``[min_len, max_len]`` per block.

    Args:
        seq: Parent DNA sequence.
        n: Number of samples to draw.
        min_len: Minimum length of each sample.
        max_len: Maximum length of each sample.
        rng: Random number generator (drives the per-window length).

    Returns:
        A list of n subsequences of varying length at contiguous positions.
    """
    samples: list[str] = []
    cursor = 0
    for _ in range(n):
        length = _draw_subseq_length(rng, min_len, max_len, len(seq) - cursor)
        samples.append(seq[cursor : cursor + length])
        cursor += max_len
    return samples


def _sample_bounded_random_with_complement(
    seq: str,
    n: int,
    min_len: int,
    max_len: int,
    rng: random.Random,
) -> list[str]:
    """Sample n subsequences from a short sequence using overlap and complement.

    Draws subsequences at uniformly random positions and random lengths
    within the valid range. Each unique subsequence is added to the
    output; its reverse complement is also added when budget remains.
    This doubles the effective diversity for short sequences such as
    viroids (~300 bp).

    A retry budget bounds the loop to ``n * 50`` iterations to protect
    against pathological inputs where diversity is exhausted before
    ``n`` unique samples are found.

    Args:
        seq: Parent DNA sequence (assumed shorter than ``n * max_len``).
        n: Number of samples to draw.
        min_len: Minimum sample length.
        max_len: Maximum sample length.
        rng: Random number generator.

    Returns:
        A list of at most n unique subsequences. Fewer samples are
        returned when the retry budget is exhausted before reaching n.
    """
    samples: list[str] = []
    seen_samples: set[str] = set()
    effective_max_len = min(max_len, len(seq))
    max_attempts = n * 50

    for _ in range(max_attempts):
        if len(samples) >= n:
            break

        sample_length = rng.randint(min_len, effective_max_len)
        start = rng.randint(0, len(seq) - sample_length)
        candidate = seq[start : start + sample_length]

        if candidate not in seen_samples:
            samples.append(candidate)
            seen_samples.add(candidate)

        if len(samples) < n:
            reverse_complement = get_complement(candidate)
            if reverse_complement not in seen_samples:
                samples.append(reverse_complement)
                seen_samples.add(reverse_complement)

    return samples
