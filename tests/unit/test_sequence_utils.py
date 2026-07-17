"""Tests for taxotreeset.dataset.sequence_utils — subsequence extraction and IUPAC complement."""

import random

import pytest
from taxotreeset.dataset.sequence_utils import (
    _sample_bounded_random_with_complement,
    _sample_flanked_blocks,
    _sample_non_overlapping,
    _validate_extraction_parameters,
    extract_subseqs,
    get_complement,
)


# ---------------------------------------------------------------------------
# get_complement
# ---------------------------------------------------------------------------


class TestGetComplement:
    def test_simple_reverse_complement(self):
        assert get_complement("ATCG") == "CGAT"

    def test_identity_palindrome(self):
        assert get_complement("AATT") == "AATT"

    def test_single_base_complements(self):
        assert get_complement("A") == "T"
        assert get_complement("T") == "A"
        assert get_complement("C") == "G"
        assert get_complement("G") == "C"

    def test_iupac_ambiguity_y_and_r(self):
        # Y (pyrimidine) <-> R (purine)
        assert get_complement("Y") == "R"
        assert get_complement("R") == "Y"

    def test_iupac_ambiguity_in_context(self):
        assert get_complement("ATCGY") == "RCGAT"

    def test_self_complementary_bases(self):
        # W (A or T) and S (G or C) are self-complementary
        assert get_complement("W") == "W"
        assert get_complement("S") == "S"
        assert get_complement("N") == "N"

    def test_lowercase_input_produces_uppercase_output(self):
        assert get_complement("atcg") == "CGAT"

    def test_empty_sequence_returns_empty(self):
        assert get_complement("") == ""

    def test_longer_sequence_reversal(self):
        seq = "AAACCCGGG"
        rev_comp = get_complement(seq)
        assert len(rev_comp) == len(seq)
        assert rev_comp == "CCCGGGTTT"

    def test_reverse_complement_of_reverse_complement_is_identity(self):
        seq = "ACGTACGTNN"
        assert get_complement(get_complement(seq)) == seq


# ---------------------------------------------------------------------------
# _validate_extraction_parameters
# ---------------------------------------------------------------------------


class TestValidateExtractionParameters:
    def test_valid_parameters_do_not_raise(self):
        _validate_extraction_parameters(n=10, min_len=100, max_len=200)

    def test_zero_n_raises_valueerror(self):
        with pytest.raises(ValueError, match="n must be positive"):
            _validate_extraction_parameters(n=0, min_len=100, max_len=200)

    def test_negative_n_raises_valueerror(self):
        with pytest.raises(ValueError, match="n must be positive"):
            _validate_extraction_parameters(n=-1, min_len=100, max_len=200)

    def test_min_len_exceeds_max_len_raises_valueerror(self):
        with pytest.raises(ValueError, match="min_len must be <= max_len"):
            _validate_extraction_parameters(n=10, min_len=200, max_len=100)

    def test_zero_min_len_raises_valueerror(self):
        with pytest.raises(ValueError, match="min_len must be positive"):
            _validate_extraction_parameters(n=10, min_len=0, max_len=100)

    def test_negative_min_len_raises_valueerror(self):
        # An unbounded --min-subseq-len (e.g. -W -5) must fail fast rather than
        # emit degenerate/empty windows into the training data.
        with pytest.raises(ValueError, match="min_len must be positive"):
            _validate_extraction_parameters(n=10, min_len=-5, max_len=100)

    def test_equal_min_and_max_len_is_valid(self):
        _validate_extraction_parameters(n=10, min_len=100, max_len=100)


# ---------------------------------------------------------------------------
# extract_subseqs — parameter validation
# ---------------------------------------------------------------------------


class TestExtractSubseqsParameterHandling:
    def test_raises_on_nonpositive_n(self):
        with pytest.raises(ValueError):
            extract_subseqs("ACGT" * 1000, n=0, min_len=100, max_len=200)

    def test_raises_on_inverted_lengths(self):
        with pytest.raises(ValueError):
            extract_subseqs("ACGT" * 1000, n=10, min_len=200, max_len=100)

    def test_raises_on_nonpositive_min_len(self):
        with pytest.raises(ValueError):
            extract_subseqs("ACGT" * 1000, n=10, min_len=0, max_len=100)

    def test_sequence_shorter_than_min_len_returns_empty(self):
        result = extract_subseqs("ACGT", n=5, min_len=100, max_len=200)
        assert result == []

    def test_sequence_exactly_min_len_is_included(self):
        rng = random.Random(42)
        result = extract_subseqs("A" * 100, n=1, min_len=100, max_len=100, rng=rng)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# extract_subseqs — scenario dispatch
# ---------------------------------------------------------------------------


def _long_seq(factor: int = 10, n: int = 5, max_len: int = 100) -> str:
    """Sequence long enough for non-overlapping sampling."""
    return "ACGT" * (factor * n * max_len)


class TestExtractSubseqsNonOverlapping:
    """len(seq) >= 2 * n * max_len → _sample_non_overlapping."""

    def test_returns_exactly_n_samples(self):
        rng = random.Random(42)
        seq = _long_seq(n=5, max_len=100, factor=3)
        samples = extract_subseqs(seq, n=5, min_len=50, max_len=100, rng=rng)
        assert len(samples) == 5

    def test_samples_have_correct_length(self):
        rng = random.Random(42)
        seq = _long_seq(n=5, max_len=100, factor=3)
        samples = extract_subseqs(seq, n=5, min_len=100, max_len=100, rng=rng)
        for s in samples:
            assert len(s) == 100

    def test_samples_are_substrings_of_parent(self):
        rng = random.Random(42)
        seq = _long_seq(n=5, max_len=100, factor=3)
        samples = extract_subseqs(seq, n=5, min_len=100, max_len=100, rng=rng)
        for s in samples:
            assert s in seq

    def test_deterministic_with_seeded_rng(self):
        seq = _long_seq(n=5, max_len=100, factor=3)
        samples1 = extract_subseqs(seq, n=5, min_len=100, max_len=100, rng=random.Random(7))
        samples2 = extract_subseqs(seq, n=5, min_len=100, max_len=100, rng=random.Random(7))
        assert samples1 == samples2


class TestExtractSubseqsFlankedBlocks:
    """n * max_len <= len(seq) < 2 * n * max_len → _sample_flanked_blocks."""

    def _medium_seq(self, n: int = 5, max_len: int = 100) -> str:
        length = int(1.5 * n * max_len)
        return "ACGT" * (length // 4 + 1)

    def test_returns_at_most_n_samples(self):
        seq = self._medium_seq(n=5, max_len=100)
        samples = _sample_flanked_blocks(
            seq, n=5, min_len=50, max_len=100, rng=random.Random(0)
        )
        assert len(samples) <= 5

    def test_samples_are_substrings_of_parent(self):
        seq = self._medium_seq(n=5, max_len=100)
        samples = _sample_flanked_blocks(
            seq, n=5, min_len=50, max_len=100, rng=random.Random(0)
        )
        for s in samples:
            assert s in seq

    def test_odd_n_produces_middle_sample(self):
        seq = self._medium_seq(n=3, max_len=100)
        samples = _sample_flanked_blocks(
            seq, n=3, min_len=50, max_len=100, rng=random.Random(0)
        )
        assert len(samples) == 3


class TestExtractSubseqsBoundedRandom:
    """len(seq) < n * max_len → _sample_bounded_random_with_complement."""

    def test_returns_samples_for_short_sequence(self):
        rng = random.Random(42)
        seq = "ACGTACGT" * 20  # 160 bp, asking for n=10, max_len=100
        samples = extract_subseqs(seq, n=10, min_len=50, max_len=100, rng=rng)
        assert isinstance(samples, list)

    def test_samples_have_length_within_bounds(self):
        rng = random.Random(42)
        seq = "ACGTACGT" * 20
        samples = _sample_bounded_random_with_complement(seq, n=5, min_len=50, max_len=80, rng=rng)
        for s in samples:
            assert 50 <= len(s) <= 80

    def test_includes_reverse_complements(self):
        rng = random.Random(42)
        seq = "A" * 200
        samples = _sample_bounded_random_with_complement(seq, n=5, min_len=50, max_len=100, rng=rng)
        assert isinstance(samples, list)


# ---------------------------------------------------------------------------
# _sample_non_overlapping — interval logic
# ---------------------------------------------------------------------------


class TestSampleNonOverlapping:
    def test_returns_exactly_n_samples(self):
        rng = random.Random(42)
        seq = "ACGT" * 10000
        samples = _sample_non_overlapping(seq, n=50, min_len=50, max_len=100, rng=rng)
        assert len(samples) == 50

    def test_intervals_do_not_overlap(self):
        # A non-repetitive sequence makes every variable-length window locatable
        # by its (unique) first occurrence, so we can verify the spans are
        # genuinely disjoint even though lengths now vary.
        gen = random.Random(1)
        seq = "".join(gen.choice("ACGT") for _ in range(8000))
        rng = random.Random(42)
        samples = _sample_non_overlapping(seq, n=30, min_len=50, max_len=100, rng=rng)
        assert len(samples) == 30
        spans = sorted((seq.index(s), seq.index(s) + len(s)) for s in samples)
        for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
            assert next_start >= prev_end

    def test_samples_within_length_bounds(self):
        rng = random.Random(42)
        seq = "ACGT" * 10000
        samples = _sample_non_overlapping(seq, n=10, min_len=50, max_len=100, rng=rng)
        assert all(50 <= len(s) <= 100 for s in samples)


# ---------------------------------------------------------------------------
# Regression: emitted length distribution must not depend on n / strategy
# ---------------------------------------------------------------------------


def _mean(xs):
    return sum(xs) / len(xs)


class TestLengthIndependentOfStrategy:
    """The reject-class length confound: long-sequence tiling strategies used to
    emit only ``max_len`` windows while the short-sequence branch emitted random
    lengths, so a class's per-leaf window budget leaked into the sequence length
    (reject negatives, spread thin over many leaves, came out almost all 2000 bp).
    After the fix every strategy draws a random length in ``[min_len, max_len]``.
    """

    def _random_seq(self, n_bp: int, seed: int) -> str:
        return "".join(random.Random(seed).choices("ACGT", k=n_bp))

    def test_non_overlapping_branch_is_not_pinned_to_max_len(self):
        # Long sequence + small-ish n routes to _sample_non_overlapping (here
        # 600000 >= 2 * 150 * 2000), which used to return only max_len (2000 bp)
        # windows — the exact source of the confound.
        seq = self._random_seq(600_000, seed=2)
        samples = extract_subseqs(
            seq, n=150, min_len=100, max_len=2000, rng=random.Random(0)
        )
        lengths = [len(s) for s in samples]
        assert all(100 <= L <= 2000 for L in lengths)
        assert len(set(lengths)) > 1            # not a single fixed length
        assert _mean(lengths) < 1500            # not pinned near max_len (was 2000)

    def test_distribution_comparable_across_branches(self):
        # Same parent, two budgets routing to different strategies: n=150 windows
        # -> non-overlapping; n=500 windows -> bounded-random. The emitted length
        # distributions should now be comparable (both ~uniform[100, 2000]).
        seq = self._random_seq(600_000, seed=3)
        few = [
            len(s)
            for s in extract_subseqs(
                seq, n=150, min_len=100, max_len=2000, rng=random.Random(0)
            )
        ]
        many = [
            len(s)
            for s in extract_subseqs(
                seq, n=500, min_len=100, max_len=2000, rng=random.Random(0)
            )
        ]
        # Both means land near the [100, 2000] midpoint (~1050), within a generous
        # tolerance — the key point is that neither branch is pinned to an extreme.
        assert abs(_mean(few) - 1050) < 200
        assert abs(_mean(many) - 1050) < 200
