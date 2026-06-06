"""Tests for taxotreeset.ranks — canonical rank ordering and boundary predicates."""

import pytest
from taxotreeset.ranks import (
    CANONICAL_RANKS_ROOT_TO_SPECIES,
    CANONICAL_RANKS_SPECIES_TO_ROOT,
    is_at_or_below_boundary,
    is_below_boundary,
    is_canonical_rank,
    rank_depth,
)


class TestIsCanonicalRank:
    def test_all_eight_major_ranks_are_canonical(self):
        for rank in CANONICAL_RANKS_SPECIES_TO_ROOT:
            assert is_canonical_rank(rank), f"{rank!r} should be canonical"

    def test_non_canonical_ranks_are_rejected(self):
        for rank in ("no_rank", "subfamily", "subgenus", "clade", "strain", "serotype"):
            assert not is_canonical_rank(rank), f"{rank!r} should not be canonical"

    def test_empty_string_is_not_canonical(self):
        assert not is_canonical_rank("")

    def test_virtual_rank_is_not_canonical(self):
        assert not is_canonical_rank("virtual_species")


class TestRankDepth:
    def test_superkingdom_is_shallowest(self):
        assert rank_depth("superkingdom") == 0

    def test_species_is_deepest(self):
        assert rank_depth("species") == 7

    def test_intermediate_ranks_are_ordered(self):
        expected = {
            "kingdom": 1,
            "phylum": 2,
            "class": 3,
            "order": 4,
            "family": 5,
            "genus": 6,
        }
        for rank, depth in expected.items():
            assert rank_depth(rank) == depth

    def test_canonical_order_root_to_species_is_strictly_increasing(self):
        depths = [rank_depth(r) for r in CANONICAL_RANKS_ROOT_TO_SPECIES]
        assert depths == sorted(depths)
        assert len(set(depths)) == len(depths)

    def test_non_canonical_returns_none(self):
        for rank in ("no_rank", "subfamily", "", "strain"):
            assert rank_depth(rank) is None


class TestIsAtOrBelowBoundary:
    def test_same_rank_as_boundary_is_true(self):
        assert is_at_or_below_boundary("order", "order")
        assert is_at_or_below_boundary("species", "species")
        assert is_at_or_below_boundary("genus", "genus")

    def test_deeper_rank_is_below_boundary(self):
        assert is_at_or_below_boundary("species", "genus")
        assert is_at_or_below_boundary("genus", "family")
        assert is_at_or_below_boundary("species", "order")
        assert is_at_or_below_boundary("family", "superkingdom")

    def test_shallower_rank_is_not_below_boundary(self):
        assert not is_at_or_below_boundary("genus", "species")
        assert not is_at_or_below_boundary("phylum", "class")
        assert not is_at_or_below_boundary("superkingdom", "order")

    def test_non_canonical_node_rank_returns_false(self):
        assert not is_at_or_below_boundary("no_rank", "order")
        assert not is_at_or_below_boundary("subfamily", "family")
        assert not is_at_or_below_boundary("", "species")

    def test_invalid_boundary_raises_valueerror(self):
        with pytest.raises(ValueError, match="canonical rank"):
            is_at_or_below_boundary("species", "no_rank")

    def test_invalid_boundary_mentions_valid_options(self):
        with pytest.raises(ValueError, match="Valid"):
            is_at_or_below_boundary("genus", "strain")


class TestIsBelowBoundary:
    def test_same_rank_as_boundary_is_false(self):
        assert not is_below_boundary("order", "order")
        assert not is_below_boundary("species", "species")

    def test_strictly_deeper_rank_is_below(self):
        assert is_below_boundary("species", "genus")
        assert is_below_boundary("genus", "family")
        assert is_below_boundary("species", "superkingdom")

    def test_shallower_rank_is_not_below(self):
        assert not is_below_boundary("genus", "species")
        assert not is_below_boundary("order", "species")

    def test_non_canonical_node_rank_returns_false(self):
        assert not is_below_boundary("no_rank", "genus")
        assert not is_below_boundary("clade", "order")

    def test_invalid_boundary_raises_valueerror(self):
        with pytest.raises(ValueError, match="canonical rank"):
            is_below_boundary("species", "strain")

    def test_distinction_from_at_or_below(self):
        assert is_at_or_below_boundary("order", "order")
        assert not is_below_boundary("order", "order")
