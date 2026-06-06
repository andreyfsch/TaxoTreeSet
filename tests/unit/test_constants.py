"""Tests for taxotreeset.core.generation.constants — protected rank predicates."""

from taxotreeset.core.generation.constants import (
    PROTECTED_RANKS,
    is_protected_rank,
    is_recursion_terminator,
)


class TestIsProtectedRank:
    def test_all_explicitly_listed_ranks_are_protected(self):
        for rank in PROTECTED_RANKS:
            assert is_protected_rank(rank), f"{rank!r} must be protected"

    def test_virtual_prefix_makes_any_rank_protected(self):
        arbitrary_virtual_ranks = [
            "virtual_misc",
            "virtual_species",
            "virtual_genus",
            "virtual_future_rank_not_in_set",
            "virtual_",
        ]
        for rank in arbitrary_virtual_ranks:
            assert is_protected_rank(rank), f"{rank!r} must be protected via virtual_ prefix"

    def test_canonical_ranks_are_not_protected(self):
        for rank in ("species", "genus", "family", "order", "class", "phylum"):
            assert not is_protected_rank(rank)

    def test_empty_string_returns_false(self):
        assert not is_protected_rank("")

    def test_none_like_falsy_string_returns_false(self):
        assert not is_protected_rank("")

    def test_partial_prefix_is_not_protected(self):
        assert not is_protected_rank("virtua_misc")
        assert not is_protected_rank("_virtual_misc")


class TestIsRecursionTerminator:
    def test_virtual_misc_terminates_recursion(self):
        assert is_recursion_terminator("virtual_misc")

    def test_virtual_low_capacity_terminates_recursion(self):
        assert is_recursion_terminator("virtual_low_capacity")

    def test_virtual_cluster_terminates_recursion(self):
        assert is_recursion_terminator("virtual_cluster")

    def test_virtual_species_terminates_recursion(self):
        assert is_recursion_terminator("virtual_species")

    def test_realm_group_does_not_terminate_recursion(self):
        assert not is_recursion_terminator("realm_group")

    def test_canonical_ranks_do_not_terminate_recursion(self):
        for rank in ("species", "genus", "family", "order", "class"):
            assert not is_recursion_terminator(rank)

    def test_empty_string_returns_false(self):
        assert not is_recursion_terminator("")

    def test_distinction_from_is_protected_rank(self):
        assert is_protected_rank("realm_group")
        assert not is_recursion_terminator("realm_group")
