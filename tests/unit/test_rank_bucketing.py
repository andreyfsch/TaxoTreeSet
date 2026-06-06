"""Tests for taxotreeset.core.generation.rank_bucketing.classify_children_by_rank."""

import pytest
from bigtree import Node
from taxotreeset.core.generation.rank_bucketing import classify_children_by_rank
from taxotreeset.core.generation.virtual_id import make_virtual_id


def _node(name, rank="species", scientific_name=None, parent=None):
    n = Node(str(name), parent=parent)
    n.rank = rank
    n.scientific_name = scientific_name or str(name)
    return n


# ---------------------------------------------------------------------------
# Trivial / degenerate cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_children_list_returns_empty(self):
        parent = _node("root", rank="genus")
        effective, buckets = classify_children_by_rank(parent, [])
        assert effective == []
        assert buckets == []

    def test_single_child_no_buckets_created(self):
        parent = _node("root", rank="genus")
        child = _node("1", rank="species", parent=parent)
        effective, buckets = classify_children_by_rank(parent, [child])
        assert effective == [child]
        assert buckets == []

    def test_all_protected_children_returned_unchanged(self):
        parent = _node("root", rank="class")
        children = [
            _node("v1", rank="virtual_misc", parent=parent),
            _node("v2", rank="virtual_low_capacity", parent=parent),
        ]
        effective, buckets = classify_children_by_rank(parent, children)
        assert set(str(c.name) for c in effective) == {"v1", "v2"}
        assert buckets == []


# ---------------------------------------------------------------------------
# Uniform rank (no bucketing needed)
# ---------------------------------------------------------------------------


class TestUniformRankNoBucketing:
    def test_all_same_rank_produces_no_buckets(self):
        parent = _node("1000", rank="family")
        children = [_node(str(i), rank="genus", parent=parent) for i in range(5)]
        effective, buckets = classify_children_by_rank(parent, children)
        assert len(effective) == 5
        assert buckets == []

    def test_all_effective_children_preserved(self):
        parent = _node("1000", rank="family")
        children = [_node(str(i), rank="genus", parent=parent) for i in range(3)]
        effective, _ = classify_children_by_rank(parent, children)
        assert set(str(c.name) for c in effective) == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# Mixed ranks — dedicated bucket (above min_subclades_per_bucket)
# ---------------------------------------------------------------------------


class TestMixedRanksDedicatedBucket:
    def _setup(self, n_canonical=7, n_noncanon=5):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(n_canonical)]
        non_canon = [
            _node(str(i + 100), rank="family", parent=parent) for i in range(n_noncanon)
        ]
        return parent, canonical, non_canon

    def test_dedicated_bucket_created_above_threshold(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        children = canonical + non_canon
        effective, buckets = classify_children_by_rank(
            parent, children, min_subclades_per_bucket=5
        )
        assert len(buckets) == 1

    def test_bucket_metadata_rank_matches_non_canonical_rank(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        _, buckets = classify_children_by_rank(parent, canonical + non_canon)
        assert buckets[0]["rank"] == "family"

    def test_bucket_node_rank_is_virtual_prefixed(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        effective, _ = classify_children_by_rank(parent, canonical + non_canon)
        virtual_nodes = [c for c in effective if getattr(c, "rank", "").startswith("virtual_")]
        assert len(virtual_nodes) == 1
        assert virtual_nodes[0].rank == "virtual_family"

    def test_canonical_children_preserved_in_effective(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        effective, _ = classify_children_by_rank(parent, canonical + non_canon)
        order_nodes = [c for c in effective if getattr(c, "rank", "") == "order"]
        assert len(order_nodes) == 7

    def test_absorbed_taxids_all_non_canonical_children(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        _, buckets = classify_children_by_rank(parent, canonical + non_canon)
        absorbed = set(buckets[0]["absorbed_taxids"])
        expected = {str(c.name) for c in non_canon}
        assert absorbed == expected

    def test_bucket_taxid_starts_with_nine(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        _, buckets = classify_children_by_rank(parent, canonical + non_canon)
        assert buckets[0]["taxid"].startswith("9")

    def test_bucket_taxid_derived_from_parent_taxid(self):
        parent, canonical, non_canon = self._setup(n_canonical=7, n_noncanon=5)
        _, buckets = classify_children_by_rank(parent, canonical + non_canon)
        expected = make_virtual_id(str(parent.name), "rank_family")
        assert buckets[0]["taxid"] == expected

    def test_modal_rank_by_count_not_insertion_order(self):
        """ID 25: rank_counts[rank] = 1 would pick 'family' (seen first) as canonical."""
        parent = _node("1000", rank="class", scientific_name="TestClass")
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        # Non-canonical families appear FIRST — correct code must still count majority
        children = non_canon + canonical
        effective, buckets = classify_children_by_rank(parent, children)
        order_nodes = [c for c in effective if getattr(c, "rank", "") == "order"]
        assert len(order_nodes) == 7
        assert len(buckets) == 1
        assert buckets[0]["rank"] == "family"


# ---------------------------------------------------------------------------
# Mixed ranks — misc bucket (below min_subclades_per_bucket)
# ---------------------------------------------------------------------------


class TestMixedRanksMiscBucket:
    def test_below_threshold_rank_goes_to_misc_bucket(self):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        rare_family = [_node(str(i + 100), rank="family", parent=parent) for i in range(2)]
        children = canonical + rare_family
        effective, buckets = classify_children_by_rank(
            parent, children, min_subclades_per_bucket=5
        )
        assert len(buckets) == 1
        assert buckets[0]["rank"] == "misc"

    def test_misc_bucket_node_rank_is_virtual_misc(self):
        parent = _node("1000", rank="class")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        rare = [_node(str(i + 100), rank="family", parent=parent) for i in range(2)]
        effective, _ = classify_children_by_rank(parent, canonical + rare, min_subclades_per_bucket=5)
        misc_nodes = [c for c in effective if getattr(c, "rank", "") == "virtual_misc"]
        assert len(misc_nodes) == 1

    def test_two_rare_ranks_merged_into_single_misc(self):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(6)]
        rare_family = [_node(str(i + 100), rank="family", parent=parent) for i in range(2)]
        rare_genus = [_node(str(i + 200), rank="genus", parent=parent) for i in range(2)]
        children = canonical + rare_family + rare_genus
        _, buckets = classify_children_by_rank(parent, children, min_subclades_per_bucket=5)
        misc_buckets = [b for b in buckets if b["rank"] == "misc"]
        assert len(misc_buckets) == 1
        assert len(misc_buckets[0]["absorbed_taxids"]) == 4


# ---------------------------------------------------------------------------
# Idempotency: protected children are preserved across repeated calls
# ---------------------------------------------------------------------------


class TestMixedWithExistingProtected:
    """canonical + non-canonical + pre-existing virtual child in one call.

    This exercises lines 273-275 of _partition_children_by_rank, which are
    only reachable when classify_children_by_rank reaches _materialize_rank_buckets
    (needs both canonical AND non-canonical non-protected children) AND at least
    one protected (virtual) child is also present.
    """

    def test_protected_child_preserved_alongside_new_bucket(self):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]
        existing_virtual = _node("virtual_999", rank="virtual_genus", parent=parent)
        children = canonical + non_canon + [existing_virtual]

        effective, buckets = classify_children_by_rank(parent, children)

        virtual_nodes = [c for c in effective if getattr(c, "rank", "").startswith("virtual_")]
        virtual_ranks = {c.rank for c in virtual_nodes}
        # The pre-existing protected node must survive alongside the newly created bucket
        assert "virtual_genus" in virtual_ranks
        assert len(virtual_nodes) >= 2

    def test_canonical_count_unaffected_by_protected_child(self):
        parent = _node("2000", rank="class", scientific_name="TestClass2")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(6)]
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]
        existing_virtual = _node("v1", rank="virtual_misc", parent=parent)
        children = canonical + non_canon + [existing_virtual]

        effective, buckets = classify_children_by_rank(parent, children)

        order_nodes = [c for c in effective if getattr(c, "rank", "") == "order"]
        assert len(order_nodes) == 6


# ---------------------------------------------------------------------------
# Dedicated-bucket metadata assertions
# ---------------------------------------------------------------------------


class TestDedicatedBucketMetadata:
    """Assert exact metadata dict fields for dedicated rank buckets.

    Kills mutants that rename dict keys or corrupt string values in
    _create_rank_specific_bucket (IDs 307-309, 336-340, 343, 345, 347).
    """

    def _setup(self):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]
        effective, buckets = classify_children_by_rank(parent, canonical + non_canon)
        return effective, buckets, non_canon

    def test_dedicated_bucket_name_is_virtual_rank_parent(self):
        _, buckets, _ = self._setup()
        assert buckets[0]["name"] == "virtual_family_TestClass"

    def test_dedicated_bucket_purpose_is_rank_prefixed(self):
        _, buckets, _ = self._setup()
        assert buckets[0]["purpose"] == "rank_family"

    def test_absorbed_children_reparented_under_dedicated_bucket(self):
        effective, _, non_canon = self._setup()
        virtual_node = next(c for c in effective if getattr(c, "rank", "") == "virtual_family")
        for child in non_canon:
            assert child.parent is virtual_node


# ---------------------------------------------------------------------------
# Misc-bucket metadata assertions
# ---------------------------------------------------------------------------


class TestMiscBucketMetadata:
    """Assert exact metadata dict fields for the misc catch-all bucket.

    Kills mutants that rename keys or corrupt constants in _create_misc_bucket
    and _MISC_BUCKET_NAME_PREFIX (IDs 283-284, 351-352, 354-356, 358).
    """

    def _setup(self):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        rare = [_node(str(i + 100), rank="family", parent=parent) for i in range(2)]
        effective, buckets = classify_children_by_rank(
            parent, canonical + rare, min_subclades_per_bucket=5
        )
        return effective, buckets, rare

    def test_misc_bucket_name_has_virtual_misc_prefix_and_parent(self):
        _, buckets, _ = self._setup()
        assert buckets[0]["name"] == "virtual_misc_TestClass"

    def test_misc_bucket_purpose_is_misc(self):
        _, buckets, _ = self._setup()
        assert buckets[0]["purpose"] == "misc"

    def test_misc_bucket_taxid_key_present_and_starts_with_nine(self):
        _, buckets, _ = self._setup()
        assert "taxid" in buckets[0]
        assert str(buckets[0]["taxid"]).startswith("9")

    def test_misc_absorbed_children_reparented_under_misc_node(self):
        effective, _, rare = self._setup()
        misc_node = next(c for c in effective if getattr(c, "rank", "") == "virtual_misc")
        for child in rare:
            assert child.parent is misc_node


# ---------------------------------------------------------------------------
# Rank counting — loop control correctness
# ---------------------------------------------------------------------------


class TestRankCountingLoopControl:
    """Protected children must be skipped (continue), not abort the loop (break).

    A protected child appearing first in the children list would zero out rank
    counts with a break-mutant, causing no bucketing to occur.
    Kills IDs 298 (continue→break in _count_ranks_excluding_protected)
    and 329 (continue→break in _partition_children_by_rank).
    """

    def test_protected_child_first_does_not_abort_bucketing(self):
        parent = _node("1000", rank="class", scientific_name="TestClass")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]
        protected = _node("virt_pre", rank="virtual_misc", parent=parent)
        children = [protected] + canonical + non_canon
        effective, buckets = classify_children_by_rank(parent, children)
        assert len(buckets) == 1
        assert buckets[0]["rank"] == "family"

    def test_protected_child_mid_list_does_not_truncate_non_canonical(self):
        parent = _node("2000", rank="class", scientific_name="TestClass2")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]
        protected = _node("virt_mid", rank="virtual_low_capacity", parent=parent)
        # Protected child between canonical (order×7) and non-canonical (family×5)
        children = canonical + [protected] + non_canon
        effective, buckets = classify_children_by_rank(parent, children)
        assert len(buckets) == 1
        assert len(buckets[0]["absorbed_taxids"]) == 5


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_protected_children_not_rebucketed_on_second_call(self):
        parent = _node("1000", rank="class")
        canonical = [_node(str(i), rank="order", parent=parent) for i in range(7)]
        non_canon = [_node(str(i + 100), rank="family", parent=parent) for i in range(5)]

        effective1, _ = classify_children_by_rank(parent, canonical + non_canon)

        effective2, buckets2 = classify_children_by_rank(parent, effective1)
        virtual_nodes = [c for c in effective2 if getattr(c, "rank", "").startswith("virtual_")]
        assert len(virtual_nodes) == 1
        assert buckets2 == []
