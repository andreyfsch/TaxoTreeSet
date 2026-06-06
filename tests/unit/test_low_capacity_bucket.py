"""Tests for taxotreeset.core.generation.low_capacity_bucket."""

import pytest
from bigtree import Node
from taxotreeset.core.generation.low_capacity_bucket import (
    make_low_capacity_bucket_node,
    make_rare_taxa_bucket_node,
    register_virtual_bucket,
)
from taxotreeset.core.generation.virtual_id import make_virtual_id


def _node(name, rank="species", scientific_name=None, parent=None):
    n = Node(str(name), parent=parent)
    n.rank = rank
    n.scientific_name = scientific_name or str(name)
    return n


# ---------------------------------------------------------------------------
# make_low_capacity_bucket_node
# ---------------------------------------------------------------------------


class TestMakeLowCapacityBucketNode:
    def test_bucket_node_is_attached_to_parent(self):
        parent = _node("1000", rank="genus", scientific_name="TestGenus")
        children = [_node(str(i), parent=parent) for i in range(3)]
        bucket, _ = make_low_capacity_bucket_node(parent, children)
        assert bucket.parent is parent

    def test_low_capacity_children_are_reparented_under_bucket(self):
        parent = _node("1000", rank="genus", scientific_name="TestGenus")
        children = [_node(str(i), parent=parent) for i in range(3)]
        bucket, _ = make_low_capacity_bucket_node(parent, children)
        for child in children:
            assert child.parent is bucket

    def test_bucket_rank_is_virtual_low_capacity(self):
        parent = _node("1000", rank="genus")
        children = [_node("11", parent=parent)]
        bucket, _ = make_low_capacity_bucket_node(parent, children)
        assert bucket.rank == "virtual_low_capacity"

    def test_bucket_taxid_starts_with_nine(self):
        parent = _node("1000")
        children = [_node("11", parent=parent)]
        bucket, meta = make_low_capacity_bucket_node(parent, children)
        assert str(bucket.name).startswith("9")
        assert meta["taxid"].startswith("9")

    def test_bucket_taxid_is_deterministic(self):
        parent1 = _node("1000", scientific_name="TestGenus")
        children1 = [_node("11", parent=parent1)]
        bucket1, meta1 = make_low_capacity_bucket_node(parent1, children1)

        parent2 = _node("1000", scientific_name="TestGenus")
        children2 = [_node("11", parent=parent2)]
        bucket2, meta2 = make_low_capacity_bucket_node(parent2, children2)

        assert str(bucket1.name) == str(bucket2.name)
        assert meta1["taxid"] == meta2["taxid"]

    def test_metadata_contains_expected_keys(self):
        parent = _node("1000", scientific_name="TestGenus")
        children = [_node("11", parent=parent)]
        _, meta = make_low_capacity_bucket_node(parent, children)
        assert "taxid" in meta
        assert "name" in meta
        assert "rank" in meta
        assert "purpose" in meta
        assert "absorbed_taxids" in meta

    def test_metadata_absorbed_taxids_matches_children(self):
        parent = _node("1000")
        children = [_node(str(i), parent=parent) for i in (11, 12, 13)]
        _, meta = make_low_capacity_bucket_node(parent, children)
        assert set(meta["absorbed_taxids"]) == {"11", "12", "13"}

    def test_explicit_parent_taxid_and_name_are_used(self):
        parent = _node("1000")
        children = [_node("11", parent=parent)]
        _, meta = make_low_capacity_bucket_node(
            parent, children, parent_taxid="9999", parent_name="CustomName"
        )
        expected_id = make_virtual_id("9999", "low_capacity")
        assert meta["taxid"] == expected_id

    def test_empty_children_list_produces_bucket_with_no_children(self):
        parent = _node("1000")
        bucket, meta = make_low_capacity_bucket_node(parent, [])
        assert meta["absorbed_taxids"] == []
        assert not bucket.children


# ---------------------------------------------------------------------------
# make_rare_taxa_bucket_node
# ---------------------------------------------------------------------------


class TestMakeRareTaxaBucketNode:
    def test_bucket_node_is_attached_to_parent(self):
        parent = _node("2000", rank="genus", scientific_name="RareGenus")
        children = [_node(str(i), parent=parent) for i in range(2)]
        bucket, _ = make_rare_taxa_bucket_node(parent, children)
        assert bucket.parent is parent

    def test_rare_taxa_children_are_reparented_under_bucket(self):
        parent = _node("2000", rank="genus")
        children = [_node(str(i), parent=parent) for i in range(2)]
        bucket, _ = make_rare_taxa_bucket_node(parent, children)
        for child in children:
            assert child.parent is bucket

    def test_bucket_rank_is_virtual_rare_taxa(self):
        parent = _node("2000")
        children = [_node("21", parent=parent)]
        bucket, _ = make_rare_taxa_bucket_node(parent, children)
        assert bucket.rank == "virtual_rare_taxa"

    def test_bucket_taxid_is_distinct_from_low_capacity_bucket(self):
        parent = _node("2000", scientific_name="SameGenus")

        lc_children = [_node("11", parent=parent)]
        _, lc_meta = make_low_capacity_bucket_node(parent, lc_children)

        rt_children = [_node("21", parent=parent)]
        _, rt_meta = make_rare_taxa_bucket_node(parent, rt_children)

        assert lc_meta["taxid"] != rt_meta["taxid"]

    def test_metadata_purpose_is_rare_taxa(self):
        parent = _node("2000")
        children = [_node("21", parent=parent)]
        _, meta = make_rare_taxa_bucket_node(parent, children)
        assert meta["purpose"] == "rare_taxa"

    def test_metadata_absorbed_taxids_matches_children(self):
        parent = _node("2000")
        children = [_node(str(i), parent=parent) for i in (21, 22)]
        _, meta = make_rare_taxa_bucket_node(parent, children)
        assert set(meta["absorbed_taxids"]) == {"21", "22"}


# ---------------------------------------------------------------------------
# register_virtual_bucket
# ---------------------------------------------------------------------------


class TestRegisterVirtualBucket:
    def _make_meta(self, virtual_id="912345", purpose="low_capacity"):
        return {
            "taxid": virtual_id,
            "name": "virtual_bucket",
            "rank": "virtual_low_capacity",
            "purpose": purpose,
            "absorbed_taxids": ["11", "12"],
        }

    def test_registers_new_bucket_in_registry(self):
        registry = {}
        meta = self._make_meta()
        register_virtual_bucket(registry, meta, "10239", "Viruses")
        assert "912345" in registry

    def test_registered_entry_has_parent_taxid(self):
        registry = {}
        meta = self._make_meta()
        register_virtual_bucket(registry, meta, "10239", "Viruses")
        assert registry["912345"]["parent_taxid"] == "10239"

    def test_early_return_when_already_registered_identically(self):
        registry = {}
        meta = self._make_meta()
        register_virtual_bucket(registry, meta, "10239", "Viruses")
        register_virtual_bucket(registry, meta, "10239", "Viruses")
        assert len(registry) == 1

    def test_raises_on_collision_with_different_parent(self):
        registry = {}
        meta = self._make_meta()
        register_virtual_bucket(registry, meta, "10239", "Viruses")
        with pytest.raises(RuntimeError, match="Virtual ID collision"):
            register_virtual_bucket(registry, meta, "99999", "Viruses")

    def test_raises_on_collision_with_different_purpose(self):
        registry = {}
        meta_a = self._make_meta(purpose="low_capacity")
        meta_b = self._make_meta(purpose="rare_taxa")
        register_virtual_bucket(registry, meta_a, "10239", "Viruses")
        with pytest.raises(RuntimeError, match="Virtual ID collision"):
            register_virtual_bucket(registry, meta_b, "10239", "Viruses")
