"""Tests for the reject-class virtual bucket (core/generation/reject_bucket.py)."""

import random
from unittest.mock import patch

from bigtree import Node

from taxotreeset.core.generation.reject_bucket import (
    build_reject_tasks,
    make_reject_bucket_node,
    sample_reject_leaves,
)


def _seq_leaf(name: str, parent: Node) -> Node:
    node = Node(str(name), parent=parent)
    node.rank = "sequence"
    node.header_id = str(name)
    node.fasta_path = "/fake/vault"
    return node


def _node(name, parent=None, rank="no_rank", sci=None):
    node = Node(str(name), parent=parent) if parent is not None else Node(str(name))
    node.rank = rank
    node.scientific_name = sci or str(name)
    return node


def _build_tree() -> dict:
    """root → K1 → {P1 → [x1,x2], P2 → [y1]}; root → K2 → [z1]."""
    root = _node("1", rank="superkingdom", sci="Root")
    k1 = _node("2", parent=root, rank="kingdom", sci="K1")
    p1 = _node("10", parent=k1, rank="family", sci="P1")
    p2 = _node("11", parent=k1, rank="family", sci="P2")
    k2 = _node("3", parent=root, rank="kingdom", sci="K2")
    nodes = {"root": root, "k1": k1, "p1": p1, "p2": p2, "k2": k2}
    nodes["x1"] = _seq_leaf("x1", p1)
    nodes["x2"] = _seq_leaf("x2", p1)
    nodes["y1"] = _seq_leaf("y1", p2)
    nodes["z1"] = _seq_leaf("z1", k2)
    return nodes


# ---------------------------------------------------------------------------
# make_reject_bucket_node
# ---------------------------------------------------------------------------


class TestMakeRejectBucketNode:
    def test_rank_name_and_detached(self):
        parent = _node("10239", sci="Viruses")
        node, meta = make_reject_bucket_node(parent, "10239", "Viruses")
        assert node.rank == "virtual_reject"
        assert node.parent is None          # detached: not attached to the tree
        assert node.scientific_name.startswith("virtual_reject")
        assert meta["purpose"] == "reject"
        assert meta["rank"] == "reject"
        assert meta["taxid"] == str(node.name)
        assert meta["absorbed_taxids"] == []  # re-parents nothing

    def test_virtual_id_is_deterministic(self):
        parent = _node("10239", sci="Viruses")
        n1, _ = make_reject_bucket_node(parent, "10239", "Viruses")
        n2, _ = make_reject_bucket_node(parent, "10239", "Viruses")
        assert n1.name == n2.name

    def test_defaults_from_parent_node(self):
        parent = _node("555", sci="SomeClade")
        node, meta = make_reject_bucket_node(parent)
        assert meta["name"].endswith("SomeClade")


# ---------------------------------------------------------------------------
# sample_reject_leaves
# ---------------------------------------------------------------------------


class TestSampleRejectLeaves:
    def test_excludes_own_leaves_includes_external(self):
        t = _build_tree()
        near, far = sample_reject_leaves(t["p1"])
        external = set(near) | set(far)
        assert t["x1"] not in external and t["x2"] not in external
        assert t["y1"] in external and t["z1"] in external

    def test_near_is_nearest_ancestor_siblings_far_is_rest(self):
        t = _build_tree()
        near, far = sample_reject_leaves(t["p1"])
        assert set(near) == {t["y1"]}   # sibling family under the same K1
        assert set(far) == {t["z1"]}    # farther clade under K2

    def test_root_has_no_external(self):
        t = _build_tree()
        assert sample_reject_leaves(t["root"]) == ([], [])

    def test_walks_up_when_parent_has_no_other_leaves(self):
        # K2 has only z1; its nearest ancestor with external leaves is root.
        t = _build_tree()
        near, far = sample_reject_leaves(t["k2"])
        assert set(near) == {t["x1"], t["x2"], t["y1"]}
        assert far == []
        assert t["z1"] not in set(near)  # own leaf excluded

    def test_caps_each_pool(self):
        root = _node("1", rank="superkingdom", sci="Root")
        head = _node("2", parent=root, rank="kingdom", sci="Head")
        _seq_leaf("h1", head)
        other = _node("3", parent=root, rank="kingdom", sci="Other")
        for i in range(50):
            _seq_leaf(f"o{i}", other)
        near, far = sample_reject_leaves(head, max_per_pool=10, rng=random.Random(1))
        assert len(near) == 10        # capped from 50
        assert far == []              # all external are near-clade (under root)
        assert all(leaf.header_id.startswith("o") for leaf in near)  # never own h1

    def test_populates_root_leaf_cache(self):
        t = _build_tree()
        sample_reject_leaves(t["p1"])
        cached = getattr(t["root"], "_reject_seq_leaves_cache", None)
        assert cached is not None
        assert {leaf.header_id for leaf in cached} == {"x1", "x2", "y1", "z1"}

    def test_uses_cached_root_leaves(self):
        # Poison the cache with an empty list: sample_reject_leaves must consult
        # it (returning no externals) rather than re-scanning root.leaves.
        t = _build_tree()
        t["root"]._reject_seq_leaves_cache = []
        assert sample_reject_leaves(t["p1"]) == ([], [])


# ---------------------------------------------------------------------------
# sample_reject_leaves — cross-domain (non-virus) gate (P4 Phase 2)
# ---------------------------------------------------------------------------


def _cross_pool(n: int = 3) -> list:
    pool = []
    for i in range(n):
        leaf = Node(f"c{i}")
        leaf.rank = "sequence"
        leaf.header_id = f"c{i}"
        leaf.fasta_path = "/fake/cross_domain_vault"
        pool.append(leaf)
    return pool


class TestCrossDomainGate:
    # _build_tree depths (bigtree, root=1): root=1, k1/k2=2, p1/p2=3.

    def test_root_head_gets_cross_domain_as_only_negatives(self):
        # The whole-tree head has no intra-tree "outside" -> cross-domain is its
        # only reject source (the P4 gap the gate closes).
        t = _build_tree()
        near, far = sample_reject_leaves(
            t["root"], cross_domain_leaves=_cross_pool(), cross_domain_max_depth=2)
        assert near == []
        assert {leaf.header_id for leaf in far} == {"c0", "c1", "c2"}

    def test_shallow_head_appends_cross_domain_to_far(self):
        t = _build_tree()  # k1 depth 2 <= gate 2
        _, far = sample_reject_leaves(
            t["k1"], cross_domain_leaves=_cross_pool(), cross_domain_max_depth=2)
        assert len([leaf for leaf in far if leaf.header_id.startswith("c")]) == 3

    def test_deep_head_excluded_from_cross_domain(self):
        t = _build_tree()  # p1 depth 3 > gate 2
        near, far = sample_reject_leaves(
            t["p1"], cross_domain_leaves=_cross_pool(), cross_domain_max_depth=2)
        assert all(not leaf.header_id.startswith("c") for leaf in near + far)

    def test_no_pool_leaves_root_empty(self):
        # Without the gate the root still has no negatives (unchanged behaviour).
        t = _build_tree()
        assert sample_reject_leaves(t["root"]) == ([], [])


# ---------------------------------------------------------------------------
# build_reject_tasks (budget split; allocation itself is covered elsewhere)
# ---------------------------------------------------------------------------


class TestBuildRejectTasks:
    @patch("taxotreeset.core.generation.reject_bucket._allocate_n_across_leaves")
    def test_splits_budget_by_ratio(self, mock_alloc):
        mock_alloc.side_effect = lambda leaves, n, m: [{"n": n}]
        near, far = [object(), object()], [object()]
        build_reject_tasks(near, far, n_reject=100, near_far_ratio=0.3, min_subseq_len=100)
        assert mock_alloc.call_args_list[0].args[1] == 30   # near = 0.3 * 100
        assert mock_alloc.call_args_list[1].args[1] == 70   # far  = remainder

    @patch("taxotreeset.core.generation.reject_bucket._allocate_n_across_leaves")
    def test_far_only_gets_full_budget(self, mock_alloc):
        mock_alloc.side_effect = lambda leaves, n, m: [{"n": n}]
        build_reject_tasks([], [object()], n_reject=100, near_far_ratio=0.5, min_subseq_len=100)
        assert len(mock_alloc.call_args_list) == 1
        assert mock_alloc.call_args_list[0].args[1] == 100

    @patch("taxotreeset.core.generation.reject_bucket._allocate_n_across_leaves")
    def test_near_only_gets_full_budget(self, mock_alloc):
        mock_alloc.side_effect = lambda leaves, n, m: [{"n": n}]
        build_reject_tasks([object()], [], n_reject=80, near_far_ratio=0.5, min_subseq_len=100)
        assert len(mock_alloc.call_args_list) == 1
        assert mock_alloc.call_args_list[0].args[1] == 80

    def test_empty_pools_return_empty(self):
        assert build_reject_tasks([], [], 100, 0.5, 100) == []

    def test_zero_budget_returns_empty(self):
        assert build_reject_tasks([object()], [object()], 0, 0.5, 100) == []

    @patch("taxotreeset.core.generation.reject_bucket._allocate_n_across_leaves")
    def test_ratio_above_one_does_not_produce_negative_far(self, mock_alloc):
        # --reject-near-far-end is an unbounded float; ratio > 1 must not make
        # n_far negative (which would feed a negative sample count to extraction).
        mock_alloc.side_effect = lambda leaves, n, m: [{"n": n}]
        build_reject_tasks(
            [object()], [object()], n_reject=100, near_far_ratio=1.5, min_subseq_len=100
        )
        allocated = [c.args[1] for c in mock_alloc.call_args_list]
        assert all(n >= 0 for n in allocated)
        assert allocated[0] == 100  # near clamped to the full budget; far = 0 (skipped)
