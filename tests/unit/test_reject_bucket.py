"""Tests for the reject-class virtual bucket (core/generation/reject_bucket.py)."""

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
