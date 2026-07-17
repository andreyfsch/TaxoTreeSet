"""Tests for taxotreeset.core.generation.distribution — per-leaf sample allocation."""

from unittest.mock import patch

from bigtree import Node
from taxotreeset.core.generation.distribution import (
    _allocate_n_across_leaves,
    _compute_leaf_share,
    _compute_leaf_share_weights,
    _resolve_child_leaves,
    distribute_n_per_class_across_leaves,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_seq_leaf(header_id, fasta_path="/fake/vault"):
    node = Node(str(header_id))
    node.rank = "sequence"
    node.header_id = header_id
    node.fasta_path = fasta_path
    return node


def make_taxon_node(name, leaves):
    parent = Node(str(name))
    parent.rank = "species"
    parent.scientific_name = str(name)
    for leaf in leaves:
        leaf.parent = parent
    return parent


# ---------------------------------------------------------------------------
# _resolve_child_leaves
# ---------------------------------------------------------------------------


class TestResolveChildLeaves:
    def test_returns_cached_leaves_when_present(self):
        leaf = make_seq_leaf("NC_001")
        child = make_taxon_node("12345", [])
        cache = {"12345": [leaf]}
        result = _resolve_child_leaves(child, "12345", cache)
        assert result == [leaf]

    def test_scans_tree_when_cache_misses(self):
        leaf = make_seq_leaf("NC_002")
        child = make_taxon_node("12346", [leaf])
        result = _resolve_child_leaves(child, "12346", {})
        assert result == [leaf]

    def test_empty_cache_and_no_leaves_returns_empty(self):
        child = make_taxon_node("12347", [])
        result = _resolve_child_leaves(child, "12347", {})
        assert result == []

    def test_only_sequence_rank_leaves_are_returned(self):
        seq_leaf = make_seq_leaf("NC_003")
        non_seq = Node("species_x")
        non_seq.rank = "species"
        parent = make_taxon_node("12348", [seq_leaf])
        non_seq.parent = parent
        result = _resolve_child_leaves(parent, "12348", {})
        assert result == [seq_leaf]


# ---------------------------------------------------------------------------
# _compute_leaf_share
# ---------------------------------------------------------------------------


class TestComputeLeafShare:
    def test_last_leaf_gets_remainder(self):
        share = _compute_leaf_share(
            n_per_class=100,
            leaf_weight=1,
            total_weight=3,
            running_sum=67,
            is_last_leaf=True,
        )
        assert share == 33

    def test_last_leaf_is_clamped_to_zero_when_overspent(self):
        share = _compute_leaf_share(
            n_per_class=100,
            leaf_weight=1,
            total_weight=3,
            running_sum=105,
            is_last_leaf=True,
        )
        assert share == 0

    def test_non_last_leaf_gets_proportional_share(self):
        share = _compute_leaf_share(
            n_per_class=100,
            leaf_weight=1,
            total_weight=4,
            running_sum=0,
            is_last_leaf=False,
        )
        assert share == 25

    def test_non_last_leaf_rounds_to_nearest(self):
        share = _compute_leaf_share(
            n_per_class=10,
            leaf_weight=1,
            total_weight=3,
            running_sum=0,
            is_last_leaf=False,
        )
        assert share == round(10 * 1 / 3)

    def test_zero_weight_leaf_gets_zero_share(self):
        share = _compute_leaf_share(
            n_per_class=100,
            leaf_weight=0,
            total_weight=100,
            running_sum=0,
            is_last_leaf=False,
        )
        assert share == 0

    def test_single_leaf_is_always_last(self):
        share = _compute_leaf_share(
            n_per_class=50,
            leaf_weight=1000,
            total_weight=1000,
            running_sum=0,
            is_last_leaf=True,
        )
        assert share == 50

    def test_non_last_share_capped_at_remaining_budget(self):
        # round(100*1/3)=33 would overspend when only 2 of the budget remain;
        # capping prevents the per-child total from exceeding n_per_class.
        share = _compute_leaf_share(
            n_per_class=100,
            leaf_weight=1,
            total_weight=3,
            running_sum=98,
            is_last_leaf=False,
        )
        assert share == 2


# ---------------------------------------------------------------------------
# _compute_leaf_share_weights
# ---------------------------------------------------------------------------


_MOCK_SEQ_PATH = "taxotreeset.core.generation.distribution._read_sequence_cached"


class TestComputeLeafShareWeights:
    def test_weight_equals_seq_len_minus_window(self):
        leaf = make_seq_leaf("NC_001")
        with patch(_MOCK_SEQ_PATH, return_value="A" * 500):
            weights = _compute_leaf_share_weights([leaf], min_subseq_len=100)
        assert weights == [500 - 100 + 1]

    def test_weight_is_zero_when_seq_shorter_than_window(self):
        leaf = make_seq_leaf("NC_001")
        with patch(_MOCK_SEQ_PATH, return_value="A" * 50):
            weights = _compute_leaf_share_weights([leaf], min_subseq_len=100)
        assert weights == [0]

    def test_zero_weight_when_fasta_path_missing(self):
        leaf = Node("orphan")
        leaf.rank = "sequence"
        leaf.header_id = "NC_orphan"
        # fasta_path not set
        weights = _compute_leaf_share_weights([leaf], min_subseq_len=10)
        assert weights == [0]

    def test_zero_weight_when_header_id_missing(self):
        leaf = Node("orphan2")
        leaf.rank = "sequence"
        leaf.fasta_path = "/fake/path"
        # header_id not set
        weights = _compute_leaf_share_weights([leaf], min_subseq_len=10)
        assert weights == [0]

    def test_multiple_leaves_returns_correct_weights(self):
        leaf_a = make_seq_leaf("NC_A", "/fake/vault")
        leaf_b = make_seq_leaf("NC_B", "/fake/vault")

        def side_effect(fasta_path, header_id):
            return "A" * 1000 if header_id == "NC_A" else "A" * 200

        with patch(_MOCK_SEQ_PATH, side_effect=side_effect):
            weights = _compute_leaf_share_weights([leaf_a, leaf_b], min_subseq_len=50)
        assert weights == [1000 - 50 + 1, 200 - 50 + 1]

    def test_empty_leaves_returns_empty_list(self):
        weights = _compute_leaf_share_weights([], min_subseq_len=100)
        assert weights == []


# ---------------------------------------------------------------------------
# _allocate_n_across_leaves
# ---------------------------------------------------------------------------


class TestAllocateNAcrossLeaves:
    def test_single_leaf_receives_full_budget(self):
        leaf = make_seq_leaf("NC_001")
        with patch(_MOCK_SEQ_PATH, return_value="A" * 500):
            tasks = _allocate_n_across_leaves([leaf], n_per_class=50, min_subseq_len=10)
        assert len(tasks) == 1
        assert tasks[0]["n"] == 50

    def test_sum_of_shares_equals_n_per_class(self):
        leaves = [make_seq_leaf(f"NC_{i}") for i in range(4)]

        def side_effect(fasta_path, header_id):
            # All equal weight
            return "A" * 300

        with patch(_MOCK_SEQ_PATH, side_effect=side_effect):
            tasks = _allocate_n_across_leaves(leaves, n_per_class=97, min_subseq_len=10)
        total = sum(t["n"] for t in tasks)
        assert total == 97

    def test_sum_equals_n_even_when_rounding_overshoots(self):
        # Regression: for n roughly between len/2 and len, every leaf's
        # round(n*w/total) rounds up, pushing the running sum past n so the
        # per-child total exceeded n_per_class (e.g. n=3 over 5 equal leaves
        # summed to 4). The remaining-budget cap keeps the sum exactly n.
        for n, n_leaves in [(3, 5), (5, 7), (7, 10), (2, 5), (1, 3), (99, 100)]:
            leaves = [make_seq_leaf(f"NC_{i}") for i in range(n_leaves)]
            with patch(_MOCK_SEQ_PATH, return_value="A" * 200):  # equal weights
                tasks = _allocate_n_across_leaves(
                    leaves, n_per_class=n, min_subseq_len=100
                )
            total = sum(t["n"] for t in tasks)
            assert total == n, f"n={n}, leaves={n_leaves} -> {total}"

    def test_unequal_weights_produce_proportional_distribution(self):
        short_leaf = make_seq_leaf("NC_short")
        long_leaf = make_seq_leaf("NC_long")

        seq_map = {"NC_short": "A" * 100, "NC_long": "A" * 900}

        def side_effect(fasta_path, header_id):
            return seq_map[header_id]

        with patch(_MOCK_SEQ_PATH, side_effect=side_effect):
            tasks = _allocate_n_across_leaves(
                [short_leaf, long_leaf], n_per_class=100, min_subseq_len=10
            )
        totals = {t["header_id"]: t["n"] for t in tasks}
        assert totals["NC_long"] > totals["NC_short"]

    def test_zero_weight_leaf_is_excluded_from_result(self):
        zero_leaf = make_seq_leaf("NC_tiny")
        good_leaf = make_seq_leaf("NC_good")

        seq_map = {"NC_tiny": "A" * 5, "NC_good": "A" * 500}

        def side_effect(fasta_path, header_id):
            return seq_map[header_id]

        with patch(_MOCK_SEQ_PATH, side_effect=side_effect):
            tasks = _allocate_n_across_leaves(
                [zero_leaf, good_leaf], n_per_class=50, min_subseq_len=100
            )
        header_ids = {t["header_id"] for t in tasks}
        assert "NC_tiny" not in header_ids
        assert "NC_good" in header_ids

    def test_tasks_contain_expected_keys(self):
        leaf = make_seq_leaf("NC_001", "/vault/path")
        with patch(_MOCK_SEQ_PATH, return_value="A" * 200):
            tasks = _allocate_n_across_leaves([leaf], n_per_class=10, min_subseq_len=10)
        assert tasks[0].keys() == {"fasta_path", "header_id", "n"}
        assert tasks[0]["fasta_path"] == "/vault/path"
        assert tasks[0]["header_id"] == "NC_001"


# ---------------------------------------------------------------------------
# distribute_n_per_class_across_leaves
# ---------------------------------------------------------------------------


class TestDistributeNPerClassAcrossLeaves:
    def test_returns_dict_keyed_by_child_taxid(self):
        leaf = make_seq_leaf("NC_001")
        child = make_taxon_node("9000", [leaf])

        with patch(_MOCK_SEQ_PATH, return_value="A" * 500):
            result = distribute_n_per_class_across_leaves(
                n_per_class=20,
                children=[child],
                parent_taxid="1000",
                parent_name="TestParent",
                leaf_cache={},
            )
        assert "9000" in result

    def test_child_with_no_leaves_gets_empty_list(self):
        child = make_taxon_node("9001", [])
        result = distribute_n_per_class_across_leaves(
            n_per_class=20,
            children=[child],
            parent_taxid="1000",
            parent_name="TestParent",
            leaf_cache={},
        )
        assert result["9001"] == []

    def test_all_children_covered_in_output(self):
        leaf_a = make_seq_leaf("NC_A")
        leaf_b = make_seq_leaf("NC_B")
        child_a = make_taxon_node("9002", [leaf_a])
        child_b = make_taxon_node("9003", [leaf_b])

        with patch(_MOCK_SEQ_PATH, return_value="A" * 500):
            result = distribute_n_per_class_across_leaves(
                n_per_class=30,
                children=[child_a, child_b],
                parent_taxid="1000",
                parent_name="TestParent",
                leaf_cache={},
            )
        assert set(result.keys()) == {"9002", "9003"}

    def test_leaf_cache_takes_precedence_over_tree_scan(self):
        cached_leaf = make_seq_leaf("NC_cached")
        real_leaf = make_seq_leaf("NC_real")
        child = make_taxon_node("9004", [real_leaf])

        with patch(_MOCK_SEQ_PATH, return_value="A" * 500):
            result = distribute_n_per_class_across_leaves(
                n_per_class=10,
                children=[child],
                parent_taxid="1000",
                parent_name="TestParent",
                leaf_cache={"9004": [cached_leaf]},
            )
        header_ids = {t["header_id"] for t in result["9004"]}
        assert "NC_cached" in header_ids
        assert "NC_real" not in header_ids

    def test_sum_of_tasks_equals_n_per_class(self):
        leaves = [make_seq_leaf(f"NC_{i}") for i in range(3)]
        child = make_taxon_node("9005", leaves)

        with patch(_MOCK_SEQ_PATH, return_value="A" * 400):
            result = distribute_n_per_class_across_leaves(
                n_per_class=73,
                children=[child],
                parent_taxid="1000",
                parent_name="TestParent",
                leaf_cache={},
            )
        total = sum(t["n"] for t in result["9005"])
        assert total == 73
