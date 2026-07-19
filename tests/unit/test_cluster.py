"""Tests for MinHash genome clustering and cluster-aware splitting (P10 Phase 1)."""

import random
from unittest.mock import patch

from taxotreeset.core._orchestration._cluster import (
    _connected_components,
    _genome_sketch,
    _jaccard,
    cluster_genomes,
)
from taxotreeset.core._orchestration._splits import _materialize_leaf_split

# Two independent random 2 kbp "genomes": near-disjoint 21-mer sets.
_SA = "".join(random.Random(1).choices("ACGT", k=2000))
_SB = "".join(random.Random(2).choices("ACGT", k=2000))
_MOCK = "taxotreeset.core._orchestration._cluster._read_single_sequence"


def _tasks(header_ids):
    return [{"fasta_path": "/vault", "header_id": h, "n": 100} for h in header_ids]


def _seq_map(**overrides):
    base = {f"a{i}": _SA for i in range(1, 4)}
    base.update({f"b{i}": _SB for i in range(1, 4)})
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# sketch / jaccard / components
# ---------------------------------------------------------------------------


class TestSketchAndJaccard:
    def test_identical_sequences_have_identical_sketch(self):
        assert _genome_sketch(_SA, 21, 200) == _genome_sketch(_SA, 21, 200)

    def test_short_sequence_has_empty_sketch(self):
        assert _genome_sketch("ACGT", 21, 200) == frozenset()

    def test_jaccard_identical_is_one(self):
        s = _genome_sketch(_SA, 21, 200)
        assert _jaccard(s, s, 200) == 1.0

    def test_jaccard_independent_is_near_zero(self):
        sa, sb = _genome_sketch(_SA, 21, 200), _genome_sketch(_SB, 21, 200)
        assert _jaccard(sa, sb, 200) < 0.1

    def test_jaccard_empty_is_zero(self):
        assert _jaccard(frozenset(), _genome_sketch(_SA, 21, 200), 200) == 0.0

    def test_connected_components_groups_by_edges(self):
        comps = _connected_components(4, [(0, 1), (2, 3)])
        assert sorted(sorted(c) for c in comps) == [[0, 1], [2, 3]]


# ---------------------------------------------------------------------------
# cluster_genomes — the self-verifying gate
# ---------------------------------------------------------------------------


class TestClusterGenomes:
    def test_two_distinct_lineages_yield_two_clusters(self):
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        with patch(_MOCK, side_effect=lambda p, h: _seq_map()[h]):
            clusters = cluster_genomes(tasks)
        assert clusters is not None
        groups = sorted(
            sorted(t["header_id"][0] for t in c) for c in clusters
        )
        assert groups == [["a", "a", "a"], ["b", "b", "b"]]

    def test_homogeneous_head_returns_none(self):
        tasks = _tasks(["a1", "a2", "a3", "a4"])
        with patch(_MOCK, side_effect=lambda p, h: _SA):
            assert cluster_genomes(tasks) is None

    def test_single_genome_returns_none(self):
        with patch(_MOCK, side_effect=lambda p, h: _SA):
            assert cluster_genomes(_tasks(["a1"])) is None

    def test_over_max_genomes_returns_none_without_reading(self):
        tasks = _tasks([f"x{i}" for i in range(5)])
        with patch(_MOCK) as m:
            assert cluster_genomes(tasks, max_genomes=3) is None
            m.assert_not_called()  # short-circuits before any sequence read

    def test_one_big_cluster_plus_singleton_returns_none(self):
        # a1..a3 identical (cluster of 3) + one lone b -> second-largest is a
        # singleton (< min_cluster_genomes) -> not actionable.
        tasks = _tasks(["a1", "a2", "a3", "b1"])
        with patch(_MOCK, side_effect=lambda p, h: _seq_map()[h]):
            assert cluster_genomes(tasks) is None


# ---------------------------------------------------------------------------
# _materialize_leaf_split — cluster-aware behaviour
# ---------------------------------------------------------------------------


class TestClusterAwareSplit:
    def test_default_off_does_not_read_sequences(self):
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        with patch(_MOCK) as m:
            _materialize_leaf_split(tasks, 0, random.Random(0))  # cluster_aware=False
        m.assert_not_called()

    def test_cluster_aware_spreads_each_lineage_across_val_and_test(self):
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        with patch(_MOCK, side_effect=lambda p, h: _seq_map()[h]):
            split = _materialize_leaf_split(
                tasks, 0, random.Random(0), cluster_aware=True
            )
        # both sub-lineages present in val AND in test (representative split)
        for name in ("val", "test"):
            lineages = {t["header_id"][0] for t in split[name]}
            assert lineages == {"a", "b"}, f"{name} missing a lineage: {lineages}"

    def test_falls_back_to_random_when_no_structure(self, ):
        tasks = _tasks(["a1", "a2", "a3", "a4", "a5", "a6"])
        with patch(_MOCK, side_effect=lambda p, h: _SA):
            split = _materialize_leaf_split(
                tasks, 0, random.Random(0), cluster_aware=True
            )
        # homogeneous -> cluster_genomes returns None -> standard split, all filled
        assert all(split[s] for s in ("train", "val", "test"))
        assert sum(len(split[s]) for s in ("train", "val", "test")) == 6

    def test_cluster_aware_and_off_agree_when_no_structure(self):
        tasks = _tasks(["a1", "a2", "a3", "a4"])
        with patch(_MOCK, side_effect=lambda p, h: _SA):
            on = _materialize_leaf_split(tasks, 0, random.Random(7), cluster_aware=True)
        off = _materialize_leaf_split(tasks, 0, random.Random(7))
        assert on == off  # homogeneous -> identical to the random split
