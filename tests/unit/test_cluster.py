"""Tests for MinHash genome clustering and cluster-aware splitting (P10 Phase 1)."""

import random
from unittest.mock import patch

import pytest

from taxotreeset.core._orchestration._cluster import (
    ClusterParams,
    _connected_components,
    _genome_sketch,
    _jaccard,
    cluster_genomes,
)
from taxotreeset.core._orchestration._splits import (
    _block_stratified_windows,
    _cluster_stratified_split,
    _even_split,
    _materialize_leaf_split,
)

_SPLIT_MOCK = "taxotreeset.core._orchestration._splits._read_single_sequence"

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

    def test_diverse_head_with_tiny_pairs_returns_none(self):
        # 26 distinct genomes + 2 near-clone pairs: each pair is < min_cluster_frac
        # of 30, so not actionable (the real 2732529 / RefSeq-diversity case).
        seqs = {
            f"s{i}": "".join(random.Random(100 + i).choices("ACGT", k=1500))
            for i in range(26)
        }
        pa = "".join(random.Random(900).choices("ACGT", k=1500))
        pb = "".join(random.Random(901).choices("ACGT", k=1500))
        seqs.update({"pa1": pa, "pa2": pa, "pb1": pb, "pb2": pb})
        tasks = _tasks(list(seqs))
        with patch(_MOCK, side_effect=lambda p, h: seqs[h]):
            assert cluster_genomes(tasks) is None

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

    def test_diverse_no_clusters_window_slices_all_covered(self):
        tasks = _tasks(["a1", "a2", "a3", "a4", "a5", "a6"])
        with patch(_MOCK, side_effect=lambda p, h: _SA):
            split = _materialize_leaf_split(
                tasks, 0, random.Random(0), cluster_aware=True
            )
        # no substantial clusters -> window-slice each genome; every genome is held
        # out in train AND val AND test (covered), so all folds are filled
        all_genomes = {t["header_id"] for t in tasks}
        for s in ("train", "val", "test"):
            assert {t["header_id"] for t in split[s]} == all_genomes

    def test_cluster_aware_window_slices_a_diverse_clade_off_does_not(self):
        # No clusters / no near-clones: cluster-aware window-slices each genome
        # (held-out windows, all covered), while --no-cluster-aware-split keeps the
        # whole-genome volume split (each genome in exactly one fold).
        seqs = {h: "".join(random.Random(i).choices("ACGT", k=6000))
                for i, h in enumerate(["a1", "a2", "a3", "a4"])}
        tasks = _tasks(["a1", "a2", "a3", "a4"])
        with patch(_MOCK, side_effect=lambda p, h: seqs[h]):
            on = _materialize_leaf_split(tasks, 0, random.Random(7), cluster_aware=True)
        off = _materialize_leaf_split(tasks, 0, random.Random(7))
        assert on != off
        for s in ("train", "val", "test"):              # on: every genome covered
            assert {t["header_id"] for t in on[s]} == set(seqs)
        off_seen = [t["header_id"] for s in ("train", "val", "test") for t in off[s]]
        assert len(off_seen) == len(set(off_seen)) == 4  # off: each in one fold


# ---------------------------------------------------------------------------
# Block-stratified positional split — the single/few-genome fix (P10 Phase 1b)
# ---------------------------------------------------------------------------


class TestEvenSplit:
    def test_sums_and_balances(self):
        assert _even_split(10, 3) == [4, 3, 3]
        assert sum(_even_split(7, 4)) == 7


class TestBlockStratifiedWindows:
    def test_interleaves_val_and_test_into_the_interior(self):
        # 1000 bp genome, max_subseq_len 100 -> 10 equal-width blocks; _label_blocks
        # spreads val/test into the interior, not the 70-85/85-100 contiguous ends.
        task = {"fasta_path": "/v", "header_id": "g1", "n": 300}
        result = {s: [] for s in ("train", "val", "test")}
        with patch(_SPLIT_MOCK, return_value="A" * 1000):
            emitted = _block_stratified_windows(task, 0, 100, result)
        assert emitted
        assert any(t["start_pct"] < 0.85 for t in result["test"])  # interior
        assert any(t["start_pct"] < 0.70 for t in result["val"])   # interior
        assert all(result[s] for s in ("train", "val", "test"))
        assert sum(t["n"] for s in result for t in result[s]) == 300      # budget kept

    def test_uses_precomputed_length_without_reading(self):
        # When the task carries "length" (attached during task distribution), the
        # block split must use it and NOT re-read the genome.
        task = {"fasta_path": "/v", "header_id": "g1", "n": 300, "length": 1000}
        result = {s: [] for s in ("train", "val", "test")}
        with patch(_SPLIT_MOCK) as m:
            emitted = _block_stratified_windows(task, 0, 100, result)  # 10 blocks
            m.assert_not_called()
        assert emitted
        assert all(result[s] for s in ("train", "val", "test"))

    def test_short_genome_falls_back(self):
        task = {"fasta_path": "/v", "header_id": "g", "n": 100}
        result = {s: [] for s in ("train", "val", "test")}
        with patch(_SPLIT_MOCK, return_value="A" * 250):  # 2 blocks < 3
            assert _block_stratified_windows(task, 0, 100, result) is False
        assert all(not result[s] for s in ("train", "val", "test"))

    def test_unreadable_genome_falls_back(self):
        result = {s: [] for s in ("train", "val", "test")}
        with patch(_SPLIT_MOCK, return_value=""):
            assert _block_stratified_windows(
                {"fasta_path": "/v", "header_id": "g", "n": 100}, 0, 100, result
            ) is False


class TestWindowSliceLengthConsistency:
    """Regression for the audit-caught confound: window LENGTH must not differ by
    split. ``extract_subseqs`` clamps each window to the bp left in its slice, so
    the old unequal-width cut (train 70%, val/test 15%) made val/test windows
    shorter than train — a length-to-class shortcut that collapsed heads to
    <= chance on val. Every split must now draw from equal-width regions and match
    in window-length distribution.
    """

    def _median_lengths(self, split, genome, max_len):
        import statistics

        from taxotreeset.dataset.sequence_utils import extract_subseqs

        med = {}
        for s in ("train", "val", "test"):
            lens = []
            for t in split[s]:
                sub = genome[int(len(genome) * t["start_pct"]):
                             int(len(genome) * t["end_pct"])]
                lens += [len(w) for w in extract_subseqs(
                    sub, t["n"], min_len=50, max_len=max_len,
                    rng=random.Random(0))]
            med[s] = statistics.median(lens) if lens else 0
        return med

    def test_three_blocks_now_use_the_block_path(self):
        # _MIN_BLOCKS_FOR_STRATIFY lowered 6 -> 3: a 3-block genome block-stratifies
        # (equal width) instead of falling back to the contiguous cut.
        task = {"fasta_path": "/v", "header_id": "g", "n": 90, "length": 300}
        result = {s: [] for s in ("train", "val", "test")}
        assert _block_stratified_windows(task, 0, 100, result) is True
        assert all(result[s] for s in ("train", "val", "test"))

    def test_block_path_medians_match_across_splits(self):
        genome = "".join(random.Random(1).choices("ACGT", k=20000))
        task = {"fasta_path": "/v", "header_id": "g", "n": 400,
                "length": len(genome)}
        split = _materialize_leaf_split(
            [task], 0, random.Random(0), cluster_aware=True, max_subseq_len=500)
        med = self._median_lengths(split, genome, 500)
        assert all(med.values())
        assert max(med.values()) - min(med.values()) <= 60, med

    def test_short_genome_thirds_medians_match(self):
        # 3000 bp, max 2000 -> 1 block < 3 -> equal-width thirds fallback.
        genome = "".join(random.Random(2).choices("ACGT", k=3000))
        task = {"fasta_path": "/v", "header_id": "g", "n": 90,
                "length": len(genome)}
        split = _materialize_leaf_split(
            [task], 0, random.Random(0), cluster_aware=True, max_subseq_len=2000)
        med = self._median_lengths(split, genome, 2000)
        assert all(med.values())
        assert max(med.values()) - min(med.values()) <= 120, med


class TestSegmentedGenomeGrouping:
    """A segmented/multi-contig genome (one accession, many sequences sharing a
    ``genome_key``) is ONE genome. Below the genome-count threshold it block-
    stratifies each segment (every segment reaches every split); at/above it a
    genome's segments never straddle folds. Regression for the Hadaka collapse
    (11 segments of one virus split whole -> disjoint segments in train vs val).
    """

    def _segments(self, genome_key, n_segs, n=300, length=20000):
        return [{"fasta_path": "/v", "header_id": f"{genome_key}_s{i}",
                 "genome_key": genome_key, "n": n, "length": length}
                for i in range(n_segs)]

    def test_single_segmented_genome_uses_window_slicing(self):
        # 11 segments of ONE genome -> 1 genome < 4 -> window-slice each segment,
        # so EVERY segment appears in train AND val AND test (not split whole).
        tasks = self._segments("GCF_1", 11)
        split = _materialize_leaf_split(
            tasks, 1, random.Random(0), cluster_aware=True, max_subseq_len=2000,
            min_genomes_for_genome_split=4)
        assert all(split[s] for s in ("train", "val", "test"))
        for s in ("train", "val", "test"):
            seg_ids = {t["header_id"] for t in split[s]}
            assert len(seg_ids) == 11, (s, seg_ids)  # every segment present

    def test_segmented_genome_never_straddles_in_genome_level(self):
        # 4 genomes, one segmented (3 segments). Genome-level split; the segmented
        # genome's 3 segments must all land in the SAME split (leakage-safe).
        tasks = (self._segments("GCF_seg", 3, length=5000)
                 + [{"fasta_path": "/v", "header_id": f"g{i}",
                     "genome_key": f"g{i}", "n": 300, "length": 5000}
                    for i in range(3)])
        split = _materialize_leaf_split(
            tasks, 1, random.Random(0), cluster_aware=False,
            min_genomes_for_genome_split=4)
        holds = {s for s in ("train", "val", "test")
                 if any(t["header_id"].startswith("GCF_seg") for t in split[s])}
        assert len(holds) == 1, f"segmented genome straddled {holds}"


class TestClusterParams:
    def test_defaults_match_module_constants(self):
        from taxotreeset.core._orchestration import _cluster as C
        cp = ClusterParams()
        assert (cp.k, cp.sketch_size, cp.jaccard_threshold) == (
            C._KMER_K, C._SKETCH_SIZE, C._JACCARD_THRESHOLD)
        assert (cp.min_cluster_genomes, cp.min_cluster_frac, cp.max_genomes) == (
            C._MIN_CLUSTER_GENOMES, C._MIN_CLUSTER_FRAC, C._MAX_GENOMES)

    def test_params_are_forwarded_to_cluster_genomes(self):
        # The dataclass fields must reach cluster_genomes under its kwarg names
        # (jaccard_threshold -> threshold), so the CLI knobs actually take effect.
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        cp = ClusterParams(
            k=15, sketch_size=64, jaccard_threshold=0.55,
            min_cluster_genomes=4, min_cluster_frac=0.25, max_genomes=99)
        with patch(
            "taxotreeset.core._orchestration._splits.cluster_genomes_adaptive",
            return_value=None,
        ) as m:
            _materialize_leaf_split(
                tasks, 0, random.Random(0), cluster_aware=True, cluster_params=cp)
        m.assert_called_once()
        kwargs = m.call_args.kwargs
        assert kwargs["k"] == 15
        assert kwargs["sketch_size"] == 64
        assert kwargs["threshold"] == 0.55
        assert kwargs["min_cluster_genomes"] == 4
        assert kwargs["min_cluster_frac"] == 0.25
        assert kwargs["max_genomes"] == 99

    def test_high_min_frac_suppresses_actionable_structure(self):
        # 3 identical a's + 3 identical b's: default fires (two size-3 clusters);
        # requiring each cluster to cover 90% of the 6 genomes disqualifies both.
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        with patch(_MOCK, side_effect=lambda p, h: _seq_map()[h]):
            fired = _cluster_stratified_split(tasks, 0, 3, ClusterParams())
            suppressed = _cluster_stratified_split(
                tasks, 0, 3, ClusterParams(min_cluster_frac=0.9))
        assert fired is not None
        assert suppressed is None

    def test_none_params_behave_like_defaults(self):
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        with patch(_MOCK, side_effect=lambda p, h: _seq_map()[h]):
            explicit = _cluster_stratified_split(tasks, 0, 3, ClusterParams())
            implicit = _cluster_stratified_split(tasks, 0, 3, None)
        assert (explicit is None) == (implicit is None)
        if explicit is not None:
            for split in ("train", "val", "test"):
                assert {t["header_id"][0] for t in explicit[split]} == {
                    t["header_id"][0] for t in implicit[split]}


class TestAdaptiveClustering:
    """``cluster_genomes_adaptive`` tries the default threshold, then relaxes if it
    zeroes — so a diverse clade (coronaviruses: every pairwise Jaccard below the
    default) recovers segregable sub-lineages instead of falling through to a
    non-representative volume split (val great, test ~chance)."""

    def test_relaxes_when_default_zeroes(self):
        from taxotreeset.core._orchestration import _cluster as C
        tasks = _tasks(["a1", "a2", "a3", "b1", "b2", "b3"])
        seen = []

        def fake_cluster_at(sketches, n, threshold, sketch_size, min_size):
            seen.append(threshold)
            return [[0, 1, 2], [3, 4, 5]] if threshold <= 0.15 else None

        with patch(_MOCK, side_effect=lambda p, h: _SA), \
             patch.object(C, "_cluster_at", side_effect=fake_cluster_at):
            out = C.cluster_genomes_adaptive(tasks)
        assert out is not None                     # relaxation recovered structure
        assert seen[0] == C._JACCARD_THRESHOLD     # tried the default first
        assert any(t <= 0.15 for t in seen)        # then relaxed lower

    def test_no_relax_when_default_finds_structure(self):
        from taxotreeset.core._orchestration import _cluster as C
        tasks = _tasks(["a1", "a2", "b1", "b2"])
        seen = []

        def fake_cluster_at(sketches, n, threshold, sketch_size, min_size):
            seen.append(threshold)
            return [[0, 1], [2, 3]]

        with patch(_MOCK, side_effect=lambda p, h: _SA), \
             patch.object(C, "_cluster_at", side_effect=fake_cluster_at):
            out = C.cluster_genomes_adaptive(tasks)
        assert out is not None
        assert seen == [C._JACCARD_THRESHOLD]      # first try succeeded, no relax

    def test_returns_none_when_even_loosest_finds_nothing(self):
        from taxotreeset.core._orchestration import _cluster as C
        tasks = _tasks(["a1", "a2", "a3"])
        with patch(_MOCK, side_effect=lambda p, h: _SA), \
             patch.object(C, "_cluster_at", return_value=None):
            assert C.cluster_genomes_adaptive(tasks) is None


class TestWindowsliceDiverse:
    """Diverse genome-level clade with no substantial clusters: window-slice EACH
    genome so val/test are held-out WINDOWS of the same genomes (all covered),
    fixing val-great / test-chance where a whole-genome split strands a
    sub-lineage."""

    def test_diverse_genome_level_window_slices_every_genome(self):
        # 5 dissimilar >= 3-block genomes, no clusters -> window-slice each genome;
        # every genome appears in train AND is held out in val AND test.
        seqmap = {h: "".join(random.Random(i).choices("ACGT", k=6000))
                  for i, h in enumerate(["g0", "g1", "g2", "g3", "g4"])}
        tasks = _tasks(list(seqmap))
        with patch(_MOCK, side_effect=lambda p, h: seqmap[h]):
            split = _materialize_leaf_split(
                tasks, 1, random.Random(0), cluster_aware=True,
                min_genomes_for_genome_split=4)
        for s in ("train", "val", "test"):
            assert {t["header_id"] for t in split[s]} == set(seqmap)  # all covered


class TestClusterAwareWindowSlicing:
    def test_on_uses_interior_blocks(self):
        # 1 genome (< 3) -> window-slicing; long genome -> block-stratified path.
        tasks = [{"fasta_path": "/v", "header_id": "g1", "n": 300}]
        with patch(_SPLIT_MOCK, return_value="A" * 1000):
            split = _materialize_leaf_split(
                tasks, 0, random.Random(0), cluster_aware=True, max_subseq_len=100
            )
        assert any(t["start_pct"] < 0.85 for t in split["test"])  # interior

    def test_off_uses_contiguous_equal_thirds_without_reading(self):
        tasks = [{"fasta_path": "/v", "header_id": "g1", "n": 300}]
        with patch(_SPLIT_MOCK) as m:
            split = _materialize_leaf_split(tasks, 0, random.Random(0))
            m.assert_not_called()
        # equal-width thirds (the length-clamp fix), not the old 0.70/0.85 cut
        assert split["val"][0]["start_pct"] == pytest.approx(1 / 3)
        assert split["test"][0]["start_pct"] == pytest.approx(2 / 3)


class TestBlockGridIsWindowLengthInvariant:
    """The held-out REGIONS of a genome must not move when the window length changes.

    Blocks were once ``length // max_subseq_len`` wide while :func:`_label_blocks`
    assigns splits by block INDEX, so a 20 kb genome cut at 250 bp (80 blocks) and the
    same genome cut at 1100 bp (18 blocks) held out different PARTS of the genome.
    Measured on head 11049 before the fix: 72% of one generation's test windows fell
    inside the other generation's train regions, which invalidated every comparison
    between a head and its regenerated version — including the reject-bucket wins.
    """

    @staticmethod
    def _regions(length: int, max_subseq_len: int) -> dict[str, list[tuple]]:
        task = {"fasta_path": "/v", "header_id": "g", "n": 1200, "length": length}
        result = {s: [] for s in ("train", "val", "test")}
        assert _block_stratified_windows(task, 0, max_subseq_len, result) is True
        return {
            s: sorted((round(t["start_pct"], 6), round(t["end_pct"], 6))
                      for t in result[s])
            for s in result
        }

    @pytest.mark.parametrize("length", [20_000, 30_000, 50_000, 200_000])
    def test_same_regions_at_250_and_1100_bp(self, length):
        assert self._regions(length, 250) == self._regions(length, 1100)

    def test_same_regions_across_a_wide_sweep_of_window_lengths(self):
        # Every window length that fits _SPLIT_TARGET_BLOCKS blocks in a 60 kb genome
        # (60000 // 12 = 5000) must produce the identical grid.
        base = self._regions(60_000, 100)
        for max_len in (150, 250, 500, 1000, 1100, 2000, 5000):
            assert self._regions(60_000, max_len) == base, f"divergiu em {max_len}"

    def test_val_and_test_regions_are_disjoint_from_train(self):
        r = self._regions(30_000, 1100)
        train = r["train"]
        for split in ("val", "test"):
            for start, end in r[split]:
                assert all(end <= ts or start >= te for ts, te in train)

    def test_window_bound_genomes_are_the_documented_exception(self):
        # 5 kb cannot hold 12 blocks of 1100 bp; the grid is window-bound there and
        # still moves. Documented in _SPLIT_TARGET_BLOCKS as irreducible.
        assert self._regions(5_000, 250) != self._regions(5_000, 1100)
