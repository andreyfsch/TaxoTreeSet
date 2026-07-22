"""Tests for clade-holdout selection, pruning, and manifest (P11-P1)."""

import random
from unittest.mock import patch

from bigtree import Node

from taxotreeset.benchmark.holdout import (
    _deepest_retained_ancestor,
    _is_eligible,
    build_holdout_manifest,
    prune_holdout,
    select_holdout_taxids,
)

_READ = "taxotreeset.benchmark.holdout._read_single_sequence"
_SA = "".join(random.Random(1).choices("ACGT", k=2000))
_SB = "".join(random.Random(2).choices("ACGT", k=2000))


def _seq(name, parent, hdr):
    return Node(name, parent=parent, rank="sequence", header_id=hdr, fasta_path="/v")


def _tree():
    # root ├─ F1(family) ├─ G1,G2,G3 (genus, 1 genome each)
    #      └─ F2(family) └─ G4 (genus, 1 genome)  [G4 is F2's only child]
    root = Node("10239", rank="superkingdom", scientific_name="Viruses")
    f1 = Node("F1", parent=root, rank="family", scientific_name="FamA")
    for g, h in (("G1", "H1"), ("G2", "H2"), ("G3", "H3")):
        node = Node(g, parent=f1, rank="genus", scientific_name=f"Gen{g}")
        _seq(f"{g}s", node, h)
    f2 = Node("F2", parent=root, rank="family", scientific_name="FamB")
    g4 = Node("G4", parent=f2, rank="genus", scientific_name="GenG4")
    _seq("G4s", g4, "H4")
    return root


def _find(root, taxid):
    return next(n for n in root.descendants if str(n.name) == taxid)


class TestEligibility:
    def test_eligible_when_parent_keeps_a_sibling(self):
        assert _is_eligible(_find(_tree(), "G1"))

    def test_ineligible_as_only_child(self):
        # removing G4 would empty F2 -> rho* undefined at the parent
        assert not _is_eligible(_find(_tree(), "G4"))

    def test_ineligible_without_genomes(self):
        root = _tree()
        Node("G9", parent=_find(root, "F1"), rank="genus")  # no sequence leaf
        assert not _is_eligible(_find(root, "G9"))


class TestSelection:
    def test_explicit_keeps_eligible_drops_ineligible(self):
        got = select_holdout_taxids(_tree(), explicit=["G1", "G4", "999"])
        assert got == {"G1"}  # G4 ineligible (only child), 999 absent

    def test_auto_samples_ceil_fraction_of_eligible(self):
        # eligible genera = {G1, G2, G3}; ceil(0.5*3) = 2
        got = select_holdout_taxids(_tree(), rank="genus", fraction=0.5, seed=0)
        assert len(got) == 2
        assert got <= {"G1", "G2", "G3"}

    def test_auto_is_seed_deterministic(self):
        # ceil(0.3 * 3 eligible) = 1
        a = select_holdout_taxids(_tree(), rank="genus", fraction=0.3, seed=7)
        b = select_holdout_taxids(_tree(), rank="genus", fraction=0.3, seed=7)
        assert a == b and len(a) == 1

    def test_dedup_drops_clade_under_a_selected_ancestor(self):
        # F1 and its child G1 both requested -> keep only F1
        assert select_holdout_taxids(_tree(), explicit=["F1", "G1"]) == {"F1"}


class TestExpectedCommitAncestor:
    def test_rho_is_the_nearest_retained_ancestor(self):
        root = _tree()
        assert _deepest_retained_ancestor(_find(root, "G1"), {"G1"}).name == "F1"

    def test_rho_skips_held_out_ancestors(self):
        root = _tree()
        # both F1 and G1 held out -> rho* for G1 is the root, not F1
        assert _deepest_retained_ancestor(_find(root, "G1"), {"G1", "F1"}).name == "10239"


class TestPruning:
    def test_prunes_only_the_held_out_subtree(self):
        root = _tree()
        n = prune_holdout(root, {"G1"})
        assert n == 1
        names = {str(x.name) for x in root.descendants}
        assert "G1" not in names and "G1s" not in names  # subtree gone
        assert {"G2", "G3", "F1"} <= names               # siblings retained

    def test_prune_is_a_noop_for_absent_taxid(self):
        assert prune_holdout(_tree(), {"999"}) == 0


class TestManifest:
    def test_records_members_rho_and_distance(self):
        seqs = {"H1": _SA, "H2": _SB, "H3": _SB, "H4": _SA}
        with patch(_READ, side_effect=lambda p, h: seqs[h]):
            manifest = build_holdout_manifest(_tree(), {"G1"}, seed=7)
        assert len(manifest) == 1
        e = manifest[0]
        assert e["taxid"] == "G1"
        assert e["rank"] == "genus"
        assert e["n_genomes"] == 1
        assert e["member_headers"] == ["H1"]
        assert e["expected_commit_taxid"] == "F1"
        assert e["expected_commit_rank"] == "family"
        assert e["nearest_retained_sibling_taxid"] in {"G2", "G3"}
        assert e["distance_bin"] is not None
        assert 0.0 <= e["distance_ani_proxy"] <= 1.0
        assert e["seed"] == 7

    def test_manifest_built_before_pruning_is_stable(self):
        # manifest is computed on the full tree; a member is always recorded
        seqs = {f"H{i}": _SA for i in range(1, 5)}
        with patch(_READ, side_effect=lambda p, h: seqs[h]):
            manifest = build_holdout_manifest(_tree(), {"G1", "G2"}, seed=0)
        assert {e["taxid"] for e in manifest} == {"G1", "G2"}
        assert all(e["member_headers"] for e in manifest)
