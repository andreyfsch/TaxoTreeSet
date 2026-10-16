"""Tests for the domain gate defaults (P4): resolution of ``auto``, the random
sample of cross-domain genomes, and the even split of the gate windows."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from taxotreeset.core._orchestration._sync import resolve_cross_domain
from taxotreeset.core.generation.reject_bucket import (
    _allocate_cross_domain,
    _cap_cross_pool,
    build_reject_tasks,
    summarise_reject_provenance,
)
from taxotreeset.core.orchestrator import DiscoveryOrchestrator, sample_reports


class TestResolveCrossDomain:
    @pytest.mark.parametrize("root, expected", [
        ("archaea", ["viruses", "bacteria", "eukaryotes"]),
        ("viruses", ["bacteria", "archaea", "eukaryotes"]),
        ("2157", ["viruses", "bacteria", "eukaryotes"]),
        ("Archaea", ["viruses", "bacteria", "eukaryotes"]),
    ])
    def test_auto_is_every_other_domain(self, root, expected):
        assert resolve_cross_domain("auto", root) == expected

    @pytest.mark.parametrize("root", ["all", "archaea,viruses", "28890", None, ""])
    def test_auto_is_off_without_a_single_domain_root(self, root):
        assert resolve_cross_domain("auto", root) == []

    def test_auto_is_off_for_plasmids(self):
        assert resolve_cross_domain("auto", "archaea", plasmids=True) == []

    @pytest.mark.parametrize("spec", [None, "none", "off", "", " NONE "])
    def test_disabled(self, spec):
        assert resolve_cross_domain(spec, "archaea") == []

    def test_explicit_list_is_kept(self):
        assert resolve_cross_domain("bacteria, eukaryotes", "archaea") == [
            "bacteria", "eukaryotes"]
        assert resolve_cross_domain(["bacteria"], "archaea") == ["bacteria"]


def _report(acc, size=1000, taxid=1):
    return {"accession": acc, "assembly_stats": {"total_sequence_length": size},
            "organism": {"tax_id": taxid}}


class TestSampleReports:
    def test_is_a_seeded_sample_not_a_prefix(self):
        reports = [_report(f"GCF_{i:04d}") for i in range(500)]
        a = sample_reports(reports, 20, seed=7)
        assert a == sample_reports(reports, 20, seed=7)
        assert [r["accession"] for r in a] != [f"GCF_{i:04d}" for i in range(20)]
        assert sample_reports(reports, 20, seed=8) != a

    def test_independent_of_stream_order(self):
        reports = [_report(f"GCF_{i:04d}") for i in range(300)]
        assert sample_reports(reports, 30, seed=1) == sample_reports(
            list(reversed(reports)), 30, seed=1)

    def test_size_cap_applies_before_the_draw(self):
        reports = [_report("big", 300_000_000)] + [_report(f"s{i}", 10) for i in range(5)]
        got = sample_reports(reports, 10, seed=0, max_genome_bp=200_000_000)
        assert {r["accession"] for r in got} == {f"s{i}" for i in range(5)}

    def test_exclusions_are_skipped_and_replaced(self):
        reports = [_report(f"GCF_{i}", taxid=i) for i in range(10)]
        got = sample_reports(reports, 5, seed=3, is_excluded=lambda r: r["organism"]["tax_id"] < 5)
        assert len(got) == 5 and all(r["organism"]["tax_id"] >= 5 for r in got)

    def test_limit_zero(self):
        assert sample_reports([_report("a")], 0) == []


class TestStreamReferenceReports:
    def _disc(self):
        disc = DiscoveryOrchestrator.__new__(DiscoveryOrchestrator)
        disc.assembly_source = "RefSeq"
        return disc

    def test_falls_back_to_all_assemblies_when_no_reference(self):
        # RefSeq flags no viral genome as reference (0 reports on 2026-10-15).
        disc = self._disc()
        calls = []

        def stream(cmd):
            calls.append(cmd)
            return [] if "--reference" in cmd else [_report("NC_1"), _report("NC_2")]

        disc._stream_reports = stream
        got = disc.stream_reference_reports("10239", 5, seed=0)
        assert {r["accession"] for r in got} == {"NC_1", "NC_2"}
        assert "--reference" in calls[0] and "--reference" not in calls[1]

    def test_excludes_by_lineage(self):
        disc = self._disc()
        disc._stream_reports = lambda cmd: [_report("ecoli", taxid=511145),
                                            _report("other", taxid=1423)]
        disc._lineage_taxids = lambda t: {"511145": {"511145", "562", "561", "2"},
                                          "1423": {"1423", "1386", "2"}}[str(t)]
        got = disc.stream_reference_reports("2", 5, exclude_taxids=["561"])
        assert [r["accession"] for r in got] == ["other"]


def _xleaf(acc, header, domain):
    return SimpleNamespace(rank="sequence", header_id=header, fasta_path="/vault.lmdb",
                           cross_domain=True, clade=f"cross_domain:{domain}",
                           accession=acc)


def _alloc_by_n(leaves, n, m):
    # share the n windows evenly over the given leaves (length-free stand-in)
    k = len(leaves)
    return [{"header_id": leaf.header_id, "n": n // k + (1 if i < n % k else 0)}
            for i, leaf in enumerate(leaves) if n // k + (1 if i < n % k else 0)]


@patch("taxotreeset.core.generation.reject_bucket._allocate_n_across_leaves",
       side_effect=_alloc_by_n)
class TestAllocateCrossDomain:
    def test_even_by_domain_then_by_genome(self, _):
        # 1 bacterium with 1 sequence, 1 bacterium with 2, 1 eukaryote with 40
        # scaffolds: by length or by leaf the eukaryote would take nearly all.
        leaves = ([_xleaf("B1", "b1", "bacteria")]
                  + [_xleaf("B2", f"b2_{i}", "bacteria") for i in range(2)]
                  + [_xleaf("E1", f"e{i}", "eukaryotes") for i in range(40)])
        tasks = _allocate_cross_domain(leaves, 60, 100)
        per_domain = {}
        for t in tasks:
            per_domain[t["cross_domain"]] = per_domain.get(t["cross_domain"], 0) + t["n"]
        assert per_domain == {"cross_domain:bacteria": 30, "cross_domain:eukaryotes": 30}
        b1 = sum(t["n"] for t in tasks if t["header_id"] == "b1")
        assert b1 == 15  # half of the bacterial share, though it has one sequence

    def test_build_reject_tasks_gives_cross_its_own_split(self, _):
        intra = [SimpleNamespace(header_id=f"v{i}") for i in range(3)]
        cross = [_xleaf("B1", "b1", "bacteria"), _xleaf("E1", "e1", "eukaryotes")]
        tasks = build_reject_tasks([], intra + cross, n_reject=100, near_far_ratio=0.5,
                                   min_subseq_len=100)
        assert sum(t["n"] for t in tasks if t.get("cross_domain")) == 50
        assert sum(t["n"] for t in tasks if not t.get("cross_domain")) == 50
        assert all(t["reject_pool"] == "far" for t in tasks)

    def test_cross_only_takes_the_whole_budget(self, _):
        cross = [_xleaf("B1", "b1", "bacteria"), _xleaf("V1", "v1", "viruses")]
        tasks = build_reject_tasks([], cross, n_reject=90, near_far_ratio=0.5,
                                   min_subseq_len=100)
        assert sum(t["n"] for t in tasks) == 90


def test_cap_cross_pool_keeps_every_genome():
    import random
    leaves = ([_xleaf("E1", f"e{i}", "eukaryotes") for i in range(500)]
              + [_xleaf(f"B{i}", f"b{i}", "bacteria") for i in range(30)])
    kept = _cap_cross_pool(leaves, 20, random.Random(0))
    assert sum(1 for leaf in kept if leaf.accession == "E1") == 20
    assert {leaf.accession for leaf in kept} >= {f"B{i}" for i in range(30)}


def test_provenance_reports_windows_and_genomes_per_domain():
    leaves = [_xleaf("B1", "b1", "bacteria"), _xleaf("B2", "b2", "bacteria"),
              _xleaf("E1", "e1", "eukaryotes")]
    tasks = [{"header_id": "b1", "n": 10, "reject_pool": "far",
              "cross_domain": "cross_domain:bacteria"},
             {"header_id": "b2", "n": 10, "reject_pool": "far",
              "cross_domain": "cross_domain:bacteria"},
             {"header_id": "e1", "n": 20, "reject_pool": "far",
              "cross_domain": "cross_domain:eukaryotes"}]
    out = summarise_reject_provenance([], leaves, tasks)
    assert out["cross_domain"] == {
        "cross_domain:bacteria": {"windows": 20, "genomes": 2},
        "cross_domain:eukaryotes": {"windows": 20, "genomes": 1}}
    assert set(out["far"]["source_clades"]) == {"cross_domain:bacteria",
                                                "cross_domain:eukaryotes"}
