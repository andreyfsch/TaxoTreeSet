"""Tests for DiscoveryOrchestrator.discover_from_reports (P9, bottom-up path)."""

import json

import pytest
from taxotreeset.core.orchestrator import DiscoveryOrchestrator, _Ancestor
from taxotreeset.io.registry import NCBIRegistry


@pytest.fixture
def mapping_file(tmp_path):
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps({"scopes": {}}), encoding="utf-8")
    return str(p)


@pytest.fixture
def orch(tmp_path, mapping_file):
    registry = NCBIRegistry(
        registry_path=str(tmp_path / "registry.json"), config_path=mapping_file)
    return DiscoveryOrchestrator(registry=registry, mapping_config={})


def _report(accession, host_taxid, organism, length=100):
    return {
        "accession": accession,
        "assembly_info": {"assembly_level": "Complete Genome"},
        "organism": {"organism_name": organism, "tax_id": host_taxid},
        "assembly_stats": {"total_sequence_length": length},
    }


class TestGroupReportsByHost:
    def test_groups_by_host_taxid(self):
        grouped = DiscoveryOrchestrator._group_reports_by_host([
            _report("A.1", 562, "E. coli"),
            _report("B.1", 562, "E. coli"),
            _report("C.1", 1280, "S. aureus"),
        ])
        assert set(grouped) == {"562", "1280"}
        assert [r["accession"] for r in grouped["562"]] == ["A.1", "B.1"]

    def test_drops_reports_without_host(self):
        grouped = DiscoveryOrchestrator._group_reports_by_host(
            [{"accession": "X.1", "organism": {}}])
        assert grouped == {}


class TestDiscoverFromReports:
    def test_registers_and_marks_downloaded(self, orch):
        # Host 562 resolves to a two-rank lineage (leaf tax_id == host, so the
        # self-node branch is skipped).
        orch._resolve_lineage = lambda taxid: [
            _Ancestor(562, "species", "Escherichia coli"),
            _Ancestor(2, "superkingdom", "Bacteria"),
        ]
        orch.discover_from_reports(
            [_report("NZ_P1.1", 562, "Escherichia coli")],
            root_id_str="plasmids",
            vault_lmdb_path="/vault/sequences.lmdb",
        )
        reg = orch.registry.registry
        # Registered under its host, with the resolved lineage.
        assert reg["taxons"]["562"] == ["NZ_P1.1"]
        assert reg["lineages"]["562"][0]["name"] == "Escherichia coli"
        # Marked downloaded against the vault it was ingested into.
        acc = reg["accessions"]["NZ_P1.1"]
        assert acc["downloaded"] is True
        assert acc["local_path"] == "/vault/sequences.lmdb"
        assert acc["headers"] == [{"id": "NZ_P1.1", "name": "Escherichia coli"}]

    def test_unresolvable_host_is_skipped_not_raised(self, orch):
        def _resolve(taxid):
            raise RuntimeError("obsolete taxid")

        orch._resolve_lineage = _resolve
        # No crash; nothing registered, nothing marked downloaded.
        orch.discover_from_reports(
            [_report("NZ_ORPHAN.1", 999999, "gone")],
            root_id_str="plasmids", vault_lmdb_path="/vault/sequences.lmdb")
        assert orch.registry.registry["accessions"] == {}

    def test_empty_reports_is_a_noop(self, orch):
        orch.discover_from_reports([], root_id_str="plasmids")
        assert orch.registry.registry["accessions"] == {}

    def test_without_vault_path_registers_but_leaves_pending(self, orch):
        orch._resolve_lineage = lambda taxid: [
            _Ancestor(562, "species", "Escherichia coli")]
        orch.discover_from_reports(
            [_report("NZ_P1.1", 562, "Escherichia coli")], root_id_str="plasmids")
        acc = orch.registry.registry["accessions"]["NZ_P1.1"]
        # Registered (present) but not marked downloaded — a normal pending entry.
        assert acc.get("downloaded") in (False, None)
