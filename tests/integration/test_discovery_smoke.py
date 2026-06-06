"""Smoke test for DiscoveryOrchestrator.discover_from_root without network.

Mocks subprocess.Popen (the NCBI Datasets CLI) and taxoniq.Taxon (the
offline lineage resolver) to verify the full discovery → registry flow
without any live network calls.

Validates that the orchestrator correctly processes synthetic assembly
reports, resolves lineages, populates the registry, and writes to disk.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from taxotreeset.core.orchestrator import DiscoveryOrchestrator
from taxotreeset.io.registry import NCBIRegistry


# ---------------------------------------------------------------------------
# synthetic NCBI reports
# ---------------------------------------------------------------------------

_SARS_COV2_TAXID = 2697049
_MHV_TAXID = 11234

_FAKE_REPORTS = [
    {
        "organism": {"tax_id": _SARS_COV2_TAXID},
        "accession": "GCF_SMOKE_001",
        "assembly_info": {"assembly_level": "Complete Genome"},
        "assembly_stats": {"total_sequence_length": 29903},
    },
    {
        "organism": {"tax_id": _MHV_TAXID},
        "accession": "GCF_SMOKE_002",
        "assembly_info": {"assembly_level": "Complete Genome"},
        "assembly_stats": {"total_sequence_length": 31526},
    },
]


def _fake_jsonlines() -> list[str]:
    return [json.dumps(r) + "\n" for r in _FAKE_REPORTS]


# ---------------------------------------------------------------------------
# fake subprocess process
# ---------------------------------------------------------------------------


def _make_fake_process(lines: list[str]) -> MagicMock:
    process = MagicMock()
    process.stdout = iter(lines)
    process.stderr = iter([])
    process.wait.return_value = 0
    return process


# ---------------------------------------------------------------------------
# fake taxoniq lineage
# ---------------------------------------------------------------------------


def _fake_ancestor(tax_id: int, rank: str, name: str) -> SimpleNamespace:
    """Build a fake taxoniq ancestor node matching the _Ancestor NamedTuple fields."""
    rank_obj = SimpleNamespace(name=rank)
    return SimpleNamespace(tax_id=tax_id, rank=rank_obj, scientific_name=name)


def _fake_lineage_for(taxid: int) -> list:
    """Return a canonical species-to-superkingdom lineage for known taxids."""
    if taxid == _SARS_COV2_TAXID:
        return [
            _fake_ancestor(_SARS_COV2_TAXID, "species", "Severe acute respiratory syndrome coronavirus 2"),
            _fake_ancestor(694002, "subgenus", "Sarbecovirus"),
            _fake_ancestor(11118, "family", "Coronaviridae"),
            _fake_ancestor(10239, "superkingdom", "Viruses"),
        ]
    if taxid == _MHV_TAXID:
        return [
            _fake_ancestor(_MHV_TAXID, "species", "Mouse hepatitis virus"),
            _fake_ancestor(11118, "family", "Coronaviridae"),
            _fake_ancestor(10239, "superkingdom", "Viruses"),
        ]
    return []


class _FakeTaxon:
    """Minimal taxoniq.Taxon stand-in."""

    def __init__(self, taxid: int):
        self._taxid = taxid
        lineage = _fake_lineage_for(taxid)
        self._lineage = lineage if lineage else []
        self._self = self._lineage[0] if self._lineage else _fake_ancestor(taxid, "no_rank", str(taxid))

    @property
    def tax_id(self) -> int:
        return self._taxid

    @property
    def rank(self):
        return self._self.rank

    @property
    def scientific_name(self) -> str:
        return self._self.scientific_name

    @property
    def ranked_lineage(self) -> list:
        return self._lineage


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def smoke_registry(tmp_path, minimal_mapping):
    registry_path = str(tmp_path / "registry.json")
    return NCBIRegistry(registry_path=registry_path, config_path=minimal_mapping)


@pytest.fixture
def smoke_orchestrator(smoke_registry, minimal_mapping):
    import json as _json
    with open(minimal_mapping, encoding="utf-8") as fh:
        mapping_config = _json.load(fh)
    return DiscoveryOrchestrator(
        registry=smoke_registry,
        mapping_config=mapping_config,
    )


@pytest.fixture
def discovered_smoke(smoke_orchestrator, smoke_registry):
    """Run discover_from_root with all external calls mocked."""
    fake_process = _make_fake_process(_fake_jsonlines())

    with (
        patch(
            "taxotreeset.core.orchestrator.subprocess.Popen",
            return_value=fake_process,
        ),
        patch(
            "taxotreeset.core.orchestrator.taxoniq.Taxon",
            side_effect=_FakeTaxon,
        ),
    ):
        smoke_orchestrator.discover_from_root(10239)

    return smoke_registry


# ---------------------------------------------------------------------------
# registry population
# ---------------------------------------------------------------------------


class TestDiscoverySmokeRegistryPopulation:
    def test_both_accessions_are_registered(self, discovered_smoke):
        accessions = discovered_smoke.registry["accessions"]
        assert "GCF_SMOKE_001" in accessions
        assert "GCF_SMOKE_002" in accessions

    def test_sars_taxon_linked_to_accession(self, discovered_smoke):
        taxons = discovered_smoke.registry["taxons"]
        assert str(_SARS_COV2_TAXID) in taxons
        assert "GCF_SMOKE_001" in taxons[str(_SARS_COV2_TAXID)]

    def test_mhv_taxon_linked_to_accession(self, discovered_smoke):
        taxons = discovered_smoke.registry["taxons"]
        assert str(_MHV_TAXID) in taxons
        assert "GCF_SMOKE_002" in taxons[str(_MHV_TAXID)]

    def test_accession_entry_has_expected_fields(self, discovered_smoke):
        info = discovered_smoke.registry["accessions"]["GCF_SMOKE_001"]
        assert "taxid" in info
        assert "is_reference" in info
        assert "total_sequence_length" in info
        assert "downloaded" in info

    def test_sars_accession_is_marked_reference(self, discovered_smoke):
        info = discovered_smoke.registry["accessions"]["GCF_SMOKE_001"]
        assert info["is_reference"] is True

    def test_sars_accession_has_correct_sequence_length(self, discovered_smoke):
        info = discovered_smoke.registry["accessions"]["GCF_SMOKE_001"]
        assert info["total_sequence_length"] == 29903

    def test_accessions_are_not_downloaded(self, discovered_smoke):
        for acc in ("GCF_SMOKE_001", "GCF_SMOKE_002"):
            assert discovered_smoke.registry["accessions"][acc]["downloaded"] is False


# ---------------------------------------------------------------------------
# lineage storage
# ---------------------------------------------------------------------------


class TestDiscoverySmokeLineageStorage:
    def test_sars_lineage_is_stored(self, discovered_smoke):
        lineages = discovered_smoke.registry["lineages"]
        assert str(_SARS_COV2_TAXID) in lineages

    def test_mhv_lineage_is_stored(self, discovered_smoke):
        lineages = discovered_smoke.registry["lineages"]
        assert str(_MHV_TAXID) in lineages

    def test_sars_lineage_contains_viruses_ancestor(self, discovered_smoke):
        lineage = discovered_smoke.registry["lineages"][str(_SARS_COV2_TAXID)]
        taxids = {entry["taxid"] for entry in lineage}
        assert "10239" in taxids

    def test_lineage_entries_have_required_keys(self, discovered_smoke):
        lineage = discovered_smoke.registry["lineages"][str(_SARS_COV2_TAXID)]
        for entry in lineage:
            assert "taxid" in entry
            assert "rank" in entry
            assert "name" in entry

    def test_sars_lineage_first_entry_is_species(self, discovered_smoke):
        lineage = discovered_smoke.registry["lineages"][str(_SARS_COV2_TAXID)]
        assert lineage[0]["taxid"] == str(_SARS_COV2_TAXID)
        assert lineage[0]["rank"] == "species"


# ---------------------------------------------------------------------------
# registry persistence
# ---------------------------------------------------------------------------


class TestDiscoverySmokeRegistryPersistence:
    def test_registry_is_saved_to_disk(self, discovered_smoke):
        import os
        assert os.path.exists(discovered_smoke.registry_path)

    def test_pending_volume_is_positive(self, discovered_smoke):
        volume = discovered_smoke.get_pending_volume()
        assert volume > 0

    def test_pending_volume_matches_total_sequence_lengths(self, discovered_smoke):
        volume = discovered_smoke.get_pending_volume()
        assert volume == 29903 + 31526
