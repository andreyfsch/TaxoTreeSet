"""Integration tests for the NCBI discovery pipeline (network-dependent).

These tests require a live connection to the NCBI APIs and a working
installation of the ``datasets`` CLI (NCBI Datasets). They are marked
with the ``network`` pytest marker so they can be run independently:

    python -m pytest tests/integration/test_ncbi_discovery.py -m network

They are NOT excluded by default — the CI environment is expected to
have network access and the datasets CLI installed.

Taxon used: SARS-CoV-2 (TaxID 2697049), a single species with exactly
one complete reference assembly (NC_045512.2 / RefSeq GCF_009858895.2).
Its small size makes the test fast and its reference status is stable.
"""

import pytest
from taxotreeset.core.orchestrator import DiscoveryOrchestrator
from taxotreeset.io.registry import NCBIRegistry


_SARS_COV2_TAXID = 2697049
_SARS_COV2_EXPECTED_ACCESSION = "GCF_009858895.2"
_SARS_COV2_EXPECTED_HEADER = "NC_045512.2"


@pytest.fixture
def ncbi_registry(tmp_path, minimal_mapping):
    registry_path = str(tmp_path / "registry.json")
    return NCBIRegistry(registry_path=registry_path, config_path=minimal_mapping)


@pytest.fixture
def discovered_registry(ncbi_registry, minimal_mapping):
    """Run discovery against NCBI for SARS-CoV-2 and return the registry."""
    import json
    with open(minimal_mapping, encoding="utf-8") as fh:
        mapping_config = json.load(fh)
    orchestrator = DiscoveryOrchestrator(
        registry=ncbi_registry,
        mapping_config=mapping_config,
    )
    orchestrator.discover_from_root(_SARS_COV2_TAXID)
    return ncbi_registry


@pytest.mark.network
class TestSarsCov2Discovery:
    def test_at_least_one_accession_discovered(self, discovered_registry):
        accessions = discovered_registry.registry["accessions"]
        assert len(accessions) >= 1

    def test_reference_accession_is_present(self, discovered_registry):
        accessions = discovered_registry.registry["accessions"]
        assert _SARS_COV2_EXPECTED_ACCESSION in accessions

    def test_accession_has_valid_total_sequence_length(self, discovered_registry):
        info = discovered_registry.registry["accessions"][_SARS_COV2_EXPECTED_ACCESSION]
        tsl = info.get("total_sequence_length")
        assert tsl is not None
        assert isinstance(tsl, int)
        assert tsl > 20_000

    def test_accession_is_marked_as_reference(self, discovered_registry):
        info = discovered_registry.registry["accessions"][_SARS_COV2_EXPECTED_ACCESSION]
        assert info["is_reference"] is True

    def test_accession_is_not_downloaded(self, discovered_registry):
        info = discovered_registry.registry["accessions"][_SARS_COV2_EXPECTED_ACCESSION]
        assert info["downloaded"] is False

    def test_lineage_stored_for_species(self, discovered_registry):
        lineages = discovered_registry.registry["lineages"]
        assert str(_SARS_COV2_TAXID) in lineages

    def test_lineage_contains_viruses_ancestor(self, discovered_registry):
        lineage = discovered_registry.registry["lineages"][str(_SARS_COV2_TAXID)]
        ancestor_taxids = {entry["taxid"] for entry in lineage}
        assert "10239" in ancestor_taxids

    def test_lineage_entries_have_required_keys(self, discovered_registry):
        lineage = discovered_registry.registry["lineages"][str(_SARS_COV2_TAXID)]
        for entry in lineage:
            assert "taxid" in entry
            assert "rank" in entry
            assert "name" in entry

    def test_registry_is_saved_to_disk(self, discovered_registry):
        import os
        assert os.path.exists(discovered_registry.registry_path)

    def test_pending_volume_is_positive(self, discovered_registry):
        volume = discovered_registry.get_pending_volume()
        assert volume > 0
