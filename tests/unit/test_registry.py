"""Tests for taxotreeset.io.registry.NCBIRegistry."""

import json
import subprocess
from unittest.mock import patch, MagicMock
import pytest
from taxotreeset.io.registry import NCBIRegistry


@pytest.fixture
def mapping_file(tmp_path):
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps({"domains": [], "redirects": {}}), encoding="utf-8")
    return str(p)


@pytest.fixture
def registry_path(tmp_path):
    return str(tmp_path / "registry.json")


@pytest.fixture
def reg(mapping_file, registry_path):
    return NCBIRegistry(registry_path=registry_path, config_path=mapping_file)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_fresh_registry_has_empty_sections(self, reg):
        assert reg.registry["taxons"] == {}
        assert reg.registry["accessions"] == {}
        assert reg.registry["lineages"] == {}
        assert reg.registry["capacities"] == {}
        assert reg.registry["last_update"] is None

    def test_loads_existing_registry_from_disk(self, mapping_file, tmp_path):
        registry_path = str(tmp_path / "existing.json")
        existing = {
            "last_update": None,
            "taxons": {"10239": ["GCF_001"]},
            "accessions": {
                "GCF_001": {
                    "taxid": "10239",
                    "organism": "Test virus",
                    "is_reference": True,
                    "total_sequence_length": 50000,
                    "downloaded": False,
                    "download_deferred": False,
                    "local_path": None,
                }
            },
            "lineages": {},
            "capacities": {},
        }
        with open(registry_path, "w", encoding="utf-8") as fh:
            json.dump(existing, fh)

        loaded = NCBIRegistry(registry_path=registry_path, config_path=mapping_file)
        assert "GCF_001" in loaded.registry["accessions"]
        assert loaded.registry["taxons"]["10239"] == ["GCF_001"]

    def test_missing_sections_are_backfilled(self, mapping_file, tmp_path):
        registry_path = str(tmp_path / "old.json")
        old_schema = {"last_update": None, "taxons": {}, "accessions": {}, "lineages": {}}
        with open(registry_path, "w", encoding="utf-8") as fh:
            json.dump(old_schema, fh)

        loaded = NCBIRegistry(registry_path=registry_path, config_path=mapping_file)
        assert "capacities" in loaded.registry
        assert loaded.registry["capacities"] == {}


# ---------------------------------------------------------------------------
# _build_accession_entry
# ---------------------------------------------------------------------------


class TestBuildAccessionEntry:
    def _report(self, level="Complete Genome", org="Test virus", seq_len="100000"):
        return {
            "assembly_info": {"assembly_level": level},
            "organism": {"organism_name": org},
            "assembly_stats": {"total_sequence_length": seq_len},
        }

    def test_complete_genome_is_reference(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report("Complete Genome"))
        assert entry["is_reference"] is True

    def test_chromosome_level_is_reference(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report("Chromosome"))
        assert entry["is_reference"] is True

    def test_scaffold_level_is_not_reference(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report("Scaffold"))
        assert entry["is_reference"] is False

    def test_total_sequence_length_is_cast_to_int(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report(seq_len="123456"))
        assert entry["total_sequence_length"] == 123456
        assert isinstance(entry["total_sequence_length"], int)

    def test_missing_seq_len_stored_as_none(self):
        report = {
            "assembly_info": {"assembly_level": "Scaffold"},
            "organism": {"organism_name": "Test"},
            "assembly_stats": {},
        }
        entry = NCBIRegistry._build_accession_entry("10239", report)
        assert entry["total_sequence_length"] is None

    def test_default_flags(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report())
        assert entry["downloaded"] is False
        assert entry["download_deferred"] is False
        assert entry["local_path"] is None

    def test_taxid_stored_as_string(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report())
        assert entry["taxid"] == "10239"

    def test_organism_name_stored_in_entry(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report(org="Test virus"))
        assert entry["organism"] == "Test virus"

    def test_organism_key_present_in_entry(self):
        entry = NCBIRegistry._build_accession_entry("10239", self._report())
        assert "organism" in entry


# ---------------------------------------------------------------------------
# store_lineage / lineage access
# ---------------------------------------------------------------------------


class TestStoreLineage:
    def test_lineage_stored_by_taxid_string(self, reg):
        lineage = [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]
        reg.store_lineage(12227, lineage)
        assert reg.registry["lineages"]["12227"] == lineage

    def test_lineage_overwrite_is_idempotent(self, reg):
        lineage = [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]
        reg.store_lineage("12227", lineage)
        reg.store_lineage("12227", lineage)
        assert len(reg.registry["lineages"]) == 1


# ---------------------------------------------------------------------------
# Capacity cache
# ---------------------------------------------------------------------------


class TestCapacityCache:
    def test_store_and_load_capacities(self, reg):
        caps = {"10239": 5000, "12227": 3000}
        reg.store_capacities(caps, min_len=100)
        loaded = reg.load_capacities(min_len=100)
        assert loaded == caps

    def test_different_min_len_stored_independently(self, reg):
        reg.store_capacities({"10239": 5000}, min_len=100)
        reg.store_capacities({"10239": 3000}, min_len=150)
        assert reg.load_capacities(100)["10239"] == 5000
        assert reg.load_capacities(150)["10239"] == 3000

    def test_load_missing_min_len_returns_empty(self, reg):
        assert reg.load_capacities(999) == {}

    def test_store_merges_without_overwriting_other_min_lens(self, reg):
        reg.store_capacities({"A": 1}, min_len=100)
        reg.store_capacities({"B": 2}, min_len=100)
        loaded = reg.load_capacities(100)
        assert loaded == {"A": 1, "B": 2}


# ---------------------------------------------------------------------------
# get_pending_volume
# ---------------------------------------------------------------------------


class TestGetPendingVolume:
    def _populate(self, reg):
        reg.registry["taxons"] = {
            "11234": ["GCF_A", "GCF_B"],
            "11235": ["GCF_C"],
        }
        reg.registry["accessions"] = {
            "GCF_A": {
                "taxid": "11234",
                "total_sequence_length": 100_000,
                "downloaded": False,
                "download_deferred": False,
            },
            "GCF_B": {
                "taxid": "11234",
                "total_sequence_length": 200_000,
                "downloaded": True,
                "download_deferred": False,
            },
            "GCF_C": {
                "taxid": "11235",
                "total_sequence_length": 50_000,
                "downloaded": False,
                "download_deferred": False,
            },
        }
        reg.registry["lineages"] = {
            "11234": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}],
            "11235": [{"taxid": "11111", "rank": "superkingdom", "name": "Bacteria"}],
        }

    def test_sum_of_pending_accessions(self, reg):
        self._populate(reg)
        assert reg.get_pending_volume() == 150_000

    def test_domain_filter_restricts_to_lineage(self, reg):
        self._populate(reg)
        vol = reg.get_pending_volume(domain_taxid="10239")
        assert vol == 100_000

    def test_domain_filter_out_of_scope_returns_zero(self, reg):
        self._populate(reg)
        vol = reg.get_pending_volume(domain_taxid="99999")
        assert vol == 0

    def test_defensive_int_cast_for_string_values(self, reg):
        reg.registry["taxons"] = {"111": ["GCF_X"]}
        reg.registry["accessions"] = {
            "GCF_X": {
                "taxid": "111",
                "total_sequence_length": "300000",
                "downloaded": False,
                "download_deferred": False,
            }
        }
        assert reg.get_pending_volume() == 300_000

    def test_none_seq_len_treated_as_zero(self, reg):
        reg.registry["taxons"] = {"111": ["GCF_X"]}
        reg.registry["accessions"] = {
            "GCF_X": {
                "taxid": "111",
                "total_sequence_length": None,
                "downloaded": False,
                "download_deferred": False,
            }
        }
        assert reg.get_pending_volume() == 0

    def test_deferred_accessions_are_counted(self, reg):
        reg.registry["taxons"] = {"111": ["GCF_D"]}
        reg.registry["accessions"] = {
            "GCF_D": {
                "taxid": "111",
                "total_sequence_length": 80_000,
                "downloaded": False,
                "download_deferred": True,
            }
        }
        assert reg.get_pending_volume() == 80_000

    def test_duplicate_accession_across_taxons_counted_once(self, reg):
        # Same accession GCF_SHARED under two taxon entries — seen set dedup (line 415)
        reg.registry["taxons"] = {
            "11234": ["GCF_SHARED"],
            "11235": ["GCF_SHARED"],
        }
        reg.registry["accessions"] = {
            "GCF_SHARED": {
                "taxid": "11234",
                "total_sequence_length": 100_000,
                "downloaded": False,
                "download_deferred": False,
            }
        }
        assert reg.get_pending_volume() == 100_000


# ---------------------------------------------------------------------------
# mark_accessions_deferred / reset_selection_flags
# ---------------------------------------------------------------------------


class TestMarkAccessionsDeferred:
    def _populate(self, reg):
        reg.registry["accessions"] = {
            "GCF_A": {"downloaded": False, "download_deferred": False},
            "GCF_B": {"downloaded": False, "download_deferred": False},
        }

    def test_marks_listed_accessions_as_deferred(self, reg):
        self._populate(reg)
        reg.mark_accessions_deferred(["GCF_A"])
        assert reg.registry["accessions"]["GCF_A"]["download_deferred"] is True
        assert reg.registry["accessions"]["GCF_B"]["download_deferred"] is False

    def test_unknown_accession_ids_are_silently_ignored(self, reg):
        self._populate(reg)
        reg.mark_accessions_deferred(["GCF_NONEXISTENT"])
        assert reg.registry["accessions"]["GCF_A"]["download_deferred"] is False


class TestResetSelectionFlags:
    def _populate(self, reg):
        reg.registry["taxons"] = {
            "11234": ["GCF_A"],
            "11235": ["GCF_B"],
        }
        reg.registry["accessions"] = {
            "GCF_A": {"downloaded": False, "download_deferred": True},
            "GCF_B": {"downloaded": True, "download_deferred": True},
        }
        reg.registry["lineages"] = {
            "11234": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}],
            "11235": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}],
        }

    def test_clears_deferred_flag_for_pending_accessions(self, reg):
        self._populate(reg)
        reg.reset_selection_flags()
        assert reg.registry["accessions"]["GCF_A"]["download_deferred"] is False

    def test_clears_deferred_flag_regardless_of_downloaded_status(self, reg):
        self._populate(reg)
        reg.reset_selection_flags()
        assert reg.registry["accessions"]["GCF_B"]["download_deferred"] is False

    def test_duplicate_accession_across_taxons_processed_once(self, reg):
        # GCF_A listed under two taxon entries — seen set dedup exercises line 465
        reg.registry["taxons"] = {
            "11234": ["GCF_A"],
            "11235": ["GCF_A"],
        }
        reg.registry["accessions"] = {
            "GCF_A": {"downloaded": False, "download_deferred": True},
        }
        reg.registry["lineages"] = {}
        reg.reset_selection_flags()
        assert reg.registry["accessions"]["GCF_A"]["download_deferred"] is False

    def test_domain_filter_restricts_reset_scope(self, reg):
        reg.registry["taxons"] = {
            "11234": ["GCF_A"],
            "22222": ["GCF_C"],
        }
        reg.registry["accessions"] = {
            "GCF_A": {"downloaded": False, "download_deferred": True},
            "GCF_C": {"downloaded": False, "download_deferred": True},
        }
        reg.registry["lineages"] = {
            "11234": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}],
            "22222": [{"taxid": "11111", "rank": "superkingdom", "name": "Bacteria"}],
        }
        reg.reset_selection_flags(domain_taxid="10239")
        assert reg.registry["accessions"]["GCF_A"]["download_deferred"] is False
        assert reg.registry["accessions"]["GCF_C"]["download_deferred"] is True


# ---------------------------------------------------------------------------
# save / _load_registry roundtrip
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_persists_to_disk(self, reg, registry_path):
        reg.registry["taxons"]["99"] = ["GCF_TEST"]
        reg.save()
        with open(registry_path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["taxons"]["99"] == ["GCF_TEST"]

    def test_save_creates_parent_directories(self, tmp_path, mapping_file):
        nested_path = str(tmp_path / "a" / "b" / "c" / "registry.json")
        r = NCBIRegistry(registry_path=nested_path, config_path=mapping_file)
        r.save()
        with open(nested_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "taxons" in data

    def test_roundtrip_preserves_all_sections(self, reg, registry_path, mapping_file):
        lineage = [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]
        reg.store_lineage("12227", lineage)
        reg.store_capacities({"12227": 5000}, min_len=100)
        reg.save()

        reloaded = NCBIRegistry(registry_path=registry_path, config_path=mapping_file)
        assert reloaded.registry["lineages"]["12227"] == lineage
        assert reloaded.load_capacities(100) == {"12227": 5000}


# ---------------------------------------------------------------------------
# _invalidate_ancestor_capacities
# ---------------------------------------------------------------------------


class TestInvalidateAncestorCapacities:
    def _populate(self, reg):
        lineage = [
            {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
            {"taxid": "11118", "rank": "family", "name": "Coronaviridae"},
        ]
        reg.registry["lineages"]["12227"] = lineage
        reg.registry["capacities"] = {
            "10239": {"100": 9000},
            "11118": {"100": 4000},
            "99999": {"100": 1000},
        }

    def test_ancestor_capacities_removed_on_invalidation(self, reg):
        self._populate(reg)
        reg._invalidate_ancestor_capacities("12227")
        assert "10239" not in reg.registry["capacities"]
        assert "11118" not in reg.registry["capacities"]

    def test_unrelated_taxid_not_removed(self, reg):
        self._populate(reg)
        reg._invalidate_ancestor_capacities("12227")
        assert "99999" in reg.registry["capacities"]

    def test_missing_lineage_is_a_noop(self, reg):
        reg.registry["capacities"]["10239"] = {"100": 5000}
        reg._invalidate_ancestor_capacities("99999")
        assert "10239" in reg.registry["capacities"]

    def test_removes_ancestor_capacity_from_cache(self, reg):
        lineage = [
            {"taxid": "2697049", "rank": "species", "name": "SARS"},
            {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
        ]
        reg.store_lineage("2697049", lineage)
        reg.store_capacities({"2697049": 5000, "10239": 8000}, min_len=100)
        reg._invalidate_ancestor_capacities("2697049")
        cache = reg.registry["capacities"]
        assert "2697049" not in cache
        assert "10239" not in cache

    def test_no_error_when_lineage_absent(self, reg):
        reg.store_capacities({"10239": 5000}, min_len=100)
        reg._invalidate_ancestor_capacities("99999")
        assert "10239" in reg.registry["capacities"]

    def test_only_affected_ancestors_removed(self, reg):
        lineage = [{"taxid": "2697049", "rank": "species", "name": "SARS"}]
        reg.store_lineage("2697049", lineage)
        reg.store_capacities({"2697049": 5000, "99999": 8000}, min_len=100)
        reg._invalidate_ancestor_capacities("2697049")
        assert "99999" in reg.registry["capacities"]


# ---------------------------------------------------------------------------
# get_pending_volume — domain filter
# ---------------------------------------------------------------------------


class TestGetPendingVolumeWithDomainFilter:
    def _populate_two_domains(self, reg):
        reg.registry["taxons"] = {
            "2697049": ["GCF_VIRUS"],
            "1234567": ["GCF_BACT"],
        }
        reg.registry["accessions"] = {
            "GCF_VIRUS": {"downloaded": False, "total_sequence_length": 30000},
            "GCF_BACT": {"downloaded": False, "total_sequence_length": 50000},
        }
        reg.registry["lineages"] = {
            "2697049": [
                {"taxid": "2697049", "rank": "species", "name": "SARS"},
                {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
            ],
            "1234567": [
                {"taxid": "1234567", "rank": "species", "name": "E.coli"},
                {"taxid": "2", "rank": "superkingdom", "name": "Bacteria"},
            ],
        }

    def test_domain_filter_includes_only_viruses(self, reg):
        self._populate_two_domains(reg)
        volume = reg.get_pending_volume(domain_taxid="10239")
        assert volume == 30000

    def test_domain_filter_none_counts_all(self, reg):
        self._populate_two_domains(reg)
        volume = reg.get_pending_volume(domain_taxid=None)
        assert volume == 80000

    def test_excludes_downloaded_from_domain_sum(self, reg):
        reg.registry["taxons"] = {"2697049": ["GCF_VIRUS"]}
        reg.registry["accessions"] = {
            "GCF_VIRUS": {"downloaded": True, "total_sequence_length": 30000},
        }
        reg.registry["lineages"] = {
            "2697049": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]
        }
        assert reg.get_pending_volume(domain_taxid="10239") == 0


# ---------------------------------------------------------------------------
# discover_taxon_metadata (subprocess-based)
# ---------------------------------------------------------------------------


class TestUpdateTaxonEntry:
    def test_report_without_accession_key_is_skipped(self, reg):
        # Report missing "accession" → triggers `if not accession: continue` (line 332)
        reg._update_taxon_entry(
            taxon_id=10239,
            assembly_data={"reports": [{"assembly_info": {}}]},
        )
        assert reg.registry["taxons"].get("10239", []) == []
        assert reg.registry["accessions"] == {}

    def _valid_report(self, accession="GCF_VALID"):
        return {
            "accession": accession,
            "assembly_info": {"assembly_level": "Complete Genome"},
            "assembly_stats": {"total_sequence_length": "29903"},
            "organism": {"organism_name": "Test virus"},
        }

    def test_valid_report_registers_accession(self, reg):
        reg._update_taxon_entry(
            taxon_id=10239,
            assembly_data={"reports": [self._valid_report()]},
        )
        assert "GCF_VALID" in reg.registry["accessions"]
        assert "GCF_VALID" in reg.registry["taxons"]["10239"]

    def test_accession_entry_is_a_dict_not_none(self, reg):
        reg._update_taxon_entry(
            taxon_id=10239,
            assembly_data={"reports": [self._valid_report()]},
        )
        entry = reg.registry["accessions"]["GCF_VALID"]
        assert isinstance(entry, dict)
        assert entry["taxid"] == "10239"

    def test_report_without_accession_does_not_stop_subsequent_reports(self, reg):
        # ID 262: continue→break; if continue, valid report after bad one is still processed.
        reg._update_taxon_entry(
            taxon_id=10239,
            assembly_data={
                "reports": [
                    {"assembly_info": {}},  # no accession key → should be skipped via continue
                    self._valid_report("GCF_AFTER_SKIP"),
                ]
            },
        )
        assert "GCF_AFTER_SKIP" in reg.registry["accessions"]


class TestDiscoverTaxonMetadata:
    def _fake_report_line(self, accession="GCF_TEST_001"):
        return json.dumps({
            "reports": [{
                "accession": accession,
                "assembly_info": {"assembly_level": "Complete Genome"},
                "assembly_stats": {"total_sequence_length": "29903"},
                "organism": {"organism_name": "Test virus"},
            }]
        })

    def test_discovers_accession_via_subprocess(self, reg):
        fake_result = MagicMock()
        fake_result.stdout = self._fake_report_line("GCF_MOCK_001")
        with patch("subprocess.run", return_value=fake_result):
            reg.discover_taxon_metadata(10239)
        assert "GCF_MOCK_001" in reg.registry["accessions"]

    def test_handles_subprocess_error_gracefully(self, reg):
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "datasets", stderr="error"),
        ):
            reg.discover_taxon_metadata(10239)
        assert reg.registry["accessions"] == {}

    def test_skips_empty_lines(self, reg):
        fake_result = MagicMock()
        fake_result.stdout = "\n\n" + self._fake_report_line("GCF_MOCK_002")
        with patch("subprocess.run", return_value=fake_result):
            reg.discover_taxon_metadata(10239)
        assert "GCF_MOCK_002" in reg.registry["accessions"]


# ---------------------------------------------------------------------------
# Pending-volume loop control (continue vs break in domain-filter and dedup)
# ---------------------------------------------------------------------------


class TestGetPendingVolumeLoopControl:
    """Verify that non-matching or duplicate entries use continue, not break.

    Kills IDs 228 (outer continue→break, domain filter) and 229/230
    (inner continue→break or not-in-seen inversion, dedup logic).
    """

    def test_domain_filter_skips_non_matching_taxon_not_first(self, reg):
        # Non-matching taxid FIRST; matching taxid SECOND.
        # With break mutant (ID 228), loop stops at non-match and never reaches match.
        reg.registry["taxons"] = {
            "non_match": ["GCF_NM"],
            "11234": ["GCF_A"],
        }
        reg.registry["accessions"] = {
            "GCF_NM": {"downloaded": False, "total_sequence_length": 500_000},
            "GCF_A": {"downloaded": False, "total_sequence_length": 100_000},
        }
        reg.registry["lineages"] = {
            "non_match": [{"taxid": "99999", "rank": "superkingdom", "name": "Other"}],
            "11234": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}],
        }
        vol = reg.get_pending_volume(domain_taxid="10239")
        assert vol == 100_000

    def test_dedup_continue_processes_subsequent_accessions_in_same_taxon(self, reg):
        # GCF_SHARED appears in both taxon lists; GCF_UNIQUE only in second.
        # With break mutant (ID 230), seeing GCF_SHARED again stops inner loop
        # before GCF_UNIQUE is processed.
        reg.registry["taxons"] = {
            "11234": ["GCF_SHARED"],
            "11235": ["GCF_SHARED", "GCF_UNIQUE"],
        }
        reg.registry["accessions"] = {
            "GCF_SHARED": {"downloaded": False, "total_sequence_length": 100_000},
            "GCF_UNIQUE": {"downloaded": False, "total_sequence_length": 50_000},
        }
        reg.registry["lineages"] = {}
        vol = reg.get_pending_volume()
        assert vol == 150_000

    def test_all_accessions_counted_without_domain_filter(self, reg):
        # Targeted at ID 229 (not-in-seen inversion): with that mutant every
        # accession is skipped because the guard fires for NEW acc_ids and we
        # never reach seen.add, so total stays 0.
        reg.registry["taxons"] = {"111": ["GCF_X", "GCF_Y"]}
        reg.registry["accessions"] = {
            "GCF_X": {"downloaded": False, "total_sequence_length": 60_000},
            "GCF_Y": {"downloaded": False, "total_sequence_length": 40_000},
        }
        reg.registry["lineages"] = {}
        assert reg.get_pending_volume() == 100_000


# ---------------------------------------------------------------------------
# reset_selection_flags loop control
# ---------------------------------------------------------------------------


class TestResetSelectionFlagsLoopControl:
    """Verify continue (not break) semantics in reset_selection_flags loops.

    Kills IDs 260 (outer continue→break) and 262 (inner continue→break),
    and ID 265 (info and … vs info or …, crashes on None info).
    """

    def test_domain_filter_skips_non_matching_taxon_not_first(self, reg):
        # Non-matching taxid FIRST; matching taxid SECOND.
        # With break mutant (ID 260), loop stops and matching taxid is never reset.
        reg.registry["taxons"] = {
            "non_match": ["GCF_NM"],
            "11234": ["GCF_A"],
        }
        reg.registry["accessions"] = {
            "GCF_NM": {"downloaded": False, "download_deferred": True},
            "GCF_A": {"downloaded": False, "download_deferred": True},
        }
        reg.registry["lineages"] = {
            "non_match": [{"taxid": "99999", "rank": "superkingdom", "name": "Other"}],
            "11234": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}],
        }
        reg.reset_selection_flags(domain_taxid="10239")
        assert reg.registry["accessions"]["GCF_A"]["download_deferred"] is False
        assert reg.registry["accessions"]["GCF_NM"]["download_deferred"] is True

    def test_dedup_continue_resets_subsequent_accession_in_same_taxon(self, reg):
        # GCF_SHARED seen in first taxon; GCF_SECOND comes after it in second taxon.
        # With break mutant (ID 262), seeing GCF_SHARED again breaks inner loop
        # before GCF_SECOND is processed.
        reg.registry["taxons"] = {
            "11234": ["GCF_SHARED"],
            "11235": ["GCF_SHARED", "GCF_SECOND"],
        }
        reg.registry["accessions"] = {
            "GCF_SHARED": {"downloaded": False, "download_deferred": True},
            "GCF_SECOND": {"downloaded": False, "download_deferred": True},
        }
        reg.registry["lineages"] = {}
        reg.reset_selection_flags()
        assert reg.registry["accessions"]["GCF_SECOND"]["download_deferred"] is False

    def test_missing_accession_entry_does_not_crash(self, reg):
        # ID 265: `if info and info.get(…)` vs `if info or info.get(…)`.
        # When acc_id is in taxons but absent from accessions, info=None.
        # The mutant evaluates `None or None.get(…)` → AttributeError.
        reg.registry["taxons"] = {"11234": ["GCF_GHOST"]}
        reg.registry["accessions"] = {}  # GCF_GHOST not registered
        reg.registry["lineages"] = {}
        reg.reset_selection_flags()  # must not raise


# ---------------------------------------------------------------------------
# accession_snapshot
# ---------------------------------------------------------------------------


class TestAccessionSnapshot:
    def test_empty_registry(self, reg):
        snap = reg.accession_snapshot()
        assert snap["n_accessions"] == 0
        assert snap["accessions"] == []
        assert len(snap["sha256"]) == 64  # sha256 hex digest

    def test_sorted_and_counted(self, reg):
        reg.registry["accessions"] = {"GCF_002.1": {}, "GCF_001.3": {}, "GCF_001.2": {}}
        snap = reg.accession_snapshot()
        assert snap["n_accessions"] == 3
        assert snap["accessions"] == ["GCF_001.2", "GCF_001.3", "GCF_002.1"]

    def test_digest_is_insertion_order_independent(self, reg):
        reg.registry["accessions"] = {"GCF_002.1": {}, "GCF_001.1": {}}
        first = reg.accession_snapshot()["sha256"]
        reg.registry["accessions"] = {"GCF_001.1": {}, "GCF_002.1": {}}
        assert reg.accession_snapshot()["sha256"] == first

    def test_digest_changes_with_accession_version(self, reg):
        reg.registry["accessions"] = {"GCF_001.1": {}}
        first = reg.accession_snapshot()["sha256"]
        reg.registry["accessions"] = {"GCF_001.2": {}}  # same assembly, new version
        assert reg.accession_snapshot()["sha256"] != first


# ---------------------------------------------------------------------------
# mark_updated
# ---------------------------------------------------------------------------


class TestMarkUpdated:
    def test_sets_parseable_utc_timestamp(self, reg):
        import datetime

        assert reg.registry["last_update"] is None
        reg.mark_updated()
        parsed = datetime.datetime.fromisoformat(reg.registry["last_update"])
        assert parsed.tzinfo is not None
