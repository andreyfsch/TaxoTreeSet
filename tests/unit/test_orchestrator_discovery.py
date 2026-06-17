"""Tests for taxotreeset.core.orchestrator — DiscoveryOrchestrator pure helpers."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from taxotreeset.core.orchestrator import DiscoveryOrchestrator, _Ancestor


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def orchestrator():
    mock_registry = MagicMock()
    mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
    return DiscoveryOrchestrator(
        registry=mock_registry,
        mapping_config={},
    )


# ---------------------------------------------------------------------------
# _build_summary_command
# ---------------------------------------------------------------------------


class TestBuildSummaryCommand:
    def test_returns_list_starting_with_datasets(self):
        cmd = DiscoveryOrchestrator._build_summary_command("10239", "complete,chromosome")
        assert cmd[0] == "datasets"

    def test_includes_taxid_in_command(self):
        cmd = DiscoveryOrchestrator._build_summary_command("10239", "complete")
        assert "10239" in cmd

    def test_includes_assembly_levels(self):
        cmd = DiscoveryOrchestrator._build_summary_command("10239", "complete,chromosome")
        assert "complete,chromosome" in cmd

    def test_includes_refseq_source(self):
        cmd = DiscoveryOrchestrator._build_summary_command("10239", "complete")
        assert "RefSeq" in cmd

    def test_includes_json_lines_flag(self):
        cmd = DiscoveryOrchestrator._build_summary_command("10239", "complete")
        assert "--as-json-lines" in cmd


# ---------------------------------------------------------------------------
# _consume_jsonlines_stream
# ---------------------------------------------------------------------------


class TestConsumeJsonlinesStream:
    def _mock_process(self, lines):
        mock = MagicMock()
        mock.stdout = lines
        return mock

    def test_groups_reports_by_taxid(self):
        lines = [
            json.dumps({"organism": {"tax_id": 12345}}) + "\n",
            json.dumps({"organism": {"tax_id": 12345}}) + "\n",
            json.dumps({"organism": {"tax_id": 99999}}) + "\n",
        ]
        process = self._mock_process(lines)
        result = DiscoveryOrchestrator._consume_jsonlines_stream(process)
        assert "12345" in result
        assert len(result["12345"]) == 2
        assert "99999" in result
        assert len(result["99999"]) == 1

    def test_skips_non_json_lines(self):
        lines = [
            "not valid json\n",
            json.dumps({"organism": {"tax_id": 12345}}) + "\n",
        ]
        process = self._mock_process(lines)
        result = DiscoveryOrchestrator._consume_jsonlines_stream(process)
        assert "12345" in result

    def test_skips_empty_lines(self):
        lines = ["\n", "   \n", json.dumps({"organism": {"tax_id": 12345}}) + "\n"]
        process = self._mock_process(lines)
        result = DiscoveryOrchestrator._consume_jsonlines_stream(process)
        assert len(result) == 1

    def test_skips_reports_without_taxid(self):
        lines = [json.dumps({"organism": {"name": "Foo"}}) + "\n"]
        process = self._mock_process(lines)
        result = DiscoveryOrchestrator._consume_jsonlines_stream(process)
        assert result == {}

    def test_returns_empty_dict_when_stdout_is_none(self):
        process = MagicMock()
        process.stdout = None
        result = DiscoveryOrchestrator._consume_jsonlines_stream(process)
        assert result == {}

    def test_empty_stream_returns_empty_dict(self):
        process = self._mock_process([])
        result = DiscoveryOrchestrator._consume_jsonlines_stream(process)
        assert result == {}


# ---------------------------------------------------------------------------
# _parse_taxonomy_classification
# ---------------------------------------------------------------------------


class TestParseTaxonomyClassification:
    def test_extracts_classification_from_valid_output(self):
        payload = {
            "taxonomy": {
                "classification": {
                    "species": {"id": "2697049", "name": "SARS-CoV-2"},
                    "family": {"id": "11118", "name": "Coronaviridae"},
                }
            }
        }
        stdout = json.dumps(payload) + "\n"
        result = DiscoveryOrchestrator._parse_taxonomy_classification(stdout)
        assert result is not None
        assert "species" in result
        assert "family" in result

    def test_returns_none_for_empty_stdout(self):
        assert DiscoveryOrchestrator._parse_taxonomy_classification("") is None

    def test_returns_none_for_missing_classification_key(self):
        payload = {"taxonomy": {"tax_id": 12345}}
        stdout = json.dumps(payload) + "\n"
        assert DiscoveryOrchestrator._parse_taxonomy_classification(stdout) is None

    def test_skips_non_json_lines(self):
        payload = {"taxonomy": {"classification": {"species": {"id": "1", "name": "S"}}}}
        stdout = "downloading...\n" + json.dumps(payload) + "\n"
        result = DiscoveryOrchestrator._parse_taxonomy_classification(stdout)
        assert result is not None

    def test_returns_none_for_all_non_json_output(self):
        assert DiscoveryOrchestrator._parse_taxonomy_classification("not json\n") is None


# ---------------------------------------------------------------------------
# _classification_node_for_rank
# ---------------------------------------------------------------------------


class TestClassificationNodeForRank:
    def _classification(self):
        return {
            "species": {"id": "2697049", "name": "SARS-CoV-2"},
            "family": {"id": "11118", "name": "Coronaviridae"},
            "superkingdom": {"id": "10239", "name": "Viruses"},
        }

    def test_returns_node_for_known_rank(self):
        result = DiscoveryOrchestrator._classification_node_for_rank(
            self._classification(), "family"
        )
        assert result == {"id": "11118", "name": "Coronaviridae"}

    def test_returns_none_for_absent_rank(self):
        result = DiscoveryOrchestrator._classification_node_for_rank(
            self._classification(), "genus"
        )
        assert result is None

    def test_superkingdom_resolves_via_alias(self):
        classification = {"acellular_root": {"id": "10239", "name": "Viruses"}}
        result = DiscoveryOrchestrator._classification_node_for_rank(
            classification, "superkingdom"
        )
        assert result is not None
        assert result["id"] == "10239"

    def test_superkingdom_falls_back_to_superkingdom_key(self):
        result = DiscoveryOrchestrator._classification_node_for_rank(
            self._classification(), "superkingdom"
        )
        assert result == {"id": "10239", "name": "Viruses"}

    def test_superkingdom_returns_none_when_both_keys_absent(self):
        result = DiscoveryOrchestrator._classification_node_for_rank({}, "superkingdom")
        assert result is None


# ---------------------------------------------------------------------------
# _sanitize_path_component
# ---------------------------------------------------------------------------


class TestSanitizePathComponent:
    def test_replaces_spaces_with_underscores(self):
        assert DiscoveryOrchestrator._sanitize_path_component("SARS CoV 2") == "SARS_CoV_2"

    def test_replaces_forward_slash_with_underscore(self):
        assert DiscoveryOrchestrator._sanitize_path_component("A/B") == "A_B"

    def test_leaves_clean_names_unchanged(self):
        assert DiscoveryOrchestrator._sanitize_path_component("Coronaviridae") == "Coronaviridae"

    def test_empty_string_returns_empty(self):
        assert DiscoveryOrchestrator._sanitize_path_component("") == ""


# ---------------------------------------------------------------------------
# _resolve_mapped_path
# ---------------------------------------------------------------------------


class TestResolveMappedPath:
    def _ancestor(self, tax_id, rank, name):
        return _Ancestor(tax_id=tax_id, rank=rank, scientific_name=name)

    def test_returns_sanitized_names_when_no_mapping(self, orchestrator):
        lineage = [
            self._ancestor(2697049, "species", "SARS CoV 2"),
            self._ancestor(10239, "superkingdom", "Viruses"),
        ]
        path = orchestrator._resolve_mapped_path(lineage, "10239")
        assert "SARS_CoV_2" in path
        assert "Viruses" in path

    def test_applies_explicit_redirection(self, orchestrator):
        orchestrator.mapping = {
            "scopes": {
                "10239": {
                    "redirections": {
                        "11118": {"target_id": "999001", "label": "Coronaviridae"},
                    },
                    "virtual_id_labels": {"999001": "Nidovirales_Group"},
                }
            }
        }
        lineage = [
            self._ancestor(11118, "family", "Coronaviridae"),
            self._ancestor(10239, "superkingdom", "Viruses"),
        ]
        path = orchestrator._resolve_mapped_path(lineage, "10239")
        assert "Nidovirales_Group" in path

    def test_empty_lineage_returns_empty_path(self, orchestrator):
        path = orchestrator._resolve_mapped_path([], "10239")
        assert path == []


# ---------------------------------------------------------------------------
# _resolve_lineage — taxoniq fallback via NCBI CLI
# ---------------------------------------------------------------------------


class TestResolveLineageFallback:
    def _orchestrator(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        return DiscoveryOrchestrator(
            registry=mock_registry,
            mapping_config={},
        )

    def test_taxoniq_keye_error_triggers_ncbi_fallback(self):
        orch = self._orchestrator()
        fallback_lineage = [_Ancestor(2697049, "species", "SARS")]

        with (
            patch(
                "taxotreeset.core.orchestrator.taxoniq.Taxon",
                side_effect=KeyError("unknown taxid"),
            ),
            patch.object(
                orch, "_fetch_lineage_via_ncbi", return_value=fallback_lineage
            ) as mock_fallback,
        ):
            result = orch._resolve_lineage(2697049)

        mock_fallback.assert_called_once_with(2697049)
        assert result == fallback_lineage

    def test_raises_runtime_error_when_both_sources_fail(self):
        orch = self._orchestrator()

        with (
            patch(
                "taxotreeset.core.orchestrator.taxoniq.Taxon",
                side_effect=KeyError("unknown taxid"),
            ),
            patch.object(orch, "_fetch_lineage_via_ncbi", return_value=[]),
            pytest.raises(RuntimeError, match="Could not resolve lineage"),
        ):
            orch._resolve_lineage(9999999)


# ---------------------------------------------------------------------------
# _process_lineage_batch — checkpoint save
# ---------------------------------------------------------------------------


class TestProcessLineageBatchCheckpoint:
    def _orchestrator(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        return DiscoveryOrchestrator(
            registry=mock_registry,
            mapping_config={},
        )

    def test_checkpoint_save_called_at_interval(self):
        orch = self._orchestrator()

        reports_by_taxid = {str(i): [{"accession": f"GCF_{i:03d}"}] for i in range(10)}

        successful_taxids = []

        def fake_register(taxid_str, reports, root_id_str):
            successful_taxids.append(taxid_str)

        with patch.object(orch, "_register_taxon", side_effect=fake_register):
            orch._build_hierarchy(
                reports_by_taxid=reports_by_taxid,
                root_id_str="10239",
                checkpoint_interval=5,
            )

        assert orch.registry.save.call_count >= 2

    def test_exception_in_register_is_skipped(self):
        orch = self._orchestrator()
        reports_by_taxid = {
            "good": [{"accession": "GCF_GOOD"}],
            "bad": [{"accession": "GCF_BAD"}],
        }

        def fake_register(taxid_str, reports, root_id_str):
            if taxid_str == "bad":
                raise RuntimeError("lineage error")

        with patch.object(orch, "_register_taxon", side_effect=fake_register):
            orch._build_hierarchy(
                reports_by_taxid=reports_by_taxid,
                root_id_str="10239",
                checkpoint_interval=100,
            )


# ---------------------------------------------------------------------------
# _stream_ncbi_summaries — OSError and empty result paths
# ---------------------------------------------------------------------------


class TestStreamNcbiSummaries:
    def _orchestrator(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        return DiscoveryOrchestrator(registry=mock_registry, mapping_config={})

    def test_returns_empty_dict_on_popen_os_error(self):
        orch = self._orchestrator()
        with patch(
            "taxotreeset.core.orchestrator.subprocess.Popen",
            side_effect=OSError("no datasets CLI"),
        ):
            result = orch._stream_ncbi_summaries("10239", "complete")
        assert result == {}

    def test_returns_empty_dict_when_no_reports_produced(self):
        orch = self._orchestrator()
        mock_process = MagicMock()
        mock_process.stdout = iter([])
        mock_process.stderr.read.return_value = "no data"
        mock_process.wait.return_value = 0

        with patch(
            "taxotreeset.core.orchestrator.subprocess.Popen",
            return_value=mock_process,
        ):
            result = orch._stream_ncbi_summaries("10239", "complete")
        assert result == {}


# ---------------------------------------------------------------------------
# _register_taxon — self_node prepend path
# ---------------------------------------------------------------------------


class TestRegisterTaxonSelfNodePrepend:
    def _make_orchestrator_and_registry(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        orch = DiscoveryOrchestrator(registry=mock_registry, mapping_config={})
        return orch, mock_registry

    def test_self_node_prepended_when_lineage_first_entry_differs(self):
        orch, mock_registry = self._make_orchestrator_and_registry()

        parent_lineage = [_Ancestor(10239, "superkingdom", "Viruses")]
        self_node = _Ancestor(9999999, "no_rank", "Strain XYZ")

        with (
            patch.object(orch, "_resolve_lineage", return_value=parent_lineage),
            patch.object(orch, "_resolve_self_node", return_value=self_node),
            patch.object(orch, "_resolve_mapped_path", return_value=["Viruses", "Strain_XYZ"]),
            patch("taxotreeset.core.orchestrator.add_path_to_tree"),
        ):
            orch._register_taxon("9999999", [{"accession": "GCF_TEST"}], "10239")

        stored_call_args = mock_registry.store_lineage.call_args
        stored_lineage = stored_call_args[0][1]
        assert stored_lineage[0]["taxid"] == "9999999"


# ---------------------------------------------------------------------------
# _resolve_self_node — taxoniq success path
# ---------------------------------------------------------------------------


class TestResolveSelfNode:
    def _orchestrator(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        return DiscoveryOrchestrator(registry=mock_registry, mapping_config={})

    def test_returns_ancestor_from_taxoniq(self):
        orch = self._orchestrator()

        mock_taxon = MagicMock()
        mock_taxon.tax_id = 12345
        mock_taxon.rank.name = "species"
        mock_taxon.scientific_name = "Fake species"

        with patch(
            "taxotreeset.core.orchestrator.taxoniq.Taxon",
            return_value=mock_taxon,
        ):
            result = orch._resolve_self_node(12345)

        assert result is not None
        assert result.tax_id == 12345
        assert result.rank == "species"

    def test_falls_back_to_ncbi_on_exception(self):
        orch = self._orchestrator()
        fallback = _Ancestor(12345, "no_rank", "Fallback")

        with (
            patch(
                "taxotreeset.core.orchestrator.taxoniq.Taxon",
                side_effect=RuntimeError("error"),
            ),
            patch.object(orch, "_fetch_self_node_via_ncbi", return_value=fallback),
        ):
            result = orch._resolve_self_node(12345)

        assert result is fallback


# ---------------------------------------------------------------------------
# _fetch_lineage_via_ncbi — subprocess-based fallback
# ---------------------------------------------------------------------------


class TestFetchLineageViaNcbi:
    def _orchestrator(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        return DiscoveryOrchestrator(registry=mock_registry, mapping_config={})

    def test_returns_empty_list_on_subprocess_error(self):
        orch = self._orchestrator()
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "datasets", stderr="err"),
        ):
            result = orch._fetch_lineage_via_ncbi(9999999)
        assert result == []

    def test_returns_empty_list_when_classification_is_none(self):
        orch = self._orchestrator()
        mock_result = MagicMock()
        mock_result.stdout = "not valid json\n"

        with patch("subprocess.run", return_value=mock_result):
            result = orch._fetch_lineage_via_ncbi(9999999)
        assert result == []

    def test_returns_ancestors_when_classification_found(self):
        orch = self._orchestrator()
        payload = {
            "taxonomy": {
                "classification": {
                    "species": {"id": "2697049", "name": "SARS-CoV-2"},
                    "superkingdom": {"id": "10239", "name": "Viruses"},
                }
            }
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(payload) + "\n"

        with patch("subprocess.run", return_value=mock_result):
            result = orch._fetch_lineage_via_ncbi(2697049)

        assert len(result) >= 1
        ranks = [a.rank for a in result]
        assert "species" in ranks or "superkingdom" in ranks


# ---------------------------------------------------------------------------
# discover_from_root — early return when no reports
# ---------------------------------------------------------------------------


class TestDiscoverFromRootEarlyReturn:
    def test_early_return_when_no_reports(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        orch = DiscoveryOrchestrator(registry=mock_registry, mapping_config={})

        with patch.object(orch, "_stream_ncbi_summaries", return_value={}):
            orch.discover_from_root(10239)

        mock_registry.save.assert_not_called()


# ---------------------------------------------------------------------------
# _log_api_key_status — when API key set
# ---------------------------------------------------------------------------


class TestLogApiKeyStatus:
    def test_logs_when_api_key_env_var_is_set(self, monkeypatch):
        monkeypatch.setenv("NCBI_API_KEY", "fake_api_key_12345")
        with patch("taxotreeset.core.orchestrator.logger") as mock_logger:
            DiscoveryOrchestrator._log_api_key_status()
        mock_logger.info.assert_called_once()

    def test_does_not_log_when_api_key_absent(self, monkeypatch):
        monkeypatch.delenv("NCBI_API_KEY", raising=False)
        with patch("taxotreeset.core.orchestrator.logger") as mock_logger:
            DiscoveryOrchestrator._log_api_key_status()
        mock_logger.info.assert_not_called()


# ---------------------------------------------------------------------------
# _parse_taxonomy_classification — blank-line skip
# ---------------------------------------------------------------------------


class TestParseTaxonomyClassificationBlankLine:
    def test_skips_blank_lines_before_valid_json(self):
        payload = {"taxonomy": {"classification": {"species": {"id": "1", "name": "S"}}}}
        stdout = "\n\n  \n" + json.dumps(payload) + "\n"
        result = DiscoveryOrchestrator._parse_taxonomy_classification(stdout)
        assert result is not None
        assert "species" in result


# ---------------------------------------------------------------------------
# _fetch_self_node_via_ncbi
# ---------------------------------------------------------------------------


class TestFetchSelfNodeViaNcbi:
    def _orchestrator(self):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}, "taxons": {}, "lineages": {}}
        return DiscoveryOrchestrator(registry=mock_registry, mapping_config={})

    def test_returns_none_on_subprocess_error(self):
        orch = self._orchestrator()
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "datasets", stderr="err"),
        ):
            result = orch._fetch_self_node_via_ncbi(9999999)
        assert result is None

    def test_returns_ancestor_from_valid_json(self):
        orch = self._orchestrator()
        payload = {
            "taxonomy": {
                "tax_id": 12345,
                "rank": "species",
                "current_scientific_name": {"name": "Fake Organism"},
            }
        }
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(payload) + "\n"

        with patch("subprocess.run", return_value=mock_result):
            result = orch._fetch_self_node_via_ncbi(12345)

        assert result is not None
        assert result.tax_id == 12345

    def test_returns_none_when_no_valid_entry_found(self):
        orch = self._orchestrator()
        mock_result = MagicMock()
        mock_result.stdout = "not json\n"

        with patch("subprocess.run", return_value=mock_result):
            result = orch._fetch_self_node_via_ncbi(9999)

        assert result is None

    def test_blank_lines_in_output_are_skipped(self):
        # Line 478: `if not line: continue` — blank line in stdout
        orch = self._orchestrator()
        payload = {
            "taxonomy": {
                "tax_id": 777,
                "rank": "genus",
                "current_scientific_name": {"name": "Test Genus"},
            }
        }
        # Blank line before the valid JSON line triggers the continue on line 478
        mock_result = MagicMock()
        mock_result.stdout = "\n" + json.dumps(payload) + "\n"

        with patch("subprocess.run", return_value=mock_result):
            result = orch._fetch_self_node_via_ncbi(777)

        assert result is not None
        assert result.tax_id == 777
