"""Tests for taxotreeset CLI modules: __main__, discover, generate.

These tests use argparse.Namespace objects directly (NOT click.testing.CliRunner)
because the CLI is built with argparse, not Click. Orchestrators are mocked to
avoid network/filesystem side effects.
"""

import argparse
import json
from unittest.mock import patch

import pytest

from taxotreeset.__main__ import build_parser, main
from taxotreeset.cli import discover, generate


# ---------------------------------------------------------------------------
# build_parser / __main__
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_returns_argument_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_discover_subcommand(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["discover", "--help"])
        assert exc_info.value.code == 0

    def test_parser_has_generate_subcommand(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate", "--help"])
        assert exc_info.value.code == 0

    def test_discover_subcommand_sets_run_attribute(self):
        parser = build_parser()
        args = parser.parse_args(["discover"])
        assert hasattr(args, "_run")
        assert args._run is discover.run

    def test_generate_subcommand_sets_run_attribute(self):
        parser = build_parser()
        args = parser.parse_args(["generate"])
        assert hasattr(args, "_run")
        assert args._run is generate.run

    def test_no_subcommand_prints_help_and_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_cluster_aware_split_is_on_by_default(self):
        parser = build_parser()
        assert parser.parse_args(["generate"]).cluster_aware_split is True

    def test_no_cluster_aware_split_opts_out(self):
        parser = build_parser()
        args = parser.parse_args(["generate", "--no-cluster-aware-split"])
        assert args.cluster_aware_split is False

    def test_cluster_aware_flags_are_mutually_exclusive(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(
                ["generate", "--cluster-aware-split", "--no-cluster-aware-split"])
        assert exc_info.value.code == 2

    def test_single_level_and_stop_at_are_mutually_exclusive(self):
        # The help promises the two cannot be combined; argparse rejects the
        # combination up front (exit 2) instead of a deep run_pipeline ValueError.
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["generate", "--single-level", "--stop-at", "genus"])
        assert exc_info.value.code == 2

    def test_main_dispatches_to_discover_run(self, tmp_path):
        mapping = tmp_path / "mapping.json"
        mapping.write_text(json.dumps({"scopes": {}}), encoding="utf-8")

        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.discover_from_root.return_value = None
            main([
                "discover",
                "--taxon-id", "10239",
                "--mapping", str(mapping),
                "--registry", str(tmp_path / "registry.json"),
            ])
        mock_orch.return_value.discover_from_root.assert_called_once()


# ---------------------------------------------------------------------------
# cli/discover.py
# ---------------------------------------------------------------------------


class TestDiscoverRun:
    def _make_args(self, tmp_path, reset=False, mapping_exists=True):
        mapping = tmp_path / "mapping.json"
        if mapping_exists:
            mapping.write_text(json.dumps({"scopes": {}}), encoding="utf-8")
        return argparse.Namespace(
            taxon_id=10239,
            mapping=str(mapping),
            registry=str(tmp_path / "registry.json"),
            reset=reset,
            all_ranks=False,
            log_level="INFO",
        )

    def test_run_calls_discover_from_root(self, tmp_path):
        args = self._make_args(tmp_path)
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.discover_from_root.return_value = None
            discover.run(args)
        mock_orch.return_value.discover_from_root.assert_called_once_with(10239)

    def test_all_ranks_flag_threads_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.all_ranks = True
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.discover_from_root.return_value = None
            discover.run(args)
        assert mock_orch.call_args.kwargs["all_ranks"] is True

    def test_run_exits_when_mapping_missing(self, tmp_path):
        args = self._make_args(tmp_path, mapping_exists=False)
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            pytest.raises(SystemExit) as exc_info,
        ):
            discover.run(args)
        assert exc_info.value.code == 1

    def test_reset_flag_removes_existing_registry(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        registry_path.write_text("{}", encoding="utf-8")
        args = self._make_args(tmp_path, reset=True)
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.discover_from_root.return_value = None
            discover.run(args)
        assert not registry_path.exists()

    def test_reset_os_error_causes_sys_exit(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        registry_path.write_text("{}", encoding="utf-8")
        args = self._make_args(tmp_path, reset=True)
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("os.remove", side_effect=OSError("permission denied")),
            pytest.raises(SystemExit) as exc_info,
        ):
            discover.run(args)
        assert exc_info.value.code == 1

    def test_exception_in_orchestrator_causes_sys_exit(self, tmp_path):
        args = self._make_args(tmp_path)
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_orch.return_value.discover_from_root.side_effect = RuntimeError("boom")
            discover.run(args)
        assert exc_info.value.code == 1

    def test_existing_registry_without_reset_logs_incremental_mode(self, tmp_path):
        # Registry file exists and reset=False → elif branch logs incremental mode (line 76)
        registry_path = tmp_path / "registry.json"
        registry_path.write_text("{}", encoding="utf-8")
        args = self._make_args(tmp_path, reset=False)
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.discover_from_root.return_value = None
            discover.run(args)
        mock_orch.return_value.discover_from_root.assert_called_once()

    def test_invalid_json_in_mapping_causes_sys_exit(self, tmp_path):
        # Mapping file exists but contains invalid JSON → lines 87-89
        mapping = tmp_path / "mapping.json"
        mapping.write_text("not valid json!!!", encoding="utf-8")
        args = argparse.Namespace(
            taxon_id=10239,
            mapping=str(mapping),
            registry=str(tmp_path / "registry.json"),
            reset=False,
            log_level="INFO",
        )
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            pytest.raises(SystemExit) as exc_info,
        ):
            discover.run(args)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# cli/generate.py
# ---------------------------------------------------------------------------


class TestGenerateRun:
    def _make_args(self, tmp_path, no_sync=False, registry_exists=True):
        registry = tmp_path / "registry.json"
        if registry_exists:
            registry.write_text("{}", encoding="utf-8")
        return argparse.Namespace(
            mapping=str(tmp_path / "mapping.json"),
            vault=str(tmp_path / "vault"),
            seed=42,
            min_num_seqs=1000,
            cutoff_percentage=98.0,
            approximate_capacity=False,
            root="viruses",
            stop_at=None,
            single_level=False,
            output_format="parquet",
            max_subseq_len=2000,
            min_subseq_len=100,
            registry=str(registry),
            output=str(tmp_path / "output"),
            min_abundance=2,
            min_subclades_per_bucket=5,
            max_n_per_class=20000,
            keep_imbalance=False,
            cluster_aware_split=False,
            cluster_jaccard_threshold=None,
            cluster_min_genomes=None,
            cluster_min_frac=None,
            min_leaves_per_class=3,
            rare_taxa_strategy="fallback",
            no_sync=no_sync,
            spill_dir=None,
            tmp_dir=None,
            workers=None,
            gpu_workers=None,
            exclude_plasmids=False,
            reject_class=False,
            reject_fraction=1.0,
            reject_near_far_start=0.5,
            reject_near_far_end=0.9,
            binary_only=False,
            binary_budget=30000,
            extract_batch_size=300,
            all_ranks=False,
            log_level="INFO",
        )

    def test_run_calls_run_pipeline(self, tmp_path):
        args = self._make_args(tmp_path)
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        mock_orch.return_value.run_pipeline.assert_called_once()

    def test_reject_flags_thread_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.reject_class = True
        args.reject_fraction = 0.5
        args.reject_near_far_start = 0.25
        args.reject_near_far_end = 0.8
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        kwargs = mock_orch.call_args.kwargs
        assert kwargs["reject_class"] is True
        assert kwargs["reject_fraction"] == 0.5
        assert kwargs["reject_near_far_start"] == 0.25
        assert kwargs["reject_near_far_end"] == 0.8

    def test_binary_flags_thread_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.binary_only = True
        args.binary_budget = 25000
        args.extract_batch_size = 128
        args.all_ranks = True
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        kwargs = mock_orch.call_args.kwargs
        assert kwargs["binary_only"] is True
        assert kwargs["binary_budget"] == 25000
        assert kwargs["binary_extract_batch_size"] == 128
        assert kwargs["all_ranks"] is True

    def test_cluster_flags_thread_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.cluster_aware_split = True
        args.cluster_jaccard_threshold = 0.5
        args.cluster_min_genomes = 4
        args.cluster_min_frac = 0.2
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        kwargs = mock_orch.call_args.kwargs
        assert kwargs["cluster_aware_split"] is True
        cp = kwargs["cluster_params"]
        assert cp.jaccard_threshold == 0.5
        assert cp.min_cluster_genomes == 4
        assert cp.min_cluster_frac == 0.2
        # unset knobs keep their defaults
        from taxotreeset.core._orchestration._cluster import _KMER_K
        assert cp.k == _KMER_K

    def test_cluster_flags_default_to_none_giving_default_params(self, tmp_path):
        args = self._make_args(tmp_path)  # cluster_* all None
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        from taxotreeset.core._orchestration._cluster import ClusterParams
        assert mock_orch.call_args.kwargs["cluster_params"] == ClusterParams()

    def test_no_sync_with_missing_registry_exits(self, tmp_path):
        args = self._make_args(tmp_path, no_sync=True, registry_exists=False)
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            pytest.raises(SystemExit) as exc_info,
        ):
            generate.run(args)
        assert exc_info.value.code == 1

    def test_nonpositive_min_subseq_len_exits(self, tmp_path):
        # Fail fast before the sync/capacity passes rather than crashing deep in
        # extraction (where _validate_extraction_parameters would raise).
        args = self._make_args(tmp_path)
        args.min_subseq_len = 0
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
            pytest.raises(SystemExit) as exc_info,
        ):
            generate.run(args)
        assert exc_info.value.code == 1
        mock_orch.assert_not_called()  # exits before building the pipeline

    def test_max_subseq_len_below_min_exits(self, tmp_path):
        args = self._make_args(tmp_path)
        args.min_subseq_len = 200
        args.max_subseq_len = 100
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
            pytest.raises(SystemExit) as exc_info,
        ):
            generate.run(args)
        assert exc_info.value.code == 1
        mock_orch.assert_not_called()

    def test_exception_in_pipeline_causes_sys_exit(self, tmp_path):
        args = self._make_args(tmp_path)
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
            pytest.raises(SystemExit) as exc_info,
        ):
            mock_orch.return_value.run_pipeline.side_effect = RuntimeError("boom")
            generate.run(args)
        assert exc_info.value.code == 1

    def test_run_pipeline_receives_correct_root(self, tmp_path):
        args = self._make_args(tmp_path)
        args.root = "bacteria"
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        call_kwargs = mock_orch.return_value.run_pipeline.call_args
        assert call_kwargs.kwargs.get("target_group") == "bacteria" or \
               (call_kwargs.args and call_kwargs.args[0] == "bacteria")

    def test_single_level_flag_uses_single_level_depth_desc(self, tmp_path):
        # Covers generate.py line 238: depth_desc = "single level (root head only)"
        args = self._make_args(tmp_path)
        args.single_level = True
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
            patch("builtins.print"),  # suppress output
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        mock_orch.return_value.run_pipeline.assert_called_once()

    def test_stop_at_flag_uses_stop_at_depth_desc(self, tmp_path):
        # Covers generate.py line 240: depth_desc = f"stop at {args.stop_at}"
        args = self._make_args(tmp_path)
        args.stop_at = "genus"
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
            patch("builtins.print"),
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        mock_orch.return_value.run_pipeline.assert_called_once()
