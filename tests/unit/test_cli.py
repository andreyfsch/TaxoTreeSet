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

    def test_benchmark_build_eval_parses_and_dispatches(self):
        from taxotreeset.cli import benchmark
        parser = build_parser()
        args = parser.parse_args([
            "benchmark", "build-eval",
            "--manifest", "m.json", "--registry", "r.json", "--output", "o.parquet",
        ])
        assert args.command == "benchmark"
        assert args.benchmark_cmd == "build-eval"
        assert args._run is benchmark.run
        assert args.track == "short"  # default track

    def test_benchmark_build_eval_long_track_parses(self):
        parser = build_parser()
        args = parser.parse_args([
            "benchmark", "build-eval", "--manifest", "m", "--registry", "r",
            "--output", "o", "--track", "long",
            "--min-read-length", "2000", "--max-read-length", "20000",
            "--del-rate", "0.03",
        ])
        assert args.track == "long"
        assert args.min_read_length == 2000 and args.max_read_length == 20000
        assert args.del_rate == 0.03

    def test_benchmark_requires_a_subcommand(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["benchmark"])

    def test_benchmark_build_eval_run_invokes_builder(self, tmp_path):
        from taxotreeset.cli import benchmark
        args = argparse.Namespace(
            benchmark_cmd="build-eval", manifest="m", registry="r",
            output=str(tmp_path / "o.parquet"), track="short", read_length=150,
            min_read_length=3000, max_read_length=30000, sub_rate=0.01,
            ins_rate=0.02, del_rate=0.02, homopolymer_factor=2.0,
            reads_per_genome=10, seed=0,
        )
        with (
            patch("taxotreeset.io.registry.NCBIRegistry") as mreg,
            patch("taxotreeset.cli.benchmark.build_eval_set",
                  return_value=(100, 2)) as mbe,
        ):
            mreg.return_value.registry = {"accessions": {}, "lineages": {}}
            benchmark.run(args)
        mbe.assert_called_once()

    def test_benchmark_score_parses_and_dispatches(self):
        from taxotreeset.cli import benchmark
        parser = build_parser()
        args = parser.parse_args([
            "benchmark", "score",
            "--eval-set", "e.parquet", "--predictions", "p.tsv",
            "--output", "o.json",
        ])
        assert args.command == "benchmark"
        assert args.benchmark_cmd == "score"
        assert args._run is benchmark.run

    def test_benchmark_score_run_writes_report(self, tmp_path):
        import json as _json

        import pyarrow as pa
        import pyarrow.parquet as pq

        from taxotreeset.cli import benchmark

        lineage = _json.dumps(
            [["S1", "species"], ["G1", "genus"], ["F1", "family"]])
        eval_p = tmp_path / "eval.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"read_id": "r1", "true_lineage": lineage,
             "expected_commit_taxid": "F1", "expected_commit_rank": "family",
             "distance_bin": "ANI<85%"},
            {"read_id": "r2", "true_lineage": lineage,
             "expected_commit_taxid": "F1", "expected_commit_rank": "family",
             "distance_bin": "ANI<85%"},
        ]), str(eval_p))
        preds = tmp_path / "preds.tsv"
        preds.write_text(
            "read_id\tpredicted_taxid\tpredicted_rank\n"
            "r1\tF1\tfamily\nr2\tGX\tgenus\n", encoding="utf-8")
        out = tmp_path / "report.json"
        args = argparse.Namespace(
            benchmark_cmd="score", eval_set=str(eval_p),
            predictions=str(preds), output=str(out), csv=None)
        benchmark.run(args)
        rep = _json.loads(out.read_text())
        assert rep["overall"]["n"] == 2
        assert rep["overall"]["correct"] == 1       # r1 backs off to F1
        assert rep["overall"]["over_commit"] == 1   # r2 -> a deeper wrong genus

    def test_benchmark_export_refs_and_parse_baseline_parse(self):
        parser = build_parser()
        a = parser.parse_args([
            "benchmark", "export-refs", "--manifest", "m.json",
            "--registry", "r.json", "--out-fasta", "ref.fa", "--out-map", "m.tsv",
        ])
        assert a.benchmark_cmd == "export-refs"
        b = parser.parse_args([
            "benchmark", "parse-baseline", "--tool", "kraken2",
            "--input", "k2.out", "--registry", "r.json", "--output", "p.parquet",
        ])
        assert b.benchmark_cmd == "parse-baseline" and b.tool == "kraken2"

    def test_benchmark_parse_baseline_run_writes_predictions(self, tmp_path):
        import pyarrow.parquet as pq

        from taxotreeset.cli import benchmark

        k2 = tmp_path / "k2.out"
        k2.write_text("C\tr1\tT2\t150\tmap\nU\tr2\t0\t150\t\n", encoding="utf-8")
        out = tmp_path / "preds.parquet"
        args = argparse.Namespace(
            benchmark_cmd="parse-baseline", tool="kraken2", input=str(k2),
            registry="r.json", output=str(out))
        with patch("taxotreeset.io.registry.NCBIRegistry") as mreg:
            mreg.return_value.registry = {
                "lineages": {"T2": [{"taxid": "T2", "rank": "species"}]}}
            benchmark.run(args)
        rows = {r["read_id"]: r for r in pq.read_table(str(out)).to_pylist()}
        assert rows["r1"]["predicted_taxid"] == "T2"
        assert rows["r1"]["predicted_rank"] == "species"
        assert rows["r2"]["predicted_taxid"] == ""   # unclassified -> abstain

    def test_benchmark_reliability_parses_and_dispatches(self):
        parser = build_parser()
        a = parser.parse_args([
            "benchmark", "reliability", "--heads", "d/", "--write",
        ])
        assert a.benchmark_cmd == "reliability" and a.write is True

    def test_benchmark_reliability_run_annotates_and_writes_back(self, tmp_path):
        import json as _json

        from taxotreeset.cli import benchmark

        head_dir = tmp_path / "1335638"
        head_dir.mkdir()
        lm = head_dir / "label_map.json"
        lm.write_text(_json.dumps({
            "head_taxid": "1335638",
            "reliability": {"belongs_genomes": 6, "a_priori_flag": "low",
                            "split_mode": "genome-level"},
        }), encoding="utf-8")
        metrics = tmp_path / "metrics.json"
        metrics.write_text(_json.dumps({
            "1335638": {"learned": True, "test_f1": 0.93,
                        "val_f1s": [0.92, 0.93, 0.93]},
        }), encoding="utf-8")
        summary = tmp_path / "summary.csv"
        args = argparse.Namespace(
            benchmark_cmd="reliability", heads=str(tmp_path),
            training_metrics=str(metrics), write=True, summary=str(summary))
        benchmark.run(args)

        merged = _json.loads(lm.read_text())["reliability"]
        assert merged["verdict"] == "reliable"          # training overrides "low"
        assert merged["verdict_source"] == "training"
        assert "1335638" in summary.read_text()

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
            plasmids=False,
            plasmid_release=None,
            vault=None,
            no_fetch=False,
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

    def test_plasmids_path_fetches_ingests_and_registers(self, tmp_path):
        args = self._make_args(tmp_path)
        args.plasmids = True
        args.vault = str(tmp_path / "vault")
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator") as mock_orch,
            patch("taxotreeset.cli.discover.fetch_release") as mock_fetch,
            patch("taxotreeset.cli.discover.iter_release_records", return_value=iter([])),
            patch(
                "taxotreeset.cli.discover.ingest_records_to_vault",
                return_value=[{"accession": "NZ_P1.1"}],
            ) as mock_ingest,
        ):
            discover.run(args)
        # Auto-fetches into <vault>/refseq_plasmid, then ingests + registers.
        mock_fetch.assert_called_once()
        assert mock_fetch.call_args.args[0].endswith("refseq_plasmid")
        mock_ingest.assert_called_once()
        mock_orch.return_value.discover_from_root.assert_not_called()
        call = mock_orch.return_value.discover_from_reports.call_args
        assert call.args[0] == [{"accession": "NZ_P1.1"}]
        assert call.kwargs["vault_lmdb_path"].endswith("sequences.lmdb")

    def test_plasmids_no_fetch_skips_download(self, tmp_path):
        release_dir = tmp_path / "vault" / "refseq_plasmid"
        release_dir.mkdir(parents=True)
        args = self._make_args(tmp_path)
        args.plasmids = True
        args.no_fetch = True
        args.vault = str(tmp_path / "vault")
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator"),
            patch("taxotreeset.cli.discover.fetch_release") as mock_fetch,
            patch("taxotreeset.cli.discover.iter_release_records", return_value=iter([])),
            patch("taxotreeset.cli.discover.ingest_records_to_vault", return_value=[]),
        ):
            discover.run(args)
        mock_fetch.assert_not_called()

    def test_plasmids_requires_vault(self, tmp_path):
        args = self._make_args(tmp_path)
        args.plasmids = True
        args.vault = None
        with (
            patch("taxotreeset.cli.discover.setup_logging"),
            patch("taxotreeset.cli.discover.NCBIRegistry"),
            patch("taxotreeset.cli.discover.DiscoveryOrchestrator"),
            pytest.raises(SystemExit) as exc_info,
        ):
            discover.run(args)
        assert exc_info.value.code == 1

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
            holdout_clades=None,
            holdout_rank=None,
            holdout_fraction=None,
            holdout_seed=0,
            holdout_manifest=None,
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
            reject_cross_domain=None,
            reject_cross_domain_sample=200,
            reject_cross_domain_depth=2,
            binary_only=False,
            binary_budget=30000,
            extract_batch_size=300,
            all_ranks=False,
            plasmids=False,
            plasmid_release=None,
            no_fetch=False,
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

    def test_plasmids_flag_threads_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.plasmids = True
        args.plasmid_release = "/data/release"
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        kwargs = mock_orch.call_args.kwargs
        assert kwargs["plasmids"] is True
        assert kwargs["plasmid_release"] == "/data/release"

    def test_plasmids_with_no_sync_is_rejected(self, tmp_path):
        args = self._make_args(tmp_path)
        args.plasmids = True
        args.no_sync = True
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            pytest.raises(SystemExit) as exc_info,
        ):
            generate.run(args)
        assert exc_info.value.code == 1

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

    def test_cross_domain_flags_thread_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.reject_cross_domain = "bacteria, archaea"
        args.reject_cross_domain_sample = 50
        args.reject_cross_domain_depth = 3
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        kwargs = mock_orch.call_args.kwargs
        assert kwargs["reject_cross_domain"] == ["bacteria", "archaea"]  # parsed
        assert kwargs["reject_cross_domain_sample"] == 50
        assert kwargs["reject_cross_domain_depth"] == 3

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

    def test_holdout_flags_thread_to_orchestrator(self, tmp_path):
        args = self._make_args(tmp_path)
        args.holdout_clades = "11234, 10509"
        args.holdout_seed = 3
        with (
            patch("taxotreeset.cli.generate.setup_logging"),
            patch("taxotreeset.cli.generate.NCBIRegistry"),
            patch("taxotreeset.cli.generate.GenerationOrchestrator") as mock_orch,
        ):
            mock_orch.return_value.run_pipeline.return_value = None
            generate.run(args)
        kwargs = mock_orch.call_args.kwargs
        assert kwargs["holdout_clades"] == ["11234", "10509"]  # parsed + trimmed
        assert kwargs["holdout_seed"] == 3

    def test_holdout_clades_and_rank_are_mutually_exclusive(self, tmp_path):
        args = self._make_args(tmp_path)
        args.holdout_clades = "11234"
        args.holdout_rank = "genus"
        with pytest.raises(SystemExit):
            generate.run(args)

    def test_holdout_rank_requires_a_fraction(self, tmp_path):
        args = self._make_args(tmp_path)
        args.holdout_rank = "genus"
        args.holdout_fraction = None
        with pytest.raises(SystemExit):
            generate.run(args)

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
