"""Integration tests for the synthetic vault pipeline.

Covers the full chain from LMDB read-back → tree construction →
capacity computation → balanced extraction plan, using only local
synthetic data (no network access required).
"""

import json
import zlib
from pathlib import Path

import pytest

from taxotreeset.core.generation.balancing import compute_balanced_extraction_plan
from taxotreeset.core.generation.capacity import compute_all_capacities
from taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from taxotreeset.dataset.utils import _read_single_sequence

_MIN_LEN = 10


# ---------------------------------------------------------------------------
# 1. Vault read-back
# ---------------------------------------------------------------------------


class TestVaultReadBack:
    def test_all_headers_are_readable(self, synthetic_env):
        lmdb_path = synthetic_env["lmdb_path"]
        sequences = synthetic_env["sequences"]
        for header_id, expected_seq in sequences.items():
            read_seq = _read_single_sequence(lmdb_path, header_id)
            assert read_seq == expected_seq, f"{header_id} read-back mismatch"

    def test_unknown_header_returns_empty_string(self, synthetic_env):
        lmdb_path = synthetic_env["lmdb_path"]
        assert _read_single_sequence(lmdb_path, "NONEXISTENT_HEADER") == ""

    def test_raw_vault_key_is_zlib_compressed(self, synthetic_env):
        from taxotreeset.dataset.utils import _get_lmdb_env
        lmdb_dir = synthetic_env["lmdb_path"]
        env = _get_lmdb_env(lmdb_dir)
        with env.begin() as txn:
            raw = txn.get(b"NC_045512")
        assert raw is not None
        decompressed = zlib.decompress(raw).decode("utf-8")
        assert decompressed == synthetic_env["sequences"]["NC_045512"]


# ---------------------------------------------------------------------------
# 2. Tree construction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree_root(synthetic_env):
    return generate_seqs_by_taxon_tree(
        registry_path=synthetic_env["registry_path"],
        vault_path=synthetic_env["vault_dir"],
        domain_taxid=synthetic_env["domain_taxid"],
        mapping_path=synthetic_env["mapping_path"],
    )


class TestTreeConstruction:
    def test_root_node_exists(self, tree_root):
        assert tree_root is not None
        assert tree_root.name == "root"

    def test_domain_anchor_is_direct_child_of_root(self, tree_root):
        child_names = {c.name for c in tree_root.children}
        assert "10239" in child_names

    def test_two_families_under_domain(self, tree_root):
        domain_node = next(c for c in tree_root.children if c.name == "10239")
        family_names = {c.name for c in domain_node.children}
        assert "11118" in family_names
        assert "12227" in family_names

    def test_correct_species_under_coronaviridae(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        corona = next(c for c in domain.children if c.name == "11118")
        species_names = {c.name for c in corona.children}
        assert "2697049" in species_names
        assert "11234" in species_names

    def test_correct_species_under_adenoviridae(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        adeno = next(c for c in domain.children if c.name == "12227")
        species_names = {c.name for c in adeno.children}
        assert "10509" in species_names

    def test_sequence_leaves_are_attached(self, tree_root):
        all_leaves = list(tree_root.leaves)
        seq_leaves = [leaf for leaf in all_leaves if getattr(leaf, "rank", "") == "sequence"]
        assert len(seq_leaves) == 3

    def test_sequence_leaves_carry_fasta_path(self, tree_root, synthetic_env):
        all_leaves = list(tree_root.leaves)
        for leaf in all_leaves:
            if getattr(leaf, "rank", "") == "sequence":
                assert hasattr(leaf, "fasta_path")
                assert leaf.fasta_path == synthetic_env["lmdb_path"]

    def test_sequence_leaves_carry_header_id(self, tree_root):
        all_leaves = list(tree_root.leaves)
        header_ids = {
            leaf.header_id
            for leaf in all_leaves
            if getattr(leaf, "rank", "") == "sequence"
        }
        assert header_ids == {"NC_045512", "NC_001846", "NC_001407"}

    def test_taxonomic_nodes_carry_rank_attribute(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        assert domain.rank == "superkingdom"

        corona = next(c for c in domain.children if c.name == "11118")
        assert corona.rank == "family"

    def test_tree_is_idempotent_when_called_twice(self, synthetic_env):
        root1 = generate_seqs_by_taxon_tree(
            registry_path=synthetic_env["registry_path"],
            vault_path=synthetic_env["vault_dir"],
            domain_taxid=synthetic_env["domain_taxid"],
            mapping_path=synthetic_env["mapping_path"],
        )
        root2 = generate_seqs_by_taxon_tree(
            registry_path=synthetic_env["registry_path"],
            vault_path=synthetic_env["vault_dir"],
            domain_taxid=synthetic_env["domain_taxid"],
            mapping_path=synthetic_env["mapping_path"],
        )
        leaves1 = {leaf.name for leaf in root1.leaves}
        leaves2 = {leaf.name for leaf in root2.leaves}
        assert leaves1 == leaves2


# ---------------------------------------------------------------------------
# 3. Capacity computation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_capacities(tree_root):
    return compute_all_capacities(tree_root, min_len=_MIN_LEN)


class TestCapacityComputation:
    def test_returns_a_non_empty_dict(self, all_capacities):
        assert isinstance(all_capacities, dict)
        assert len(all_capacities) > 0

    def test_all_non_leaf_taxon_nodes_have_entries(self, all_capacities):
        expected_taxids = {"root", "10239", "11118", "12227", "2697049", "11234", "10509"}
        for taxid in expected_taxids:
            assert taxid in all_capacities, f"Missing capacity for taxid {taxid}"

    def test_all_capacities_are_positive(self, all_capacities):
        for taxid, cap in all_capacities.items():
            assert cap > 0, f"Zero or negative capacity for {taxid}"

    def test_parent_capacity_ge_child_capacity(self, all_capacities):
        assert all_capacities["11118"] >= all_capacities["2697049"]
        assert all_capacities["11118"] >= all_capacities["11234"]
        assert all_capacities["12227"] >= all_capacities["10509"]
        assert all_capacities["10239"] >= all_capacities["11118"]
        assert all_capacities["10239"] >= all_capacities["12227"]

    def test_species_capacity_matches_sequence_length_minus_window(
        self, all_capacities, synthetic_env
    ):
        seq = synthetic_env["sequences"]["NC_045512"]
        expected_upper_bound = len(seq) - _MIN_LEN + 1
        assert all_capacities["2697049"] <= expected_upper_bound

    def test_sequence_leaves_absent_from_capacities(self, all_capacities):
        header_ids = {"NC_045512", "NC_001846", "NC_001407"}
        for hid in header_ids:
            assert hid not in all_capacities

    def test_root_capacity_is_largest(self, all_capacities):
        root_cap = all_capacities["root"]
        for taxid, cap in all_capacities.items():
            assert root_cap >= cap, f"root capacity should be >= {taxid}: {root_cap} < {cap}"


# ---------------------------------------------------------------------------
# 4. Full pipeline: tree → capacity → balanced plan
# ---------------------------------------------------------------------------


class TestBalancingWithRealCapacities:
    def _get_domain_node_and_families(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        families = list(domain.children)
        return domain, families

    def test_plan_has_valid_structure(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert "scenario" in plan
        assert "n_per_class" in plan
        assert "retained_children" in plan
        assert "low_capacity_children" in plan
        assert "capacities" in plan
        assert "rare_taxa_children" in plan

    def test_plan_scenario_is_a_known_value(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["scenario"] in ("level_all", "level_all_capped", "cutoff_applied")

    def test_n_per_class_is_positive(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["n_per_class"] > 0

    def test_retained_plus_low_capacity_equals_total(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        total = len(plan["retained_children"]) + len(plan["low_capacity_children"])
        assert total == len(families)

    def test_plan_with_low_min_num_seqs_triggers_level_all(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        min_cap = min(all_capacities.get(str(f.name), 0) for f in families)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            min_num_seqs=1,
            max_n_per_class=min_cap + 1,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["scenario"] == "level_all"
        assert plan["n_per_class"] == min_cap

    def test_plan_without_capacity_override_uses_real_sequences(
        self, tree_root
    ):
        domain = next(c for c in tree_root.children if c.name == "10239")
        corona = next(c for c in domain.children if c.name == "11118")
        species = list(corona.children)

        plan = compute_balanced_extraction_plan(
            parent_node=corona,
            children=species,
            leaf_cache={},
            min_len=_MIN_LEN,
            use_exact_capacity=True,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["n_per_class"] > 0
        for taxid, cap in plan["capacities"].items():
            assert cap > 0, f"Zero capacity for {taxid} without override"


# ---------------------------------------------------------------------------
# 5. Full pipeline output contracts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline_output(synthetic_env, tmp_path_factory):
    """Run the full pipeline end-to-end and return output paths."""
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator
    from taxotreeset.io.registry import NCBIRegistry

    output_dir = str(tmp_path_factory.mktemp("pipeline_out"))
    registry = NCBIRegistry(registry_path=synthetic_env["registry_path"])

    orch = GenerationOrchestrator(
        registry=registry,
        vault_path=synthetic_env["vault_dir"],
        output_dir=output_dir,
        config_path=synthetic_env["mapping_path"],
        min_subseq_len=_MIN_LEN,
        max_subseq_len=200,  # must be >= builder._DEFAULT_MIN_SUBSEQ_LEN (100)
        max_n_per_class=100,
        min_leaves_per_class=1,
        rare_taxa_strategy="keep",
        n_gpu_workers=0,
        n_workers=1,
    )
    orch.run_pipeline(target_group="viruses", sync=False, abundance_threshold=1)
    return {"output_dir": output_dir}


@pytest.fixture(scope="module")
def binary_pipeline_output(synthetic_env, tmp_path_factory):
    """Run the full pipeline in --binary-only mode and return output paths."""
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator
    from taxotreeset.io.registry import NCBIRegistry

    output_dir = str(tmp_path_factory.mktemp("binary_out"))
    registry = NCBIRegistry(registry_path=synthetic_env["registry_path"])
    orch = GenerationOrchestrator(
        registry=registry,
        vault_path=synthetic_env["vault_dir"],
        output_dir=output_dir,
        config_path=synthetic_env["mapping_path"],
        min_subseq_len=_MIN_LEN,
        max_subseq_len=200,
        max_n_per_class=100,
        min_leaves_per_class=1,
        rare_taxa_strategy="keep",
        n_gpu_workers=0,
        n_workers=1,
        binary_only=True,
        binary_budget=60,
    )
    orch.run_pipeline(target_group="viruses", sync=False, abundance_threshold=1)
    return {"output_dir": output_dir}


class TestBinaryOnlyGeneration:
    def test_metadata_records_binary(self, binary_pipeline_output):
        meta = json.loads((Path(binary_pipeline_output["output_dir"])
                           / "run_metadata_viruses.json").read_text())
        assert meta["parameters"]["binary_only"] is True
        assert meta["parameters"]["binary_budget"] == 60

    def test_heads_are_two_class(self, binary_pipeline_output):
        import pandas as pd
        out = Path(binary_pipeline_output["output_dir"])
        parquets = list(out.rglob("train.parquet"))
        assert parquets, "no binary head datasets produced"
        for p in parquets:
            labels = set(int(x) for x in pd.read_parquet(p)["class_idx"].unique())
            assert labels <= {0, 1}, f"{p} not binary: {labels}"
            assert labels == {0, 1}, f"{p} missing a class: {labels}"

    def test_every_head_has_nonempty_test(self, binary_pipeline_output):
        # The window-slicing fallback must give every node a valid split — no
        # viability gating, no empty test (the bug that broke a 3-genome node).
        import pandas as pd
        out = Path(binary_pipeline_output["output_dir"])
        for train_p in out.rglob("train.parquet"):
            test_p = train_p.parent / "test.parquet"
            assert test_p.exists(), f"{train_p.parent} has train but no test split"
            assert len(pd.read_parquet(test_p)) > 0, f"{test_p} is empty"

    def test_single_child_nodes_collapse_to_passthrough(self, binary_pipeline_output):
        # A binary head for a single-child node is redundant with its only child
        # (identical subtree), so it must NOT be emitted — it collapses to a
        # passthrough. Adenoviridae 12227 has exactly one species (10509).
        out = Path(binary_pipeline_output["output_dir"])
        passthroughs = json.loads(
            (out / "passthroughs_viruses.json").read_text())
        assert passthroughs.get("12227") == "10509", (
            "single-child Adenoviridae should collapse to its only species")
        # No passthrough taxid may have a head; the child keeps its head.
        for taxid in passthroughs:
            assert not list(out.rglob(f"*/{taxid}/train.parquet")), (
                f"passthrough {taxid} wrongly got a belongs/not-belongs head")
        assert list(out.rglob("*/10509/train.parquet")), (
            "the passthrough's child (10509) must keep its head")


@pytest.fixture(scope="module")
def binary_pipeline_output_batched(synthetic_env, tmp_path_factory):
    """Same binary run but with batch_size=1, forcing many extraction flushes."""
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator
    from taxotreeset.io.registry import NCBIRegistry

    output_dir = str(tmp_path_factory.mktemp("binary_out_batched"))
    registry = NCBIRegistry(registry_path=synthetic_env["registry_path"])
    orch = GenerationOrchestrator(
        registry=registry,
        vault_path=synthetic_env["vault_dir"],
        output_dir=output_dir,
        config_path=synthetic_env["mapping_path"],
        min_subseq_len=_MIN_LEN,
        max_subseq_len=200,
        max_n_per_class=100,
        min_leaves_per_class=1,
        rare_taxa_strategy="keep",
        n_gpu_workers=0,
        n_workers=1,
        binary_only=True,
        binary_budget=60,
        binary_extract_batch_size=1,
    )
    orch.run_pipeline(target_group="viruses", sync=False, abundance_threshold=1)
    return {"output_dir": output_dir}


class TestBinaryBatchedExtractionParity:
    """Streaming extraction in batches must not change the produced datasets."""

    def _head_rowcounts(self, output_dir):
        import pandas as pd
        out = Path(output_dir)
        counts = {}
        for parquet in out.rglob("*.parquet"):
            rel = parquet.relative_to(out)
            counts[str(rel)] = len(pd.read_parquet(parquet))
        return counts

    def test_same_head_files_and_rowcounts(
        self, binary_pipeline_output, binary_pipeline_output_batched
    ):
        single = self._head_rowcounts(binary_pipeline_output["output_dir"])
        batched = self._head_rowcounts(binary_pipeline_output_batched["output_dir"])
        assert batched, "batched run produced no parquet files"
        assert set(batched) == set(single), (
            "batched extraction produced a different set of head files"
        )
        assert batched == single, (
            "batched extraction changed per-file row counts (non-deterministic "
            "across batch boundaries)"
        )

    def test_batched_metadata_records_all_heads(
        self, binary_pipeline_output, binary_pipeline_output_batched
    ):
        def n_heads(output_dir):
            meta = json.loads((Path(output_dir)
                               / "run_metadata_viruses.json").read_text())
            return meta["summary"]["n_heads"]
        single = n_heads(binary_pipeline_output["output_dir"])
        batched = n_heads(binary_pipeline_output_batched["output_dir"])
        # n_heads must reflect the manifest, not the (now-empty) extraction_jobs.
        assert single > 0 and batched > 0, "binary run reported zero heads"
        assert batched == single


class TestFullPipelineOutputContracts:
    def test_run_metadata_file_exists(self, pipeline_output):
        path = Path(pipeline_output["output_dir"]) / "run_metadata_viruses.json"
        assert path.exists(), "run_metadata_viruses.json not written"

    def test_run_metadata_required_keys(self, pipeline_output):
        path = Path(pipeline_output["output_dir"]) / "run_metadata_viruses.json"
        meta = json.loads(path.read_text())
        for key in ("taxotreeset_version", "generated_at", "elapsed_seconds",
                    "parameters", "summary", "heads"):
            assert key in meta, f"Missing top-level key: {key}"

    def test_run_metadata_parameters_match_config(self, pipeline_output):
        path = Path(pipeline_output["output_dir"]) / "run_metadata_viruses.json"
        meta = json.loads(path.read_text())
        params = meta["parameters"]
        assert params["root"] == "viruses"
        assert params["min_subseq_len"] == _MIN_LEN
        assert params["max_n_per_class"] == 100
        assert params["stop_at"] is None
        # reject-bucket provenance (incl. the depth-scaled near/far ratio) must be
        # recorded so a dataset's reject constitution is reproducible from metadata.
        for key in ("reject_class", "reject_fraction",
                    "reject_near_far_start", "reject_near_far_end"):
            assert key in params, f"missing reject provenance key: {key}"

    def test_run_metadata_summary_counts(self, pipeline_output):
        path = Path(pipeline_output["output_dir"]) / "run_metadata_viruses.json"
        meta = json.loads(path.read_text())
        summary = meta["summary"]
        assert summary["n_heads"] > 0
        assert summary["n_classes_total"] > 0
        assert summary["n_taxa_in_tree"] > 0
        assert summary["n_accessions"] == 3

    def test_run_metadata_by_rank_present(self, pipeline_output):
        path = Path(pipeline_output["output_dir"]) / "run_metadata_viruses.json"
        meta = json.loads(path.read_text())
        by_rank = meta["summary"]["by_rank"]
        assert isinstance(by_rank, dict)
        assert len(by_rank) > 0
        for rank, stats in by_rank.items():
            assert stats["n_heads"] > 0
            assert stats["total_classes"] > 0
            assert stats["median_n_per_class"] >= 0
            assert stats["median_capacity"] >= 0

    def test_run_metadata_heads_have_class_list(self, pipeline_output):
        path = Path(pipeline_output["output_dir"]) / "run_metadata_viruses.json"
        meta = json.loads(path.read_text())
        for head in meta["heads"]:
            assert "taxid" in head
            assert "rank" in head
            assert "n_classes" in head
            assert isinstance(head["classes"], list)
            assert len(head["classes"]) == head["n_classes"]

    def test_parquet_files_exist(self, pipeline_output):
        parquet_files = list(Path(pipeline_output["output_dir"]).rglob("*.parquet"))
        assert len(parquet_files) > 0, "No parquet files written"

    def test_parquet_required_columns(self, pipeline_output):
        import pyarrow.parquet as pq
        for parquet_file in Path(pipeline_output["output_dir"]).rglob("*.parquet"):
            schema_names = pq.read_schema(parquet_file).names
            assert "seq" in schema_names, f"Missing 'seq' column in {parquet_file}"
            assert "class_idx" in schema_names, f"Missing 'class_idx' column in {parquet_file}"

    def test_label_map_file_exists_in_each_head_dir(self, pipeline_output):
        output_dir = Path(pipeline_output["output_dir"])
        manifest_path = output_dir / "manifest_viruses.json"
        manifest = json.loads(manifest_path.read_text())
        for v in manifest.values():
            head_dir = Path(v["directory_path"])
            label_map_path = head_dir / "label_map.json"
            assert label_map_path.exists(), f"label_map.json missing in {head_dir}"

    def test_label_map_huggingface_format(self, pipeline_output):
        output_dir = Path(pipeline_output["output_dir"])
        manifest_path = output_dir / "manifest_viruses.json"
        manifest = json.loads(manifest_path.read_text())
        for taxid, v in manifest.items():
            head_dir = Path(v["directory_path"])
            label_map = json.loads((head_dir / "label_map.json").read_text())
            assert label_map["head_taxid"] == taxid
            assert "id2label" in label_map
            assert "label2id" in label_map
            assert "classes" in label_map
            # id2label and label2id must be consistent
            for idx_str, name in label_map["id2label"].items():
                assert label_map["label2id"][name] == int(idx_str)
            # classes sorted by class_idx, no gaps
            idxs = [c["class_idx"] for c in label_map["classes"]]
            assert idxs == list(range(len(idxs)))

    def test_parquet_class_idx_within_manifest_bounds(self, pipeline_output):
        import pyarrow.parquet as pq
        output_dir = Path(pipeline_output["output_dir"])
        manifest_path = output_dir / "manifest_viruses.json"
        manifest = json.loads(manifest_path.read_text())

        # Build a map: head directory → valid class_idx set
        dir_to_valid: dict[str, set[int]] = {}
        for v in manifest.values():
            head_dir = str(v["directory_path"])
            valid = {lv["class_idx"] for lv in v["labels"].values()}
            dir_to_valid[head_dir] = valid

        for parquet_file in output_dir.rglob("*.parquet"):
            table = pq.read_table(parquet_file, columns=["class_idx"])
            # Identify the owning head directory (parent of train/val/test split dir)
            head_dir = str(parquet_file.parent.parent)
            valid_idxs = dir_to_valid.get(head_dir, set())
            if not valid_idxs:
                continue
            for idx in table["class_idx"].to_pylist():
                assert idx in valid_idxs, (
                    f"class_idx {idx} not in manifest for {head_dir}"
                )


# ---------------------------------------------------------------------------
# --single-level <taxid>: regenerate a single head, negatives from the whole tree
# ---------------------------------------------------------------------------


def _single_level_orch(synthetic_env, output_dir, **overrides):
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator
    from taxotreeset.io.registry import NCBIRegistry

    registry = NCBIRegistry(registry_path=synthetic_env["registry_path"])
    kwargs = dict(
        registry=registry,
        vault_path=synthetic_env["vault_dir"],
        output_dir=output_dir,
        config_path=synthetic_env["mapping_path"],
        min_subseq_len=_MIN_LEN,
        max_subseq_len=200,
        max_n_per_class=100,
        min_leaves_per_class=1,
        rare_taxa_strategy="keep",
        n_gpu_workers=0,
        n_workers=1,
    )
    kwargs.update(overrides)
    return GenerationOrchestrator(**kwargs)


@pytest.fixture(scope="module")
def binary_single_level_output(synthetic_env, tmp_path_factory):
    """``--binary-only`` + ``single_level='11118'``: only the Coronaviridae head.

    11118 (Coronaviridae) is a branching family; its not-belongs negatives can
    only come from OUTSIDE its subtree (the Adenoviridae leaf 10509). Scheduling
    just this head must still sample that external pool, so the head is a drop-in
    for the one a full run emits.
    """
    output_dir = str(tmp_path_factory.mktemp("binary_single_out"))
    orch = _single_level_orch(
        synthetic_env, output_dir, binary_only=True, binary_budget=60)
    orch.run_pipeline(
        target_group="viruses", sync=False, abundance_threshold=1,
        single_level="11118")
    return {"output_dir": output_dir}


def _head_11118_class_counts(output_dir):
    import pandas as pd

    manifest = json.loads(
        (Path(output_dir) / "manifest_viruses.json").read_text())
    head_dir = Path(manifest["11118"]["directory_path"])
    return {
        split: dict(sorted(
            pd.read_parquet(head_dir / f"{split}.parquet")["class_idx"]
            .value_counts().items()))
        for split in ("train", "val", "test")
    }


class TestBinarySingleLevelTarget:
    def test_only_the_target_head_is_scheduled(self, binary_single_level_output):
        out = Path(binary_single_level_output["output_dir"])
        manifest = json.loads((out / "manifest_viruses.json").read_text())
        assert set(manifest) == {"11118"}, (
            f"single-level run scheduled heads other than 11118: {set(manifest)}")
        heads = {p.parent.name for p in out.rglob("train.parquet")}
        assert heads == {"11118"}, f"expected only the 11118 head dir, got {heads}"

    def test_target_head_keeps_out_of_subtree_negatives(
        self, binary_single_level_output
    ):
        # label 0 = not-belongs, drawn from OUTSIDE 11118. Its presence proves the
        # reject pool is still the whole tree, not the (empty) 11118 subtree.
        import pandas as pd
        out = Path(binary_single_level_output["output_dir"])
        train = next(out.rglob("*/11118/train.parquet"))
        labels = set(int(x) for x in pd.read_parquet(train)["class_idx"].unique())
        assert labels == {0, 1}, f"single-level head lost a class: {labels}"

    def test_head_is_identical_to_the_full_run(
        self, binary_single_level_output, binary_pipeline_output
    ):
        # Drop-in parity: per-split, per-class window counts must match the 11118
        # head a full binary run emits (same tree, same reject pool, same per-head
        # seed) — the whole point of the flag. Equal label-0 counts prove the
        # negatives are the same out-of-subtree windows, not a shrunken pool.
        single = _head_11118_class_counts(binary_single_level_output["output_dir"])
        full = _head_11118_class_counts(binary_pipeline_output["output_dir"])
        assert single == full, (
            f"single-level 11118 head differs from full run: {single} != {full}")


@pytest.fixture(scope="module")
def multi_single_level_output(synthetic_env, tmp_path_factory):
    """Multi-class + ``single_level='11118'``: only the Coronaviridae decision point."""
    output_dir = str(tmp_path_factory.mktemp("multi_single_out"))
    orch = _single_level_orch(synthetic_env, output_dir)
    orch.run_pipeline(
        target_group="viruses", sync=False, abundance_threshold=1,
        single_level="11118")
    return {"output_dir": output_dir}


@pytest.fixture(scope="module")
def binary_holdout_output(synthetic_env, tmp_path_factory):
    """--binary-only with Mouse hepatitis virus (11234) withheld as a novel clade."""
    output_dir = str(tmp_path_factory.mktemp("binary_holdout_out"))
    orch = _single_level_orch(
        synthetic_env, output_dir, binary_only=True, binary_budget=60,
        holdout_clades=["11234"])
    orch.run_pipeline(
        target_group="viruses", sync=False, abundance_threshold=1)
    return {"output_dir": output_dir}


class TestCladeHoldout:
    def test_held_out_clade_gets_no_head(self, binary_holdout_output):
        out = Path(binary_holdout_output["output_dir"])
        heads = {p.parent.name for p in out.rglob("train.parquet")}
        assert heads, "no heads produced at all"
        assert "11234" not in heads, "withheld clade must not be trained on"

    def test_manifest_records_expected_commit_rank(self, binary_holdout_output):
        out = Path(binary_holdout_output["output_dir"])
        mani = json.loads(
            (out / "benchmark_manifest_viruses.json").read_text())
        entries = {e["taxid"]: e for e in mani["holdout"]}
        assert "11234" in entries
        e = entries["11234"]
        # a read from held-out MHV should back off to Coronaviridae (11118)
        assert e["expected_commit_taxid"] == "11118"
        assert e["n_genomes"] >= 1
        assert e["member_headers"]  # NC_001846 recorded before pruning

    def test_manifest_params_echoed(self, binary_holdout_output):
        out = Path(binary_holdout_output["output_dir"])
        mani = json.loads(
            (out / "benchmark_manifest_viruses.json").read_text())
        assert mani["params"]["holdout_clades"] == ["11234"]
        assert mani["n_holdout_clades"] == 1

    def test_eval_set_built_from_the_holdout_manifest(
        self, binary_holdout_output, synthetic_env, tmp_path_factory
    ):
        # P1 -> P2: the manifest's held-out clade becomes labeled novel reads.
        import pyarrow.parquet as pq
        from taxotreeset.benchmark.eval_set import build_eval_set
        from taxotreeset.io.registry import NCBIRegistry

        reg = NCBIRegistry(registry_path=synthetic_env["registry_path"])
        manifest = Path(binary_holdout_output["output_dir"]) \
            / "benchmark_manifest_viruses.json"
        eval_path = str(tmp_path_factory.mktemp("evalset") / "eval.parquet")
        n_reads, n_clades = build_eval_set(
            str(manifest), reg.registry["accessions"], reg.registry["lineages"],
            eval_path, read_length=150, reads_per_genome=5, seed=0)
        assert n_clades == 1 and n_reads > 0
        rows = pq.read_table(eval_path).to_pylist()
        assert all(r["held_out_taxid"] == "11234" for r in rows)
        # a read from held-out MHV should back off to Coronaviridae (11118)
        assert all(r["expected_commit_taxid"] == "11118" for r in rows)
        assert all(len(r["seq"]) == 150 for r in rows)

    def test_full_loop_holdout_to_eval_to_score(
        self, binary_holdout_output, synthetic_env, tmp_path_factory
    ):
        # P1 -> P2 -> P4: build the eval set, then score two synthetic classifiers.
        import pyarrow.parquet as pq
        from taxotreeset.benchmark.eval_set import build_eval_set
        from taxotreeset.benchmark.scorer import score_reads
        from taxotreeset.io.registry import NCBIRegistry

        reg = NCBIRegistry(registry_path=synthetic_env["registry_path"])
        manifest = Path(binary_holdout_output["output_dir"]) \
            / "benchmark_manifest_viruses.json"
        eval_path = str(tmp_path_factory.mktemp("evalscore") / "eval.parquet")
        build_eval_set(
            str(manifest), reg.registry["accessions"], reg.registry["lineages"],
            eval_path, read_length=150, reads_per_genome=5, seed=0)
        rows = pq.read_table(eval_path).to_pylist()
        assert rows

        # a classifier that always backs off to rho* (11118) is perfect
        perfect = {r["read_id"]: ("11118", "family") for r in rows}
        assert score_reads(rows, perfect)["overall"]["correct_rate"] == 1.0
        # one that commits to a deeper wrong taxon over-commits every read
        bad = {r["read_id"]: ("99999", "genus") for r in rows}
        assert score_reads(rows, bad)["overall"]["over_commit_rate"] == 1.0


class TestMultiSingleLevelTarget:
    def test_only_the_interior_target_head_is_scheduled(
        self, multi_single_level_output
    ):
        # Reaches an interior node (11118) directly and stops: the root (10239)
        # head above it and the species heads below it are NOT emitted.
        out = Path(multi_single_level_output["output_dir"])
        manifest = json.loads((out / "manifest_viruses.json").read_text())
        assert set(manifest) == {"11118"}, (
            f"expected only the 11118 head, got {set(manifest)}")

    def test_target_head_is_a_real_multiclass_head(self, multi_single_level_output):
        # The head has its own classes (the two Coronaviridae species) + whatever
        # bucketing added — i.e. a genuine decision point, not a degenerate stub.
        out = Path(multi_single_level_output["output_dir"])
        manifest = json.loads((out / "manifest_viruses.json").read_text())
        assert len(manifest["11118"]["labels"]) >= 2, (
            f"11118 head is not multi-class: {manifest['11118']['labels']}")
