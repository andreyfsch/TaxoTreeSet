"""Sidecar-file writers for the generation orchestrator.

Extracted from ``generation_orchestrator.py``: per-head ``label_map.json``, the
run-metadata + reproducible accession snapshot, and the manifest / passthrough /
virtual-id registry artifacts. ``_write_label_maps`` is pure given the scheduling
artifacts; the run-metadata / artifact writers read orchestrator config through a
``ctx`` handle (the orchestrator instance). The logger name is shared with the
orchestrator so log output is unchanged.
"""

import dataclasses
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator

logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")


def _capture_tool_versions() -> dict[str, str]:
    """Capture versions of the external tools that determine the data snapshot.

    Records the NCBI ``datasets`` CLI, the taxoniq taxonomy package, and the
    Python runtime, so a generated dataset can be reproduced against the same
    tooling. Each lookup degrades to ``"unknown"`` on failure (e.g. the CLI is
    absent on a ``--no-sync`` run).

    Returns:
        Mapping of tool name to version string.
    """
    import platform
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        completed = subprocess.run(
            ["datasets", "--version"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        datasets_cli = (completed.stdout or completed.stderr).strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        datasets_cli = "unknown"

    try:
        taxoniq_version = _pkg_version("taxoniq")
    except PackageNotFoundError:
        taxoniq_version = "unknown"

    return {
        "datasets_cli": datasets_cli,
        "taxoniq": taxoniq_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _write_label_maps(scheduling_artifacts: dict[str, Any]) -> None:
    """Write label_map.json into every head's output directory.

    Each head directory becomes self-contained: the parquet files
    carry integer class indices, and label_map.json provides the
    id2label / label2id mappings in the HuggingFace-standard format
    so fine-tuning scripts can load the head without touching the
    root-level manifest.

    Args:
        scheduling_artifacts: Output of _schedule_pipeline_jobs.
    """
    master_manifest = scheduling_artifacts.get("master_manifest", {})
    n_written = 0
    for taxid, v in master_manifest.items():
        head_dir = v.get("directory_path", "")
        if not head_dir:
            continue
        labels = v.get("labels", {})
        classes = sorted(
            [
                {
                    "class_idx": lv.get("class_idx", -1),
                    "taxid": label_taxid,
                    "name": lv.get("name", label_taxid),
                    "rank": lv.get("rank", "unknown"),
                    "n_windows": lv.get("n_windows", 0),
                }
                for label_taxid, lv in labels.items()
            ],
            key=lambda x: x["class_idx"],
        )
        # Disambiguate duplicate class names so id2label / label2id stay
        # bijective — two distinct taxa can carry the same NCBI scientific
        # name (homonyms) or sanitize to the same string, and a bare
        # ``{name: idx}`` dict would silently drop the earlier class.
        seen_names: set[str] = set()
        for c in classes:
            if c["name"] in seen_names:
                c["name"] = f"{c['name']} ({c['taxid']})"
            seen_names.add(c["name"])
        id2label = {str(c["class_idx"]): c["name"] for c in classes}
        label2id = {c["name"]: c["class_idx"] for c in classes}
        balance_mode = v.get("balance_mode", "undersample")
        label_map = {
            "head_taxid": taxid,
            "head_name": v.get("scientific_name", taxid),
            "head_rank": v.get("rank", "unknown"),
            "balance_mode": balance_mode,
            "id2label": id2label,
            "label2id": label2id,
            "classes": classes,
        }
        if balance_mode == "keep":
            # "balanced" class weights (total / (n_classes * n_c)) so a trainer
            # can offset the on-disk imbalance in its loss (or drive oversampling).
            total = sum(c["n_windows"] for c in classes)
            k = len(classes)
            if total and k:
                label_map["class_weights"] = {
                    str(c["class_idx"]): round(total / (k * c["n_windows"]), 4)
                    for c in classes
                    if c["n_windows"]
                }
        os.makedirs(head_dir, exist_ok=True)
        label_map_path = os.path.join(head_dir, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)
        n_written += 1
    logger.info("Label maps written for %d heads.", n_written)


def _write_run_metadata(
    ctx: "GenerationOrchestrator",
    target_group: str,
    scheduling_artifacts: dict[str, Any],
    n_taxa: int,
    n_cap: int,
    abundance_threshold: int,
    t_pipeline_start: float,
) -> None:
    """Write run_metadata_{target_group}.json to the output directory.

    The file captures every parameter needed to reproduce the run,
    a summary of what was generated, and a per-head breakdown with
    class lists. Downstream tools (training scripts, inference
    cascade, evaluation harness) can read this file instead of
    reconstructing parameters from CLI history.

    Args:
        ctx: The orchestrator, read for config and the registry.
        target_group: Root group name used in the filename.
        scheduling_artifacts: Output of _schedule_pipeline_jobs.
        n_taxa: Total descendant taxa in the taxonomic tree.
        n_cap: Number of nodes with a computed capacity.
        abundance_threshold: Minimum sequence count per class.
        t_pipeline_start: monotonic timestamp from pipeline start.
    """
    import datetime

    try:
        from importlib.metadata import version as _pkg_version
        pkg_version = _pkg_version("taxotreeset")
    except Exception:
        pkg_version = "unknown"

    tools = _capture_tool_versions()
    snapshot = ctx.registry.accession_snapshot()

    master_manifest = scheduling_artifacts.get("master_manifest", {})
    # One manifest entry per head, for both the multi-class and the
    # (batch-extracted, so extraction_jobs is empty) binary paths.
    n_heads = len(master_manifest)
    n_classes_total = sum(len(v.get("labels", {})) for v in master_manifest.values())

    heads = []
    for taxid, v in master_manifest.items():
        labels = v.get("labels", {})
        heads.append({
            "taxid": taxid,
            "name": v.get("scientific_name", taxid),
            "rank": v.get("rank", "unknown"),
            "directory": v.get("directory_path", ""),
            "scenario": v.get("scenario", ""),
            "n_per_class": v.get("n_per_class", 0),
            "n_leaves": v.get("num_leaves", 0),
            "n_classes": len(labels),
            "classes": [
                {
                    "taxid": label_taxid,
                    "name": lv.get("name", label_taxid),
                    "rank": lv.get("rank", "unknown"),
                    "class_idx": lv.get("class_idx", -1),
                }
                for label_taxid, lv in labels.items()
            ],
        })

    # Aggregate per-rank statistics for quick dataset inspection.
    import statistics as _stats
    from collections import defaultdict as _dd
    _by_rank: dict[str, list] = _dd(list)
    _caps = ctx._all_capacities or {}
    for head in heads:
        _by_rank[head["rank"]].append({
            "npc": head["n_per_class"],
            "nc": head["n_classes"],
            "cap": _caps.get(head["taxid"], 0),
        })
    by_rank = {}
    for rank, items in sorted(_by_rank.items()):
        npc_vals = [i["npc"] for i in items]
        cap_vals = [i["cap"] for i in items if i["cap"] > 0]
        by_rank[rank] = {
            "n_heads": len(items),
            "total_classes": sum(i["nc"] for i in items),
            "median_n_per_class": _stats.median(npc_vals) if npc_vals else 0,
            "median_capacity": int(_stats.median(cap_vals)) if cap_vals else 0,
        }

    metadata = {
        "taxotreeset_version": pkg_version,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - t_pipeline_start, 1),
        "provenance": {
            "tools": tools,
            "registry_last_update": ctx.registry.registry.get("last_update"),
            "accession_snapshot": {
                "file": f"accession_snapshot_{target_group}.json",
                "sha256": snapshot["sha256"],
                "n_accessions": snapshot["n_accessions"],
            },
        },
        "parameters": {
            "root": target_group,
            "min_subseq_len": ctx.min_subseq_len,
            "max_subseq_len": ctx.max_subseq_len,
            "max_n_per_class": ctx.max_n_per_class,
            "min_leaves_per_class": ctx.min_leaves_per_class,
            "min_num_seqs": ctx.min_num_seqs,
            "cutoff_percentage": ctx.cutoff_percentage,
            "rare_taxa_strategy": ctx.rare_taxa_strategy,
            "seed": ctx.seed,
            "abundance_threshold": abundance_threshold,
            "stop_at": ctx._depth_boundary,
            "single_level": ctx._single_level,
            "single_level_taxid": ctx._single_level_taxid,
            "output_format": ctx.output_format,
            "reject_class": ctx.reject_class,
            "reject_fraction": ctx.reject_fraction,
            "reject_near_far_start": ctx.reject_near_far_start,
            "reject_near_far_end": ctx.reject_near_far_end,
            "binary_only": ctx.binary_only,
            "binary_budget": ctx.binary_budget,
            "all_ranks": ctx.all_ranks,
            "cluster_aware_split": ctx.cluster_aware_split,
            "cluster_params": (
                dataclasses.asdict(ctx.cluster_params)
                if ctx.cluster_aware_split else None
            ),
        },
        "summary": {
            "n_heads": n_heads,
            "n_classes_total": n_classes_total,
            "n_taxa_in_tree": n_taxa,
            "n_nodes_with_capacity": n_cap,
            "n_accessions": len(ctx.registry.registry.get("accessions", {})),
            "by_rank": by_rank,
        },
        "heads": heads,
    }

    os.makedirs(ctx.output_dir, exist_ok=True)
    metadata_path = os.path.join(
        ctx.output_dir, f"run_metadata_{target_group}.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata written to %s", metadata_path)

    # Full reproducible accession snapshot (versioned accessions + digest),
    # written alongside the metadata so the run can be re-fetched and cited.
    snapshot_path = os.path.join(
        ctx.output_dir, f"accession_snapshot_{target_group}.json"
    )
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": metadata["generated_at"],
                "tools": tools,
                "n_accessions": snapshot["n_accessions"],
                "sha256": snapshot["sha256"],
                "accessions": snapshot["accessions"],
            },
            f,
            indent=2,
        )
    logger.info("Accession snapshot written to %s", snapshot_path)


def _persist_scheduling_artifacts(
    ctx: "GenerationOrchestrator",
    target_group: str,
    scheduling_artifacts: dict[str, Any],
) -> None:
    """Write the manifest, passthrough map, and virtual ID registry.

    These sidecar files are critical metadata that allows
    downstream consumers (training scripts, inference cascade)
    to interpret the generated dataset.

    Args:
        ctx: The orchestrator, read for the output directory.
        target_group: CLI-friendly group name used in filenames.
        scheduling_artifacts: Output of _schedule_pipeline_jobs.
    """
    os.makedirs(ctx.output_dir, exist_ok=True)

    manifest_path = os.path.join(ctx.output_dir, f"manifest_{target_group}.json")
    passthroughs_path = os.path.join(
        ctx.output_dir, f"passthroughs_{target_group}.json"
    )
    virtual_registry_path = os.path.join(
        ctx.output_dir, f"virtual_id_registry_{target_group}.json"
    )

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(
            scheduling_artifacts["master_manifest"],
            manifest_file,
            indent=2,
        )

    with open(passthroughs_path, "w", encoding="utf-8") as passthrough_file:
        json.dump(
            scheduling_artifacts["passthrough_map"],
            passthrough_file,
            indent=2,
        )

    with open(virtual_registry_path, "w", encoding="utf-8") as virtual_file:
        json.dump(
            {"virtual_ids": scheduling_artifacts["virtual_id_registry"]},
            virtual_file,
            indent=2,
        )

    logger.info(
        f"Persisted manifest "
        f"({len(scheduling_artifacts['master_manifest'])} heads), "
        f"passthroughs "
        f"({len(scheduling_artifacts['passthrough_map'])}), "
        f"virtual IDs "
        f"({len(scheduling_artifacts['virtual_id_registry'])})."
    )
