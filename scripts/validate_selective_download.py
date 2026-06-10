"""Integration validation for the selective download feature.

Exercises the full selective download pipeline against a real eukaryotic
scope in NCBI RefSeq to verify that:

  1. Discovery correctly populates metadata (total_sequence_length) for
     all reference assemblies in the scope.
  2. When the aggregate pending volume exceeds the threshold, the
     selection pass runs and defers low-priority accessions.
  3. Stage 1 (download) fetches only the selected subset, not the full
     scope.
  4. The bottom-up capacity pass correctly computes real capacities for
     the downloaded subset.
  5. The refinement pass detects shortfalls (if any) and undefers
     additional accessions for a second download round.

Usage::

    # Dry-run: discovery + selection only, no download
    python scripts/validate_selective_download.py \\
        --taxon 50557 \\
        --stop-at order \\
        --output-dir ~/taxotreeset_insecta_test \\
        --dry-run

    # Full run: discovery + selection + download + capacity + refinement
    python scripts/validate_selective_download.py \\
        --taxon 50557 \\
        --stop-at order \\
        --output-dir ~/taxotreeset_insecta_test

    # Larger scope that is certain to exceed the threshold
    python scripts/validate_selective_download.py \\
        --taxon 7742 \\
        --stop-at class \\
        --max-n-per-class 5000 \\
        --output-dir ~/taxotreeset_vertebrata_test

Taxon suggestions (NCBI TaxIDs):
    50557   Insecta       genomes 150-600 MB, probably 50-200 GB total
    7088    Lepidoptera   genomes 200-600 MB, probably 30-120 GB total
    7147    Diptera       genomes 100-300 MB, probably 10-80 GB total
    7742    Vertebrata    genomes 0.5-3 GB,   definitely > 500 GB total
    40674   Mammalia      genomes 2-3.5 GB,   definitely > 200 GB total
    8782    Aves          genomes 1-1.5 GB,   definitely > 150 GB total

Expected runtimes (rough estimates for the full run):
    Insecta + stop_at=order:  2-5 hours
    Diptera + stop_at=order:  1-3 hours
    Mammalia + stop_at=order: 4-12 hours
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

from taxotreeset.core.generation.capacity import compute_all_capacities
from taxotreeset.core.generation_orchestrator import (
    GenerationOrchestrator,
    _MAX_REFINEMENT_ROUNDS,
    _SELECTIVE_DOWNLOAD_THRESHOLD_BYTES,
)
from taxotreeset.core.orchestrator import DiscoveryOrchestrator
from taxotreeset.io.registry import NCBIRegistry
from taxotreeset.io.downloader import NCBIDownloader
from taxotreeset.logging_utils import get_ui_logger

ui_logger = get_ui_logger()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.2f} GiB"


def _fmt_count(n: int) -> str:
    return f"{n:,}"


def _print_section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _discovery_stage(
    registry: NCBIRegistry,
    domain_taxid: str,
    mapping_path: str,
) -> None:
    _print_section(f"Stage 1 — Discovery (TaxID {domain_taxid})")
    t0 = time.monotonic()

    with open(mapping_path, encoding="utf-8") as fh:
        mapping_config = json.load(fh)

    discovery = DiscoveryOrchestrator(
        registry=registry,
        mapping_config=mapping_config,
    )
    discovery.discover_from_root(int(domain_taxid))
    registry.save()

    elapsed = time.monotonic() - t0
    n_taxons = len(registry.registry["taxons"])
    n_accessions = len(registry.registry["accessions"])
    print(f"  Taxons discovered : {_fmt_count(n_taxons)}")
    print(f"  Accessions found  : {_fmt_count(n_accessions)}")
    print(f"  Elapsed           : {elapsed:.1f}s")


def _volume_check(
    registry: NCBIRegistry,
    domain_taxid: str,
    threshold: int,
) -> tuple[int, bool]:
    volume = registry.get_pending_volume(domain_taxid)
    exceeds = volume >= threshold
    _print_section("Volume Check")
    print(f"  Total pending volume : {_gib(volume)}")
    print(f"  Threshold            : {_gib(threshold)}")
    print(f"  Selective download   : {'ACTIVATED' if exceeds else 'not needed'}")
    return volume, exceeds


def _selection_stage(
    orch: GenerationOrchestrator,
    registry: NCBIRegistry,
    domain_taxid: str,
) -> dict[str, int]:
    _print_section("Stage 2 — Selective Download Selection")
    t0 = time.monotonic()

    orch._run_selective_download(domain_taxid)

    accessions = registry.registry["accessions"]
    n_total = len(accessions)
    n_selected = sum(
        1 for info in accessions.values()
        if not info.get("downloaded") and not info.get("download_deferred")
    )
    n_deferred = sum(
        1 for info in accessions.values()
        if info.get("download_deferred")
    )
    n_already = sum(
        1 for info in accessions.values()
        if info.get("downloaded")
    )

    selected_vol = sum(
        (info.get("total_sequence_length") or 0)
        for info in accessions.values()
        if not info.get("downloaded") and not info.get("download_deferred")
    )
    deferred_vol = sum(
        (info.get("total_sequence_length") or 0)
        for info in accessions.values()
        if info.get("download_deferred")
    )

    elapsed = time.monotonic() - t0
    print(f"  Total accessions   : {_fmt_count(n_total)}")
    print(f"  Already downloaded : {_fmt_count(n_already)}")
    print(f"  Selected for DL    : {_fmt_count(n_selected)}  ({_gib(selected_vol)})")
    print(f"  Deferred           : {_fmt_count(n_deferred)}  ({_gib(deferred_vol)})")
    print(f"  Elapsed            : {elapsed:.1f}s")

    return {
        "n_total": n_total,
        "n_selected": n_selected,
        "n_deferred": n_deferred,
        "selected_vol_bytes": selected_vol,
        "deferred_vol_bytes": deferred_vol,
    }


def _download_stage(downloader: NCBIDownloader) -> None:
    _print_section("Stage 3 — Download (selected subset only)")
    t0 = time.monotonic()
    downloader.download_all_pending()
    elapsed = time.monotonic() - t0
    print(f"  Elapsed : {elapsed:.1f}s  ({elapsed/60:.1f} min)")


def _capacity_stage(
    orch: GenerationOrchestrator,
    domain_taxid: str,
) -> dict[str, int]:
    _print_section("Stage 4 — Tree Build + Capacity Computation")
    t0 = time.monotonic()

    tree_root = orch._build_target_tree(domain_taxid)
    if tree_root is None or not tree_root.children:
        print("  ERROR: no tree built — no downloaded accessions?")
        return {}

    all_capacities = compute_all_capacities(tree_root, orch.min_subseq_len)
    elapsed = time.monotonic() - t0

    n_nodes = len(all_capacities)
    max_cap = max(all_capacities.values(), default=0)
    min_cap = min(all_capacities.values(), default=0)

    print(f"  Non-sequence nodes : {_fmt_count(n_nodes)}")
    print(f"  Capacity range     : {_fmt_count(min_cap)} – {_fmt_count(max_cap)} unique k-mers")
    print(f"  Elapsed            : {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    orch._all_capacities = all_capacities
    orch.registry.store_capacities(all_capacities, orch.min_subseq_len)
    orch.registry.save()

    return all_capacities


def _refinement_stage(
    orch: GenerationOrchestrator,
    domain_taxid: str,
    all_capacities: dict[str, int],
) -> None:
    _print_section("Stage 5 — Refinement Check")

    tree_root = orch._build_target_tree(domain_taxid)
    if tree_root is None:
        print("  Skipped: tree unavailable.")
        return

    orch._all_capacities = all_capacities
    orch._selective_download_active = True

    for round_num in range(1, _MAX_REFINEMENT_ROUNDS + 1):
        t0 = time.monotonic()
        needed = orch._run_refinement_pass(domain_taxid, tree_root)
        elapsed = time.monotonic() - t0

        if not needed:
            print(f"  Round {round_num}: no shortfalls — refinement complete. ({elapsed:.1f}s)")
            break

        print(
            f"  Round {round_num}: shortfall detected, undeferred additional accessions "
            f"({elapsed:.1f}s) — would trigger another download round."
        )

        accessions = orch.registry.registry["accessions"]
        n_newly_pending = sum(
            1 for info in accessions.values()
            if not info.get("downloaded") and not info.get("download_deferred")
        )
        print(f"           Accessions now pending for download: {_fmt_count(n_newly_pending)}")
    else:
        print(
            f"  Reached maximum refinement rounds ({_MAX_REFINEMENT_ROUNDS}); "
            "some labels may remain below target."
        )


def _save_report(
    report: dict,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "selective_download_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Report saved to: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--taxon",
        default="50557",
        help="NCBI TaxID of the eukaryotic scope. Default: 50557 (Insecta).",
    )
    parser.add_argument(
        "--stop-at",
        default="order",
        dest="stop_at",
        help="Canonical rank at which cascade stops. Default: order.",
    )
    parser.add_argument(
        "--max-n-per-class",
        type=int,
        default=10_000,
        dest="max_n_per_class",
        help="Hard cap on subsequences per training label. Default: 10000.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=_SELECTIVE_DOWNLOAD_THRESHOLD_BYTES,
        help=(
            "Selective download activation threshold in bytes. "
            f"Default: {_SELECTIVE_DOWNLOAD_THRESHOLD_BYTES} "
            f"({_gib(_SELECTIVE_DOWNLOAD_THRESHOLD_BYTES)})."
        ),
    )
    parser.add_argument(
        "--min-subseq-len",
        type=int,
        default=100,
        dest="min_subseq_len",
        help="Sliding-window size for capacity measurement. Default: 100.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.expanduser("~"), "taxotreeset_selective_download_test"),
        dest="output_dir",
        help="Directory for the test registry, vault, and report.",
    )
    parser.add_argument(
        "--mapping",
        default="configs/mapping.json",
        help="Path to configs/mapping.json. Default: configs/mapping.json.",
    )
    parser.add_argument(
        "--spill-dir",
        default=None,
        dest="spill_dir",
        help=(
            "Directory for temporary spill files created during the bottom-up "
            "capacity pass. Defaults to the OS temp directory (usually on the "
            "system drive). Set to a path on a large external drive to avoid "
            "filling the system disk on large scopes."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "Run discovery and selection only. Skip download, capacity "
            "computation, and refinement. Use this to validate the selection "
            "logic before committing to a multi-GiB download."
        ),
    )
    args = parser.parse_args()

    domain_taxid = str(args.taxon)
    registry_path = os.path.join(args.output_dir, "registry.json")
    vault_path = os.path.join(args.output_dir, "vault")
    datasets_path = os.path.join(args.output_dir, "datasets")

    print("\nTaxoTreeSet — Selective Download Validation")
    print(f"  Scope TaxID      : {domain_taxid}")
    print(f"  Stop at rank     : {args.stop_at}")
    print(f"  max_n_per_class  : {_fmt_count(args.max_n_per_class)}")
    print(f"  Threshold        : {_gib(args.threshold)}")
    print(f"  min_subseq_len   : {args.min_subseq_len}")
    print(f"  Output directory : {args.output_dir}")
    print(f"  Spill directory  : {args.spill_dir or '(OS default /tmp)'}")
    mode = "DRY-RUN (no download)" if args.dry_run else "FULL (with download)"
    print(f"  Mode             : {mode}")
    print(f"  Started at       : {datetime.now().isoformat(timespec='seconds')}")

    os.makedirs(args.output_dir, exist_ok=True)

    registry = NCBIRegistry(
        registry_path=registry_path,
        config_path=args.mapping,
    )

    orch = GenerationOrchestrator(
        registry=registry,
        vault_path=vault_path,
        output_dir=datasets_path,
        config_path=args.mapping,
        min_subseq_len=args.min_subseq_len,
        max_n_per_class=args.max_n_per_class,
        selective_download_threshold=args.threshold,
        spill_dir=args.spill_dir,
    )
    orch._depth_boundary = args.stop_at
    orch._single_level = False
    orch._selective_download_active = False

    report: dict = {
        "taxon": domain_taxid,
        "stop_at": args.stop_at,
        "max_n_per_class": args.max_n_per_class,
        "threshold_bytes": args.threshold,
        "min_subseq_len": args.min_subseq_len,
        "dry_run": args.dry_run,
        "started_at": datetime.now().isoformat(),
    }

    wall_t0 = time.monotonic()

    # Stage 1: discovery
    _discovery_stage(registry, domain_taxid, args.mapping)
    report["n_accessions_discovered"] = len(registry.registry["accessions"])
    report["n_taxons_discovered"] = len(registry.registry["taxons"])

    # Volume check
    volume, exceeds = _volume_check(registry, domain_taxid, args.threshold)
    report["pending_volume_bytes"] = volume
    report["selective_download_activated"] = exceeds

    if not exceeds:
        print(
            f"\n  WARNING: pending volume ({_gib(volume)}) is below the threshold "
            f"({_gib(args.threshold)}). Selective download will not activate.\n"
            "  Try a larger taxon or lower the --threshold.\n"
        )

    # Stage 2: selection
    selection_stats = _selection_stage(orch, registry, domain_taxid)
    report.update(selection_stats)

    if args.dry_run:
        print(
            "\n  DRY-RUN mode: stopping before download.\n"
            "  Re-run without --dry-run to proceed with the full pipeline."
        )
        report["status"] = "dry_run_complete"
        _save_report(report, args.output_dir)
        return

    # Stage 3: download (selected only)
    downloader = NCBIDownloader(registry=registry, vault_path=vault_path)
    _download_stage(downloader)

    n_downloaded = sum(
        1 for info in registry.registry["accessions"].values()
        if info.get("downloaded")
    )
    downloaded_vol = sum(
        (info.get("total_sequence_length") or 0)
        for info in registry.registry["accessions"].values()
        if info.get("downloaded")
    )
    print(f"  Downloaded accessions : {_fmt_count(n_downloaded)} ({_gib(downloaded_vol)})")
    report["n_downloaded"] = n_downloaded
    report["downloaded_vol_bytes"] = downloaded_vol

    # Stage 4: tree build + capacity
    all_capacities = _capacity_stage(orch, domain_taxid)
    if not all_capacities:
        report["status"] = "failed_no_capacities"
        _save_report(report, args.output_dir)
        sys.exit(1)

    report["n_capacity_nodes"] = len(all_capacities)
    report["max_capacity"] = max(all_capacities.values(), default=0)
    report["min_capacity"] = min(all_capacities.values(), default=0)

    # Stage 5: refinement check
    _refinement_stage(orch, domain_taxid, all_capacities)

    total_elapsed = time.monotonic() - wall_t0
    print(f"\n  Total wall-clock time: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)")
    report["total_elapsed_seconds"] = total_elapsed
    report["status"] = "complete"
    _save_report(report, args.output_dir)


if __name__ == "__main__":
    main()
