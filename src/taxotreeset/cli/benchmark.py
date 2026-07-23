"""``taxotreeset benchmark`` — open-set benchmark subcommands.

Currently exposes ``build-eval``: turn a clade-holdout manifest (from
``generate --holdout-*``) into a labeled set of novel reads for open-set scoring.
Later phases (long-noisy track, scorer, k-mer baselines) extend this subcommand.
"""

import argparse
import csv
import json
import logging

from taxotreeset.benchmark.baselines import (
    export_retained_reference,
    parse_kraken2_output,
    taxid_rank_map,
)
from taxotreeset.benchmark.eval_set import build_eval_set
from taxotreeset.benchmark.scorer import report_csv_rows, score_reads

logger = logging.getLogger("TaxoTreeSet.Benchmark.CLI")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    subs = parser.add_subparsers(dest="benchmark_cmd", metavar="<command>")
    subs.required = True

    be = subs.add_parser(
        "build-eval",
        help="Build the open-set eval read set from a clade-holdout manifest.",
    )
    be.add_argument(
        "--manifest", "-m", required=True,
        help="benchmark_manifest_<scope>.json produced by generate --holdout-*.",
    )
    be.add_argument(
        "--registry", "-r", required=True,
        help="Registry (frozen snapshot) used for the holdout run.",
    )
    be.add_argument(
        "--output", "-o", required=True,
        help="Destination parquet for the labeled eval reads.",
    )
    be.add_argument(
        "--read-length", type=int, default=150,
        help="Fixed read length in bp (short/Illumina-like track; default 150).",
    )
    be.add_argument(
        "--reads-per-genome", type=int, default=200,
        help="Reads sampled per held-out genome (default 200).",
    )
    be.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0).")

    sc = subs.add_parser(
        "score",
        help="Score a classifier's predictions on the open-set eval read set.",
    )
    sc.add_argument(
        "--eval-set", "-e", required=True,
        help="Eval-read parquet from `benchmark build-eval`.",
    )
    sc.add_argument(
        "--predictions", "-p", required=True,
        help="Per-read predictions: parquet or (t)sv with columns "
        "read_id, predicted_taxid, predicted_rank (empty taxid = abstain).",
    )
    sc.add_argument(
        "--output", "-o", required=True, help="Destination JSON report.",
    )
    sc.add_argument(
        "--csv", default=None,
        help="Also write a flattened per-group CSV to this path.",
    )

    ex = subs.add_parser(
        "export-refs",
        help="Export the retained-only reference FASTA for a k-mer baseline "
        "index (held-out clades excluded).",
    )
    ex.add_argument("--manifest", "-m", required=True, help="Holdout manifest.")
    ex.add_argument("--registry", "-r", required=True, help="Registry snapshot.")
    ex.add_argument(
        "--out-fasta", required=True,
        help="Destination FASTA (taxid-labeled: >seq|kraken:taxid|<taxid>).",
    )
    ex.add_argument(
        "--out-map", required=True, help="Destination seqid->taxid TSV map.",
    )

    pb = subs.add_parser(
        "parse-baseline",
        help="Convert a k-mer tool's per-read output into scorer predictions.",
    )
    pb.add_argument(
        "--tool", choices=["kraken2"], default="kraken2",
        help="Baseline tool whose output format to parse (default kraken2).",
    )
    pb.add_argument("--input", "-i", required=True, help="Tool per-read output.")
    pb.add_argument("--registry", "-r", required=True, help="Registry snapshot.")
    pb.add_argument(
        "--output", "-o", required=True,
        help="Destination predictions parquet for `benchmark score`.",
    )


def run(args: argparse.Namespace) -> None:
    if args.benchmark_cmd == "build-eval":
        _run_build_eval(args)
    elif args.benchmark_cmd == "score":
        _run_score(args)
    elif args.benchmark_cmd == "export-refs":
        _run_export_refs(args)
    elif args.benchmark_cmd == "parse-baseline":
        _run_parse_baseline(args)
    else:  # pragma: no cover - argparse enforces a valid subcommand
        raise SystemExit(f"unknown benchmark command: {args.benchmark_cmd!r}")


def _run_build_eval(args: argparse.Namespace) -> None:
    from taxotreeset.io.registry import NCBIRegistry

    if args.read_length <= 0 or args.reads_per_genome <= 0:
        logger.error("--read-length and --reads-per-genome must be positive.")
        raise SystemExit(1)

    registry = NCBIRegistry(registry_path=args.registry)
    accessions = registry.registry.get("accessions", {})
    lineages = registry.registry.get("lineages", {})

    logger.info("Building open-set eval reads from %s ...", args.manifest)
    n_reads, n_clades = build_eval_set(
        args.manifest, accessions, lineages, args.output,
        read_length=args.read_length,
        reads_per_genome=args.reads_per_genome,
        seed=args.seed,
    )
    logger.info(
        "Wrote %s eval reads from %s held-out clade(s) -> %s",
        f"{n_reads:,}", f"{n_clades:,}", args.output,
    )


def _load_predictions(path: str) -> dict[str, tuple[str | None, str | None]]:
    """Load per-read predictions (parquet or delimited) into a dict."""
    preds: dict[str, tuple[str | None, str | None]] = {}
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        for row in pq.read_table(path).to_pylist():
            preds[str(row["read_id"])] = (
                row.get("predicted_taxid") or None,
                row.get("predicted_rank") or None,
            )
        return preds
    delimiter = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            preds[str(row["read_id"])] = (
                (row.get("predicted_taxid") or "").strip() or None,
                (row.get("predicted_rank") or "").strip() or None,
            )
    return preds


def _run_score(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    eval_rows = pq.read_table(args.eval_set).to_pylist()
    predictions = _load_predictions(args.predictions)
    logger.info(
        "Scoring %s eval reads against %s predictions ...",
        f"{len(eval_rows):,}", f"{len(predictions):,}",
    )
    report = score_reads(eval_rows, predictions)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    ov = report["overall"]
    logger.info(
        "Open-set score: correct-back-off %.1f%%  over-commit %.1f%%  "
        "abstain %.1f%%  (n=%s) -> %s",
        100 * ov["correct_rate"], 100 * ov["over_commit_rate"],
        100 * ov["abstain_rate"], f"{ov['n']:,}", args.output,
    )
    if args.csv:
        rows = report_csv_rows(report)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote per-group CSV -> %s", args.csv)


def _run_export_refs(args: argparse.Namespace) -> None:
    from taxotreeset.io.registry import NCBIRegistry

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    held_out = {str(e["taxid"]) for e in manifest.get("holdout", [])}
    registry = NCBIRegistry(registry_path=args.registry)
    n = export_retained_reference(
        held_out,
        registry.registry.get("accessions", {}),
        registry.registry.get("lineages", {}),
        args.out_fasta,
        args.out_map,
    )
    logger.info(
        "Exported %s retained genomes (%s held-out clade(s) excluded) -> %s",
        f"{n:,}", f"{len(held_out):,}", args.out_fasta,
    )


def _run_parse_baseline(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from taxotreeset.io.registry import NCBIRegistry

    registry = NCBIRegistry(registry_path=args.registry)
    taxid_rank = taxid_rank_map(registry.registry.get("lineages", {}))
    with open(args.input, encoding="utf-8") as f:
        predictions = parse_kraken2_output(f, taxid_rank)
    rows = [
        {"read_id": rid, "predicted_taxid": taxid or "", "predicted_rank": rank or ""}
        for rid, (taxid, rank) in predictions.items()
    ]
    pq.write_table(pa.Table.from_pylist(rows), args.output)
    n_classified = sum(1 for t, _ in predictions.values() if t)
    logger.info(
        "Parsed %s reads (%s classified, %s abstained) -> %s",
        f"{len(rows):,}", f"{n_classified:,}",
        f"{len(rows) - n_classified:,}", args.output,
    )
