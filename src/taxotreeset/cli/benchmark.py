"""``taxotreeset benchmark`` — open-set benchmark subcommands.

Currently exposes ``build-eval``: turn a clade-holdout manifest (from
``generate --holdout-*``) into a labeled set of novel reads for open-set scoring.
Later phases (long-noisy track, scorer, k-mer baselines) extend this subcommand.
"""

import argparse
import csv
import json
import logging

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


def run(args: argparse.Namespace) -> None:
    if args.benchmark_cmd == "build-eval":
        _run_build_eval(args)
    elif args.benchmark_cmd == "score":
        _run_score(args)
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
