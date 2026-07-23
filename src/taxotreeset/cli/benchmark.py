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
from taxotreeset.benchmark.eval_set import ErrorModel, build_eval_set
from taxotreeset.benchmark.reliability import annotate_reliability
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
        "--track", choices=["short", "long"], default="short",
        help="short = fixed-length Illumina-like reads; long = variable-length "
        "ONT/PacBio-like reads with an indel/homopolymer error model.",
    )
    be.add_argument(
        "--read-length", type=int, default=150,
        help="Fixed read length in bp for the short track (default 150).",
    )
    be.add_argument(
        "--min-read-length", type=int, default=3000,
        help="Min read length for the long track (default 3000).",
    )
    be.add_argument(
        "--max-read-length", type=int, default=30000,
        help="Max read length for the long track (default 30000).",
    )
    be.add_argument("--sub-rate", type=float, default=0.01,
                    help="Long track: substitution rate per base (default 0.01).")
    be.add_argument("--ins-rate", type=float, default=0.02,
                    help="Long track: insertion rate per base (default 0.02).")
    be.add_argument("--del-rate", type=float, default=0.02,
                    help="Long track: deletion rate per base (default 0.02).")
    be.add_argument(
        "--homopolymer-factor", type=float, default=2.0,
        help="Long track: indel-rate multiplier inside homopolymer runs (2.0).",
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

    rl = subs.add_parser(
        "reliability",
        help="Annotate each head's reliability from its a-priori data properties "
        "and (optionally) its training behaviour.",
    )
    rl.add_argument(
        "--heads", required=True,
        help="Dataset directory whose head label_map.json files to annotate.",
    )
    rl.add_argument(
        "--training-metrics", default=None,
        help="Optional JSON {taxid: {test_f1, val_f1s: [...], learned}} — the "
        "a-posteriori signal that determines the verdict.",
    )
    rl.add_argument(
        "--write", action="store_true",
        help="Write the merged reliability block back into each label_map.json.",
    )
    rl.add_argument(
        "--summary", default=None, help="Write a per-head summary CSV here.",
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
    elif args.benchmark_cmd == "reliability":
        _run_reliability(args)
    else:  # pragma: no cover - argparse enforces a valid subcommand
        raise SystemExit(f"unknown benchmark command: {args.benchmark_cmd!r}")


def _run_build_eval(args: argparse.Namespace) -> None:
    from taxotreeset.io.registry import NCBIRegistry

    if args.reads_per_genome <= 0:
        logger.error("--reads-per-genome must be positive.")
        raise SystemExit(1)
    if args.track == "long":
        min_len, max_len = args.min_read_length, args.max_read_length
        error_model = ErrorModel(
            sub_rate=args.sub_rate, ins_rate=args.ins_rate,
            del_rate=args.del_rate, homopolymer_factor=args.homopolymer_factor,
        )
    else:
        min_len = max_len = args.read_length
        error_model = None
    if min_len <= 0 or max_len < min_len:
        logger.error("read length range must be positive with max >= min.")
        raise SystemExit(1)

    registry = NCBIRegistry(registry_path=args.registry)
    accessions = registry.registry.get("accessions", {})
    lineages = registry.registry.get("lineages", {})

    logger.info(
        "Building open-set eval reads (%s track) from %s ...",
        args.track, args.manifest,
    )
    n_reads, n_clades = build_eval_set(
        args.manifest, accessions, lineages, args.output,
        min_len=min_len, max_len=max_len, error_model=error_model,
        track=args.track, reads_per_genome=args.reads_per_genome, seed=args.seed,
    )
    logger.info(
        "Wrote %s %s-track eval reads from %s held-out clade(s) -> %s",
        f"{n_reads:,}", args.track, f"{n_clades:,}", args.output,
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


def _run_reliability(args: argparse.Namespace) -> None:
    import glob
    import os

    training: dict = {}
    if args.training_metrics:
        with open(args.training_metrics, encoding="utf-8") as f:
            training = json.load(f)

    summary: list[dict] = []
    verdict_counts: dict[str, int] = {}
    for path in sorted(glob.glob(
        os.path.join(args.heads, "**", "label_map.json"), recursive=True
    )):
        with open(path, encoding="utf-8") as f:
            label_map = json.load(f)
        taxid = str(label_map.get("head_taxid", ""))
        merged = annotate_reliability(
            label_map.get("reliability"), training.get(taxid))
        verdict_counts[merged["verdict"]] = verdict_counts.get(merged["verdict"], 0) + 1
        if args.write:
            label_map["reliability"] = merged
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(label_map, f, indent=2)
            os.replace(tmp, path)
        summary.append({
            "taxid": taxid,
            "belongs_genomes": (merged.get("belongs_genomes")),
            "a_priori_flag": merged.get("a_priori_flag"),
            "verdict": merged["verdict"],
            "verdict_source": merged["verdict_source"],
            "val_f1_std": (merged.get("posterior") or {}).get("val_f1_std"),
            "val_test_gap": (merged.get("posterior") or {}).get("val_test_gap"),
        })

    logger.info(
        "Annotated %s heads%s. Verdicts: %s",
        f"{len(summary):,}", " (written back)" if args.write else "",
        ", ".join(f"{k}={v}" for k, v in sorted(verdict_counts.items())),
    )
    if args.summary and summary:
        with open(args.summary, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
        logger.info("Wrote reliability summary -> %s", args.summary)
