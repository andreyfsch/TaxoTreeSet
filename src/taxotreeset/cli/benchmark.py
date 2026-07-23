"""``taxotreeset benchmark`` — open-set benchmark subcommands.

Currently exposes ``build-eval``: turn a clade-holdout manifest (from
``generate --holdout-*``) into a labeled set of novel reads for open-set scoring.
Later phases (long-noisy track, scorer, k-mer baselines) extend this subcommand.
"""

import argparse
import logging

from taxotreeset.benchmark.eval_set import build_eval_set

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


def run(args: argparse.Namespace) -> None:
    if args.benchmark_cmd == "build-eval":
        _run_build_eval(args)
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
