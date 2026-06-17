"""The ``separability`` subcommand: k-mer separability diagnostic.

Runs a k-mer + logistic-regression baseline on every head of a generated
dataset, writes the macro-F1 into each ``label_map.json`` under
``kmer_separability``, and optionally exports an aggregate CSV. This is a
post-generation diagnostic and requires the optional ``diagnose`` extra
(scikit-learn).
"""
import argparse
import csv
import logging
import sys

from taxotreeset.dataset import separability
from taxotreeset.logging_utils import get_ui_logger, setup_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the separability-specific arguments on a subparser.

    Args:
        parser: The subparser to populate.
    """
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Root directory of a generated dataset (the tree containing "
        "per-head label_map.json and parquet splits).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=separability.DEFAULT_K,
        help="k-mer length used as the feature space (4**k features).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=separability.DEFAULT_MAX_TRAIN,
        help="Class-balanced cap on training rows per head, for speed.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=separability.DEFAULT_MAX_TEST,
        help="Cap on test rows per head, for speed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for subsampling and the classifier.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write the aggregate results as CSV.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not modify label_map.json files; only report results.",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the separability survey from parsed arguments.

    Args:
        args: Parsed CLI arguments for the separability subcommand.
    """
    setup_logging("separability.log", level=getattr(logging, args.log_level))
    ui = get_ui_logger()

    ui.info("Running k-mer separability survey on %s (k=%d)",
            args.dataset_dir, args.k)
    try:
        rows = separability.survey_dataset(
            args.dataset_dir,
            k=args.k,
            max_train=args.max_train,
            max_test=args.max_test,
            seed=args.seed,
            write=not args.no_write,
        )
    except ImportError as exc:
        ui.error("%s", exc)
        sys.exit(1)

    if not rows:
        ui.warning("No heads found under %s", args.dataset_dir)
        sys.exit(1)

    if args.csv:
        fields = list(rows[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        ui.info("Aggregate CSV written to %s", args.csv)

    scored = [r for r in rows if r.get("test_f1_macro") is not None]
    if scored:
        mean_f1 = sum(r["test_f1_macro"] for r in scored) / len(scored)
        ui.info("Scored %d heads | mean test macro-F1 = %.3f%s",
                len(scored), mean_f1,
                "" if not args.no_write else " (label_map.json not modified)")
