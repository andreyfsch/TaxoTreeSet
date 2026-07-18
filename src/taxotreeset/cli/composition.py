"""The ``composition`` subcommand: per-head compositional-confound audit.

Reports per-class sequence length and nucleotide composition for every head of a
generated dataset and flags virtual classes whose GC content is an outlier
relative to the canonical classes, writing a ``composition_audit`` summary into
each ``label_map.json`` and optionally an aggregate CSV. This is a
post-generation diagnostic (backlog P6); numpy-only, no optional dependency.
"""
import argparse
import csv
import logging
import sys

from taxotreeset.dataset import composition
from taxotreeset.logging_utils import get_ui_logger, setup_logging


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the composition-specific arguments on a subparser.

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
        "--split",
        type=str,
        default="train",
        help="Which split to read per head (default: train).",
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
    """Execute the compositional-confound audit from parsed arguments.

    Args:
        args: Parsed CLI arguments for the composition subcommand.
    """
    setup_logging("composition.log", level=getattr(logging, args.log_level))
    ui = get_ui_logger()

    ui.info("Auditing compositional confounds on %s (split=%s)",
            args.dataset_dir, args.split)
    rows = composition.survey_dataset(
        args.dataset_dir, split=args.split, write=not args.no_write
    )

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

    n_flagged = sum(1 for r in rows if r["n_flagged_virtual"])
    ui.info("Audited %d heads | %d have a flagged virtual class%s",
            len(rows), n_flagged,
            "" if not args.no_write else " (label_map.json not modified)")
    for r in rows:
        if r["n_flagged_virtual"]:
            ui.warning("  head %s (%s): compositional-outlier virtual class(es): %s",
                       r["head_taxid"], r["head_rank"], r["flagged_virtual"])
