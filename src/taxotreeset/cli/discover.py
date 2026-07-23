"""The ``discover`` subcommand: taxonomic mapping and registry build.

Scans NCBI taxonomy from a root TaxID and compiles the inventory
registry that the ``generate`` subcommand later consumes.
"""
import argparse
import json
import logging
import os
import sys

from taxotreeset import paths
from taxotreeset.logging_utils import setup_logging
from taxotreeset.core.orchestrator import DiscoveryOrchestrator
from taxotreeset.io.plasmid_release import (
    fetch_release,
    ingest_records_to_vault,
    iter_release_records,
)
from taxotreeset.io.registry import NCBIRegistry

# Sub-directory of the vault where the RefSeq plasmid release is synced by default.
_PLASMID_RELEASE_SUBDIR = "refseq_plasmid"

# Scope key the plasmid host tree registers under. It has no NCBI TaxID (plasmid
# is not a taxon); a user may add a "plasmids" scope with redirections to
# mapping.json, otherwise host names pass through unmapped.
_PLASMID_SCOPE_KEY = "plasmids"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the discover-specific arguments on a subparser.

    Args:
        parser: The subparser to populate.
    """
    parser.add_argument(
        "--taxon-id", "-t",
        type=int,
        default=10239,
        help="NCBI TaxID of the biological root to start mapping from "
        "(e.g. 10239 for Viruses, 2 for Bacteria)",
    )
    parser.add_argument(
        "--mapping", "-m",
        type=str,
        default="configs/mapping.json",
        help="Path to the JSON file mapping scopes and fallback "
        "redirections",
    )
    parser.add_argument(
        "--registry", "-r",
        type=str,
        default=str(paths.default_registry_path()),
        help="Destination path for the inventory/registry file",
    )
    parser.add_argument(
        "--reset", "-f",
        action="store_true",
        help="If set, forces deletion of the old registry before "
        "starting a new discovery run",
    )
    parser.add_argument(
        "--all-ranks",
        action="store_true",
        help="Resolve lineages at FULL NCBI granularity (subgenus, subfamily, "
        "suborder, clade, ...) via taxoniq's full lineage, instead of only the "
        "8 canonical ranks. Intermediate taxa become heads where they branch "
        "(single-child sub-ranks are still collapsed by passthroughs).",
    )
    parser.add_argument(
        "--plasmids",
        action="store_true",
        help="Bottom-up plasmid discovery instead of walking --taxon-id: fetch "
        "the RefSeq plasmid release, ingest each plasmid sequence into the vault, "
        "and register it under its host organism's lineage. Requires --vault.",
    )
    parser.add_argument(
        "--vault",
        type=str,
        default=None,
        metavar="DIR",
        help="Vault directory to ingest plasmid sequences into (the LMDB is "
        "<DIR>/sequences.lmdb, matching the downloader). Required with --plasmids.",
    )
    parser.add_argument(
        "--plasmid-release",
        type=str,
        default=None,
        metavar="DIR",
        help="Where to store/read the RefSeq plasmid release files (default: "
        "<vault>/" + _PLASMID_RELEASE_SUBDIR + "). Reused across runs (the fetch "
        "is resumable and md5-verified).",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="With --plasmids, skip the download and use the release files "
        "already present in the release directory (offline / pre-fetched).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the discovery workflow from parsed arguments.

    Args:
        args: Parsed CLI arguments for the discover subcommand.
    """
    setup_logging("discovery.log", level=getattr(logging, args.log_level))
    logger = logging.getLogger("TaxoTreeSet.Discover.CLI")

    # Idempotency: handle the existing registry file.
    if args.reset and os.path.exists(args.registry):
        try:
            os.remove(args.registry)
            logger.info(
                "Flag --reset enabled. Old registry removed: %s",
                args.registry,
            )
        except OSError as exc:
            logger.error(
                "Could not remove the old registry at %s: %s",
                args.registry, exc,
            )
            sys.exit(1)
    elif os.path.exists(args.registry):
        logger.info(
            "Registry found at %s. Incremental/append mode active.",
            args.registry,
        )

    if not os.path.exists(args.mapping):
        logger.error("Mapping file missing at %s", args.mapping)
        sys.exit(1)
    try:
        with open(args.mapping, "r", encoding="utf-8") as handle:
            mapping_config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Error reading the mapping JSON file: %s", exc)
        sys.exit(1)

    if args.plasmids:
        _validate_plasmid_args(args, logger)

    try:
        registry = NCBIRegistry(
            registry_path=args.registry,
            config_path=args.mapping,
        )
        orchestrator = DiscoveryOrchestrator(
            registry=registry,
            mapping_config=mapping_config,
            all_ranks=args.all_ranks,
        )

        if args.plasmids:
            root_label = _run_plasmid_discovery(orchestrator, args, logger)
        else:
            logger.info("Starting taxonomic scan for TaxID: %s", args.taxon_id)
            orchestrator.discover_from_root(args.taxon_id)
            root_label = str(args.taxon_id)
        logger.info("Discovery process finished successfully.")

        print("\n" + "=" * 50)
        print("   Taxonomic Mapping Complete")
        print(f"   Root Processed     : {root_label}")
        print(f"   Registry Updated   : {args.registry}")
        print("=" * 50 + "\n")
    except Exception as exc:
        logger.error(
            "Critical failure during discovery: %s", exc, exc_info=True
        )
        sys.exit(1)


def _validate_plasmid_args(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Fail fast on an invalid plasmid-discovery invocation."""
    if not args.vault:
        logger.error("--plasmids requires --vault (where to ingest sequences).")
        sys.exit(1)
    if args.no_fetch and not os.path.isdir(_plasmid_release_dir(args)):
        logger.error(
            "--no-fetch set but the release directory is missing at %s",
            _plasmid_release_dir(args))
        sys.exit(1)


def _plasmid_release_dir(args: argparse.Namespace) -> str:
    """The release directory: the override, else a default under the vault."""
    return args.plasmid_release or os.path.join(
        args.vault, _PLASMID_RELEASE_SUBDIR)


def _run_plasmid_discovery(
    orchestrator: DiscoveryOrchestrator,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> str:
    """Fetch + ingest the RefSeq plasmid release, then register by host lineage.

    Returns a human-readable label for the run summary.
    """
    release_dir = _plasmid_release_dir(args)
    if not args.no_fetch:
        logger.info("Fetching the RefSeq plasmid release into %s", release_dir)
        fetch_release(release_dir)

    lmdb_path = os.path.join(args.vault, "sequences.lmdb")
    logger.info("Ingesting plasmid sequences from %s into %s", release_dir, lmdb_path)
    reports = ingest_records_to_vault(
        iter_release_records(release_dir), lmdb_path)
    orchestrator.discover_from_reports(
        reports,
        root_id_str=_PLASMID_SCOPE_KEY,
        vault_lmdb_path=lmdb_path,
    )
    return f"RefSeq plasmid release ({len(reports)} record(s))"
