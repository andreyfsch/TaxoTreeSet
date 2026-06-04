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
from taxotreeset.io.registry import NCBIRegistry


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

    try:
        registry = NCBIRegistry(
            registry_path=args.registry,
            config_path=args.mapping,
        )
        orchestrator = DiscoveryOrchestrator(
            registry=registry,
            mapping_config=mapping_config,
        )
        logger.info("Starting taxonomic scan for TaxID: %s", args.taxon_id)
        orchestrator.discover_from_root(args.taxon_id)
        logger.info("Discovery process finished successfully.")

        print("\n" + "=" * 50)
        print("   Taxonomic Mapping Complete")
        print(f"   Root Processed     : {args.taxon_id}")
        print(f"   Registry Updated   : {args.registry}")
        print("=" * 50 + "\n")
    except Exception as exc:
        logger.error(
            "Critical failure during discovery: %s", exc, exc_info=True
        )
        sys.exit(1)
