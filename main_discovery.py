import argparse
import json
import logging
import os
import sys
from taxotreeset.io.registry import NCBIRegistry
from taxotreeset.core.orchestrator import DiscoveryOrchestrator


def setup_logging():
    """Configure telemetry by writing to a file and printing to the terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("discovery.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger("TaxoTreeSet.CLI")

    # 1. Command-line argument parser configuration (CLI)
    parser = argparse.ArgumentParser(
        description="TaxoTreeSet - Taxonomic mapping engine and registry generation for ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--taxon-id", "-t",
        type=int,
        default=10239,
        help="NCBI TaxID of the biological root to start mapping from (e.g. 10239 for Viruses, 2 for Bacteria)"
    )
    parser.add_argument(
        "--mapping", "-m",
        type=str,
        default="configs/mapping.json",
        help="Path to the JSON file mapping scopes and fallback redirections"
    )
    parser.add_argument(
        "--registry", "-r",
        type=str,
        default="data/registry.json",
        help="Destination path for the inventory/registry file"
    )
    parser.add_argument(
        "--reset", "-f",
        action="store_true",
        help="If set, forces deletion of the old registry.json before starting a new discovery run"
    )

    args = parser.parse_args()

    # 2. Idempotency handling (managing the old registry file)
    if args.reset and os.path.exists(args.registry):
        try:
            os.remove(args.registry)
            logger.info(
                f"Flag --reset enabled. Old registry removed successfully: {args.registry}")
        except Exception as e:
            logger.error(
                f"Could not remove the old registry at {args.registry}: {e}")
            sys.exit(1)
    elif os.path.exists(args.registry):
        logger.info(
            f"Registry file found at {args.registry}. Incremental/append mode active.")

    # 3. Validate the scope mapping file
    if not os.path.exists(args.mapping):
        logger.error(
            f"Critical error: mapping file missing at {args.mapping}")
        sys.exit(1)

    try:
        with open(args.mapping, "r", encoding="utf-8") as f:
            mapping_config = json.load(f)
    except Exception as e:
        logger.error(f"Error reading the mapping JSON file: {e}")
        sys.exit(1)

    # 4. Safe initialization of the structural components
    try:
        # NCBIRegistry loads the file if it exists (incremental mode)
        # or instantiates an empty dictionary if the file does not exist
        registry = NCBIRegistry(
            config_path=args.mapping,
            registry_path=args.registry
        )

        orchestrator = DiscoveryOrchestrator(
            registry=registry,
            mapping_config=mapping_config
        )

        # 5. Run the workflow
        logger.info(
            f"Starting taxonomic scan for TaxID: {args.taxon_id}")
        orchestrator.discover_from_root(args.taxon_id)

        logger.info("Discovery process finished successfully.")
        print("\n" + "="*50)
        print("   Taxonomic Mapping Complete")
        print(f"   Root Processed     : {args.taxon_id}")
        print(f"   Registry Updated   : {args.registry}")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(
            f"Critical failure during pipeline execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
