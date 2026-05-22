import argparse
import logging
import sys
import os
from src.taxotreeset.io.registry import NCBIRegistry
from src.taxotreeset.core.generation_orchestrator import GenerationOrchestrator

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("generation.log", encoding="utf-8", mode="w"),  # mode='w' zera o log a cada run
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(
        description="TaxoTreeSet - Hierarchical Dataset Generation for Cascaded LoRA Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilita logs DEBUG para diagnóstico"
    )
    
    parser.add_argument(
        "--rank", "-g",
        type=str,
        default="viruses",
        choices=["viruses", "bacteria", "archaea", "eukaryotes", "all"],
        help="Target biological domain scope to isolate and compile the cascaded hierarchy for"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output storage optimization format for deep learning engines"
    )
    parser.add_argument(
        "--window", "-w",
        type=int,
        default=2000,
        help="Sliding window context size (bp) extracted for DNABERT tokens processing"
    )
    parser.add_argument(
        "--registry", "-r",
        type=str,
        default="data/registry.json",
        help="Path to the compiled inventory registry manifest input"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/datasets",
        help="Target directory to dump the completed training shards and sidecar manifests"
    )
    parser.add_argument(
        "--min-abundance", "-a",
        type=int,
        default=2,
        help="Minimum sequence abundance required for a taxon node to avoid fallback redirection"
    )
    parser.add_argument(
        "--min-subclades-per-bucket",
        type=int,
        default=5,
        help="Mínimo de subclados (filhos) necessários para um rank anômalo "
            "ter bucket virtual próprio. Abaixo disso, mescla no bucket 'misc' do pai."
    )

    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("TaxoTreeSet.Generation.CLI")

    if not os.path.exists(args.registry):
        logger.error(f"Missing inventory blueprint at {args.registry}. Please execute 'main_discovery.py' first.")
        sys.exit(1)

    try:
        logger.info("Initializing active Metadata Registry map repository...")
        registry = NCBIRegistry(registry_path=args.registry)

        logger.info("Assembling pipeline execution master blocks...")
        pipeline = GenerationOrchestrator(
            registry=registry,
            vault_path="data/vault",
            output_dir=args.output,
            max_subseq_len=args.window,
            output_format=args.format,
            min_subclades_per_bucket=args.min_subclades_per_bucket
        )

        logger.info(f"Executing downstream generation matrix targets for domain group: '{args.rank}' down to Genus")
        pipeline.run_pipeline(
            target_group=args.rank,
            abundance_threshold=args.min_abundance
        )

        print("\n" + "="*60)
        print("   🎉 CASCADED DATASET PRODUCTION SUCCEEDED!")
        print(f"   Target Domain Group: {args.rank.upper()}")
        print(f"   Depth Boundary     : GENUS (Fixed Floor)")
        print(f"   Output Encoding    : {args.format.upper()}")
        print(f"   Destination Vault  : {args.output}")
        print("="*60 + "\n")

    except Exception as e:
        logger.error(f"Critical execution block interrupted production pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()