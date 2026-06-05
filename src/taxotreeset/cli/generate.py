"""The ``generate`` subcommand: cascaded dataset production.

Consumes the inventory registry built by ``discover`` and emits the
balanced, hierarchically structured training shards.
"""
import argparse
import logging
import os
import sys

from taxotreeset import paths
from taxotreeset.logging_utils import setup_logging
from taxotreeset.core.generation_orchestrator import GenerationOrchestrator
from taxotreeset.io.registry import NCBIRegistry


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the generate-specific arguments on a subparser.

    Args:
        parser: The subparser to populate.
    """
    parser.add_argument(
        "--mapping", "-m",
        type=str,
        default="configs/mapping.json",
        help="Path to the JSON file mapping scopes and fallback "
        "redirections",
    )
    parser.add_argument(
        "--vault", "-v",
        type=str,
        default=str(paths.default_vault_path()),
        help="Path to LMDB vault storing genome sequences",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility of splits and subseq "
        "sampling",
    )
    parser.add_argument(
        "--min-num-seqs",
        type=int,
        default=1000,
        help="Threshold of unique subseqs per class for global leveling "
        "without cutoff. Below this, a percentile cutoff is applied and a "
        "virtual_low_capacity bucket is created (default: 1000).",
    )
    parser.add_argument(
        "--cutoff-percentage",
        type=float,
        default=98.0,
        help="Percentage of children retained when cutoff is applied. "
        "The lowest (100-p)%% by capacity go into the low_capacity "
        "bucket (default: 98.0).",
    )
    parser.add_argument(
        "--approximate-capacity",
        action="store_true",
        help="Use a Bloom filter to estimate node capacity (~12MB memory, "
        "~1%% false-positive rate). Without this flag, capacity is computed "
        "exactly via 2-bit-packed deduplication, which is memory-bounded "
        "(spilling supernodes to disk) and runs on modest hardware. Use "
        "this flag to trade exactness for speed on very large runs.",
    )
    parser.add_argument(
        "--root", "-g",
        type=str,
        default="viruses",
        help="Root of the taxonomy to generate from: a domain shortcut "
        "(viruses, bacteria, archaea, eukaryotes), a numeric NCBI TaxID, "
        "or a clade scientific name (e.g. Caudoviricetes).",
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        default=None,
        help="Canonical rank where the cascade stops creating heads "
        "(species, genus, family, order, class, phylum, kingdom, "
        "superkingdom). Nodes deeper than this become training labels "
        "but not heads of their own. Defaults to the deepest rank.",
    )
    parser.add_argument(
        "--single-level",
        action="store_true",
        help="Generate only the root's head: its direct children become "
        "training labels with no further recursion. Cannot be combined "
        "with --stop-at.",
    )
    parser.add_argument(
        "--output-format", "-f",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output storage optimization format for deep learning engines",
    )
    parser.add_argument(
        "--max-subseq-len", "-w",
        type=int,
        default=2000,
        help="Sliding window context size (bp) extracted for tokenization",
    )
    parser.add_argument(
        "--registry", "-r",
        type=str,
        default=str(paths.default_registry_path()),
        help="Path to the compiled inventory registry manifest input",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="taxotreeset-datasets",
        help="Target directory for the completed training shards and "
        "sidecar manifests",
    )
    parser.add_argument(
        "--min-abundance", "-a",
        type=int,
        default=2,
        help="Minimum sequence abundance required for a taxon node to "
        "avoid fallback redirection",
    )
    parser.add_argument(
        "--min-subclades-per-bucket",
        type=int,
        default=5,
        help="Minimum number of subclades (children) required for an "
        "anomalous rank to get its own virtual bucket. Below this, it is "
        "merged into the parent's 'misc' bucket.",
    )
    parser.add_argument(
        "--max-n-per-class",
        type=int,
        default=20000,
        help="Absolute ceiling on subseqs per class in any head. Limits "
        "explosion on heads with large genomes (jumbo phages, etc). "
        "Heads with capacity above this value are capped; others remain "
        "intact (default: 20000).",
    )
    parser.add_argument(
        "--min-leaves-per-class",
        type=int,
        default=3,
        help="Minimum number of sequence leaves a child must have to "
        "remain a standalone training class. Children below this floor "
        "carry too little signal and, under the 'fallback' strategy, are "
        "diverted into a virtual_rare_taxa bucket (default: 3).",
    )
    parser.add_argument(
        "--rare-taxa-strategy",
        type=str,
        choices=["fallback", "keep"],
        default="fallback",
        help="How to handle children below --min-leaves-per-class. "
        "'fallback': group them into a virtual_rare_taxa bucket that "
        "becomes a single fallback label under the head (recommended for "
        "out-of-distribution-aware classification). 'keep': retain every "
        "child as its own class regardless of leaf count "
        "(default: fallback).",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the NCBI sync step and use the existing vault as-is "
        "(faster iteration on an already-populated vault).",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the generation pipeline from parsed arguments.

    Args:
        args: Parsed CLI arguments for the generate subcommand.
    """
    setup_logging(
        "generation.log",
        level=getattr(logging, args.log_level),
    )
    logger = logging.getLogger("TaxoTreeSet.Generation.CLI")

    if not os.path.exists(args.registry) and args.no_sync:
        logger.error(
            "Missing inventory registry at %s and --no-sync was given, "
            "so there is nothing to generate from. Run 'taxotreeset "
            "discover' first, or drop --no-sync to build it via sync.",
            args.registry,
        )
        sys.exit(1)

    try:
        logger.info("Initializing active metadata registry...")
        registry = NCBIRegistry(registry_path=args.registry)

        logger.info("Assembling pipeline execution blocks...")
        pipeline = GenerationOrchestrator(
            registry=registry,
            vault_path=args.vault,
            output_dir=args.output,
            config_path=args.mapping,
            max_subseq_len=args.max_subseq_len,
            seed=args.seed,
            output_format=args.output_format,
            min_subclades_per_bucket=args.min_subclades_per_bucket,
            min_num_seqs=args.min_num_seqs,
            cutoff_percentage=args.cutoff_percentage,
            max_n_per_class=args.max_n_per_class,
            use_exact_capacity=not args.approximate_capacity,
            min_leaves_per_class=args.min_leaves_per_class,
            rare_taxa_strategy=args.rare_taxa_strategy,
        )

        logger.info(
            "Generating cascaded hierarchy from root '%s'", args.root,
        )
        pipeline.run_pipeline(
            target_group=args.root,
            abundance_threshold=args.min_abundance,
            sync=not args.no_sync,
            stop_at=args.stop_at,
            single_level=args.single_level,
        )

        print("\n" + "=" * 60)
        print("   CASCADED DATASET PRODUCTION SUCCEEDED")
        print(f"   Generation Root    : {args.root}")
        if args.single_level:
            depth_desc = "single level (root head only)"
        elif args.stop_at:
            depth_desc = f"stop at {args.stop_at}"
        else:
            depth_desc = "full depth (to species)"
        print(f"   Depth Boundary     : {depth_desc}")
        print(f"   Output Encoding    : {args.output_format.upper()}")
        print(f"   Destination        : {args.output}")
        print("=" * 60 + "\n")
    except Exception as exc:
        logger.error(
            "Critical failure during generation: %s", exc, exc_info=True
        )
        sys.exit(1)
