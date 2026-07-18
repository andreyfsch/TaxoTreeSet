"""Entry point for ``python -m taxotreeset``.

Builds the top-level argument parser with the ``discover`` and
``generate`` subcommands and dispatches to the selected one.
"""
import argparse
import sys

from taxotreeset.cli import composition, discover, generate, separability


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser with all subcommands.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="taxotreeset",
        description="TaxoTreeSet - balanced hierarchical genomic datasets "
        "from NCBI RefSeq for cascaded fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{discover,generate,separability,composition}",
        help="Subcommand to run.",
    )

    # Arguments shared by every subcommand, attached via parents=[...].
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Verbosity written to the log file. The terminal always "
        "shows only warnings, errors, and progress.",
    )

    discover_parser = subparsers.add_parser(
        "discover",
        parents=[common],
        help="Scan NCBI taxonomy and build the inventory registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    discover.add_arguments(discover_parser)
    discover_parser.set_defaults(_run=discover.run)

    generate_parser = subparsers.add_parser(
        "generate",
        parents=[common],
        help="Produce the cascaded training dataset from the registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    generate.add_arguments(generate_parser)
    generate_parser.set_defaults(_run=generate.run)

    separability_parser = subparsers.add_parser(
        "separability",
        parents=[common],
        help="Score k-mer separability per head and enrich label_map.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    separability.add_arguments(separability_parser)
    separability_parser.set_defaults(_run=separability.run)

    composition_parser = subparsers.add_parser(
        "composition",
        parents=[common],
        help="Audit per-head compositional confounds in virtual classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    composition.add_arguments(composition_parser)
    composition_parser.set_defaults(_run=composition.run)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the selected subcommand.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when None).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(1)
    args._run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
