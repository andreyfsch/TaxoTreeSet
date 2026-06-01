"""Shared logging configuration for the CLI subcommands."""
import logging
import sys

from taxotreeset import paths


def setup_logging(log_filename: str, level: int = logging.INFO) -> None:
    """Configure root logging to a file under the state dir and to stdout.

    Args:
        log_filename: Name of the log file, written under the XDG state
            directory (e.g. ``"discovery.log"``).
        level: Logging level for the root logger.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                paths.log_path(log_filename), encoding="utf-8", mode="w"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
