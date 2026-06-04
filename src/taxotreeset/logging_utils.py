"""Shared logging configuration for the CLI subcommands.

Two audiences are separated:

- The **log file** under the XDG state directory receives the full
  diagnostic stream at the user-selected level (default INFO). This is
  the complete record for troubleshooting.
- The **terminal** stays quiet: the root logger's stream handler only
  surfaces warnings and errors, so the detailed INFO telemetry no longer
  floods stdout. User-facing progress is shown by tqdm bars and by a
  small number of stage milestones emitted through a dedicated UI logger.

The UI logger (``TaxoTreeSet.UI``) carries those milestones. It prints to
the terminal at INFO regardless of the root stream level, and also
propagates to the file handler so the milestones appear in the log too.
"""
import logging
import sys

from taxotreeset import paths

_UI_LOGGER_NAME = "TaxoTreeSet.UI"


class _TqdmCompatibleStreamHandler(logging.StreamHandler):
    """Stream handler that writes through tqdm to avoid breaking bars.

    When a progress bar is active, writing directly to the stream would
    corrupt the bar's line. Routing the record through ``tqdm.write``
    keeps the bar intact. Falls back to the normal stream write if tqdm
    is unavailable.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            super().emit(record)


def setup_logging(log_filename: str, level: int = logging.INFO) -> None:
    """Configure file and terminal logging for a CLI run.

    Args:
        log_filename: Name of the log file, written under the XDG state
            directory (e.g. ``"discovery.log"``).
        level: Level for the log file (and the diagnostic stream). The
            terminal still only shows warnings and errors plus the
            user-facing milestones.
    """
    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler: full diagnostic stream at the requested level.
    file_handler = logging.FileHandler(
        paths.log_path(log_filename), encoding="utf-8", mode="w"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    # Terminal handler on the root logger: warnings and errors only, so
    # routine INFO telemetry does not flood the terminal.
    stream_handler = _TqdmCompatibleStreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(min(level, logging.WARNING))
    # Reset handlers so repeated calls (e.g. in tests) do not stack.
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # UI logger: user-facing milestones to the terminal at INFO, while
    # still propagating to the root's file handler for the record.
    ui_logger = logging.getLogger(_UI_LOGGER_NAME)
    ui_logger.setLevel(logging.INFO)
    ui_logger.handlers.clear()
    ui_stream = _TqdmCompatibleStreamHandler(stream=sys.stdout)
    ui_stream.setLevel(logging.INFO)
    ui_stream.setFormatter(logging.Formatter("%(message)s"))
    ui_logger.addHandler(ui_stream)
    ui_logger.propagate = True  # also recorded in the log file


def get_ui_logger() -> logging.Logger:
    """Return the dedicated user-facing (terminal) logger.

    Use this for the small set of milestones a user should see on the
    terminal (e.g. stage banners), keeping ordinary diagnostic logging on
    module loggers that only reach the log file.

    Returns:
        The ``TaxoTreeSet.UI`` logger.
    """
    return logging.getLogger(_UI_LOGGER_NAME)
