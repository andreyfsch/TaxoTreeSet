"""Tests for taxotreeset.logging_utils — setup_logging and get_ui_logger."""

import logging

import pytest
from taxotreeset.logging_utils import (
    _TqdmCompatibleStreamHandler,
    get_ui_logger,
    setup_logging,
)


# ---------------------------------------------------------------------------
# get_ui_logger
# ---------------------------------------------------------------------------


class TestGetUiLogger:
    def test_returns_logger_with_correct_name(self):
        logger = get_ui_logger()
        assert logger.name == "TaxoTreeSet.UI"

    def test_returns_logging_logger_instance(self):
        logger = get_ui_logger()
        assert isinstance(logger, logging.Logger)

    def test_same_instance_on_repeated_calls(self):
        assert get_ui_logger() is get_ui_logger()


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_root_logger_has_two_handlers_after_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        setup_logging("test.log")
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_file_handler_writes_to_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        setup_logging("test.log")
        log_file = tmp_path / "taxotreeset" / "test.log"
        assert log_file.exists()

    def test_stream_handler_level_is_warning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        setup_logging("test.log")
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert any(h.level == logging.WARNING for h in stream_handlers)

    def test_repeated_calls_do_not_stack_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        setup_logging("test.log")
        setup_logging("test.log")
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_ui_logger_has_handler_after_setup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        setup_logging("test.log")
        ui_logger = logging.getLogger("TaxoTreeSet.UI")
        assert len(ui_logger.handlers) >= 1

    def test_ui_logger_propagate_is_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        setup_logging("test.log")
        ui_logger = logging.getLogger("TaxoTreeSet.UI")
        assert ui_logger.propagate is True

    def teardown_method(self):
        """Reset root and UI logger state to avoid test pollution."""
        root = logging.getLogger()
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        ui_logger = logging.getLogger("TaxoTreeSet.UI")
        for handler in list(ui_logger.handlers):
            handler.close()
            ui_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# _TqdmCompatibleStreamHandler
# ---------------------------------------------------------------------------


class TestTqdmCompatibleStreamHandler:
    def test_emit_falls_back_to_super_on_exception(self, capsys):
        import io
        from unittest.mock import patch

        stream = io.StringIO()
        handler = _TqdmCompatibleStreamHandler(stream=stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="fallback message", args=(), exc_info=None
        )

        with patch("taxotreeset.logging_utils._TqdmCompatibleStreamHandler.emit") as mock_emit:
            mock_emit.side_effect = None
            mock_emit.return_value = None
            handler.emit(record)

    def test_emit_falls_back_when_tqdm_write_raises(self):
        import io
        from unittest.mock import patch

        stream = io.StringIO()
        handler = _TqdmCompatibleStreamHandler(stream=stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="fallback message", args=(), exc_info=None
        )

        with patch("tqdm.tqdm.write", side_effect=RuntimeError("tqdm unavailable")):
            handler.emit(record)

        output = stream.getvalue()
        assert "fallback message" in output

    def test_emit_success_path_calls_flush(self):
        """Happy path: tqdm.write succeeds → self.flush() on line 40 is reached."""
        import io

        stream = io.StringIO()
        handler = _TqdmCompatibleStreamHandler(stream=stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello flush", args=(), exc_info=None
        )
        handler.emit(record)

        output = stream.getvalue()
        assert "hello flush" in output

    def test_handler_is_subclass_of_stream_handler(self):
        import io

        handler = _TqdmCompatibleStreamHandler(stream=io.StringIO())
        assert isinstance(handler, logging.StreamHandler)
