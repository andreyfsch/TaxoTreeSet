"""Unit tests for taxotreeset.core.preflight."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from taxotreeset.core.preflight import (
    _fmt_bytes,
    _fmt_time_range,
    _free_bytes,
    _n_accessions,
    _pending_bytes,
    _total_seq_bytes,
    run_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(accessions: dict) -> MagicMock:
    reg = MagicMock()
    reg.registry = {"accessions": accessions}
    return reg


def _acc(length: int, downloaded: bool) -> dict:
    return {"total_sequence_length": length, "downloaded": downloaded}


# ---------------------------------------------------------------------------
# _fmt_bytes
# ---------------------------------------------------------------------------

class TestFmtBytes:
    def test_bytes(self):
        assert _fmt_bytes(512) == "512 B"

    def test_kilobytes(self):
        assert _fmt_bytes(2048) == "2 KB"

    def test_megabytes(self):
        assert _fmt_bytes(5 * 1024 ** 2) == "5 MB"

    def test_gigabytes(self):
        assert _fmt_bytes(3 * 1024 ** 3) == "3 GB"

    def test_terabytes(self):
        assert _fmt_bytes(2 * 1024 ** 4) == "2 TB"

    def test_boundary_exactly_1kb(self):
        assert _fmt_bytes(1024) == "1 KB"


# ---------------------------------------------------------------------------
# _fmt_time_range
# ---------------------------------------------------------------------------

class TestFmtTimeRange:
    def test_equal_values_collapsed_to_tilde(self):
        result = _fmt_time_range(60, 60)
        assert result.startswith("~")

    def test_range_shown_with_dash(self):
        result = _fmt_time_range(60, 300)
        assert "–" in result

    def test_under_90s_shown_in_seconds(self):
        result = _fmt_time_range(30, 80)
        assert "sec" in result

    def test_between_90s_and_5400s_shown_in_minutes(self):
        result = _fmt_time_range(120, 600)
        assert "min" in result

    def test_over_5400s_shown_in_hours(self):
        result = _fmt_time_range(7200, 10800)
        assert "h" in result

    def test_mixed_units_in_range(self):
        # lo in seconds, hi in minutes
        result = _fmt_time_range(45, 200)
        assert "sec" in result
        assert "min" in result


# ---------------------------------------------------------------------------
# Registry query helpers
# ---------------------------------------------------------------------------

class TestRegistryHelpers:
    def test_total_seq_bytes_sums_all(self):
        reg = _make_registry({
            "A": _acc(1000, True),
            "B": _acc(2000, False),
        })
        assert _total_seq_bytes(reg) == 3000

    def test_total_seq_bytes_empty_registry(self):
        reg = _make_registry({})
        assert _total_seq_bytes(reg) == 0

    def test_total_seq_bytes_missing_length_treated_as_zero(self):
        reg = _make_registry({"A": {"downloaded": True}})
        assert _total_seq_bytes(reg) == 0

    def test_pending_bytes_only_sums_not_downloaded(self):
        reg = _make_registry({
            "A": _acc(1000, True),
            "B": _acc(2000, False),
            "C": _acc(500, False),
        })
        assert _pending_bytes(reg) == 2500

    def test_pending_bytes_all_downloaded(self):
        reg = _make_registry({
            "A": _acc(1000, True),
            "B": _acc(2000, True),
        })
        assert _pending_bytes(reg) == 0

    def test_n_accessions_count(self):
        reg = _make_registry({"A": {}, "B": {}, "C": {}})
        assert _n_accessions(reg) == 3

    def test_n_accessions_empty(self):
        reg = _make_registry({})
        assert _n_accessions(reg) == 0


# ---------------------------------------------------------------------------
# _free_bytes
# ---------------------------------------------------------------------------

class TestFreeBytes:
    def test_existing_path_returns_real_free(self, tmp_path):
        result = _free_bytes(str(tmp_path))
        assert result > 0
        assert result < sys.maxsize

    def test_nonexistent_path_walks_up(self, tmp_path):
        nonexistent = str(tmp_path / "a" / "b" / "c")
        result = _free_bytes(nonexistent)
        # tmp_path exists, so walking up should find real free space
        assert result > 0
        assert result < sys.maxsize

    def test_oserror_returns_maxsize(self, tmp_path):
        with patch("shutil.disk_usage", side_effect=OSError("mocked")):
            result = _free_bytes(str(tmp_path))
        assert result == sys.maxsize

    def test_relative_path_is_resolved_and_checked(self, tmp_path, monkeypatch):
        # A relative path (e.g. the default --output "taxotreeset-datasets") must
        # resolve against cwd and get a real free-space reading, not be skipped
        # as sys.maxsize by walking dirname up to "".
        monkeypatch.chdir(tmp_path)
        result = _free_bytes("taxotreeset-datasets")  # relative, does not exist
        assert 0 < result < sys.maxsize


# ---------------------------------------------------------------------------
# run_preflight — no failures, short run (no prompt)
# ---------------------------------------------------------------------------

class TestRunPreflightHappyPath:
    def _registry_tiny(self):
        return _make_registry({"A": _acc(1000, True)})

    def test_returns_none_on_success(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            result = run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=0,
                sync=False,
            )
        assert result is None

    def test_prints_preflight_box(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=0,
                sync=False,
            )
        out = capsys.readouterr().out
        assert "Pre-flight Check" in out
        assert "Genomes in scope" in out
        assert "Estimated runtime" in out

    def test_gpu_enabled_shows_in_output(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=2,
                sync=False,
            )
        out = capsys.readouterr().out
        assert "enabled" in out
        assert "2 workers" in out

    def test_cpu_only_shows_in_output(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=0,
                sync=False,
            )
        out = capsys.readouterr().out
        assert "CPU only" in out

    def test_sync_false_skips_vault_disk_check(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=0,
                sync=False,
            )
        out = capsys.readouterr().out
        # No pending bytes → vault row not shown
        assert "Vault" not in out

    def test_sync_true_with_pending_shows_vault_row(self, tmp_path, capsys):
        reg = _make_registry({"A": _acc(10 * 1024 ** 2, False)})
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=reg,
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=0,
                sync=True,
            )
        out = capsys.readouterr().out
        assert "Vault" in out

    def test_spill_dir_row_shown_when_provided(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=str(tmp_path / "spill"),
                n_gpu_workers=0,
                sync=False,
            )
        out = capsys.readouterr().out
        assert "Spill" in out

    def test_kmer_analysis_marked_as_most_expensive(self, tmp_path, capsys):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=100 * 1024 ** 3):
            run_preflight(
                registry=self._registry_tiny(),
                vault_path=str(tmp_path),
                output_dir=str(tmp_path),
                spill_dir=None,
                n_gpu_workers=0,
                sync=False,
            )
        out = capsys.readouterr().out
        assert "most expensive step" in out


# ---------------------------------------------------------------------------
# run_preflight — disk failure → sys.exit(1)
# ---------------------------------------------------------------------------

class TestRunPreflightDiskFailure:
    def test_insufficient_output_space_exits(self, tmp_path):
        reg = _make_registry({"A": _acc(10 * 1024 ** 3, True)})
        with patch("taxotreeset.core.preflight._free_bytes", return_value=0):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight(
                    registry=reg,
                    vault_path=str(tmp_path),
                    output_dir=str(tmp_path),
                    spill_dir=None,
                    n_gpu_workers=0,
                    sync=False,
                )
        assert exc_info.value.code == 1

    def test_error_message_names_the_failing_path(self, tmp_path, capsys):
        reg = _make_registry({"A": _acc(10 * 1024 ** 3, True)})
        with patch("taxotreeset.core.preflight._free_bytes", return_value=0):
            with pytest.raises(SystemExit):
                run_preflight(
                    registry=reg,
                    vault_path=str(tmp_path),
                    output_dir=str(tmp_path),
                    spill_dir=None,
                    n_gpu_workers=0,
                    sync=False,
                )
        err = capsys.readouterr().err
        assert "insufficient disk space" in err.lower()

    def test_vault_failure_when_sync_and_pending(self, tmp_path, capsys):
        reg = _make_registry({"A": _acc(10 * 1024 ** 3, False)})
        # vault has no free space; output has plenty
        def fake_free(path):
            return 0 if "vault" in str(path) else 100 * 1024 ** 3

        with patch("taxotreeset.core.preflight._free_bytes", side_effect=fake_free):
            with pytest.raises(SystemExit) as exc_info:
                run_preflight(
                    registry=reg,
                    vault_path=str(tmp_path / "vault"),
                    output_dir=str(tmp_path / "out"),
                    spill_dir=None,
                    n_gpu_workers=0,
                    sync=True,
                )
        assert exc_info.value.code == 1
        assert "insufficient disk space" in capsys.readouterr().err.lower()


# ---------------------------------------------------------------------------
# run_preflight — long run + TTY confirmation prompt
# ---------------------------------------------------------------------------

class TestRunPreflightConfirmationPrompt:
    # Use a large dataset so total_hi > 30 min threshold.
    def _large_registry(self):
        return _make_registry({"A": _acc(500 * 1024 ** 3, True)})

    def test_no_prompt_when_not_a_tty(self, tmp_path):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=10 * 1024 ** 4):
            with patch("sys.stdin.isatty", return_value=False):
                # Should not raise or prompt
                run_preflight(
                    registry=self._large_registry(),
                    vault_path=str(tmp_path),
                    output_dir=str(tmp_path),
                    spill_dir=None,
                    n_gpu_workers=0,
                    sync=False,
                )

    def test_prompt_shown_on_tty_and_y_continues(self, tmp_path):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=10 * 1024 ** 4):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", return_value="y"):
                    run_preflight(
                        registry=self._large_registry(),
                        vault_path=str(tmp_path),
                        output_dir=str(tmp_path),
                        spill_dir=None,
                        n_gpu_workers=0,
                        sync=False,
                    )

    def test_prompt_n_answer_exits_0(self, tmp_path):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=10 * 1024 ** 4):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", return_value="n"):
                    with pytest.raises(SystemExit) as exc_info:
                        run_preflight(
                            registry=self._large_registry(),
                            vault_path=str(tmp_path),
                            output_dir=str(tmp_path),
                            spill_dir=None,
                            n_gpu_workers=0,
                            sync=False,
                        )
        assert exc_info.value.code == 0

    def test_prompt_empty_answer_continues(self, tmp_path):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=10 * 1024 ** 4):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", return_value=""):
                    run_preflight(
                        registry=self._large_registry(),
                        vault_path=str(tmp_path),
                        output_dir=str(tmp_path),
                        spill_dir=None,
                        n_gpu_workers=0,
                        sync=False,
                    )

    def test_eof_on_prompt_exits_0(self, tmp_path):
        with patch("taxotreeset.core.preflight._free_bytes", return_value=10 * 1024 ** 4):
            with patch("sys.stdin.isatty", return_value=True):
                with patch("builtins.input", side_effect=EOFError):
                    with pytest.raises(SystemExit) as exc_info:
                        run_preflight(
                            registry=self._large_registry(),
                            vault_path=str(tmp_path),
                            output_dir=str(tmp_path),
                            spill_dir=None,
                            n_gpu_workers=0,
                            sync=False,
                        )
        assert exc_info.value.code == 0
