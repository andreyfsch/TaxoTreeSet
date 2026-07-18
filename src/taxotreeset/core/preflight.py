"""Pre-flight checks for the generation pipeline.

Runs before Stage 1 to:
1. Verify there is enough disk space on each target filesystem.
2. Estimate total runtime and warn the user when it exceeds a threshold.

If disk space is insufficient the function raises SystemExit with an
explanatory message.  If estimated runtime exceeds the warning threshold
and stdin is a TTY, the user is asked to confirm before proceeding.
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from taxotreeset.io.registry import NCBIRegistry

_WARN_THRESHOLD_SECS = 30 * 60  # show confirmation prompt above 30 min


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.0f} PB"


def _fmt_time_range(lo: float, hi: float) -> str:
    """Return a human-readable duration range."""
    def _single(s: float) -> str:
        if s < 90:
            return f"{s:.0f} sec"
        if s < 5400:
            return f"{s / 60:.0f} min"
        return f"{s / 3600:.1f} h"

    lo_s, hi_s = _single(lo), _single(hi)
    if lo_s == hi_s:
        return f"~{lo_s}"
    return f"{lo_s} – {hi_s}"


# ---------------------------------------------------------------------------
# Estimation logic
# ---------------------------------------------------------------------------

def _total_seq_bytes(registry: "NCBIRegistry") -> int:
    return sum(
        int(v.get("total_sequence_length") or 0)
        for v in registry.registry["accessions"].values()
    )


def _pending_bytes(registry: "NCBIRegistry") -> int:
    return sum(
        int(v.get("total_sequence_length") or 0)
        for v in registry.registry["accessions"].values()
        if not v.get("downloaded", False)
    )


def _n_accessions(registry: "NCBIRegistry") -> int:
    return len(registry.registry["accessions"])


def _free_bytes(path: str) -> int:
    """Return free bytes on the filesystem containing path.

    The path is resolved to absolute first (against the cwd) so a relative
    path — e.g. the default ``--output taxotreeset-datasets`` — resolves to a
    real filesystem instead of walking up to ``""`` and skipping the check.
    Walks up to the first existing ancestor when path does not yet exist.
    Returns a large sentinel (sys.maxsize) if the check is not possible so
    that a missing path does not cause a false disk-space failure.
    """
    check = os.path.abspath(path)
    while check and not os.path.exists(check):
        parent = os.path.dirname(check)
        if parent == check:
            return sys.maxsize
        check = parent
    try:
        return shutil.disk_usage(check).free
    except OSError:
        return sys.maxsize


# ---------------------------------------------------------------------------
# Report rendering and gating
# ---------------------------------------------------------------------------

def _format_preflight_report(
    n_acc: int,
    total_bytes: int,
    disk_checks: list[tuple[str, str, int, int]],
    time_rows: list[tuple[str, float, float]],
    total_lo: float,
    total_hi: float,
    n_gpu_workers: int | None,
    gpu_on: bool,
) -> str:
    """Build the boxed pre-flight summary text.

    Args:
        n_acc: Number of genomes in scope.
        total_bytes: Total sequence data in scope.
        disk_checks: ``(label, path, needed, free)`` rows to display.
        time_rows: ``(label, lo_secs, hi_secs)`` runtime estimate rows.
        total_lo: Optimistic total runtime in seconds.
        total_hi: Pessimistic total runtime in seconds.
        n_gpu_workers: Configured GPU worker count (0 or None = CPU only).
        gpu_on: Whether GPU acceleration is enabled.

    Returns:
        The fully rendered, multi-line report (without surrounding blank lines).
    """
    W = 62
    sep = "─" * W

    def _row(label: str, value: str, flag: str = "") -> str:
        flag_part = f"  {flag}" if flag else ""
        pad = W - 4 - len(label) - len(value) - len(flag_part)
        return f"║  {label}{' ' * max(pad, 1)}{value}{flag_part}  ║"

    lines: list[str] = [
        f"╔{'═' * W}╗",
        f"║{'  TaxoTreeSet — Pre-flight Check':^{W}}║",
        f"║{sep}║",
        _row("Genomes in scope:", f"{n_acc:,}"),
        _row("Total sequence data:", _fmt_bytes(total_bytes)),
        f"║{sep}║",
        f"║  {'Disk requirements (estimated)':<{W - 2}}║",
    ]

    for lbl, path, need, free in disk_checks:
        short = path if len(path) <= 32 else "…" + path[-31:]
        status = "✗ INSUFFICIENT" if free < need else "✓"
        # Two-line entry: path on first line, figures on second
        lines.append(f"║  {lbl}  {short:<{W - len(lbl) - 4}}║")
        fig = f"{_fmt_bytes(need)} needed  /  {_fmt_bytes(free)} free"
        lines.append(_row(f"    {fig}", status))

    lines += [
        f"║{sep}║",
        f"║  {'Estimated runtime':<{W - 2}}║",
    ]

    for label, lo, hi in time_rows:
        marker = "  ◀ most expensive step" if "k-mer" in label else ""
        lines.append(_row(f"  {label}", _fmt_time_range(lo, hi), marker))

    workers = n_gpu_workers or 0
    gpu_value = (
        f"enabled ({n_gpu_workers} worker{'s' if workers > 1 else ''})"
        if gpu_on else "disabled (CPU only)"
    )
    lines += [
        _row("  " + "─" * 36, ""),
        _row("  Total", _fmt_time_range(total_lo, total_hi)),
        f"║{sep}║",
        _row("GPU acceleration:", gpu_value),
        f"╚{'═' * W}╝",
    ]
    return "\n".join(lines)


def _abort_if_insufficient_disk(
    failures: list[tuple[str, str, int, int]],
) -> None:
    """Print disk-shortage details and ``exit(1)`` when any check failed."""
    if not failures:
        return
    print("ERROR: insufficient disk space on the following paths:", file=sys.stderr)
    for lbl, path, need, free in failures:
        shortage = need - free
        print(
            f"  {lbl.strip()}: {path}\n"
            f"    → need {_fmt_bytes(need)}, "
            f"have {_fmt_bytes(free)}, "
            f"short by {_fmt_bytes(shortage)}",
            file=sys.stderr,
        )
    print(
        "\nFix: free up space on the affected drives, use a different "
        "--spill-dir / --output / --vault path, or reduce the dataset scope.",
        file=sys.stderr,
    )
    sys.exit(1)


def _confirm_long_run(total_hi: float) -> None:
    """Prompt for confirmation on a TTY when the run may be long; exit on no."""
    if not (total_hi > _WARN_THRESHOLD_SECS and sys.stdin.isatty()):
        return
    try:
        answer = input("This run may take a while. Proceed? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer in ("n", "no"):
        print("Aborted.", file=sys.stderr)
        sys.exit(0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_preflight(
    registry: "NCBIRegistry",
    vault_path: str,
    output_dir: str,
    spill_dir: str | None,
    n_gpu_workers: int | None,
    sync: bool,
) -> None:
    """Print a pre-flight summary and abort or prompt if needed.

    Args:
        registry: Populated NCBIRegistry instance.
        vault_path: LMDB vault directory.
        output_dir: Parquet output directory.
        spill_dir: Spill directory, or None if not configured.
        n_gpu_workers: Number of GPU workers (0 or None = CPU only).
        sync: Whether Stage 1 will download new data.
    """
    total_bytes = _total_seq_bytes(registry)
    dl_bytes = _pending_bytes(registry) if sync else 0
    n_acc = _n_accessions(registry)

    # ---- disk estimates ------------------------------------------------
    vault_need = int(dl_bytes * 1.5)           # LMDB overhead on new data
    spill_need = int(total_bytes * 3.0)        # pre-dedup worst case
    output_need = int(total_bytes * 0.05)      # parquets ≈ 5 % of raw

    disk_checks: list[tuple[str, str, int, int]] = []
    # (label, path, needed, free)
    if vault_need > 0:
        disk_checks.append(("Vault ", vault_path, vault_need,
                             _free_bytes(vault_path)))
    if spill_dir:
        disk_checks.append(("Spill ", spill_dir, spill_need,
                             _free_bytes(spill_dir)))
    disk_checks.append(("Output", output_dir, output_need,
                         _free_bytes(output_dir)))

    # Each path is checked against its own free space independently. When vault,
    # spill, and output share one filesystem the combined footprint could still
    # overflow it while every individual check passes; this advisory estimate does
    # not sum needs per device.
    failures = [(lbl, path, need, free)
                for lbl, path, need, free in disk_checks
                if free < need]

    # ---- time estimates ------------------------------------------------
    gpu_on = bool(n_gpu_workers)

    dl_lo = dl_bytes / (20 * 1024 ** 2)    # optimistic: 20 MB/s
    dl_hi = dl_bytes / (5 * 1024 ** 2)     # pessimistic: 5 MB/s

    if gpu_on:
        # GPU handles small genomes fast; large chromosomes spill to CPU.
        # lo: dominated by GPU-accelerated leaves (~100 MB/s effective)
        # hi: large spill-heavy datasets drop to ~5 MB/s effective
        km_lo = total_bytes / (100 * 1024 ** 2)
        km_hi = total_bytes / (5 * 1024 ** 2)
    else:
        km_lo = total_bytes / (20 * 1024 ** 2)
        km_hi = total_bytes / (2 * 1024 ** 2)

    build_lo = n_acc * 0.2
    build_hi = n_acc * 1.0

    total_lo = dl_lo + km_lo + build_lo
    total_hi = dl_hi + km_hi + build_hi

    time_rows: list[tuple[str, float, float]] = []
    if dl_bytes > 0:
        time_rows.append(("Downloading sequences", dl_lo, dl_hi))
    time_rows.append(("Unique k-mer analysis", km_lo, km_hi))
    time_rows.append(("Building datasets    ", build_lo, build_hi))

    # ---- display -------------------------------------------------------
    report = _format_preflight_report(
        n_acc, total_bytes, disk_checks, time_rows,
        total_lo, total_hi, n_gpu_workers, gpu_on,
    )
    print("\n" + report + "\n", flush=True)

    # ---- hard abort on disk failure, then confirm long runs ------------
    _abort_if_insufficient_disk(failures)
    _confirm_long_run(total_hi)
