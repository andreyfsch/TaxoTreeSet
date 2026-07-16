"""Leaf-phase checkpointing and spill-directory cleanup for the bottom-up pass.

The bottom-up capacity computation runs Phase 1 (per-leaf key extraction) then
Phase 2 (bottom-up merge). Phase 1 can be expensive, so its per-leaf accumulators
are checkpointed to ``spill_dir`` after it completes; a resumed run with the same
``spill_dir`` detects the checkpoint and skips already-computed leaves, making the
leaf phase resumable across SLURM job boundaries. This module holds that
checkpoint I/O plus the cleanup of the ``tts_capacity_*`` spill directories.

Extracted from ``capacity.py`` (P7 Part C) and re-exported there. ``_NodeCapacityKeys``
is imported lazily inside ``_load_leaf_checkpoint`` to avoid a package import cycle
(capacity imports this module at load time).
"""

import logging

# Filename for the leaf-phase checkpoint written to spill_dir after Phase 1
# completes. A subsequent run with the same spill_dir detects this file and
# skips already-computed leaves, making the leaf phase resumable across SLURM
# job boundaries.
_LEAF_CHECKPOINT_FNAME: str = "capacity_leaf_checkpoint.json"


def _save_leaf_checkpoint(
    leaf_accumulators: dict,
    spill_dir: str,
    min_len: int,
    void_dtype,
) -> None:
    """Persist leaf accumulators to disk so a resumed run can skip Phase 1.

    Disk-mode accumulators already live in spill_dir as bucket files; only
    their paths need to be recorded.  Memory-mode accumulators are serialised
    to ``capacity_leaf_{i}.bin`` files in spill_dir so they survive the
    process boundary.

    Args:
        leaf_accumulators: Mapping of str(leaf_name) to accumulator.
        spill_dir: Directory where checkpoint and auxiliary files are written.
        min_len: Window size used to produce the accumulators; stored in the
            checkpoint so a resume with a different min_len is detected and
            rejected.
        void_dtype: Numpy void dtype for packed keys.
    """
    import json
    import os

    checkpoint: dict = {"min_len": min_len, "leaves": {}}
    for idx, (name, acc) in enumerate(leaf_accumulators.items()):
        if acc._on_disk:
            checkpoint["leaves"][name] = {
                "mode": "disk",
                "buckets": acc._bucket_paths,
                "tmp_dir": acc._tmp_dir,
                "amb": acc._ambiguous_count,
                "kb": acc._key_bytes,
            }
        else:
            fpath = os.path.join(spill_dir, f"capacity_leaf_{idx}.bin")
            if acc._pure_keys is not None and acc._pure_keys.shape[0]:
                acc._pure_keys.tofile(fpath)
            else:
                open(fpath, "wb").close()
            checkpoint["leaves"][name] = {
                "mode": "memory",
                "file": fpath,
                "amb": acc._ambiguous_count,
                "kb": acc._key_bytes,
            }

    checkpoint_path = os.path.join(spill_dir, _LEAF_CHECKPOINT_FNAME)
    with open(checkpoint_path, "w", encoding="utf-8") as fh:
        json.dump(checkpoint, fh)


def _load_leaf_checkpoint(
    spill_dir: str,
    min_len: int,
    void_dtype,
) -> dict | None:
    """Load a previously saved leaf checkpoint if valid.

    Returns None when no checkpoint exists, the checkpoint was produced
    with a different min_len, or any referenced file is missing (which
    indicates the spill_dir was partially cleaned between runs).

    Args:
        spill_dir: Directory that may contain the checkpoint file.
        min_len: Window size expected in the checkpoint.
        void_dtype: Numpy void dtype for packed keys.

    Returns:
        Dict mapping str(leaf_name) to a reconstructed _NodeCapacityKeys,
        or None when the checkpoint cannot be used.
    """
    import json
    import os

    import numpy as np

    # lazy to avoid a capacity <-> _spill import cycle (see module docstring)
    from taxotreeset.core.generation.capacity import _NodeCapacityKeys

    checkpoint_path = os.path.join(spill_dir, _LEAF_CHECKPOINT_FNAME)
    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, encoding="utf-8") as fh:
        checkpoint = json.load(fh)

    if checkpoint.get("min_len") != min_len:
        return None

    result: dict = {}
    for name, info in checkpoint["leaves"].items():
        kb = info["kb"]
        amb = info["amb"]
        if info["mode"] == "disk":
            buckets = info["buckets"]
            tmp_dir = info["tmp_dir"]
            if not all(os.path.exists(b) for b in buckets):
                return None
            result[name] = _NodeCapacityKeys(None, amb, kb, buckets, tmp_dir)
        else:
            fpath = info["file"]
            if not os.path.exists(fpath):
                return None
            raw = np.fromfile(fpath, dtype=void_dtype)
            result[name] = _NodeCapacityKeys(
                raw.copy() if raw.size else np.empty((0,), dtype=void_dtype),
                amb,
                kb,
            )

    return result or None


def _delete_leaf_checkpoint(spill_dir: str) -> None:
    """Remove the leaf checkpoint and its auxiliary bin files.

    Called after a successful Phase 2 so stale checkpoints do not
    interfere with future runs in the same spill_dir.
    """
    import os

    checkpoint_path = os.path.join(spill_dir, _LEAF_CHECKPOINT_FNAME)
    if not os.path.exists(checkpoint_path):
        return

    try:
        import json
        with open(checkpoint_path, encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        for info in checkpoint.get("leaves", {}).values():
            if info.get("mode") == "memory":
                fpath = info.get("file", "")
                if fpath and os.path.exists(fpath):
                    os.remove(fpath)
        os.remove(checkpoint_path)
    except Exception:
        pass


def _cleanup_spill_dirs(spill_dir: str) -> None:
    """Remove tts_capacity_* bucket dirs AND stranded spill files in spill_dir.

    Sweeps three kinds of leftover:
    - ``tts_capacity_*/`` prefix-bucket directories;
    - the eviction flat-bin ``tts_capacity_flatbins_*.bin`` (a single file that
      can reach many GB);
    - the memory-mode leaf-checkpoint bins ``capacity_leaf_*.bin``.

    Only a *successful* run deletes the loose files explicitly (in
    ``_BottomUpCapacityComputer._cleanup``), so a directory-only sweep leaked
    what a crashed run left behind — a fresh run over the same spill_dir never
    reclaimed it, and the flat-bin can be many gigabytes. The file globs are
    specific (not every ``tts_capacity_*`` entry) so unrelated files in
    spill_dir are never touched; the checkpoint JSON is left to
    ``_delete_leaf_checkpoint`` (its load already guards missing bins).

    Called at the start of a fresh run (evict prior failed-run leftovers) and
    after a successful Phase 2. Errors are logged as warnings and never
    propagate — a cleanup failure must not abort an otherwise successful run.
    """
    import glob
    import os
    import shutil

    log = logging.getLogger("TaxoTreeSet.Core.Generation.Capacity")
    removed = 0

    for path in glob.glob(os.path.join(spill_dir, "tts_capacity_*")):
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                removed += 1
            else:
                log.warning("[bottom-up] Could not fully remove spill dir %s", path)
        except OSError as exc:
            log.warning("[bottom-up] Could not remove spill dir %s: %s", path, exc)

    for pattern in ("tts_capacity_flatbins_*.bin", "capacity_leaf_*.bin"):
        for path in glob.glob(os.path.join(spill_dir, pattern)):
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError as exc:
                log.warning("[bottom-up] Could not remove spill file %s: %s", path, exc)

    if removed:
        log.info("[bottom-up] Removed %d spill artifact(s) from %s.", removed, spill_dir)
