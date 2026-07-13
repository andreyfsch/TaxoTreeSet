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
    """Remove all tts_capacity_* directories inside spill_dir.

    Called in two situations:
    - At the start of a fresh run (no valid checkpoint found) to evict
      directories left behind by previous failed runs.
    - After a successful Phase 2 to remove directories created by the
      current run once their data has been merged and is no longer needed.

    Errors are logged as warnings and never propagate — a failure to clean
    up is unfortunate but must not abort an otherwise successful pipeline.
    """
    import glob
    import os
    import shutil

    pattern = os.path.join(spill_dir, "tts_capacity_*")
    removed = 0
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path, ignore_errors=True)
            if not os.path.exists(path):
                removed += 1
            else:
                logging.getLogger("TaxoTreeSet.Core.Generation.Capacity").warning(
                    "[bottom-up] Could not fully remove spill dir %s", path
                )
        except OSError as exc:
            logging.getLogger("TaxoTreeSet.Core.Generation.Capacity").warning(
                "[bottom-up] Could not remove spill dir %s: %s", path, exc
            )
    if removed:
        logging.getLogger("TaxoTreeSet.Core.Generation.Capacity").info(
            "[bottom-up] Removed %d spill dir(s) from %s.", removed, spill_dir
        )
