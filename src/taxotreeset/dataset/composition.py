"""Compositional-confound audit for generated heads (backlog P6).

Virtual buckets (``virtual_low_capacity`` / ``virtual_rare_taxa`` /
``virtual_misc`` / ``virtual_reject`` / ``virtual_not_belongs``) aggregate
taxonomically heterogeneous taxa, so a head could separate a virtual class from
the canonical ones for **non-phylogenetic** reasons — a compositional artifact
(GC or nucleotide-frequency skew) rather than genuine clade signal. This
diagnostic reports, per head, the per-class sequence length and nucleotide
composition, and flags virtual classes whose GC content is an outlier relative to
the head's canonical classes.

Length is reported as a sanity check: ``extract_subseqs`` draws each window's
length uniformly in ``[min_len, max_len]`` regardless of class, so per-class
length distributions should be comparable — a divergence would flag a regression
of the length-confound fix.

numpy-only; unlike the separability diagnostic it needs no scikit-learn.
"""

import json
import logging
import os
from typing import Any

import numpy as np

from taxotreeset.dataset.separability import _iter_label_maps, _read_split

logger = logging.getLogger(__name__)

_VIRTUAL_RANK_PREFIX = "virtual_"
# |z| above which a virtual class's mean GC is flagged as a compositional outlier
# relative to the canonical classes (used when the head has >= 2 canonical classes).
_GC_Z_FLAG = 2.0
# Fallback for binary / single-canonical-class heads where a z-score is undefined:
# flag a virtual class whose mean GC differs from the canonical reference by more
# than this many GC-fraction points.
_GC_GAP_FLAG = 0.05


def _class_composition(seqs: list[str]) -> dict[str, Any]:
    """Summarize length and nucleotide composition for one class's sequences.

    Args:
        seqs: The class's subsequences (already uppercased at extraction).

    Returns:
        Dict with ``n_rows``, length mean/std, GC mean/std, and the mean A/C/G/T
        fractions across all bases in the class.
    """
    lengths = np.array([len(s) for s in seqs], dtype=float)
    gc_per_seq: list[float] = []
    base_counts = np.zeros(4, dtype=float)  # A, C, G, T
    total_bases = 0
    for seq in seqs:
        s = seq.upper()
        a, c, g, t = s.count("A"), s.count("C"), s.count("G"), s.count("T")
        base_counts += (a, c, g, t)
        n = len(s)
        total_bases += n
        if n:
            gc_per_seq.append((g + c) / n)
    gc = np.array(gc_per_seq) if gc_per_seq else np.array([0.0])
    acgt = (base_counts / total_bases) if total_bases else np.zeros(4)
    return {
        "n_rows": int(len(seqs)),
        "len_mean": float(lengths.mean()) if lengths.size else 0.0,
        "len_std": float(lengths.std()) if lengths.size else 0.0,
        "gc_mean": float(gc.mean()),
        "gc_std": float(gc.std()),
        "acgt_fraction": [float(x) for x in acgt],
    }


def audit_head(head_dir: str, split: str = "train") -> dict[str, Any]:
    """Audit one head for a compositional confound in its virtual classes.

    Groups the split's rows by class, summarizes each class's length and
    composition, then compares every virtual class's mean GC against the head's
    canonical classes: a z-score when there are >= 2 canonical classes, else a raw
    GC gap (binary / single-canonical heads). Virtual classes beyond the threshold
    are flagged as possible non-phylogenetic separators.

    Args:
        head_dir: Head directory with a ``<split>.parquet`` and ``label_map.json``.
        split: Which split to read (default ``"train"``).

    Returns:
        A report dict with head metadata, the per-class composition list, the
        canonical GC reference, and the list of flagged virtual class names.
    """
    seqs, labels = _read_split(head_dir, split)
    with open(os.path.join(head_dir, "label_map.json"), encoding="utf-8") as fh:
        label_map = json.load(fh)
    class_meta = {int(c["class_idx"]): c for c in label_map.get("classes", [])}

    per_class: list[dict[str, Any]] = []
    for cls in sorted({int(x) for x in labels.tolist()}):
        rows = [seqs[i] for i, lab in enumerate(labels.tolist()) if int(lab) == cls]
        comp = _class_composition(rows)
        meta = class_meta.get(cls, {})
        rank = meta.get("rank", "unknown")
        comp.update({
            "class_idx": cls,
            "name": meta.get("name", str(cls)),
            "rank": rank,
            "is_virtual": str(rank).startswith(_VIRTUAL_RANK_PREFIX),
        })
        per_class.append(comp)

    canonical_gc = np.array(
        [c["gc_mean"] for c in per_class if not c["is_virtual"]]
    )
    ref_mean = float(canonical_gc.mean()) if canonical_gc.size else 0.0
    ref_std = float(canonical_gc.std()) if canonical_gc.size else 0.0

    flagged: list[str] = []
    for c in per_class:
        c["gc_z_vs_canonical"] = None
        c["gc_gap_vs_canonical"] = None
        if not c["is_virtual"] or canonical_gc.size == 0:
            continue
        gap = c["gc_mean"] - ref_mean
        c["gc_gap_vs_canonical"] = float(gap)
        if canonical_gc.size >= 2 and ref_std > 0:
            z = gap / ref_std
            c["gc_z_vs_canonical"] = float(z)
            is_outlier = abs(z) > _GC_Z_FLAG
        else:
            # binary / single canonical class: no meaningful spread for a z-score
            is_outlier = abs(gap) > _GC_GAP_FLAG
        if is_outlier:
            c["gc_confound_flag"] = True
            flagged.append(c["name"])

    return {
        "head_taxid": label_map.get("head_taxid", os.path.basename(head_dir)),
        "head_name": label_map.get("head_name", ""),
        "head_rank": label_map.get("head_rank", ""),
        "split": split,
        "n_classes": len(per_class),
        "n_virtual": sum(1 for c in per_class if c["is_virtual"]),
        "canonical_gc_mean": ref_mean,
        "canonical_gc_std": ref_std,
        "n_flagged_virtual": len(flagged),
        "flagged_virtual": flagged,
        "per_class": per_class,
    }


def enrich_label_map(head_dir: str, report: dict[str, Any]) -> None:
    """Write a compact composition-audit summary into a head's ``label_map.json``.

    Stores only the head-level summary (flags + canonical GC reference), not the
    full per-class table, under ``composition_audit``. The write is atomic (temp
    sibling + ``os.replace``), matching the separability diagnostic.

    Args:
        head_dir: Head directory containing ``label_map.json``.
        report: The dict returned by :func:`audit_head`.
    """
    summary = {
        "split": report["split"],
        "n_virtual": report["n_virtual"],
        "n_flagged_virtual": report["n_flagged_virtual"],
        "flagged_virtual": report["flagged_virtual"],
        "canonical_gc_mean": report["canonical_gc_mean"],
        "canonical_gc_std": report["canonical_gc_std"],
    }
    path = os.path.join(head_dir, "label_map.json")
    with open(path, encoding="utf-8") as fh:
        label_map = json.load(fh)
    label_map["composition_audit"] = summary
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)
    os.replace(tmp_path, path)


def survey_dataset(
    dataset_dir: str,
    split: str = "train",
    write: bool = True,
) -> list[dict[str, Any]]:
    """Audit every head under ``dataset_dir`` for a compositional confound.

    Walks the tree for ``label_map.json`` files, audits each head, and (unless
    ``write`` is False) records a compact summary into each ``label_map.json``.

    Args:
        dataset_dir: Root of a generated dataset tree.
        split: Which split to read per head.
        write: When True, persist the summary into each ``label_map.json``.

    Returns:
        One summary row dict per audited head, suitable for an aggregate table.
        Heads whose ``<split>`` parquet is missing are skipped with a warning.
    """
    rows: list[dict[str, Any]] = []
    for label_map_path in sorted(_iter_label_maps(dataset_dir)):
        head_dir = os.path.dirname(label_map_path)
        try:
            report = audit_head(head_dir, split=split)
        except FileNotFoundError:
            logger.warning("Skipping %s: missing the %s split", head_dir, split)
            continue
        if write:
            enrich_label_map(head_dir, report)
        rows.append({
            "head_taxid": report["head_taxid"],
            "head_name": report["head_name"],
            "head_rank": report["head_rank"],
            "n_classes": report["n_classes"],
            "n_virtual": report["n_virtual"],
            "n_flagged_virtual": report["n_flagged_virtual"],
            "flagged_virtual": ";".join(report["flagged_virtual"]),
            "canonical_gc_mean": round(report["canonical_gc_mean"], 4),
        })
    n_flagged = sum(1 for r in rows if r["n_flagged_virtual"])
    logger.info(
        "Composition audit: %d heads, %d with a flagged virtual class.",
        len(rows), n_flagged,
    )
    return rows
