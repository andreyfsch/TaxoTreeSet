"""GC-only cascade: the same traversal, with every head replaced by a GC rule.

Why this exists. A GC-content threshold, fitted per head on that head's own train
split, recovers a median 76% of the fine-tuned DNABERT-2 head's test f1_macro
(measured across the 42 trained pilot heads; in five of them the threshold WINS).
Per-head f1 cannot say whether that matters, because the cascade's accuracy is not
a monotone function of its heads' accuracy — what prunes is rejection by siblings,
not confidence. So the question has to be asked end-to-end: does the whole cascade
beat a cascade of the same shape whose heads are one line of arithmetic?

The comparison is only meaningful if NOTHING else differs. This module therefore
injects a NodeInferer rather than reimplementing traversal: the registry, the
reject margin, the false-reject weighting, the deepest-survivor rule, true-path
consistency and passthrough collapsing are the production ones, untouched. The
only substituted component is the per-node decision.

Calibration. BFSTraverser computes softmax(logits / node.temperature), where the
temperature was fitted for the DNABERT-2 head. Applying it to GC log-odds would
distort them, so infer() pre-multiplies by that same temperature and the division
cancels — the GC head is served at its own logistic calibration. Handicapping the
baseline would make a win meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phylocascadeglm._inferer import NodeInferer

_GC_BASES = ("G", "C", "g", "c")


def gc_content(seq: str) -> float:
    """Fraction of G/C bases in ``seq`` (0.0 for an empty string)."""
    if not seq:
        return 0.0
    return sum(seq.count(b) for b in _GC_BASES) / len(seq)


def _fit_logistic_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit ``P(y=1) = sigmoid(a*x + b)`` by Newton-Raphson on one feature.

    A closed dependency-free fit rather than sklearn, so the baseline can run
    anywhere the bundle does. Returns ``(0.0, 0.0)`` — an uninformative head that
    always emits p=0.5 — if the split is degenerate (one class, or no variance).

    Args:
        x: Feature values, shape (n,).
        y: Binary labels in {0, 1}, shape (n,).

    Returns:
        ``(a, b)``: slope and intercept in log-odds space.
    """
    if len(x) < 10 or len(set(y.tolist())) < 2 or float(np.std(x)) < 1e-9:
        return 0.0, 0.0
    # Standardise for conditioning, then map the coefficients back.
    mu, sd = float(np.mean(x)), float(np.std(x))
    z = (x - mu) / sd
    a = b = 0.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(a * z + b)))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        r = y - p
        # 2x2 normal equations for [a, b].
        h11 = float(np.sum(w * z * z))
        h12 = float(np.sum(w * z))
        h22 = float(np.sum(w))
        g1 = float(np.sum(r * z))
        g2 = float(np.sum(r))
        det = h11 * h22 - h12 * h12
        if abs(det) < 1e-12:
            break
        da = (h22 * g1 - h12 * g2) / det
        db = (h11 * g2 - h12 * g1) / det
        a += da
        b += db
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return a / sd, b - a * mu / sd


def fit_gc_heads(
    taxids: list[str],
    dataset_index: dict[str, Path],
    *,
    max_rows: int = 40000,
    verbose: bool = True,
) -> dict[str, tuple[float, float]]:
    """Fit one GC logistic head per taxid, on that head's OWN train split.

    Fitting on train only mirrors what the fine-tuned head saw; val and test stay
    untouched so the end-to-end comparison is honest.

    Args:
        taxids: Heads to fit (typically ``registry`` keys).
        dataset_index: taxid -> directory holding ``train.parquet``.
        max_rows: Row cap per head, for speed.
        verbose: Print per-head progress.

    Returns:
        taxid -> ``(slope, intercept)`` in log-odds space, for P(belongs).
    """
    import pyarrow.parquet as pq

    out: dict[str, tuple[float, float]] = {}
    for i, taxid in enumerate(taxids, 1):
        d = dataset_index.get(str(taxid))
        train = d / "train.parquet" if d else None
        if train is None or not train.exists():
            out[str(taxid)] = (0.0, 0.0)
            continue
        tbl = pq.read_table(train, columns=["seq", "class_idx"])
        seqs = tbl.column("seq").to_pylist()[:max_rows]
        labs = tbl.column("class_idx").to_pylist()[:max_rows]
        x = np.array([gc_content(s) for s in seqs], dtype=float)
        y = np.array(labs, dtype=float)
        out[str(taxid)] = _fit_logistic_1d(x, y)
        if verbose and i % 25 == 0:
            print(f"  fitted {i}/{len(taxids)} GC heads", flush=True)
    return out


def build_dataset_index(datasets_root: Path) -> dict[str, Path]:
    """Map taxid -> dataset directory by scanning for ``train.parquet``.

    The all-ranks dataset tree nests one directory per taxon, so the taxid is the
    directory name. Later matches overwrite earlier ones; taxids are unique in
    the tree, so order does not matter.
    """
    index: dict[str, Path] = {}
    for train in Path(datasets_root).rglob("train.parquet"):
        index[train.parent.name] = train.parent
    return index


class GCInferer(NodeInferer):
    """A NodeInferer whose verdict is a per-head logistic on GC content.

    Emits logits shaped like the head it replaces: the reject bucket's class_idx
    gets 0.0 and the taxon's gets the log-odds, so the traverser's own
    ``_binary_probs`` reads them exactly as it reads a real head's.
    """

    def __init__(self, registry, coefs: dict[str, tuple[float, float]]) -> None:
        """
        Args:
            registry: The bundle's ``AdapterRegistry`` — supplies each node's
                class layout and temperature.
            coefs: taxid -> ``(slope, intercept)`` from :func:`fit_gc_heads`.
        """
        self._registry = registry
        self._coefs = coefs

    def infer(self, windows: list[str], taxid: str) -> np.ndarray:
        node = self._registry.get(str(taxid))
        n = int(getattr(node, "num_labels", 2))
        logits = np.zeros(n, dtype=float)

        a, b = self._coefs.get(str(taxid), (0.0, 0.0))
        gc = float(np.mean([gc_content(w) for w in windows])) if windows else 0.0
        z = a * gc + b

        # Cancel the traverser's division by the DNABERT-2 temperature, so the
        # GC head is scored at its own calibration rather than someone else's.
        z *= float(getattr(node, "temperature", 1.0) or 1.0)

        belong = next((c for c in node.classes if not c.is_bucket), None)
        if belong is not None and belong.class_idx < n:
            logits[belong.class_idx] = z
        return logits

    def close(self) -> None:
        return None


def load_or_fit(
    cache_path: Path,
    taxids: list[str],
    dataset_index: dict[str, Path],
) -> dict[str, tuple[float, float]]:
    """Fit the GC heads, caching to ``cache_path`` so repeat runs are instant."""
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}
    coefs = fit_gc_heads(taxids, dataset_index)
    cache_path.write_text(json.dumps({k: list(v) for k, v in coefs.items()}, indent=1))
    return coefs
