"""k-mer separability diagnostic for generated heads.

For each head (a directory holding ``train``/``val``/``test`` parquet files and a
``label_map.json``), a k-mer + logistic-regression baseline is fit on the train
split and scored on the test split. This baseline upper-bounds what a
sequence-only model can learn from a head and, empirically, tracks DNABERT-2
fine-tuning performance closely on the hard, fine-grained heads while
underestimating the easy high-rank ones. The resulting macro-F1 is written back
into each ``label_map.json`` under ``kmer_separability`` so the metric travels
with the dataset.

scikit-learn is required and ships as the optional ``diagnose`` extra
(``pip install taxotreeset[diagnose]``); it is imported lazily so the base
package keeps no hard dependency on it.
"""
import itertools
import json
import logging
import os
from typing import Any

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_K = 4
DEFAULT_MAX_TRAIN = 4000
DEFAULT_MAX_TEST = 3000


def _require_sklearn():
    """Import scikit-learn or raise a helpful error.

    Returns:
        A tuple ``(CountVectorizer, LogisticRegression, normalize, metrics)``.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import normalize
        from sklearn import metrics
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "The separability diagnostic requires scikit-learn. Install it with "
            "'pip install taxotreeset[diagnose]'."
        ) from exc
    return CountVectorizer, LogisticRegression, normalize, metrics


def _kmer_vocabulary(k: int) -> list[str]:
    """Return all ``4**k`` ACGT k-mers in a fixed order."""
    return ["".join(p) for p in itertools.product("ACGT", repeat=k)]


def _read_split(head_dir: str, split: str) -> tuple[list[str], np.ndarray]:
    """Read a parquet split's sequences and class indices.

    Args:
        head_dir: Head directory containing the parquet files.
        split: Split name without extension (``"train"``, ``"val"``, ``"test"``).

    Returns:
        A tuple ``(sequences, class_indices)``.
    """
    table = pq.read_table(
        os.path.join(head_dir, f"{split}.parquet"), columns=["seq", "class_idx"]
    )
    seqs = table.column("seq").to_pylist()
    labels = np.asarray(table.column("class_idx").to_pylist())
    return seqs, labels


def _balanced_subsample(labels: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Indices of a class-balanced subsample of at most ``max_n`` rows."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per = max(1, max_n // len(classes))
    picked: list[int] = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) > per:
            cls_idx = rng.choice(cls_idx, per, replace=False)
        picked.extend(int(i) for i in cls_idx)
    return np.array(sorted(picked))


def compute_head_separability(
    head_dir: str,
    k: int = DEFAULT_K,
    max_train: int = DEFAULT_MAX_TRAIN,
    max_test: int = DEFAULT_MAX_TEST,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a k-mer + logistic-regression baseline and score it on the test split.

    Args:
        head_dir: Head directory with ``train``/``test`` parquet and a
            ``label_map.json``.
        k: k-mer length (feature space is ``4**k`` normalised frequencies).
        max_train: Balanced cap on training rows, for speed.
        max_test: Cap on test rows, for speed.
        seed: Seed for subsampling and the classifier.

    Returns:
        A ``kmer_separability`` metric dict with keys ``k``, ``classifier``,
        ``test_accuracy``, ``test_f1_macro``, ``chance_accuracy``,
        ``accuracy_lift``, ``n_train_sampled`` and ``n_test``. The accuracy
        fields are ``None`` when the train split has fewer than two classes.
    """
    CountVectorizer, LogisticRegression, normalize, metrics = _require_sklearn()

    seqs_tr, y_tr = _read_split(head_dir, "train")
    seqs_te, y_te = _read_split(head_dir, "test")

    n_classes = int(len(np.unique(np.concatenate([y_tr, y_te])))) if len(y_te) else int(len(np.unique(y_tr)))
    chance = round(1.0 / n_classes, 4) if n_classes else None

    if len(np.unique(y_tr)) < 2:
        return {
            "k": k, "classifier": "logistic_regression",
            "test_accuracy": None, "test_f1_macro": None,
            "chance_accuracy": chance, "accuracy_lift": None,
            "n_train_sampled": int(len(y_tr)), "n_test": int(len(y_te)),
        }

    tr_idx = _balanced_subsample(y_tr, max_train, seed)
    te_idx = (_balanced_subsample(y_te, max_test, seed)
              if len(y_te) > max_test else np.arange(len(y_te)))

    vec = CountVectorizer(analyzer="char", ngram_range=(k, k),
                          vocabulary=_kmer_vocabulary(k), lowercase=False)
    x_tr = normalize(vec.transform(s.upper() for s in
                                   (seqs_tr[i] for i in tr_idx)), norm="l1")
    x_te = normalize(vec.transform(s.upper() for s in
                                   (seqs_te[i] for i in te_idx)), norm="l1")

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(x_tr, y_tr[tr_idx])
    pred = clf.predict(x_te)

    acc = round(float(metrics.accuracy_score(y_te[te_idx], pred)), 4)
    f1 = round(float(metrics.f1_score(y_te[te_idx], pred, average="macro")), 4)
    return {
        "k": k, "classifier": "logistic_regression",
        "test_accuracy": acc, "test_f1_macro": f1,
        "chance_accuracy": chance,
        "accuracy_lift": round(acc - chance, 4) if chance is not None else None,
        "n_train_sampled": int(len(tr_idx)), "n_test": int(len(te_idx)),
    }


def enrich_label_map(head_dir: str, metric: dict[str, Any]) -> None:
    """Write the separability metric into a head's ``label_map.json`` in place.

    The existing content is preserved; only the ``kmer_separability`` key is
    added or replaced.

    Args:
        head_dir: Head directory containing ``label_map.json``.
        metric: The metric dict from :func:`compute_head_separability`.
    """
    path = os.path.join(head_dir, "label_map.json")
    with open(path, "r", encoding="utf-8") as fh:
        label_map = json.load(fh)
    label_map["kmer_separability"] = metric
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)


def survey_dataset(
    dataset_dir: str,
    k: int = DEFAULT_K,
    max_train: int = DEFAULT_MAX_TRAIN,
    max_test: int = DEFAULT_MAX_TEST,
    seed: int = 0,
    write: bool = True,
) -> list[dict[str, Any]]:
    """Compute separability for every head under ``dataset_dir``.

    Walks the tree for ``label_map.json`` files, scores each head, and (unless
    ``write`` is False) enriches each ``label_map.json`` in place.

    Args:
        dataset_dir: Root of a generated dataset tree.
        k: k-mer length.
        max_train: Balanced cap on training rows per head.
        max_test: Cap on test rows per head.
        seed: Seed for subsampling and the classifier.
        write: When True, persist the metric into each ``label_map.json``.

    Returns:
        One row dict per scored head, with the head metadata and metric fields,
        suitable for building an aggregate table.
    """
    head_dirs = sorted(
        os.path.dirname(p)
        for p in _iter_label_maps(dataset_dir)
    )
    rows: list[dict[str, Any]] = []
    for head_dir in head_dirs:
        with open(os.path.join(head_dir, "label_map.json"), "r",
                  encoding="utf-8") as fh:
            lm = json.load(fh)
        try:
            metric = compute_head_separability(
                head_dir, k=k, max_train=max_train, max_test=max_test, seed=seed)
        except FileNotFoundError:
            logger.warning("Skipping %s: missing a parquet split", head_dir)
            continue
        if write:
            enrich_label_map(head_dir, metric)
        rows.append({
            "head_taxid": lm.get("head_taxid", os.path.basename(head_dir)),
            "head_name": lm.get("head_name", ""),
            "head_rank": lm.get("head_rank", ""),
            "n_classes": len(lm.get("classes", [])),
            **metric,
        })
    logger.info("Scored %d heads under %s", len(rows), dataset_dir)
    return rows


def _iter_label_maps(dataset_dir: str):
    """Yield paths to every ``label_map.json`` under ``dataset_dir``."""
    for root, _dirs, files in os.walk(dataset_dir):
        if "label_map.json" in files:
            yield os.path.join(root, "label_map.json")
