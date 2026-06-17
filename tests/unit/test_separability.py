"""Tests for the k-mer separability diagnostic and its CLI subcommand."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxotreeset.dataset import separability

sklearn = pytest.importorskip("sklearn")


def _write_parquet(path, seqs, labels):
    pq.write_table(pa.table({"seq": seqs, "class_idx": labels}), str(path))


def _make_head(tmp_path, *, separable=True, single_class=False):
    """Create a head dir with train/val/test parquet and a label_map.json.

    When ``separable`` the two classes have disjoint k-mer content (class 0 is
    A/T-rich, class 1 is G/C-rich), so the baseline should score near-perfectly.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    seqs, labels = [], []
    classes = [0] if single_class else [0, 1]
    for cls in classes:
        bases = "ACGT" if not separable else ("AT" if cls == 0 else "GC")
        for _ in range(120):
            seqs.append("".join(rng.choice(list(bases), size=200)))
            labels.append(cls)
    for split in ("train", "val", "test"):
        _write_parquet(tmp_path / f"{split}.parquet", seqs, labels)

    label_map = {
        "head_taxid": "999", "head_name": "TestHead", "head_rank": "genus",
        "id2label": {"0": "A", "1": "B"},
        "label2id": {"A": 0, "B": 1},
        "classes": [
            {"class_idx": 0, "taxid": "10", "name": "A", "rank": "species"},
            {"class_idx": 1, "taxid": "11", "name": "B", "rank": "species"},
        ],
    }
    (tmp_path / "label_map.json").write_text(json.dumps(label_map, indent=2))
    return tmp_path


class TestHelpers:
    def test_kmer_vocabulary_size_and_order(self):
        vocab = separability._kmer_vocabulary(2)
        assert len(vocab) == 16
        assert vocab[0] == "AA" and vocab[-1] == "TT"

    def test_balanced_subsample_caps_and_balances(self):
        labels = np.array([0] * 100 + [1] * 100)
        idx = separability._balanced_subsample(labels, max_n=40, seed=0)
        picked = labels[idx]
        assert len(idx) == 40
        assert (picked == 0).sum() == (picked == 1).sum() == 20


class TestComputeHeadSeparability:
    def test_separable_classes_score_high(self, tmp_path):
        head = _make_head(tmp_path, separable=True)
        m = separability.compute_head_separability(str(head), k=3)
        assert m["test_f1_macro"] > 0.9
        assert m["chance_accuracy"] == 0.5
        assert m["accuracy_lift"] > 0.3
        assert m["k"] == 3

    def test_single_class_train_returns_none_metrics(self, tmp_path):
        head = _make_head(tmp_path, single_class=True)
        m = separability.compute_head_separability(str(head))
        assert m["test_f1_macro"] is None
        assert m["test_accuracy"] is None
        assert m["n_test"] > 0

    def test_missing_split_raises(self, tmp_path):
        head = _make_head(tmp_path)
        (tmp_path / "test.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            separability.compute_head_separability(str(head))


class TestEnrichLabelMap:
    def test_enrich_adds_key_and_preserves_content(self, tmp_path):
        head = _make_head(tmp_path)
        before = json.loads((tmp_path / "label_map.json").read_text())
        metric = {"k": 4, "test_f1_macro": 0.5}
        separability.enrich_label_map(str(head), metric)
        after = json.loads((tmp_path / "label_map.json").read_text())
        assert after["kmer_separability"] == metric
        for key in before:
            assert after[key] == before[key]

    def test_enrich_is_idempotent(self, tmp_path):
        head = _make_head(tmp_path)
        separability.enrich_label_map(str(head), {"k": 4, "v": 1})
        separability.enrich_label_map(str(head), {"k": 4, "v": 2})
        after = json.loads((tmp_path / "label_map.json").read_text())
        assert after["kmer_separability"]["v"] == 2


class TestSurveyDataset:
    def test_survey_enriches_and_returns_rows(self, tmp_path):
        head_a = _make_head(tmp_path / "a", separable=True)
        head_b = _make_head(tmp_path / "b", separable=False)
        rows = separability.survey_dataset(str(tmp_path), k=3)
        assert len(rows) == 2
        assert all("test_f1_macro" in r and "head_taxid" in r for r in rows)
        # Metric persisted to disk.
        lm = json.loads((head_a / "label_map.json").read_text())
        assert "kmer_separability" in lm

    def test_survey_no_write_leaves_label_map_untouched(self, tmp_path):
        head = _make_head(tmp_path / "a", separable=True)
        rows = separability.survey_dataset(str(tmp_path), k=3, write=False)
        assert len(rows) == 1
        lm = json.loads((head / "label_map.json").read_text())
        assert "kmer_separability" not in lm


class TestCLIWiring:
    def test_parser_has_separability_subcommand(self):
        from taxotreeset.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["separability", "/some/dir"])
        from taxotreeset.cli import separability as sep_cli
        assert args._run is sep_cli.run
        assert args.dataset_dir == "/some/dir"
        assert args.k == separability.DEFAULT_K
