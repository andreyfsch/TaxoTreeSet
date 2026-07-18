"""Tests for the compositional-confound audit (dataset/composition.py, P6)."""

import json
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from taxotreeset.dataset import composition


def _seq(gc: float, length: int, rng: random.Random) -> str:
    """A length-bp sequence with expected GC fraction ``gc``."""
    out = []
    for _ in range(length):
        if rng.random() < gc:
            out.append(rng.choice(["G", "C"]))
        else:
            out.append(rng.choice(["A", "T"]))
    return "".join(out)


def _write_parquet(path, seqs, labels):
    pq.write_table(pa.table({"seq": seqs, "class_idx": labels}), str(path))


def _make_head(tmp_path, class_specs, split="train"):
    """Create a head dir. class_specs: list of (class_idx, rank, name, gc)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    seqs, labels = [], []
    classes = []
    for idx, rank, name, gc in class_specs:
        for _ in range(60):
            seqs.append(_seq(gc, 200, rng))
            labels.append(idx)
        classes.append(
            {"class_idx": idx, "taxid": str(1000 + idx), "name": name, "rank": rank}
        )
    _write_parquet(tmp_path / f"{split}.parquet", seqs, labels)
    label_map = {
        "head_taxid": "999", "head_name": "Head", "head_rank": "genus",
        "id2label": {str(c["class_idx"]): c["name"] for c in classes},
        "label2id": {c["name"]: c["class_idx"] for c in classes},
        "classes": classes,
    }
    (tmp_path / "label_map.json").write_text(json.dumps(label_map, indent=2))
    return tmp_path


# ---------------------------------------------------------------------------
# _class_composition
# ---------------------------------------------------------------------------


class TestClassComposition:
    def test_gc_and_acgt_and_length(self):
        comp = composition._class_composition(["ACGTACGT", "ACGTACGT"])
        assert comp["n_rows"] == 2
        assert comp["gc_mean"] == pytest.approx(0.5)
        assert comp["acgt_fraction"] == pytest.approx([0.25, 0.25, 0.25, 0.25])
        assert comp["len_mean"] == pytest.approx(8.0)

    def test_all_gc_sequence(self):
        comp = composition._class_composition(["GCGCGC"])
        assert comp["gc_mean"] == pytest.approx(1.0)

    def test_empty_is_safe(self):
        comp = composition._class_composition([])
        assert comp["n_rows"] == 0
        assert comp["gc_mean"] == 0.0


# ---------------------------------------------------------------------------
# audit_head
# ---------------------------------------------------------------------------


class TestAuditHead:
    def test_flags_gc_skewed_virtual_class(self, tmp_path):
        head = _make_head(tmp_path, [
            (0, "species", "spA", 0.45),
            (1, "species", "spB", 0.55),
            (2, "virtual_low_capacity", "virtual_low_capacity_X", 0.90),
        ])
        report = composition.audit_head(str(head))
        assert report["n_virtual"] == 1
        assert report["n_flagged_virtual"] == 1
        assert "virtual_low_capacity_X" in report["flagged_virtual"]
        virt = next(c for c in report["per_class"] if c["is_virtual"])
        assert virt["gc_z_vs_canonical"] is not None
        assert abs(virt["gc_z_vs_canonical"]) > 2.0

    def test_does_not_flag_comparable_virtual_class(self, tmp_path):
        head = _make_head(tmp_path, [
            (0, "species", "spA", 0.45),
            (1, "species", "spB", 0.55),
            (2, "virtual_misc", "virtual_misc_X", 0.50),
        ])
        report = composition.audit_head(str(head))
        assert report["n_flagged_virtual"] == 0

    def test_binary_head_uses_gap_fallback_and_flags(self, tmp_path):
        # One canonical class (belongs) + one virtual (not_belongs): no z-score
        # spread, so the raw GC gap decides.
        head = _make_head(tmp_path, [
            (0, "virtual_not_belongs", "not_belongs_H", 0.50),
            (1, "species", "spA", 0.90),
        ])
        report = composition.audit_head(str(head))
        assert report["n_flagged_virtual"] == 1
        virt = next(c for c in report["per_class"] if c["is_virtual"])
        assert virt["gc_z_vs_canonical"] is None      # undefined with 1 canonical
        assert virt["gc_gap_vs_canonical"] is not None
        assert abs(virt["gc_gap_vs_canonical"]) > composition._GC_GAP_FLAG

    def test_binary_head_comparable_not_flagged(self, tmp_path):
        head = _make_head(tmp_path, [
            (0, "virtual_not_belongs", "not_belongs_H", 0.50),
            (1, "species", "spA", 0.50),
        ])
        report = composition.audit_head(str(head))
        assert report["n_flagged_virtual"] == 0

    def test_reports_per_class_length(self, tmp_path):
        head = _make_head(tmp_path, [(0, "species", "spA", 0.5), (1, "species", "spB", 0.5)])
        report = composition.audit_head(str(head))
        assert all(c["len_mean"] == pytest.approx(200.0) for c in report["per_class"])

    def test_missing_split_raises(self, tmp_path):
        head = _make_head(tmp_path, [(0, "species", "spA", 0.5), (1, "species", "spB", 0.5)])
        (head / "train.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            composition.audit_head(str(head))


# ---------------------------------------------------------------------------
# enrich_label_map / survey_dataset
# ---------------------------------------------------------------------------


class TestEnrichAndSurvey:
    def test_enrich_writes_summary_atomically(self, tmp_path):
        head = _make_head(tmp_path, [
            (0, "species", "spA", 0.45),
            (1, "species", "spB", 0.55),
            (2, "virtual_low_capacity", "virtual_low_capacity_X", 0.90),
        ])
        report = composition.audit_head(str(head))
        composition.enrich_label_map(str(head), report)
        assert not (head / "label_map.json.tmp").exists()
        after = json.loads((head / "label_map.json").read_text())
        assert after["composition_audit"]["n_flagged_virtual"] == 1
        assert "virtual_low_capacity_X" in after["composition_audit"]["flagged_virtual"]
        # per-class detail is NOT persisted (compact summary only)
        assert "per_class" not in after["composition_audit"]

    def test_survey_enriches_and_returns_rows(self, tmp_path):
        _make_head(tmp_path / "a", [
            (0, "species", "spA", 0.45), (1, "species", "spB", 0.55),
            (2, "virtual_misc", "virtual_misc_X", 0.90),
        ])
        _make_head(tmp_path / "b", [(0, "species", "s1", 0.5), (1, "species", "s2", 0.5)])
        rows = composition.survey_dataset(str(tmp_path))
        assert len(rows) == 2
        flagged_total = sum(r["n_flagged_virtual"] for r in rows)
        assert flagged_total == 1
        lm = json.loads((tmp_path / "a" / "label_map.json").read_text())
        assert "composition_audit" in lm

    def test_survey_no_write_leaves_label_map_untouched(self, tmp_path):
        head = _make_head(tmp_path / "a", [(0, "species", "s1", 0.5), (1, "species", "s2", 0.5)])
        composition.survey_dataset(str(tmp_path), write=False)
        lm = json.loads((head / "label_map.json").read_text())
        assert "composition_audit" not in lm


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCLIWiring:
    def test_parser_has_composition_subcommand(self):
        from taxotreeset.__main__ import build_parser
        parser = build_parser()
        args = parser.parse_args(["composition", "/some/dir"])
        from taxotreeset.cli import composition as comp_cli
        assert args._run is comp_cli.run
        assert args.dataset_dir == "/some/dir"
        assert args.split == "train"
