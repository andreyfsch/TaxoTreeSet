"""Tests for the open-set eval-set builder (benchmark, P11-P2)."""

import json
import random
from unittest.mock import patch

import pyarrow.parquet as pq

from taxotreeset.benchmark.eval_set import (
    _header_index,
    build_eval_reads,
    build_eval_set,
)

_READ = "taxotreeset.benchmark.eval_set._read_single_sequence"
_SEQ = "".join(random.Random(0).choices("ACGT", k=3000))

_ACC = {
    "acc1": {
        "taxid": "T1", "local_path": "/vault/seq.lmdb",
        "headers": [{"id": "H1"}, {"id": "H2"}],
    }
}
_LIN = {
    "T1": [
        {"taxid": "T1", "rank": "species"},
        {"taxid": "G1", "rank": "genus"},
        {"taxid": "F1", "rank": "family"},
    ]
}
_ENTRIES = [
    {
        "taxid": "G1", "expected_commit_taxid": "F1",
        "expected_commit_rank": "family", "distance_bin": "ANI 85-90%",
        "member_headers": ["H1", "H2"],
    }
]


class TestHeaderIndex:
    def test_maps_header_to_taxon_and_path(self):
        idx = _header_index(_ACC)
        assert idx["H1"] == ("T1", "/vault/seq.lmdb")
        assert idx["H2"] == ("T1", "/vault/seq.lmdb")


class TestBuildEvalReads:
    def test_labels_each_read_with_lineage_rho_and_bin(self):
        with patch(_READ, return_value=_SEQ):
            rows = build_eval_reads(
                _ENTRIES, _ACC, _LIN,
                read_length=150, reads_per_genome=5, seed=0)
        assert len(rows) == 10  # 2 genomes x 5 reads
        assert {r["source_header"] for r in rows} == {"H1", "H2"}
        r = rows[0]
        assert len(r["seq"]) == 150
        assert r["true_leaf_taxid"] == "T1"
        assert r["held_out_taxid"] == "G1"
        assert r["expected_commit_taxid"] == "F1"
        assert r["expected_commit_rank"] == "family"
        assert r["distance_bin"] == "ANI 85-90%"
        lineage = json.loads(r["true_lineage"])
        assert ["G1", "genus"] in lineage and ["F1", "family"] in lineage

    def test_is_seed_deterministic(self):
        with patch(_READ, return_value=_SEQ):
            a = build_eval_reads(_ENTRIES, _ACC, _LIN, reads_per_genome=5, seed=3)
            b = build_eval_reads(_ENTRIES, _ACC, _LIN, reads_per_genome=5, seed=3)
        assert [x["seq"] for x in a] == [x["seq"] for x in b]

    def test_skips_header_absent_from_registry(self):
        entries = [{**_ENTRIES[0], "member_headers": ["HX"]}]
        with patch(_READ, return_value=_SEQ):
            assert build_eval_reads(entries, _ACC, _LIN) == []

    def test_skips_unreadable_and_too_short(self):
        with patch(_READ, return_value=""):
            assert build_eval_reads(_ENTRIES, _ACC, _LIN) == []
        with patch(_READ, return_value="ACGT"):  # shorter than read_length
            assert build_eval_reads(_ENTRIES, _ACC, _LIN, read_length=150) == []


class TestBuildEvalSet:
    def test_writes_parquet_from_manifest(self, tmp_path):
        manifest = tmp_path / "benchmark_manifest_viruses.json"
        manifest.write_text(json.dumps({"holdout": _ENTRIES}), encoding="utf-8")
        out = tmp_path / "eval.parquet"
        with patch(_READ, return_value=_SEQ):
            n_reads, n_clades = build_eval_set(
                str(manifest), _ACC, _LIN, str(out),
                read_length=150, reads_per_genome=4, seed=0)
        assert n_clades == 1
        assert n_reads == 8  # 2 genomes x 4 reads
        table = pq.read_table(str(out))
        assert set(table.column_names) >= {
            "seq", "true_leaf_taxid", "held_out_taxid",
            "expected_commit_taxid", "expected_commit_rank", "distance_bin",
        }
        assert table.num_rows == 8
