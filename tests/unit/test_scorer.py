"""Tests for the open-set benchmark scorer (P11-P4)."""

import json

from taxotreeset.benchmark.scorer import (
    classify_outcome,
    report_csv_rows,
    score_reads,
)

# true lineage leaf -> root; rho* = family F1
_LIN = [
    ["S1", "species"],
    ["G1", "genus"],
    ["F1", "family"],
    ["SK", "superkingdom"],
]


class TestClassifyOutcome:
    def _c(self, pred_taxid, pred_rank):
        return classify_outcome(_LIN, "F1", "family", pred_taxid, pred_rank)

    def test_abstain_when_no_prediction(self):
        assert self._c(None, None) == "abstain"

    def test_correct_when_commits_at_rho(self):
        assert self._c("F1", "family") == "correct"

    def test_too_shallow_when_ancestor_of_rho(self):
        assert self._c("SK", "superkingdom") == "too_shallow"

    def test_over_commit_on_true_path_below_rho(self):
        # the true genus/species live under the held-out clade -> not a valid label
        assert self._c("G1", "genus") == "over_commit"
        assert self._c("S1", "species") == "over_commit"

    def test_over_commit_off_path_deeper_than_rho(self):
        # a wrong genus (retained sibling) is deeper than rho* (family)
        assert self._c("GX", "genus") == "over_commit"

    def test_misroute_off_path_same_rank(self):
        assert self._c("FX", "family") == "misroute"

    def test_misroute_off_path_shallower(self):
        assert self._c("OX", "order") == "misroute"


def _rows():
    common = {
        "true_lineage": _LIN,
        "expected_commit_taxid": "F1",
        "expected_commit_rank": "family",
    }
    return [
        {"read_id": "r1", "distance_bin": "ANI 85-90%", **common},
        {"read_id": "r2", "distance_bin": "ANI 85-90%", **common},
        {"read_id": "r3", "distance_bin": "ANI<85%", **common},
    ]


class TestScoreReads:
    def test_aggregates_overall_rank_and_bin(self):
        preds = {"r1": ("F1", "family"), "r2": ("GX", "genus")}  # r3 -> abstain
        rep = score_reads(_rows(), preds)
        ov = rep["overall"]
        assert ov["n"] == 3
        assert (ov["correct"], ov["over_commit"], ov["abstain"]) == (1, 1, 1)
        assert ov["correct_rate"] == round(1 / 3, 4)
        assert rep["by_expected_commit_rank"]["family"]["n"] == 3
        assert rep["by_distance_bin"]["ANI 85-90%"]["n"] == 2
        assert rep["by_distance_bin"]["ANI<85%"]["abstain"] == 1

    def test_accepts_json_string_lineage(self):
        rows = [{**_rows()[0], "true_lineage": json.dumps(_LIN)}]
        rep = score_reads(rows, {"r1": ("F1", "family")})
        assert rep["overall"]["correct"] == 1

    def test_missing_prediction_counts_as_abstain(self):
        rep = score_reads(_rows(), {})
        assert rep["overall"]["abstain"] == 3


class TestReportCsv:
    def test_flattens_all_groups(self):
        rep = score_reads(_rows(), {"r1": ("F1", "family")})
        rows = report_csv_rows(rep)
        assert {r["group"] for r in rows} == {
            "overall", "by_expected_commit_rank", "by_distance_bin"}
        assert all("correct_rate" in r and "n" in r for r in rows)
