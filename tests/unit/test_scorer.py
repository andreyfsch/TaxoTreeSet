"""Tests for the open-set benchmark scorer (P11-P4)."""

import json

import pytest

from taxotreeset.benchmark.scorer import (
    classify_outcome,
    hierarchical_prf,
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


class TestHierarchicalPRF:
    """The project's headline measure, pinned.

    It had no implementation in any of the three repositories until 2026-08-17 --
    the harness mentioned F(0.5) only in comments -- so the convention could not be
    audited, and one convention error ran the length of the investigation: 6,950 of
    12,350 eval reads carried an empty true lineage, contributing nothing to recall
    while still charging the precision denominator of whichever tool answered.
    """

    @staticmethod
    def _rows():
        # a -> b -> c, stored LEAF-FIRST as the eval set does
        return [{"read_id": "r1",
                 "true_lineage": [["c", "species"], ["b", "genus"], ["a", "family"]]}]

    def test_exact_commit_is_perfect(self):
        out = hierarchical_prf(self._rows(), {"r1": ("c", "species")})
        assert out["precision"] == 1.0
        assert out["recall"] == 1.0
        assert out["f_beta"] == 1.0

    def test_correct_ancestor_earns_partial_credit(self):
        # committing at b names {a, b}: all correct, but misses c
        out = hierarchical_prf(self._rows(), {"r1": ("b", "genus")})
        assert out["precision"] == 1.0
        assert out["recall"] == pytest.approx(2 / 3)
        # beta=0.5 favours the precise-but-shallow call over a deep wrong one
        assert out["f_beta"] > 0.85

    def test_abstention_costs_recall_only(self):
        out = hierarchical_prf(self._rows(), {})
        assert out["precision"] == 0.0
        assert out["recall"] == 0.0
        assert out["f_beta"] == 0.0

    def test_beta_weights_precision_four_times(self):
        rows = self._rows()
        shallow = hierarchical_prf(rows, {"r1": ("a", "family")}, beta=0.5)
        # same read scored with beta=2 (recall-weighted) must rank it lower
        recall_weighted = hierarchical_prf(rows, {"r1": ("a", "family")}, beta=2.0)
        assert shallow["f_beta"] > recall_weighted["f_beta"]

    def test_empty_truth_raises_rather_than_scoring_zero(self):
        # this is the bug that ran the whole investigation: an unscoreable read
        # must not be silently counted, because it charges precision and credits
        # nothing, penalising exactly the tools that answer everywhere
        with pytest.raises(ValueError, match="empty true_lineage"):
            hierarchical_prf([{"read_id": "r1", "true_lineage": []}],
                             {"r1": ("c", "species")})

    def test_accepts_json_encoded_lineages(self):
        rows = [{"read_id": "r1",
                 "true_lineage": json.dumps([["c", "species"], ["b", "genus"]])}]
        out = hierarchical_prf(rows, {"r1": ("c", "species")})
        assert out["recall"] == 1.0
