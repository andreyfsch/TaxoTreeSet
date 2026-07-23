"""Open-set scorer for the clade-holdout benchmark (P11-P4).

Grades a classifier's per-read predictions on the novel-read eval set (P11-P2)
against the expected commit rank ``rho*`` recorded per clade. Each read gets one
outcome:

- **correct** — committed exactly at ``rho*`` (the deepest retained ancestor): the
  right open-set answer, backing off from the novel clade to its known parent;
- **over_commit** — committed *deeper* than ``rho*`` (a confident wrong call on a
  clade that isn't in the label space): the dangerous failure;
- **too_shallow** — committed at a proper ancestor of ``rho*`` (safe but
  under-specified);
- **misroute** — committed off the true path at ``rho*``'s level or shallower (a
  wrong sibling, not deeper);
- **abstain** — no commit.

Rates are aggregated overall, per ``rho*`` rank, and per divergence bin. The scorer
is pure: predictions are ``read_id -> (taxid, rank)``; how a classifier produces
them is out of scope. See ``docs/clade_holdout_benchmark.md``.
"""

import json
from collections import defaultdict
from typing import Any

from taxotreeset.ranks import rank_depth

_OUTCOMES: tuple[str, ...] = (
    "correct", "over_commit", "too_shallow", "misroute", "abstain",
)


def classify_outcome(
    true_lineage: list,
    rho_taxid: str | None,
    rho_rank: str | None,
    pred_taxid: str | None,
    pred_rank: str | None,
) -> str:
    """Classify one read's prediction against ``rho*``.

    Args:
        true_lineage: The genome's lineage as ``[[taxid, rank], ...]`` ordered
            leaf -> root (as stored by the eval-set builder).
        rho_taxid / rho_rank: The expected commit node (deepest retained ancestor).
        pred_taxid / pred_rank: The classifier's committed taxon and its rank;
            ``pred_taxid`` falsy means the classifier abstained.

    Returns:
        One of :data:`_OUTCOMES`.
    """
    if not pred_taxid:
        return "abstain"
    if str(pred_taxid) == str(rho_taxid):
        return "correct"
    # Depth along the true path: root = 0 ... leaf = len - 1.
    root_to_leaf = list(reversed(true_lineage))
    depth = {str(node[0]): i for i, node in enumerate(root_to_leaf)}
    rho_depth = depth.get(str(rho_taxid))
    pred_depth = depth.get(str(pred_taxid))
    if pred_depth is not None and rho_depth is not None:
        # On the true path but not rho* itself.
        return "too_shallow" if pred_depth < rho_depth else "over_commit"
    # Off the true path (a wrong taxon): decide by rank depth.
    pr, rr = rank_depth(pred_rank or ""), rank_depth(rho_rank or "")
    if pr is not None and rr is not None and pr > rr:
        return "over_commit"
    return "misroute"


def _empty_bucket() -> dict[str, int]:
    bucket = {o: 0 for o in _OUTCOMES}
    bucket["n"] = 0
    return bucket


def _rates(bucket: dict[str, int]) -> dict[str, Any]:
    n = bucket["n"] or 1
    out: dict[str, Any] = {"n": bucket["n"]}
    for outcome in _OUTCOMES:
        out[outcome] = bucket[outcome]
        out[f"{outcome}_rate"] = round(bucket[outcome] / n, 4)
    return out


def score_reads(
    eval_rows: list[dict], predictions: dict[str, tuple[str | None, str | None]]
) -> dict[str, Any]:
    """Aggregate per-read outcomes overall, per rho* rank, and per distance bin.

    Args:
        eval_rows: Rows from the eval set (each with ``read_id``, ``true_lineage``,
            ``expected_commit_taxid``, ``expected_commit_rank``, ``distance_bin``).
        predictions: ``read_id -> (pred_taxid, pred_rank)``. A read absent from the
            map counts as an abstention.

    Returns:
        A report dict with ``overall``, ``by_expected_commit_rank``, and
        ``by_distance_bin`` sections, each carrying counts and rates.
    """
    overall = _empty_bucket()
    by_rank: dict[str, dict] = defaultdict(_empty_bucket)
    by_bin: dict[str, dict] = defaultdict(_empty_bucket)
    for row in eval_rows:
        lineage = row["true_lineage"]
        if isinstance(lineage, str):
            lineage = json.loads(lineage)
        pred_taxid, pred_rank = predictions.get(row["read_id"], (None, None))
        outcome = classify_outcome(
            lineage,
            row.get("expected_commit_taxid"),
            row.get("expected_commit_rank"),
            pred_taxid,
            pred_rank,
        )
        rank_key = str(row.get("expected_commit_rank") or "unknown")
        bin_key = str(row.get("distance_bin") or "unknown")
        for bucket in (overall, by_rank[rank_key], by_bin[bin_key]):
            bucket[outcome] += 1
            bucket["n"] += 1
    return {
        "overall": _rates(overall),
        "by_expected_commit_rank": {k: _rates(v) for k, v in sorted(by_rank.items())},
        "by_distance_bin": {k: _rates(v) for k, v in sorted(by_bin.items())},
    }


def report_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a report into per-group rows for CSV export."""
    rows: list[dict[str, Any]] = []
    for group, key in (
        ("overall", None),
        ("by_expected_commit_rank", "expected_commit_rank"),
        ("by_distance_bin", "distance_bin"),
    ):
        section = report[group]
        items = [("", section)] if key is None else section.items()
        for name, stats in items:
            row = {"group": group, "key": name}
            row.update(stats)
            rows.append(row)
    return rows
