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


def load_ncbi_parents(nodes_dmp: "str | Path") -> dict[str, str]:
    """Read ``taxid -> parent_taxid`` from an NCBI ``nodes.dmp``.

    Supplies :func:`hierarchical_prf` with a COMPLETE ancestor source. Without it
    the parent map can only be inferred from the eval set's own lineages, which
    silently destroys hierarchical credit for any prediction that lands off the
    true path -- see that function's "WHY THE TAXONOMY IS REQUIRED".
    """
    parents: dict[str, str] = {}
    with open(nodes_dmp, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            fields = [x.strip() for x in line.split("|")]
            if len(fields) >= 2 and fields[0]:
                parents[fields[0]] = fields[1]
    return parents


def hierarchical_prf(
    eval_rows: list[dict],
    predictions: dict[str, tuple[str | None, str | None]],
    beta: float = 0.5,
    taxonomy_parents: dict[str, str] | None = None,
    strict: bool = True,
) -> dict[str, float]:
    """Micro-averaged hierarchical precision/recall/F_beta over lineage sets.

    The project's headline measure. A prediction is credited for every taxon it
    names that lies on the true lineage, so a commit at a correct ANCESTOR earns
    partial credit where exact match scores zero — three pilot heads land on a
    correct ancestor 93-97% of the time. beta=0.5 weights precision four times
    recall: for this tool abstaining beats misrouting.

    Predicted set = the committed taxon plus its ancestors, reconstructed from the
    parent links implied by the eval set's own lineages. True set = the read's
    lineage. Micro-averaged: intersections, predicted sizes and true sizes are
    summed over reads before the ratio, so every taxon counts once regardless of
    how deep its read's lineage runs.

    WHY THIS LIVES HERE. Until 2026-08-17 the project's headline numbers came from
    an implementation that is in no file of any of the three repositories -- the
    harness only ever mentioned F(0.5) in comments. Nobody could audit the
    convention, and one convention error had already gone unnoticed for the whole
    investigation: 6,950 of 12,350 eval reads carried an EMPTY true lineage (a
    silent ``lineages.get(taxid, [])`` in the eval builder), which contributed
    nothing to recall while still charging the precision denominator of any tool
    that answered. Tools that abstain were spared; the cascade, which answers
    everywhere, absorbed the whole penalty. Repairing the truth moved
    PhyloCascadeGLM from 0.353 to 0.685 and Kraken2 from 0.802 to 0.653.

    A read whose true lineage is empty raises rather than scoring silently: an
    empty truth cannot be scored, and treating it as zero is what produced the
    error above.

    WHY THE TAXONOMY IS REQUIRED. The parent map used to be inferred only from the
    lineages present in ``eval_rows``. A predicted taxon that is NOT on some eval
    read's true lineage therefore had no reconstructable ancestors, and
    ``ancestors()`` returned it as a singleton -- so a tool that answered with the
    right family but the wrong species scored zero instead of partial credit. The
    bug is invisible in aggregate and strongly directional: the cascade commits at
    nodes of the true path (which the eval lineages do contain) and kept its
    credit, while exact-match baselines that miss the species collapsed to
    singletons. Measured on the 1,200-read sample at 150 bp, divergence band
    >=0.10: fixing ONLY the prediction side moves Kraken2 from 0.257 to 0.611 and
    Kaiju from 0.581 to 0.716, and leaves PhyloCascadeGLM at 0.675 -- i.e. it
    erased a lead that did not exist. Pass ``taxonomy_parents``
    (:func:`load_ncbi_parents`); ``strict`` refuses to score without it.

    Args:
        eval_rows: Eval rows carrying ``read_id`` and ``true_lineage``.
        predictions: ``read_id -> (taxid, rank)``. Absent reads count as
            abstentions and contribute to the true side only.
        beta: F-measure beta. 0.5 (default) weights precision 4x recall.
        taxonomy_parents: complete ``taxid -> parent`` map. Required for correct
            hierarchical credit on off-path predictions.
        strict: raise when ``taxonomy_parents`` is absent. Set False only to
            reproduce a historical (biased) number, never to report a new one.

    Returns:
        ``{"precision", "recall", "f_beta", "beta", "n_reads", "off_path_singletons"}``.
        ``off_path_singletons`` counts predictions whose ancestors could not be
        reconstructed -- it must be 0 for the score to mean what the name says.

    Raises:
        ValueError: if any row's true lineage is empty, or if ``strict`` and no
            taxonomy was supplied.
    """
    if taxonomy_parents is None and strict:
        raise ValueError(
            "hierarchical_prf needs taxonomy_parents (see load_ncbi_parents); "
            "without it off-path predictions collapse to singletons and the "
            "score is biased toward tools that commit on the true path"
        )
    def _lineage(row: dict) -> list[str]:
        lin = row["true_lineage"]
        if isinstance(lin, str):
            lin = json.loads(lin)
        return [str(t) for t, _rank in lin]

    # Os elos vindos das eval_rows ficam num dict PEQUENO e a taxonomia completa
    # entra como fallback. Copiar a taxonomia (2,9 M entradas) a cada chamada
    # dominava o custo -- um bootstrap de 21.600 chamadas nao terminava. O
    # resultado e identico: as eval_rows tem precedencia nos dois casos.
    locais: dict[str, str] = {}
    for row in eval_rows:
        chain = _lineage(row)[::-1]          # stored leaf-first; walk root-first
        for above, below in zip(chain, chain[1:]):
            locais[below] = above
    _completa = taxonomy_parents or {}

    class _Pais:
        """Vista somente-leitura: eval_rows na frente, taxonomia completa atras."""
        __slots__ = ()
        def get(self, k, default=None):
            v = locais.get(k)
            return v if v is not None else _completa.get(k, default)
        def __contains__(self, k):
            return k in locais or k in _completa

    parent = _Pais()

    def ancestors(taxid: str) -> set[str]:
        out, seen = set(), set()
        while taxid and taxid not in seen:
            seen.add(taxid)
            out.add(taxid)
            taxid = parent.get(taxid)
        return out

    inter = pred_total = true_total = singletons = 0
    for row in eval_rows:
        truth = set(_lineage(row))
        if not truth:
            raise ValueError(
                f"read {row.get('read_id')!r} has an empty true_lineage; an "
                "unscoreable read must not be silently counted as zero "
                "(see evaluation/repair_eval_lineages.py)"
            )
        got = predictions.get(row["read_id"])
        pred = ancestors(str(got[0])) if got and got[0] else set()
        if len(pred) == 1 and str(got[0]) not in parent:
            singletons += 1          # sem ancestral reconstruivel: credito perdido
        inter += len(pred & truth)
        pred_total += len(pred)
        true_total += len(truth)

    precision = inter / pred_total if pred_total else 0.0
    recall = inter / true_total if true_total else 0.0
    b2 = beta * beta
    f = ((1 + b2) * precision * recall / (b2 * precision + recall)
         if (precision + recall) else 0.0)
    return {"precision": precision, "recall": recall, "f_beta": f,
            "beta": beta, "n_reads": len(eval_rows),
            "off_path_singletons": singletons}
