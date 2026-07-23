"""Per-head reliability annotation (P12).

A per-node reliability signal lets a downstream classifier weight or gate its
decisions (e.g. stay permissive at a low-reliability node instead of trusting its
call). Reliability is **two-source**, and the order matters:

- the **a-priori** data properties (belongs-genome counts + split mode, emitted into
  ``label_map.json`` at generation time) *predict and explain* reliability — few
  belongs genomes force a 1-genome, high-variance val;
- the **a-posteriori** training behaviour *determines* it — val-f1 instability, the
  val<->test gap, and whether the head actually learned.

``annotate_reliability`` merges the two: with training metrics the verdict is
training-driven; without them it falls back to the a-priori flag. The *policy*
(how a classifier acts on the verdict) is out of scope. Training metrics are a
generic ``{test_f1, val_f1s, learned}`` summary; how a trainer produces them is up
to it. See ``docs/BACKLOG.md`` P12.
"""

import statistics
from typing import Any

# val-f1 standard deviation above this across evals = unstable training
_UNSTABLE_STD: float = 0.08
# |test_f1 - final val_f1| above this = a non-representative val split
_LARGE_GAP: float = 0.10


def _posterior(training: dict) -> dict[str, Any]:
    """A-posteriori signals from a head's training metrics."""
    val_f1s = [v for v in (training.get("val_f1s") or [])
               if isinstance(v, (int, float))]
    test_f1 = training.get("test_f1")
    val_f1_std = round(statistics.pstdev(val_f1s), 4) if len(val_f1s) >= 2 else 0.0
    val_test_gap = (
        round(abs(test_f1 - val_f1s[-1]), 4)
        if val_f1s and isinstance(test_f1, (int, float)) else None
    )
    return {
        "test_f1": test_f1,
        "val_f1_std": val_f1_std,
        "val_test_gap": val_test_gap,
        "learned": training.get("learned"),
        "n_evals": len(val_f1s),
    }


def _verdict(posterior: dict) -> str:
    if posterior.get("learned") is False:
        return "unreliable"
    non_representative = (
        posterior["val_f1_std"] > _UNSTABLE_STD
        or (posterior["val_test_gap"] is not None
            and posterior["val_test_gap"] > _LARGE_GAP)
    )
    return "noisy-metrics" if non_representative else "reliable"


def annotate_reliability(
    apriori: dict | None, training: dict | None = None
) -> dict[str, Any]:
    """Merge the a-priori data props with the a-posteriori training behaviour.

    Training behaviour *determines* the verdict when present (``verdict_source =
    "training"``); otherwise the a-priori flag is the fallback (``"a_priori"``).
    Returns a reliability dict extending ``apriori``. Verdict scale:
    ``reliable`` / ``noisy-metrics`` / ``unreliable``.
    """
    result = dict(apriori or {})
    if training:
        posterior = _posterior(training)
        result["posterior"] = posterior
        result["verdict"] = _verdict(posterior)
        result["verdict_source"] = "training"
    else:
        low = (apriori or {}).get("a_priori_flag") == "low"
        result["verdict"] = "noisy-metrics" if low else "reliable"
        result["verdict_source"] = "a_priori"
    return result
