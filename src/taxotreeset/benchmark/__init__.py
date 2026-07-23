"""Open-set benchmark tooling for TaxoTreeSet.

Phase 1 (this module set) covers clade-holdout *generation*: selecting whole
clades to withhold from training, pruning them from the tree before head
scheduling, and recording a manifest with each held-out clade's expected commit
rank and divergence to the nearest retained relative. Later phases (eval-set
builder, scorer, baseline runners) are tracked as P11 in ``docs/BACKLOG.md`` and
specified in ``docs/clade_holdout_benchmark.md``.
"""

from taxotreeset.benchmark.eval_set import build_eval_reads, build_eval_set
from taxotreeset.benchmark.holdout import (
    build_holdout_manifest,
    prune_holdout,
    select_holdout_taxids,
)
from taxotreeset.benchmark.scorer import classify_outcome, score_reads

__all__ = [
    "build_eval_reads",
    "build_eval_set",
    "build_holdout_manifest",
    "classify_outcome",
    "prune_holdout",
    "score_reads",
    "select_holdout_taxids",
]
