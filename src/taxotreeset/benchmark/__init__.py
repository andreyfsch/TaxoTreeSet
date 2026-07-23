"""Open-set benchmark tooling for TaxoTreeSet.

Phase 1 (this module set) covers clade-holdout *generation*: selecting whole
clades to withhold from training, pruning them from the tree before head
scheduling, and recording a manifest with each held-out clade's expected commit
rank and divergence to the nearest retained relative. Later phases (eval-set
builder, scorer, baseline runners) are tracked as P11 in ``docs/BACKLOG.md`` and
specified in ``docs/clade_holdout_benchmark.md``.
"""

from taxotreeset.benchmark.baselines import (
    export_retained_reference,
    parse_kraken2_output,
    taxid_rank_map,
)
from taxotreeset.benchmark.eval_set import (
    ErrorModel,
    apply_errors,
    build_eval_reads,
    build_eval_set,
)
from taxotreeset.benchmark.holdout import (
    build_holdout_manifest,
    prune_holdout,
    select_holdout_taxids,
)
from taxotreeset.benchmark.reliability import annotate_reliability
from taxotreeset.benchmark.scorer import classify_outcome, score_reads

__all__ = [
    "ErrorModel",
    "annotate_reliability",
    "apply_errors",
    "build_eval_reads",
    "build_eval_set",
    "build_holdout_manifest",
    "classify_outcome",
    "export_retained_reference",
    "parse_kraken2_output",
    "prune_holdout",
    "score_reads",
    "select_holdout_taxids",
    "taxid_rank_map",
]
