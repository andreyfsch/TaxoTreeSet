"""Generation algorithms decomposed by responsibility."""

from taxotreeset.core.generation.balancing import (
    compute_balanced_extraction_plan,
)
from taxotreeset.core.generation.capacity import compute_node_capacity
from taxotreeset.core.generation.distribution import (
    distribute_n_per_class_across_leaves,
)
from taxotreeset.core.generation.low_capacity_bucket import (
    make_low_capacity_bucket_node,
    make_rare_taxa_bucket_node,
    register_virtual_bucket,
)
from taxotreeset.core.generation.rank_bucketing import (
    classify_children_by_rank,
)
from taxotreeset.core.generation.reject_bucket import (
    build_reject_tasks,
    make_reject_bucket_node,
    sample_reject_leaves,
)
from taxotreeset.core.generation.virtual_id import make_virtual_id

__all__ = [
    "build_reject_tasks",
    "classify_children_by_rank",
    "compute_balanced_extraction_plan",
    "compute_node_capacity",
    "distribute_n_per_class_across_leaves",
    "make_low_capacity_bucket_node",
    "make_rare_taxa_bucket_node",
    "make_reject_bucket_node",
    "make_virtual_id",
    "register_virtual_bucket",
    "sample_reject_leaves",
]
