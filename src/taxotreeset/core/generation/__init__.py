"""Generation algorithms decomposed by responsibility."""

from src.taxotreeset.core.generation.balancing import (
    compute_balanced_extraction_plan,
)
from src.taxotreeset.core.generation.capacity import compute_node_capacity
from src.taxotreeset.core.generation.distribution import (
    distribute_n_per_class_across_leaves,
)
from src.taxotreeset.core.generation.low_capacity_bucket import (
    make_low_capacity_bucket_node,
    register_virtual_bucket,
)
from src.taxotreeset.core.generation.rank_bucketing import (
    classify_children_by_rank,
)
from src.taxotreeset.core.generation.virtual_id import make_virtual_id

__all__ = [
    "classify_children_by_rank",
    "compute_balanced_extraction_plan",
    "compute_node_capacity",
    "distribute_n_per_class_across_leaves",
    "make_low_capacity_bucket_node",
    "make_virtual_id",
    "register_virtual_bucket",
]
