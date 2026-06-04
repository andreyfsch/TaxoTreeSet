"""Per-class balancing for the cascaded LoRA training shards.

This module implements the balancing layer that decides how many
unique subsequences (``n_per_class``) to extract from each
trainable child of a taxonomic head. The goal is to produce
Parquet shards where every class is represented equally during
training, so the LoRA fine-tune does not learn skewed priors
biased toward better-sequenced clades.

The balancing layer handles two distinct scenarios driven by the
distribution of children's capacities:

1. **Level-all scenario**: every child meets or exceeds the
   ``min_num_seqs`` threshold. ``n_per_class`` is set to the
   minimum capacity across children, then optionally capped at
   ``max_n_per_class`` to prevent dataset explosion on jumbo-
   genome heads (the cap-and-scenario distinction is preserved
   in the manifest for downstream interpretation).

2. **Cutoff scenario**: at least one child falls below
   ``min_num_seqs``. The children are sorted by capacity, the
   ``cutoff_percentage`` percentile is taken as the cut value,
   and children above the cutoff are retained as training labels
   while children below are absorbed into a low-capacity bucket
   created by the caller. ``n_per_class`` is then set to the
   minimum capacity among the retained children.

The function returns a structured plan describing the scenario,
the chosen ``n_per_class``, and the retained vs absorbed children.
The caller (the generation orchestrator) is responsible for
materializing the low-capacity bucket when needed and for
distributing samples across each child's sequence leaves.

Typical usage::

    from taxotreeset.core.generation.balancing import (
        compute_balanced_extraction_plan,
    )

    plan = compute_balanced_extraction_plan(
        parent_node=some_node,
        children=effective_children_list,
        leaf_cache={},
        min_num_seqs=1000,
        cutoff_percentage=98.0,
        use_exact_capacity=False,
        max_n_per_class=20_000,
    )
    if plan["low_capacity_children"]:
        # create the low-capacity bucket and re-parent children
        ...
"""

import logging

from taxotreeset.core.generation.capacity import compute_node_capacity
from taxotreeset.core.generation.constants import (
    DEFAULT_CUTOFF_PERCENTAGE,
    DEFAULT_MAX_N_PER_CLASS,
    DEFAULT_MIN_NUM_SEQS,
    DEFAULT_USE_EXACT_CAPACITY,
    DEFAULT_MIN_LEAVES_PER_CLASS,
    DEFAULT_RARE_TAXA_STRATEGY,
)

logger = logging.getLogger("TaxoTreeSet.Core.Generation.Balancing")

_DEFAULT_MIN_SUBSEQ_LEN: int = 100
_SCENARIO_LEVEL_ALL: str = "level_all"
_SCENARIO_LEVEL_ALL_CAPPED: str = "level_all_capped"
_SCENARIO_CUTOFF_APPLIED: str = "cutoff_applied"


def compute_balanced_extraction_plan(
    parent_node,
    children: list,
    leaf_cache: dict,
    min_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
    min_num_seqs: int = DEFAULT_MIN_NUM_SEQS,
    cutoff_percentage: float = DEFAULT_CUTOFF_PERCENTAGE,
    use_exact_capacity: bool = DEFAULT_USE_EXACT_CAPACITY,
    max_n_per_class: int = DEFAULT_MAX_N_PER_CLASS,
    min_leaves_per_class: int = DEFAULT_MIN_LEAVES_PER_CLASS,
    rare_taxa_strategy: str = DEFAULT_RARE_TAXA_STRATEGY,
    progress_callback=None,
) -> dict:
    """Build a per-class extraction plan that balances training subseqs.

    For a given parent node and its direct children, decides how many
    unique subsequences to extract from each child so that all
    training labels under the parent are represented equally in the
    resulting Parquet shard.

    Args:
        parent_node: Parent bigtree Node whose children are balanced.
        children: List of effective children (output of the rank-
            aware bucketing pass).
        leaf_cache: Per-node cache of sequence leaves keyed by taxid
            string. Used by the capacity computation to avoid
            re-scanning the tree.
        min_len: Sliding window size for capacity computation.
            Defaults to 100 bp.
        min_num_seqs: Threshold below which the cutoff scenario is
            triggered. When the minimum capacity across children is
            at or above this value, the level-all scenario applies.
        cutoff_percentage: Percentile of children retained when
            cutoff applies (e.g., 98.0 means the lowest 2% of
            children by capacity are absorbed into the low-capacity
            bucket).
        use_exact_capacity: True for exact set-union computation,
            False for Bloom filter approximation.
        max_n_per_class: Hard ceiling on n_per_class to prevent
            dataset explosion on jumbo-genome heads.

    Returns:
        Dictionary with keys:
            - 'scenario': one of 'level_all', 'level_all_capped',
              'cutoff_applied'.
            - 'n_per_class': number of unique subsequences per child.
            - 'retained_children': list of children that remain as
              training labels.
            - 'low_capacity_children': list of children below the cutoff
              (empty in level-all scenarios). The caller is
              responsible for creating the low-capacity bucket and
              re-parenting these children.
            - 'capacities': dict mapping child name -> capacity.
    """
    capacity_mode = "exact" if use_exact_capacity else "approximate"
    parent_name = getattr(parent_node, "name", "?")

    # Phase 0: divert rare taxa (children with too few sequence leaves)
    # into a fallback bucket before any capacity work. Gated on retaining
    # at least two eligible children (decision A): if fewer than two clear
    # the leaf-count floor, no diversion happens and every child stays.
    eligible_children, rare_taxa_children = _partition_by_leaf_count(
        children=children,
        leaf_cache=leaf_cache,
        min_leaves_per_class=min_leaves_per_class,
        rare_taxa_strategy=rare_taxa_strategy,
    )
    if rare_taxa_children:
        logger.info(
            f"  [RARE-TAXA] {parent_name}: diverting "
            f"{len(rare_taxa_children)} children below "
            f"{min_leaves_per_class}-leaf floor; "
            f"{len(eligible_children)} eligible remain."
        )

    capacities = _compute_children_capacities(
        children=eligible_children,
        min_len=min_len,
        leaf_cache=leaf_cache,
        capacity_mode=capacity_mode,
        max_n_per_class=max_n_per_class,
        progress_callback=progress_callback,
    )
    if not capacities:
        plan = _empty_extraction_plan()
        plan["rare_taxa_children"] = rare_taxa_children
        return plan
    min_capacity = min(capacities.values())
    if min_capacity >= min_num_seqs:
        plan = _build_level_all_plan(
            children=eligible_children,
            capacities=capacities,
            min_capacity=min_capacity,
            max_n_per_class=max_n_per_class,
            parent_name=parent_name,
        )
    else:
        plan = _build_cutoff_plan(
            children=eligible_children,
            capacities=capacities,
            cutoff_percentage=cutoff_percentage,
            max_n_per_class=max_n_per_class,
            parent_name=parent_name,
        )
    plan["rare_taxa_children"] = rare_taxa_children
    return plan


def _count_sequence_leaves(child, leaf_cache: dict) -> int:
    """Count the sequence-rank leaves descending from a child node.

    Uses the per-node leaf cache when available to avoid a full
    subtree scan; falls back to scanning ``child.leaves`` on a miss.

    Args:
        child: bigtree Node to count leaves for.
        leaf_cache: Per-node cache keyed by taxid string.

    Returns:
        Number of sequence-rank leaves under the child.
    """
    cached = leaf_cache.get(str(child.name))
    if cached is not None:
        return len(cached)
    return sum(
        1 for leaf in child.leaves if getattr(leaf, "rank", "") == "sequence"
    )


def _partition_by_leaf_count(
    children: list,
    leaf_cache: dict,
    min_leaves_per_class: int,
    rare_taxa_strategy: str,
) -> tuple[list, list]:
    """Split children into leaf-count-eligible and rare-taxa groups.

    Under the 'fallback' strategy, children with fewer than
    ``min_leaves_per_class`` sequence leaves are diverted into the
    rare-taxa group. The split is gated (decision A): when fewer than
    two children clear the floor, the threshold is not applied and all
    children are returned as eligible, preventing a head from
    degenerating into a single rare_taxa label.

    Under the 'keep' strategy, the split is a no-op: all children are
    eligible regardless of leaf count.

    Args:
        children: Effective children of the parent.
        leaf_cache: Per-node leaf cache keyed by taxid string.
        min_leaves_per_class: Minimum sequence-leaf count to remain a
            standalone training label.
        rare_taxa_strategy: 'fallback' to divert rare children,
            'keep' to retain every child.

    Returns:
        Two-tuple ``(eligible_children, rare_taxa_children)``.
    """
    if rare_taxa_strategy != "fallback":
        return list(children), []

    eligible: list = []
    rare: list = []
    for child in children:
        if _count_sequence_leaves(child, leaf_cache) >= min_leaves_per_class:
            eligible.append(child)
        else:
            rare.append(child)

    if len(eligible) < 2:
        return list(children), []

    return eligible, rare

def _compute_children_capacities(
    children: list,
    min_len: int,
    leaf_cache: dict,
    capacity_mode: str,
    max_n_per_class: int,
    progress_callback=None,
) -> dict[str, int]:
    """Compute the capacity of every child and return as a dict.

    Args:
        children: List of children to evaluate.
        min_len: Sliding window size.
        leaf_cache: Per-node leaf cache.
        capacity_mode: 'exact' or 'approximate'.
        max_n_per_class: Used as the early-termination ceiling.

    Returns:
        Dictionary mapping child name (string) to its capacity.
    """
    capacities: dict[str, int] = {}
    for child in children:
        child_name = str(child.name)
        capacities[child_name] = compute_node_capacity(
            child,
            min_len,
            leaf_cache,
            mode=capacity_mode,
            max_useful=max_n_per_class,
        )
        if progress_callback is not None:
            progress_callback()
    return capacities


def _empty_extraction_plan() -> dict:
    """Return a no-op extraction plan when there are no children.

    Returns:
        A plan dict with all fields populated as empty/zero values.
    """
    return {
        "scenario": _SCENARIO_LEVEL_ALL,
        "n_per_class": 0,
        "retained_children": [],
        "low_capacity_children": [],
        "rare_taxa_children": [],
        "capacities": {},
    }


def _build_level_all_plan(
    children: list,
    capacities: dict[str, int],
    min_capacity: int,
    max_n_per_class: int,
    parent_name: str,
) -> dict:
    """Build the level_all (or level_all_capped) extraction plan.

    Sets n_per_class to min_capacity. If min_capacity exceeds the
    hard cap, clamps and marks the scenario as 'level_all_capped'
    so the manifest reflects the clamping decision.

    Args:
        children: All children of the parent (all retained).
        capacities: Mapping of child name -> capacity.
        min_capacity: Minimum capacity across children.
        max_n_per_class: Hard ceiling on n_per_class.
        parent_name: Parent node name (for logging).

    Returns:
        Extraction plan dictionary.
    """
    if min_capacity > max_n_per_class:
        n_per_class = max_n_per_class
        scenario = _SCENARIO_LEVEL_ALL_CAPPED
        logger.info(
            f"  [BALANCE] {parent_name}: scenario=level_all_capped "
            f"(min_cap={min_capacity:,} > cap={max_n_per_class:,}); "
            f"n_per_class={n_per_class:,}"
        )
    else:
        n_per_class = min_capacity
        scenario = _SCENARIO_LEVEL_ALL
        logger.info(
            f"  [BALANCE] {parent_name}: scenario=level_all; "
            f"n_per_class={n_per_class:,}"
        )

    return {
        "scenario": scenario,
        "n_per_class": n_per_class,
        "retained_children": list(children),
        "low_capacity_children": [],
        "rare_taxa_children": [],
        "capacities": capacities,
    }


def _build_cutoff_plan(
    children: list,
    capacities: dict[str, int],
    cutoff_percentage: float,
    max_n_per_class: int,
    parent_name: str,
) -> dict:
    """Build the cutoff_applied extraction plan.

    Determines the percentile cutoff value, partitions children into
    retained (above) and low-data (at or below), and sets
    n_per_class to the minimum capacity among the retained children
    (capped at max_n_per_class).

    Args:
        children: All children of the parent.
        capacities: Mapping of child name -> capacity.
        cutoff_percentage: Percentile to keep (e.g., 98.0).
        max_n_per_class: Hard ceiling on n_per_class.
        parent_name: Parent node name (for logging).

    Returns:
        Extraction plan dictionary.
    """
    cutoff_value = _compute_percentile_cutoff(
        sorted_capacities=sorted(capacities.values()),
        cutoff_percentage=cutoff_percentage,
    )

    retained_children, low_capacity_children = _partition_by_cutoff(
        children=children,
        capacities=capacities,
        cutoff_value=cutoff_value,
    )

    n_per_class = _compute_n_per_class_from_retained(
        retained_children=retained_children,
        capacities=capacities,
        max_n_per_class=max_n_per_class,
    )

    logger.info(
        f"  [BALANCE] {parent_name}: scenario=cutoff_applied; "
        f"cutoff_value={cutoff_value:,}; "
        f"retained={len(retained_children)}; "
        f"low_capacity={len(low_capacity_children)}; "
        f"n_per_class={n_per_class:,}"
    )

    return {
        "scenario": _SCENARIO_CUTOFF_APPLIED,
        "n_per_class": n_per_class,
        "retained_children": retained_children,
        "low_capacity_children": low_capacity_children,
        "rare_taxa_children": [],
        "capacities": capacities,
    }


def _compute_percentile_cutoff(
    sorted_capacities: list[int],
    cutoff_percentage: float,
) -> int:
    """Compute the capacity value at the given percentile cutoff.

    The cutoff is computed so that children with capacity strictly
    greater than the cutoff value are retained, while those at or
    below the value are absorbed. The (100 - p)% lowest-capacity
    children fall in the absorbed group.

    Note on precision: the computation uses integer truncation
    (``int(len * (1 - p/100))``), so floating-point representation
    issues may cause the cutoff index to round down by one in edge
    cases. For example, with 10 children and ``cutoff_percentage=90.0``,
    ``1.0 - 0.90`` is ``0.0999...9`` (not 0.1 exactly), so
    ``int(0.999...) = 0`` instead of 1. In practice this means the
    function retains slightly more children than the nominal
    percentage suggests, which is the safer direction for the
    cutoff scenario (we prefer to keep marginal classes rather than
    drop them). This semantics is preserved from the original
    pre-refactor implementation.

    Args:
        sorted_capacities: Capacities sorted in ascending order.
        cutoff_percentage: Percentile (e.g., 98.0) to keep.

    Returns:
        The capacity value at the cutoff index. Returns 0 when the
        input is empty.
    """
    if not sorted_capacities:
        return 0
    cutoff_index = max(
        0, int(len(sorted_capacities) * (1.0 - cutoff_percentage / 100.0))
    )
    return sorted_capacities[cutoff_index]


def _partition_by_cutoff(
    children: list,
    capacities: dict[str, int],
    cutoff_value: int,
) -> tuple[list, list]:
    """Partition children into retained and low-data sets.

    Args:
        children: All children of the parent.
        capacities: Mapping of child name -> capacity.
        cutoff_value: Value computed by _compute_percentile_cutoff.

    Returns:
        Two-tuple ``(retained_children, low_capacity_children)``.
    """
    retained_children: list = []
    low_capacity_children: list = []
    for child in children:
        child_capacity = capacities[str(child.name)]
        if child_capacity >= cutoff_value:
            retained_children.append(child)
        else:
            low_capacity_children.append(child)
    return retained_children, low_capacity_children


def _compute_n_per_class_from_retained(
    retained_children: list,
    capacities: dict[str, int],
    max_n_per_class: int,
) -> int:
    """Compute n_per_class as the minimum capacity among retained children.

    The result is clamped at ``max_n_per_class`` to prevent the
    cutoff scenario from producing excessively large training
    labels (which could happen on heads with one huge child and
    many small ones that get absorbed into the low-capacity
    bucket).

    Args:
        retained_children: Children above the cutoff.
        capacities: Mapping of child name -> capacity.
        max_n_per_class: Hard ceiling.

    Returns:
        Clamped n_per_class value. Returns 0 when no children
        survived the cutoff.
    """
    if not retained_children:
        return 0
    retained_capacities = [capacities[str(child.name)] for child in retained_children]
    return min(min(retained_capacities), max_n_per_class)
