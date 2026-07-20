"""Per-leaf sample distribution for the balanced extraction plan.

After the balancing layer decides ``n_per_class`` (samples per
training label) and the rank-aware bucketing produces the final
list of training labels, this module translates those decisions
into concrete per-leaf extraction tasks.

The distribution is proportional to each leaf's "share weight":
the maximum number of unique sliding-window subsequences the leaf
can contribute, given its sequence length and the minimum window
size. Longer sequences receive a larger fraction of the budget;
short sequences receive less. This avoids over-sampling tiny
genomes that would otherwise be repeatedly re-extracted in the
sampling routine of ``sequence_utils``.

Rounding errors from the proportional split are absorbed by the
final leaf in each child's group, which receives whatever is left
of the budget after the previous leaves take their shares. This
guarantees the sum of per-leaf samples equals exactly
``n_per_class`` for each child.

The module is a pure transformation layer: it consumes a balanced
plan and an explicit per-node leaf cache, and produces a dict of
per-child task lists ready for the orchestrator to enrich with
class indices and split fractions.

Typical usage::

    from taxotreeset.core.generation.distribution import (
        distribute_n_per_class_across_leaves,
    )

    per_child_tasks = distribute_n_per_class_across_leaves(
        n_per_class=plan["n_per_class"],
        children=retained_children,
        parent_taxid=str(parent_node.name),
        parent_name="Caudoviricetes",
        leaf_cache=leaf_cache,
    )
"""

import logging

from taxotreeset.core.generation.capacity import _read_sequence_cached

logger = logging.getLogger("TaxoTreeSet.Core.Generation.Distribution")

_DEFAULT_MIN_SUBSEQ_LEN: int = 100


def distribute_n_per_class_across_leaves(
    n_per_class: int,
    children: list,
    parent_taxid: str,
    parent_name: str,
    leaf_cache: dict,
    min_subseq_len: int = _DEFAULT_MIN_SUBSEQ_LEN,
    per_child_n: dict[str, int] | None = None,
) -> dict[str, list[dict]]:
    """Distribute n_per_class samples across each child's sequence leaves.

    For each child in ``children``, computes its sequence leaves
    (using the cache when available, falling back to a tree scan),
    weighs each leaf by its contributable subseq count, and assigns
    per-leaf sample budgets proportionally.

    The function is the central allocator that translates the
    balancing decision ("n_per_class samples per training label")
    into concrete per-leaf extraction tasks ("read this header,
    extract this many samples").

    Args:
        n_per_class: Target sample count per training label, as
            decided by the balancing layer.
        children: List of effective children (training labels) for
            this parent. May include real taxon nodes and virtual
            buckets created by rank-aware or low-capacity bucketing.
        parent_taxid: Parent's TaxID, currently unused but reserved
            for future logging and task tagging.
        parent_name: Parent's human-readable name, currently unused
            but reserved for future logging.
        leaf_cache: Per-node cache mapping child taxid string to its
            list of sequence leaves. Cache misses fall back to a
            full tree scan.
        min_subseq_len: Minimum subseq length, used to compute each
            leaf's share weight. Defaults to 100 bp.
        per_child_n: Optional per-child target sample count keyed by child
            taxid string. When given, each child uses its own target instead of
            the shared ``n_per_class`` (a child absent from the map falls back to
            ``n_per_class``). This is how the opt-in "keep-imbalance" mode lets
            each class keep up to its own capacity instead of the sibling minimum.

    Returns:
        Dictionary mapping child taxid string to a list of per-leaf
        task dicts. Each task dict has keys 'fasta_path', 'header_id',
        and 'n' (the assigned sample count for that leaf).
    """
    _ = parent_taxid, parent_name  # reserved for future use

    distributed: dict[str, list[dict]] = {}

    for child in children:
        child_taxid = str(child.name)
        child_leaves = _resolve_child_leaves(child, child_taxid, leaf_cache)

        if not child_leaves:
            distributed[child_taxid] = []
            continue

        child_n = (
            per_child_n.get(child_taxid, n_per_class)
            if per_child_n is not None
            else n_per_class
        )
        distributed[child_taxid] = _allocate_n_across_leaves(
            child_leaves=child_leaves,
            n_per_class=child_n,
            min_subseq_len=min_subseq_len,
        )

    return distributed


def _resolve_child_leaves(
    child,
    child_taxid: str,
    leaf_cache: dict,
) -> list:
    """Resolve a child's sequence leaves, preferring the cache.

    Args:
        child: bigtree child Node.
        child_taxid: String representation of the child's TaxID.
        leaf_cache: Per-node cache.

    Returns:
        List of sequence leaf Nodes under the child. Empty list
        when none are found.
    """
    cached_leaves = leaf_cache.get(child_taxid, [])
    if cached_leaves:
        return cached_leaves

    return [leaf for leaf in child.leaves if getattr(leaf, "rank", "") == "sequence"]


def _allocate_n_across_leaves(
    child_leaves: list,
    n_per_class: int,
    min_subseq_len: int,
) -> list[dict]:
    """Allocate n_per_class samples across a child's sequence leaves.

    Computes per-leaf weights (the count of contributable subseqs)
    and distributes the budget proportionally. The last leaf
    absorbs any rounding remainder so the per-child sum equals
    exactly ``n_per_class``.

    Leaves with zero weight (e.g., sequence shorter than
    ``min_subseq_len``) receive zero allocation but still appear
    in the iteration; they are filtered out of the output.

    Args:
        child_leaves: Sequence leaves of the child.
        n_per_class: Target sample count for this child.
        min_subseq_len: Minimum subseq length for share weighting.

    Returns:
        List of per-leaf task dicts. Leaves with zero allocation
        are omitted from the result.
    """
    leaf_weights = _compute_leaf_share_weights(
        child_leaves, min_subseq_len=min_subseq_len
    )
    total_weight = sum(leaf_weights) or 1

    per_leaf_tasks: list[dict] = []
    running_sum = 0

    for leaf_index, leaf in enumerate(child_leaves):
        is_last_leaf = leaf_index == len(child_leaves) - 1
        share = _compute_leaf_share(
            n_per_class=n_per_class,
            leaf_weight=leaf_weights[leaf_index],
            total_weight=total_weight,
            running_sum=running_sum,
            is_last_leaf=is_last_leaf,
        )
        running_sum += share

        if share == 0:
            continue

        per_leaf_tasks.append(
            {
                "fasta_path": getattr(leaf, "fasta_path", ""),
                "header_id": getattr(leaf, "header_id", ""),
                "n": share,
                # Genome length, recovered for free from the share weight
                # (weight = len - min_subseq_len + 1, and share > 0 implies
                # weight > 0). Lets the cluster-aware block-stratified split read
                # the length here instead of re-reading the genome.
                "length": leaf_weights[leaf_index] + min_subseq_len - 1,
            }
        )

    return per_leaf_tasks


def _compute_leaf_share(
    n_per_class: int,
    leaf_weight: int,
    total_weight: int,
    running_sum: int,
    is_last_leaf: bool,
) -> int:
    """Compute a single leaf's share of the n_per_class budget.

    Non-last leaves receive a proportional share rounded to the
    nearest integer. The last leaf receives whatever remains of
    the budget after the previous leaves took their shares,
    absorbing rounding errors so the per-child total equals
    exactly ``n_per_class``.

    Args:
        n_per_class: Target per-child sample count.
        leaf_weight: This leaf's share weight (contributable subseqs).
        total_weight: Sum of all share weights for this child.
        running_sum: Cumulative share allocated to previous leaves.
        is_last_leaf: True when this is the final leaf in the loop.

    Returns:
        The integer sample count assigned to this leaf. Never
        negative; clamped to zero when the running sum has already
        reached n_per_class.
    """
    if is_last_leaf:
        return max(0, n_per_class - running_sum)
    share = int(round(n_per_class * leaf_weight / total_weight))
    # Cap at the remaining budget: without this, accumulated round-ups (when
    # many leaves each round n*w/total up) can push the running sum past
    # n_per_class, and the last leaf can only absorb a non-negative remainder —
    # so the per-child total would exceed n_per_class. Capping keeps the sum
    # exactly n_per_class.
    return min(share, max(0, n_per_class - running_sum))


def _compute_leaf_share_weights(
    leaves: list,
    min_subseq_len: int,
) -> list[int]:
    """Compute per-leaf weights for proportional sample apportionment.

    Each leaf's weight is its potential count of unique sliding-
    window subseqs:

        weight = max(0, len(sequence) - min_subseq_len + 1)

    Leaves whose underlying sequence cannot be read receive zero
    weight, effectively excluding them from the proportional split.

    Args:
        leaves: List of sequence leaf nodes.
        min_subseq_len: Sliding window size.

    Returns:
        List of integer weights, one per input leaf, in the same
        order. Sum of zero is possible when all leaves are unreadable.
    """
    weights: list[int] = []
    for leaf in leaves:
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            weights.append(0)
            continue
        sequence = _read_sequence_cached(fasta_path, header_id)
        weights.append(max(0, len(sequence) - min_subseq_len + 1))
    return weights
