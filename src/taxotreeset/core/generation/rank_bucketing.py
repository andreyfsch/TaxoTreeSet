"""Rank-aware bucketing for taxonomic nodes with heterogeneous children.

This module implements the rank-aware bucketing layer (formerly
referred to as "Op3" during development), which handles the
biological reality that the NCBI Taxonomy frequently violates the
expected uniform-rank-per-level convention. The phenomenon is most
visible in viral taxonomy after the 2022 ICTV reorganization, where
clades like Caudoviricetes contain children at multiple ranks
simultaneously (some are genera, others are species or families).

If left unprocessed, such mixed-rank parents would produce
training heads with inconsistent label semantics: the model would
be asked to classify a sequence as either a genus or a species
within the same softmax, which is biologically incoherent. The
rank-aware bucketing layer fixes this by:

1. Identifying the modal (canonical) rank among the parent's
   children, ignoring those that are already virtual buckets.

2. Separating children with non-canonical ranks into virtual
   buckets, grouped by their actual rank (e.g., a
   ``virtual_species`` bucket holds all the species-rank children
   when the parent is otherwise a class node).

3. Merging ranks below ``min_subclades_per_bucket`` into a single
   generic ``virtual_misc`` bucket to avoid creating many tiny
   training labels.

After processing, all retained children have either the canonical
rank or a ``virtual_*`` rank label, so downstream balancing and
training operate on a uniform-rank set of labels.

The mechanism is **idempotent**: children whose rank is already a
PROTECTED_RANK (the four virtual rank labels) are preserved as-is.
Re-running classify_children_by_rank on a previously processed
parent produces the same result.

Typical usage::

    from src.taxotreeset.core.generation.rank_bucketing import (
        classify_children_by_rank,
    )

    effective_children, new_virtual_buckets = classify_children_by_rank(
        parent_node=some_node,
        children=list_of_children,
        min_subclades_per_bucket=5,
    )
"""

import logging
from collections import Counter, defaultdict

from bigtree import Node

from src.taxotreeset.core.generation.constants import is_protected_rank
from src.taxotreeset.core.generation.low_capacity_bucket import (
    _make_virtual_bucket_node,
)
from src.taxotreeset.core.generation.virtual_id import make_virtual_id

logger = logging.getLogger("TaxoTreeSet.Core.Generation.RankBucketing")

_DEFAULT_MIN_SUBCLADES_PER_BUCKET: int = 5
_MISC_RANK_LABEL: str = "virtual_misc"
_MISC_PURPOSE: str = "misc"
_MISC_BUCKET_NAME_PREFIX: str = "virtual_misc"


def classify_children_by_rank(
    parent_node,
    children: list,
    min_subclades_per_bucket: int = _DEFAULT_MIN_SUBCLADES_PER_BUCKET,
) -> tuple[list, list[dict]]:
    """Apply rank-aware bucketing to a parent's children.

    Identifies the modal rank among non-protected children and
    separates any children with non-canonical ranks into virtual
    buckets. Ranks with fewer than ``min_subclades_per_bucket``
    subclades are merged into a generic ``virtual_misc`` bucket.

    Children for which ``is_protected_rank`` returns True (already-virtual
    buckets from a previous bucketing pass or from the curated
    fallback layer) are preserved as-is, keeping the function
    idempotent.

    Args:
        parent_node: Parent bigtree Node whose children are bucketed.
        children: List of direct children of the parent.
        min_subclades_per_bucket: Minimum subclade count for a
            non-canonical rank to receive its own dedicated bucket.
            Children of rarer ranks (below this threshold) are
            merged into ``virtual_misc``.

    Returns:
        Two-tuple ``(effective_children, new_virtual_buckets)``:
            - effective_children: list of children to use for the
              subsequent balancing pass (canonical-rank children +
              protected children + newly created virtual buckets).
            - new_virtual_buckets: list of metadata dicts (one per
              newly created bucket) with keys 'taxid', 'name',
              'rank', 'purpose', 'absorbed_taxids'.
    """
    if not children:
        return [], []

    canonical_rank = _resolve_canonical_rank(children)
    if canonical_rank is None:
        return list(children), []

    rank_counts = _count_ranks_excluding_protected(children)
    non_canonical_ranks = {rank for rank in rank_counts if rank != canonical_rank}

    if not non_canonical_ranks:
        return list(children), []

    return _materialize_rank_buckets(
        parent_node=parent_node,
        children=children,
        canonical_rank=canonical_rank,
        rank_counts=rank_counts,
        min_subclades_per_bucket=min_subclades_per_bucket,
    )


def _count_ranks_excluding_protected(children: list) -> Counter:
    """Count occurrences of each rank among non-protected children.

    Children for which ``is_protected_rank`` returns True are excluded from the
    count because they should not influence the modal rank decision
    (they are bucket nodes from previous passes, not real taxa).

    Args:
        children: List of direct children to inspect.

    Returns:
        Counter mapping rank string to occurrence count among
        non-protected children.
    """
    rank_counts: Counter = Counter()
    for child in children:
        child_rank = getattr(child, "rank", "") or ""
        if is_protected_rank(child_rank):
            continue
        rank_counts[child_rank] += 1
    return rank_counts


def _resolve_canonical_rank(children: list) -> str | None:
    """Identify the modal rank among the parent's non-protected children.

    The modal rank is the one with the highest count among children
    not already in virtual buckets. When no non-protected children
    exist (e.g., parent has only virtual buckets), returns None.

    Args:
        children: List of direct children to inspect.

    Returns:
        The canonical rank label, or None when no non-protected
        children are present.
    """
    rank_counts = _count_ranks_excluding_protected(children)
    if not rank_counts:
        return None
    return rank_counts.most_common(1)[0][0]


def _materialize_rank_buckets(
    parent_node,
    children: list,
    canonical_rank: str,
    rank_counts: Counter,
    min_subclades_per_bucket: int,
) -> tuple[list, list[dict]]:
    """Build virtual buckets for non-canonical-rank children.

    Partitions the children into three groups:

    1. **Canonical**: children whose rank matches the modal rank.
       Preserved as direct training labels.

    2. **Protected**: children already in virtual buckets.
       Preserved as direct training labels (idempotency).

    3. **Non-canonical**: children with ranks differing from the
       canonical. Grouped by rank into buckets:
       - Ranks with >= ``min_subclades_per_bucket`` children get
         their own dedicated ``virtual_<rank>`` bucket.
       - Ranks with fewer children are merged into a single
         ``virtual_misc`` bucket.

    Args:
        parent_node: bigtree Node of the parent.
        children: All children of the parent.
        canonical_rank: The modal rank determined earlier.
        rank_counts: Counter of rank -> child count.
        min_subclades_per_bucket: Threshold for dedicated buckets.

    Returns:
        Two-tuple (effective_children, new_virtual_buckets).
    """
    parent_taxid = str(parent_node.name)
    parent_name = getattr(parent_node, "scientific_name", parent_taxid)

    canonical_children, protected_children, rank_groups = _partition_children_by_rank(
        children, canonical_rank
    )

    dedicated_ranks, misc_children = _split_dedicated_versus_misc(
        rank_groups=rank_groups,
        min_subclades_per_bucket=min_subclades_per_bucket,
    )

    new_buckets: list[dict] = []
    effective_children = list(canonical_children) + list(protected_children)

    for rank, bucket_children in dedicated_ranks.items():
        bucket_node, bucket_meta = _create_rank_specific_bucket(
            parent_node=parent_node,
            parent_taxid=parent_taxid,
            parent_name=parent_name,
            rank=rank,
            absorbed_children=bucket_children,
        )
        effective_children.append(bucket_node)
        new_buckets.append(bucket_meta)

    if misc_children:
        misc_node, misc_meta = _create_misc_bucket(
            parent_node=parent_node,
            parent_taxid=parent_taxid,
            parent_name=parent_name,
            absorbed_children=misc_children,
        )
        effective_children.append(misc_node)
        new_buckets.append(misc_meta)

    if new_buckets:
        logger.info(
            f"  [RANK-BUCKETING] {parent_name}: "
            f"canonical={canonical_rank} ({rank_counts[canonical_rank]}); "
            f"created {len(new_buckets)} virtual bucket(s) for "
            "non-canonical ranks."
        )

    return effective_children, new_buckets


def _partition_children_by_rank(
    children: list,
    canonical_rank: str,
) -> tuple[list, list, dict[str, list]]:
    """Partition children into canonical, protected, and rank-grouped sets.

    Args:
        children: All children of the parent.
        canonical_rank: The modal rank.

    Returns:
        Three-tuple ``(canonical, protected, by_rank)``:
            - canonical: children whose rank matches canonical_rank.
            - protected: children for which ``is_protected_rank`` returns True.
            - by_rank: dict mapping non-canonical rank -> list of
              children with that rank.
    """
    canonical_children: list = []
    protected_children: list = []
    by_rank: dict[str, list] = defaultdict(list)

    for child in children:
        child_rank = getattr(child, "rank", "") or ""
        if is_protected_rank(child_rank):
            protected_children.append(child)
            continue
        if child_rank == canonical_rank:
            canonical_children.append(child)
            continue
        by_rank[child_rank].append(child)

    return canonical_children, protected_children, by_rank


def _split_dedicated_versus_misc(
    rank_groups: dict[str, list],
    min_subclades_per_bucket: int,
) -> tuple[dict[str, list], list]:
    """Separate ranks into those getting dedicated buckets versus misc.

    Args:
        rank_groups: Map of non-canonical rank -> list of children.
        min_subclades_per_bucket: Threshold for dedicated buckets.

    Returns:
        Two-tuple ``(dedicated_ranks, misc_children)``:
            - dedicated_ranks: subset of rank_groups passing the
              subclade-count threshold.
            - misc_children: flat list of children from ranks below
              the threshold.
    """
    dedicated_ranks: dict[str, list] = {}
    misc_children: list = []

    for rank, bucket_children in rank_groups.items():
        if len(bucket_children) >= min_subclades_per_bucket:
            dedicated_ranks[rank] = bucket_children
        else:
            misc_children.extend(bucket_children)

    return dedicated_ranks, misc_children


def _create_rank_specific_bucket(
    parent_node,
    parent_taxid: str,
    parent_name: str,
    rank: str,
    absorbed_children: list,
) -> tuple[Node, dict]:
    """Create a virtual bucket for one specific non-canonical rank.

    The bucket's TaxID is generated deterministically from the
    parent and the purpose string ``rank_<rank>``, so the same
    bucket always receives the same ID across runs.

    Args:
        parent_node: bigtree parent Node.
        parent_taxid: Parent's TaxID.
        parent_name: Parent's human-readable name.
        rank: The non-canonical rank being bucketed.
        absorbed_children: Children with this non-canonical rank,
            re-parented under the bucket.

    Returns:
        Two-tuple ``(bucket_node, bucket_metadata)``.
    """
    purpose = f"rank_{rank}"
    virtual_id = make_virtual_id(parent_taxid, purpose)
    bucket_name = f"virtual_{rank}_{parent_name}"

    bucket_node = _make_virtual_bucket_node(
        virtual_id=virtual_id,
        parent_taxid=parent_taxid,
        parent_name=parent_name,
        rank=f"virtual_{rank}",
        scientific_name=bucket_name,
        parent_node=parent_node,
    )

    for child in absorbed_children:
        child.parent = bucket_node

    metadata = {
        "taxid": virtual_id,
        "name": bucket_name,
        "rank": rank,
        "purpose": purpose,
        "absorbed_taxids": [str(child.name) for child in absorbed_children],
    }
    return bucket_node, metadata


def _create_misc_bucket(
    parent_node,
    parent_taxid: str,
    parent_name: str,
    absorbed_children: list,
) -> tuple[Node, dict]:
    """Create the catch-all ``virtual_misc`` bucket for rare ranks.

    Used when individual non-canonical ranks have fewer than
    ``min_subclades_per_bucket`` children each. Their children are
    pooled into this single bucket.

    Args:
        parent_node: bigtree parent Node.
        parent_taxid: Parent's TaxID.
        parent_name: Parent's human-readable name.
        absorbed_children: Children from below-threshold ranks.

    Returns:
        Two-tuple ``(bucket_node, bucket_metadata)``.
    """
    virtual_id = make_virtual_id(parent_taxid, _MISC_PURPOSE)
    bucket_name = f"{_MISC_BUCKET_NAME_PREFIX}_{parent_name}"

    bucket_node = _make_virtual_bucket_node(
        virtual_id=virtual_id,
        parent_taxid=parent_taxid,
        parent_name=parent_name,
        rank=_MISC_RANK_LABEL,
        scientific_name=bucket_name,
        parent_node=parent_node,
    )

    for child in absorbed_children:
        child.parent = bucket_node

    metadata = {
        "taxid": virtual_id,
        "name": bucket_name,
        "rank": _MISC_PURPOSE,
        "purpose": _MISC_PURPOSE,
        "absorbed_taxids": [str(child.name) for child in absorbed_children],
    }
    return bucket_node, metadata
