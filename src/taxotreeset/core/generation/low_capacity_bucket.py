"""Low-capacity bucket creation for the balancing layer.

The low-capacity bucket is a synthetic Node introduced by the
balancing layer when at least one of a parent's children lacks
sufficient genomic material to meet the per-class subsequence
threshold. Children that fall below the percentile cutoff are
re-parented under this bucket, and the bucket itself becomes a
training label in the parent's head.

This mechanism, formerly referred to as "Op_B" during development,
preserves the integrity of the cascaded classifier when some
classes are too sparse to train independently. Rather than dropping
the sparse classes (which would lose information) or letting them
unbalance the head (which would skew training), they are grouped
into a single bucket that the model learns to predict as "any of
these low-data taxa".

The module also exposes the lower-level ``_make_virtual_bucket_node``
factory, which is shared with the rank-aware bucketing module.
Centralizing the Node construction here ensures all virtual buckets
carry a consistent set of attributes (taxid name, rank label,
scientific_name, parent_taxid, parent_name) regardless of which
mechanism created them.

Typical usage::

    from src.taxotreeset.core.generation.low_capacity_bucket import (
        make_low_capacity_bucket_node,
    )

    bucket_node, bucket_metadata = make_low_capacity_bucket_node(
        parent_node=some_node,
        low_capacity_children=children_below_cutoff,
    )
"""

import logging

from bigtree import Node

from src.taxotreeset.core.generation.virtual_id import make_virtual_id

logger = logging.getLogger("TaxoTreeSet.Core.Generation.LowCapacityBucket")

_LOW_CAPACITY_PURPOSE: str = "low_capacity"
_LOW_CAPACITY_RANK: str = "virtual_low_capacity"
_BUCKET_NAME_PREFIX: str = "virtual_low_capacity"


def make_low_capacity_bucket_node(
    parent_node,
    low_capacity_children: list,
    parent_taxid: str | None = None,
    parent_name: str | None = None,
) -> tuple[Node, dict]:
    """Create the low-capacity bucket absorbing under-capacity children.

    Called by the balancing layer when the cutoff scenario applies.
    The bucket itself becomes a training label in the parent's head;
    the absorbed children are re-parented under the bucket and are
    no longer direct training labels of the parent.

    The bucket's virtual TaxID is generated deterministically from
    the parent TaxID and the purpose string 'low_capacity', so the
    same bucket always receives the same ID across pipeline runs
    (enabling stable cross-references in manifests).

    Args:
        parent_node: bigtree parent Node under which the bucket is
            inserted.
        low_capacity_children: List of children below the capacity
            cutoff. These children are re-parented under the new
            bucket node by mutating their ``.parent`` attribute.
        parent_taxid: Parent's TaxID. Defaults to ``parent_node.name``.
        parent_name: Parent's human-readable scientific name.
            Defaults to ``parent_node.scientific_name``.

    Returns:
        Two-tuple ``(bucket_node, bucket_metadata)``:
            - bucket_node: the newly created Node, already attached
              to parent_node with all the children re-parented.
            - bucket_metadata: dict with keys 'taxid', 'name', 'rank',
              'purpose', 'absorbed_taxids' suitable for inclusion in
              the virtual ID registry.
    """
    resolved_parent_taxid = parent_taxid or str(parent_node.name)
    resolved_parent_name = parent_name or getattr(
        parent_node, "scientific_name", resolved_parent_taxid
    )

    virtual_id = make_virtual_id(resolved_parent_taxid, _LOW_CAPACITY_PURPOSE)
    bucket_name = f"{_BUCKET_NAME_PREFIX}_{resolved_parent_name}"

    bucket_node = _make_virtual_bucket_node(
        virtual_id=virtual_id,
        parent_taxid=resolved_parent_taxid,
        parent_name=resolved_parent_name,
        rank=_LOW_CAPACITY_RANK,
        scientific_name=bucket_name,
        parent_node=parent_node,
    )

    for child in low_capacity_children:
        child.parent = bucket_node

    metadata = {
        "taxid": virtual_id,
        "name": bucket_name,
        "rank": _LOW_CAPACITY_PURPOSE,
        "purpose": _LOW_CAPACITY_PURPOSE,
        "absorbed_taxids": [str(child.name) for child in low_capacity_children],
    }
    return bucket_node, metadata


def _make_virtual_bucket_node(
    virtual_id: str,
    parent_taxid: str,
    parent_name: str,
    rank: str,
    scientific_name: str,
    parent_node,
) -> Node:
    """Construct a virtual bucket Node attached to the given parent.

    This factory is shared between low-capacity and rank-aware
    bucketing. Centralizing the construction guarantees that all
    virtual buckets carry the same set of attributes regardless of
    which bucketing mechanism creates them.

    Args:
        virtual_id: Virtual TaxID (9xxxxxxxx) for the bucket.
        parent_taxid: Parent's TaxID, stored on the node for logging.
        parent_name: Parent's scientific name, stored for logging.
        rank: Virtual rank label. One of 'virtual_low_capacity',
            'virtual_misc', or 'virtual_<concrete_rank>' (e.g.,
            'virtual_species', 'virtual_family').
        scientific_name: Human-readable name for the bucket.
        parent_node: bigtree parent Node under which the bucket
            attaches.

    Returns:
        Newly created Node, already wired as a child of parent_node.
    """
    bucket_node = Node(virtual_id, parent=parent_node)
    bucket_node.rank = rank
    bucket_node.scientific_name = scientific_name
    bucket_node.parent_taxid = parent_taxid
    bucket_node.parent_name = parent_name
    return bucket_node
