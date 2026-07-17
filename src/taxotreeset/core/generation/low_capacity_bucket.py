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

    from taxotreeset.core.generation.low_capacity_bucket import (
        make_low_capacity_bucket_node,
    )

    bucket_node, bucket_metadata = make_low_capacity_bucket_node(
        parent_node=some_node,
        low_capacity_children=children_below_cutoff,
    )
"""

import logging

from bigtree import Node

from taxotreeset.core.generation.virtual_id import make_virtual_id

logger = logging.getLogger("TaxoTreeSet.Core.Generation.LowCapacityBucket")

_LOW_CAPACITY_PURPOSE: str = "low_capacity"
_LOW_CAPACITY_RANK: str = "virtual_low_capacity"
_BUCKET_NAME_PREFIX: str = "virtual_low_capacity"

_RARE_TAXA_PURPOSE: str = "rare_taxa"
_RARE_TAXA_RANK: str = "virtual_rare_taxa"
_RARE_TAXA_NAME_PREFIX: str = "virtual_rare_taxa"


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
    return _make_absorbing_bucket_node(
        parent_node=parent_node,
        absorbed_children=low_capacity_children,
        purpose=_LOW_CAPACITY_PURPOSE,
        rank=_LOW_CAPACITY_RANK,
        name_prefix=_BUCKET_NAME_PREFIX,
        parent_taxid=parent_taxid,
        parent_name=parent_name,
    )


def make_rare_taxa_bucket_node(
    parent_node,
    rare_taxa_children: list,
    parent_taxid: str | None = None,
    parent_name: str | None = None,
) -> tuple[Node, dict]:
    """Create the rare-taxa bucket absorbing low-leaf-count children.

    Called by the balancing layer (under the 'fallback' rare-taxa
    strategy) for children whose sequence-leaf count falls below
    ``DEFAULT_MIN_LEAVES_PER_CLASS``. Such children carry too few
    distinct training examples to learn a generalizable boundary, so
    rather than letting them dilute the head with near-empty classes,
    they are grouped into a single fallback label. A classifier trained
    on this head learns to route rare or novel inputs to the bucket
    instead of forcing them into an under-supported specific class.

    This is semantically distinct from the low-capacity bucket:
    low-capacity groups children with insufficient *quantity* of
    extractable subsequences (capacity below the percentile cutoff),
    whereas rare-taxa groups children with insufficient *diversity*
    of source sequences (too few leaves). A child may have high
    capacity yet few leaves (e.g., a single very long genome), in
    which case it is rare but not low-capacity.

    The bucket's virtual TaxID is generated deterministically from
    the parent TaxID and the purpose string 'rare_taxa', so the same
    bucket always receives the same ID across pipeline runs.

    Args:
        parent_node: bigtree parent Node under which the bucket is
            inserted.
        rare_taxa_children: List of children below the leaf-count
            floor. These children are re-parented under the new
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
    return _make_absorbing_bucket_node(
        parent_node=parent_node,
        absorbed_children=rare_taxa_children,
        purpose=_RARE_TAXA_PURPOSE,
        rank=_RARE_TAXA_RANK,
        name_prefix=_RARE_TAXA_NAME_PREFIX,
        parent_taxid=parent_taxid,
        parent_name=parent_name,
    )


def _make_absorbing_bucket_node(
    parent_node,
    absorbed_children: list,
    purpose: str,
    rank: str,
    name_prefix: str,
    parent_taxid: str | None,
    parent_name: str | None,
) -> tuple[Node, dict]:
    """Create a virtual bucket that absorbs (re-parents) a set of children.

    Shared body of ``make_low_capacity_bucket_node`` and
    ``make_rare_taxa_bucket_node``, which differ only in the purpose / rank /
    name-prefix constants. The bucket gets a deterministic virtual TaxID from
    ``(parent_taxid, purpose)``, is attached under the parent, and the absorbed
    children are re-parented under it.

    Args:
        parent_node: bigtree parent Node under which the bucket is inserted.
        absorbed_children: Children re-parented under the new bucket.
        purpose: Stable purpose string driving the deterministic virtual ID and
            recorded as the metadata ``rank``/``purpose``.
        rank: The node's ``virtual_*`` rank label.
        name_prefix: Prefix for the bucket's human-readable name.
        parent_taxid: Parent's TaxID. Defaults to ``parent_node.name``.
        parent_name: Parent's scientific name. Defaults to
            ``parent_node.scientific_name``.

    Returns:
        Two-tuple ``(bucket_node, bucket_metadata)``.
    """
    resolved_parent_taxid = parent_taxid or str(parent_node.name)
    resolved_parent_name = parent_name or getattr(
        parent_node, "scientific_name", resolved_parent_taxid
    )

    virtual_id = make_virtual_id(resolved_parent_taxid, purpose)
    bucket_name = f"{name_prefix}_{resolved_parent_name}"

    bucket_node = _make_virtual_bucket_node(
        virtual_id=virtual_id,
        parent_taxid=resolved_parent_taxid,
        parent_name=resolved_parent_name,
        rank=rank,
        scientific_name=bucket_name,
        parent_node=parent_node,
    )

    for child in absorbed_children:
        child.parent = bucket_node

    metadata = {
        "taxid": virtual_id,
        "name": bucket_name,
        "rank": purpose,
        "purpose": purpose,
        "absorbed_taxids": [str(child.name) for child in absorbed_children],
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


def register_virtual_bucket(
    virtual_id_registry: dict,
    bucket_metadata: dict,
    parent_taxid: str,
    parent_name: str,
) -> None:
    """Register a virtual bucket in the registry with collision detection.

    Defends against statistical collisions in ``make_virtual_id``,
    which truncates SHA256 to 8 hex chars and is therefore not
    collision-free in principle. Re-registering the same virtual ID
    with a different (parent_taxid, purpose) tuple raises
    ``RuntimeError`` because it would silently overwrite a previously
    created bucket's metadata.

    Re-registering with the same (parent_taxid, purpose) is a no-op,
    preserving idempotency when the orchestrator visits a parent
    multiple times.

    Args:
        virtual_id_registry: The registry dict to populate (mutated).
        bucket_metadata: Dict returned by ``make_low_capacity_bucket_node``
            or by the rank-aware bucketing helpers. Must contain
            'taxid', 'name', 'rank', 'purpose', 'absorbed_taxids'.
        parent_taxid: Parent TaxID hosting the bucket.
        parent_name: Parent's scientific name (human-readable).

    Raises:
        RuntimeError: When a collision is detected, i.e. the virtual
            ID already maps to a different (parent_taxid, purpose)
            pair in the registry.

    Example:
        >>> registry = {}
        >>> bucket_meta = {
        ...     "taxid": "912345678",
        ...     "name": "virtual_misc_X",
        ...     "rank": "virtual_misc",
        ...     "purpose": "misc",
        ...     "absorbed_taxids": ["1", "2"],
        ... }
        >>> register_virtual_bucket(registry, bucket_meta, "10239", "Viruses")
        >>> registry["912345678"]["parent_taxid"]
        '10239'
    """
    virtual_id = bucket_metadata["taxid"]
    purpose = bucket_metadata["purpose"]

    existing = virtual_id_registry.get(virtual_id)
    if existing is not None:
        if (
            existing.get("parent_taxid") != parent_taxid
            or existing.get("purpose") != purpose
        ):
            raise RuntimeError(
                f"Virtual ID collision: {virtual_id} is already registered "
                f"as (parent={existing.get('parent_taxid')}, "
                f"purpose={existing.get('purpose')}); attempted to "
                f"reassign as (parent={parent_taxid}, purpose={purpose}). "
                "This indicates either a statistical collision in "
                "make_virtual_id() or a logic bug at the call site."
            )
        return

    virtual_id_registry[virtual_id] = {
        "parent_taxid": parent_taxid,
        "parent_name": parent_name,
        "name": bucket_metadata["name"],
        "rank": bucket_metadata["rank"],
        "purpose": purpose,
        "absorbed_taxids": bucket_metadata["absorbed_taxids"],
    }

