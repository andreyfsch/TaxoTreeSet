"""Deterministic virtual TaxID generation for synthetic bucket nodes.

The generation pipeline introduces synthetic ("virtual") nodes into
the taxonomic tree to absorb classes that lack proper taxonomic
placement (low-capacity bucketing) or do not match their siblings'
rank (rank-aware bucketing). These nodes need stable identifiers
that:

1. Cannot collide with real NCBI TaxIDs.
2. Reproduce across pipeline runs.
3. Encode the bucket's purpose without requiring a sidecar registry
   to interpret.

This module solves the problem by generating 9-character identifiers
prefixed with the digit '9' (which no real NCBI TaxID starts with as
of the latest taxonomy) and a deterministic 8-digit suffix derived
from a SHA-256 hash of the parent TaxID and the bucket's purpose.

Same inputs always produce the same output, so cross-references in
manifests, virtual ID registries, and training shards remain stable
across executions.

Typical usage::

    from src.taxotreeset.core.generation.virtual_id import make_virtual_id

    bucket_taxid = make_virtual_id(
        parent_taxid="10239",  # Viruses
        purpose="low_capacity",
    )
    # bucket_taxid is something like '956419858'
"""

import hashlib

_VIRTUAL_ID_PROJECTION_SPACE: int = 100_000_000
_VIRTUAL_ID_HASH_HEX_PREFIX_LENGTH: int = 8


def make_virtual_id(parent_taxid: str, purpose: str) -> str:
    """Generate a deterministic 9-digit virtual TaxID.

    Builds the identifier by hashing the string ``"{parent}:{purpose}"``
    with SHA-256, projecting the leading 8 hex characters into a
    100-million-element integer space, and prefixing with '9'. The
    '9' prefix is the project convention that distinguishes virtual
    IDs from real NCBI TaxIDs.

    Determinism is guaranteed by SHA-256's stability: identical
    inputs across runs (or across machines) always produce the same
    output. This is essential for downstream consumers that
    cross-reference virtual TaxIDs in manifests and Parquet shards.

    Args:
        parent_taxid: Parent node's TaxID (real or virtual). Used to
            scope the virtual ID to its parent context, so sibling
            buckets with the same purpose under different parents
            receive distinct IDs.
        purpose: Short identifier for the bucket type. Common values:
            'low_capacity' (LowCapacityBucket),
            'misc' (catch-all rank bucket),
            'rank_species', 'rank_genus', 'rank_family', etc.
            (rank-specific buckets).

    Returns:
        A 9-character string starting with '9'. The remaining 8
        digits are derived deterministically from the input pair.

    Example:
        >>> make_virtual_id("10239", "low_capacity")
        '956419858'
        >>> make_virtual_id("10239", "low_capacity")  # idempotent
        '956419858'
        >>> make_virtual_id("10239", "misc")  # different purpose
        '9...different...'
    """
    key_bytes = f"{parent_taxid}:{purpose}".encode("utf-8")
    digest = hashlib.sha256(key_bytes).hexdigest()
    suffix = (
        int(digest[:_VIRTUAL_ID_HASH_HEX_PREFIX_LENGTH], 16)
        % _VIRTUAL_ID_PROJECTION_SPACE
    )
    return f"9{suffix:08d}"
