"""Module-level constants shared across the generation subpackage.

This module centralizes constants that are consumed by multiple
modules of the generation subpackage to avoid duplication and to
provide a single point of edition for tuning the pipeline.

Three groups of constants are defined:

1. **Protected ranks**: rank labels marking nodes that should not
   be reprocessed by the rank-aware bucketing pass. These ensure
   idempotency when classify_children_by_rank is called repeatedly
   over the same tree.

2. **Balancing defaults**: default values for the per-class
   balancing layer. All are overridable via CLI arguments.

3. **Bloom filter sizing**: parameters that govern the capacity
   approximation Bloom filter (false-positive rate and expected
   insertion count). The actual bit array size and hash count are
   derived from these at allocation time.
"""

PROTECTED_RANKS: frozenset[str] = frozenset(
    {
        "realm_group",
        "virtual_cluster",
        "virtual_bucket",
        "virtual_low_capacity",
        "virtual_misc",
        "virtual_no_rank",
        "virtual_species",
        "virtual_genus",
        "virtual_family",
        "virtual_order",
        "virtual_class",
        "virtual_phylum",
        "virtual_kingdom",
        "virtual_subfamily",
        "virtual_subgenus",
        "virtual_subspecies",
        "virtual_superkingdom",
        "virtual_superfamily",
        "virtual_superorder",
        "virtual_clade",
        "virtual_subclass",
        "virtual_subphylum",
        "virtual_unknown",
    }
)
"""Ranks marking nodes that must not be reprocessed by the rank-aware
bucketing pass.

The set lists every concrete rank label currently produced by the
bucketing layer plus the legacy ``virtual_bucket`` and
``virtual_cluster`` labels for backward compatibility with manifests
generated prior to the granular-rank refactoring. The
``is_protected_rank`` helper below should be preferred over
membership tests against this set, because it additionally handles
arbitrary ``virtual_<rank>`` labels that may be introduced as the
NCBI Taxonomy adds new ranks (this guarantees the idempotency of
classify_children_by_rank regardless of future rank additions)."""


def is_protected_rank(rank: str) -> bool:
    """Return True when a rank label marks a node as protected.

    A rank is protected when it is explicitly listed in
    ``PROTECTED_RANKS`` or when it follows the ``virtual_<rank>``
    naming convention used by the rank-aware bucketing layer.

    Centralizing the test here ensures consistent behavior across
    every caller in the generation subpackage.

    Args:
        rank: Rank label to check. Empty strings and ``None`` are
            treated as non-protected.

    Returns:
        True when the rank is protected and the corresponding node
        should be preserved as-is by ``classify_children_by_rank``.

    Example:
        >>> is_protected_rank("virtual_misc")
        True
        >>> is_protected_rank("virtual_subgenus")  # not in the set
        True
        >>> is_protected_rank("species")
        False
        >>> is_protected_rank("realm_group")
        True
    """
    if not rank:
        return False
    if rank in PROTECTED_RANKS:
        return True
    return rank.startswith("virtual_")


def is_recursion_terminator(rank: str) -> bool:
    """Return True when a rank label marks a recursion terminator.

    Distinct from ``is_protected_rank``: this predicate governs only
    whether the cascaded head-scheduling traversal should *recurse
    into* a node's children. The rank-aware bucketing layer uses
    ``is_protected_rank`` instead, which has different semantics.

    A rank terminates recursion when it labels a virtual bucket
    (``virtual_*``). These buckets are synthetic containers whose
    classification job is already covered by the parent head, so
    descending into them would create redundant labels.

    Other protected ranks like ``realm_group`` do NOT terminate
    recursion. They are curated semantic groupings (e.g., the
    Archaeal_Viruses_Group fallback for viruses) that preserve a
    valid NCBI subtree underneath. Their children must become heads
    of their own.

    Args:
        rank: Rank label to check. Empty strings and ``None`` are
            treated as non-terminating.

    Returns:
        True when descending into the node's children would create
        meaningless heads. False otherwise.

    Example:
        >>> is_recursion_terminator("virtual_misc")
        True
        >>> is_recursion_terminator("virtual_low_capacity")
        True
        >>> is_recursion_terminator("realm_group")  # curated fallback
        False
        >>> is_recursion_terminator("virtual_cluster")  # k-means bucket
        True
        >>> is_recursion_terminator("genus")
        False
    """
    if not rank:
        return False
    return rank.startswith("virtual_")

DEFAULT_MIN_NUM_SEQS: int = 1000
DEFAULT_CUTOFF_PERCENTAGE: float = 98.0
DEFAULT_USE_EXACT_CAPACITY: bool = True
DEFAULT_MAX_N_PER_CLASS: int = 20_000

BLOOM_FALSE_POSITIVE_RATE: float = 0.01
BLOOM_EXPECTED_INSERTIONS: int = 10_000_000
