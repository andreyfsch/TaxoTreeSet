"""Canonical taxonomic rank ordering shared across the package.

A single source of truth for the major ranks and their order, used by
discovery (lineage resolution) and by generation (depth-bounded
scheduling). The NCBI rank enum's numeric values are arbitrary
identifiers, not depth, so the order is defined explicitly here.

Only the eight major ("canonical") ranks are ordered. Intermediate or
non-canonical ranks (subfamily, no_rank, clade, etc.) are deliberately
excluded: depth decisions are made against the major ranks a lineage
always passes through.
"""

# Major ranks ordered from the most specific (species) up to the
# broadest (superkingdom). This is the order lineage resolution emits
# (species to root).
CANONICAL_RANKS_SPECIES_TO_ROOT: tuple[str, ...] = (
    "species",
    "genus",
    "family",
    "order",
    "class",
    "phylum",
    "kingdom",
    "superkingdom",
)

# The same ranks ordered from broadest to most specific. "Depth" reads
# naturally in this direction: superkingdom is shallow, species is deep.
CANONICAL_RANKS_ROOT_TO_SPECIES: tuple[str, ...] = tuple(
    reversed(CANONICAL_RANKS_SPECIES_TO_ROOT)
)

# Depth index per canonical rank: larger means deeper (more specific).
# superkingdom -> 0, ..., species -> 7.
_RANK_DEPTH: dict[str, int] = {
    rank: depth
    for depth, rank in enumerate(CANONICAL_RANKS_ROOT_TO_SPECIES)
}


def is_canonical_rank(rank: str) -> bool:
    """Return True if the rank is one of the ordered major ranks.

    Args:
        rank: Rank label to test.

    Returns:
        True when the rank participates in the canonical ordering.
    """
    return rank in _RANK_DEPTH


def rank_depth(rank: str) -> int | None:
    """Return a rank's depth index, or None if it is not canonical.

    Depth grows from the root: superkingdom is 0, species is 7. Useful
    for comparing how deep two ranks sit relative to each other.

    Args:
        rank: Rank label to look up.

    Returns:
        The depth index, or None when the rank is not canonical.
    """
    return _RANK_DEPTH.get(rank)


def is_at_or_below_boundary(rank: str, boundary: str) -> bool:
    """Return True if ``rank`` is at or deeper than ``boundary``.

    Used to stop depth-bounded traversal: a node at or below the chosen
    boundary rank is the deepest level still materialized. Non-canonical
    ranks return False (they do not define a boundary crossing on their
    own and are handled by the caller's other rules).

    Args:
        rank: The node's rank.
        boundary: The chosen depth-boundary rank.

    Returns:
        True when ``rank`` is canonical and sits at or below
        ``boundary`` in depth.

    Raises:
        ValueError: If ``boundary`` is not a canonical rank.
    """
    boundary_depth = _RANK_DEPTH.get(boundary)
    if boundary_depth is None:
        raise ValueError(
            f"Boundary must be a canonical rank, got {boundary!r}. "
            f"Valid: {list(CANONICAL_RANKS_ROOT_TO_SPECIES)}."
        )
    node_depth = _RANK_DEPTH.get(rank)
    if node_depth is None:
        return False
    return node_depth >= boundary_depth
