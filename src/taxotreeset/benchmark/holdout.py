"""Clade-holdout selection, pruning, and manifest (open-set benchmark, P11-P1).

To measure *open-set* generalization, whole clades are withheld from training so
their genomes appear only in a downstream evaluation set. This module:

- **selects** the clades to withhold (explicit TaxIDs, or a seeded fraction of the
  clades at a given rank), keeping the parent a decision point and the held-out
  clade non-empty;
- **records a manifest** for each held-out clade — its members, its *expected
  commit rank* ``rho*`` (the deepest ancestor that survives pruning, i.e. the
  rank a classifier should back off to for a read from this novel clade), and its
  divergence to the nearest retained relative (a MinHash/Mash ANI proxy, binned);
- **prunes** the held-out subtrees so the generation pipeline trains only on the
  retained set.

Selection is done on the *full* tree before pruning; the caller prunes afterward.
See ``docs/clade_holdout_benchmark.md`` for the surrounding design.
"""

import math
import random
from typing import Any

from bigtree import Node

from taxotreeset.core._orchestration._cluster import (
    _KMER_K,
    _SKETCH_SIZE,
    _genome_sketch,
    _jaccard,
)
from taxotreeset.dataset.utils import _read_single_sequence

_SEQUENCE_RANK = "sequence"

# Nearest-retained-relative divergence bins, on an ANI proxy derived from the
# MinHash Jaccard via the Mash distance. Ordered high-similarity first.
_ANI_BINS: tuple[tuple[float, str], ...] = (
    (0.95, "ANI>=95%"),
    (0.90, "ANI 90-95%"),
    (0.85, "ANI 85-90%"),
    (0.0, "ANI<85%"),
)


def _taxonomic_children(node: Node) -> list[Node]:
    return [c for c in node.children if getattr(c, "rank", "") != _SEQUENCE_RANK]


def _sequence_leaves(node: Node) -> list[Node]:
    return [leaf for leaf in node.leaves if getattr(leaf, "rank", "") == _SEQUENCE_RANK]


def _is_eligible(node: Node) -> bool:
    """A holdout candidate must have genomes and leave its parent still branching."""
    if not _sequence_leaves(node):
        return False
    parent = node.parent
    if parent is None:
        return False
    # Removing this clade must leave >= 1 other taxonomic child, so the parent
    # remains a labelable decision point (i.e. rho* is well-defined at the parent).
    return len(_taxonomic_children(parent)) >= 2


def _dedup_nested(scope_node: Node, taxids: set[str]) -> set[str]:
    """Drop any selected clade that lies under another selected clade."""
    by_taxid = {str(n.name): n for n in scope_node.descendants}
    kept: set[str] = set()
    for taxid in taxids:
        node = by_taxid.get(taxid)
        if node is None:
            continue
        if {str(a.name) for a in node.ancestors} & taxids:
            continue  # an ancestor is also held out — this one is already covered
        kept.add(taxid)
    return kept


def select_holdout_taxids(
    scope_node: Node,
    *,
    explicit: list[str] | None = None,
    rank: str | None = None,
    fraction: float | None = None,
    seed: int = 0,
) -> set[str]:
    """Return the set of clade TaxIDs to withhold from training.

    Two modes:
      - ``explicit``: withhold exactly these taxonomic nodes (those that exist
        under ``scope_node`` and are eligible);
      - ``rank`` + ``fraction``: seeded uniform sample of ``ceil(fraction * n)``
        eligible clades at ``rank`` (results can still be binned by divergence via
        the manifest; stratified-by-bin selection is a future refinement).

    Nested selections are de-duplicated (an outer clade subsumes inner ones).
    """
    if explicit:
        wanted = {str(t) for t in explicit}
        selected = {
            str(n.name)
            for n in scope_node.descendants
            if str(n.name) in wanted and _is_eligible(n)
        }
    elif rank:
        eligible = [
            n
            for n in scope_node.descendants
            if getattr(n, "rank", "") == rank and _is_eligible(n)
        ]
        n_take = max(0, math.ceil((fraction or 0.0) * len(eligible)))
        rng = random.Random(seed)
        chosen = rng.sample(eligible, min(n_take, len(eligible)))
        selected = {str(n.name) for n in chosen}
    else:
        selected = set()
    return _dedup_nested(scope_node, selected)


def _deepest_retained_ancestor(node: Node, holdout: set[str]) -> Node | None:
    """The nearest ancestor of ``node`` that survives pruning (``rho*``)."""
    ancestor = node.parent
    while ancestor is not None:
        if (
            str(ancestor.name) not in holdout
            and getattr(ancestor, "rank", "") != _SEQUENCE_RANK
        ):
            return ancestor
        ancestor = ancestor.parent
    return None


def _representative(node: Node) -> tuple[str, str] | None:
    """Deterministic representative genome (smallest header_id) of a clade."""
    leaves = _sequence_leaves(node)
    if not leaves:
        return None
    leaf = min(leaves, key=lambda leaf: getattr(leaf, "header_id", ""))
    fasta_path = getattr(leaf, "fasta_path", "")
    header_id = getattr(leaf, "header_id", "")
    if not fasta_path or not header_id:
        return None
    return fasta_path, header_id


def _clade_sketch(node: Node, k: int, sketch_size: int) -> frozenset[int]:
    rep = _representative(node)
    if rep is None:
        return frozenset()
    seq = _read_single_sequence(rep[0], rep[1])
    if not seq:
        return frozenset()
    return _genome_sketch(seq, k, sketch_size)


def _ani_proxy(jaccard: float, k: int) -> float:
    """Mash-distance ANI proxy in [0, 1] from a MinHash Jaccard estimate."""
    if jaccard <= 0.0:
        return 0.0
    mash_d = -(1.0 / k) * math.log(2.0 * jaccard / (1.0 + jaccard))
    return max(0.0, min(1.0, 1.0 - mash_d))


def _ani_bin(ani: float) -> str:
    for threshold, label in _ANI_BINS:
        if ani >= threshold:
            return label
    return _ANI_BINS[-1][1]


def _nearest_retained_relative(
    node: Node, rho: Node | None, holdout: set[str], k: int, sketch_size: int
) -> tuple[str | None, float | None, float | None, str | None]:
    """Nearest retained sibling clade under ``rho`` and the divergence to it."""
    if rho is None:
        return None, None, None, None
    node_sketch = _clade_sketch(node, k, sketch_size)
    if not node_sketch:
        return None, None, None, None
    own_lineage = {str(a.name) for a in node.ancestors} | {str(node.name)}
    best_taxid: str | None = None
    best_jaccard = -1.0
    for child in _taxonomic_children(rho):
        child_taxid = str(child.name)
        if child_taxid in holdout or child_taxid in own_lineage:
            continue
        sketch = _clade_sketch(child, k, sketch_size)
        if not sketch:
            continue
        jaccard = _jaccard(node_sketch, sketch, sketch_size)
        if jaccard > best_jaccard:
            best_taxid, best_jaccard = child_taxid, jaccard
    if best_taxid is None:
        return None, None, None, None
    ani = _ani_proxy(best_jaccard, k)
    return best_taxid, round(best_jaccard, 4), round(ani, 4), _ani_bin(ani)


def build_holdout_manifest(
    scope_node: Node,
    holdout_taxids: set[str],
    *,
    seed: int = 0,
    k: int = _KMER_K,
    sketch_size: int = _SKETCH_SIZE,
) -> list[dict[str, Any]]:
    """Build the per-clade manifest (call on the FULL tree, before pruning).

    Each entry records the held-out clade, its members, its expected commit rank
    ``rho*`` (``expected_commit_*``), and the nearest retained relative with a
    binned ANI proxy — the reproducibility contract for the benchmark.
    """
    by_taxid = {str(n.name): n for n in scope_node.descendants}
    manifest: list[dict[str, Any]] = []
    for taxid in sorted(holdout_taxids):
        node = by_taxid.get(taxid)
        if node is None:
            continue
        leaves = _sequence_leaves(node)
        rho = _deepest_retained_ancestor(node, holdout_taxids)
        sib_taxid, jaccard, ani, ani_bin = _nearest_retained_relative(
            node, rho, holdout_taxids, k, sketch_size
        )
        manifest.append(
            {
                "taxid": taxid,
                "rank": getattr(node, "rank", "unknown"),
                "name": getattr(node, "scientific_name", taxid),
                "n_genomes": len(leaves),
                "member_headers": sorted(
                    getattr(leaf, "header_id", "") for leaf in leaves
                ),
                "expected_commit_taxid": str(rho.name) if rho is not None else None,
                "expected_commit_rank": (
                    getattr(rho, "rank", None) if rho is not None else None
                ),
                "nearest_retained_sibling_taxid": sib_taxid,
                "distance_jaccard": jaccard,
                "distance_ani_proxy": ani,
                "distance_bin": ani_bin,
                "seed": seed,
            }
        )
    return manifest


def prune_holdout(scope_node: Node, holdout_taxids: set[str]) -> int:
    """Detach the held-out subtrees from the tree in place; return the count."""
    by_taxid = {str(n.name): n for n in scope_node.descendants}
    pruned = 0
    for taxid in holdout_taxids:
        node = by_taxid.get(taxid)
        if node is not None and node.parent is not None:
            node.parent = None  # detach the whole subtree
            pruned += 1
    return pruned
