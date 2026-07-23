"""Reject-class virtual bucket: out-of-subtree negatives for a head.

Unlike the rare-taxa / low-capacity buckets (which absorb a parent's *own*
under-supported children), the reject bucket is a training label whose windows
come from sequence leaves **outside** the head's subtree: ``near`` = the nearest
ancestor's other-branch leaves (sibling/cousin clades), ``far`` = the remaining
leaves elsewhere in the tree. It teaches the head an explicit "does not belong
here" output, so the cascade can reject a mis-routed or out-of-distribution input
instead of confidently forcing it into one of the real classes.

The reject node is **detached** — it is not attached to the taxonomic tree and
no leaves are re-parented under it. It is purely a class entry appended to the
head's training labels; its per-leaf extraction tasks are injected directly by
the orchestrator. Its ``virtual_reject`` rank makes the existing helpers
(``is_recursion_terminator`` / ``is_protected_rank``, which match any
``virtual_*`` rank) treat it as a terminal fallback automatically.

Scope: negatives are sampled from the same vault (intra-clade), and — for shallow
heads via the **cross-domain gate** — from a pool of non-virus genomes (other
domains of life). The intra-clade negatives teach "not in this clade"; the
cross-domain pool teaches "not a virus at all", which the root/shallow heads need
(they have no intra-tree "outside") so the cascade can reject a foreign,
non-virus input at the top instead of forcing it down the tree. See P4 in
``docs/BACKLOG.md``.
"""

import logging
import random

from bigtree import Node

from taxotreeset.core.generation.distribution import _allocate_n_across_leaves
from taxotreeset.core.generation.virtual_id import make_virtual_id

logger = logging.getLogger("TaxoTreeSet.Core.Generation.RejectBucket")

_REJECT_PURPOSE: str = "reject"
_REJECT_RANK: str = "virtual_reject"
_REJECT_NAME_PREFIX: str = "virtual_reject"
_SEQUENCE_RANK: str = "sequence"
_DEFAULT_MAX_REJECT_LEAVES_PER_POOL: int = 1000
"""Cap on sampled leaves per pool (near / far).

A head's external pool can hold tens of thousands of sequence leaves; weighing
them all (each requires reading the sequence) for every head would be
prohibitive. Capping to a bounded, randomly-sampled subset keeps the cost flat
while still giving ample diversity — the per-leaf budget from
``_allocate_n_across_leaves`` only needs a few windows from each of a few hundred
leaves to reach a balanced reject class."""


def make_reject_bucket_node(
    parent_node,
    parent_taxid: str | None = None,
    parent_name: str | None = None,
) -> tuple[Node, dict]:
    """Create a detached reject-class node for a head.

    The node carries the same attribute set as the other virtual buckets
    (``rank``, ``scientific_name``, ``parent_taxid``, ``parent_name``) so it can
    be appended to ``retained_children`` and consumed by the extraction job, but
    it is **not** attached to the tree and absorbs no children — its training
    windows are external negatives supplied separately.

    The virtual TaxID is derived deterministically from the parent TaxID and the
    purpose string ``"reject"`` (via :func:`make_virtual_id`), so the same head
    always yields the same reject ID across runs.

    Args:
        parent_node: bigtree parent Node (the head this reject class belongs to).
        parent_taxid: Parent's TaxID. Defaults to ``parent_node.name``.
        parent_name: Parent's scientific name. Defaults to
            ``parent_node.scientific_name``.

    Returns:
        Two-tuple ``(reject_node, metadata)``. ``metadata`` has the keys expected
        by :func:`register_virtual_bucket` (``taxid``, ``name``, ``rank``,
        ``purpose``, ``absorbed_taxids``); ``absorbed_taxids`` is empty because no
        children are re-parented.
    """
    resolved_parent_taxid = parent_taxid or str(parent_node.name)
    resolved_parent_name = parent_name or getattr(
        parent_node, "scientific_name", resolved_parent_taxid
    )

    virtual_id = make_virtual_id(resolved_parent_taxid, _REJECT_PURPOSE)
    bucket_name = f"{_REJECT_NAME_PREFIX}_{resolved_parent_name}"

    node = Node(virtual_id)  # detached: never part of the taxonomic tree
    node.rank = _REJECT_RANK
    node.scientific_name = bucket_name
    node.parent_taxid = resolved_parent_taxid
    node.parent_name = resolved_parent_name

    metadata = {
        "taxid": virtual_id,
        "name": bucket_name,
        "rank": _REJECT_PURPOSE,
        "purpose": _REJECT_PURPOSE,
        "absorbed_taxids": [],
    }
    return node, metadata


def _cap_pool(pool: list, max_per_pool: int, rng: random.Random) -> list:
    """Randomly down-sample ``pool`` to ``max_per_pool`` leaves (or return as-is)."""
    if max_per_pool and len(pool) > max_per_pool:
        return rng.sample(pool, max_per_pool)
    return pool


def sample_reject_leaves(
    current_node,
    max_per_pool: int = _DEFAULT_MAX_REJECT_LEAVES_PER_POOL,
    rng: random.Random | None = None,
    cross_domain_leaves: list | None = None,
    cross_domain_max_depth: int = 0,
) -> tuple[list, list]:
    """Partition the tree's sequence leaves into ``near`` and ``far`` negatives.

    ``near`` = sequence leaves under the nearest ancestor of ``current_node`` that
    has any leaf outside ``current_node`` (i.e. the closest sibling/cousin clades);
    ``far`` = all other external sequence leaves. Leaves under ``current_node``
    itself (the head's genuine members) are excluded from both. Each pool is
    randomly capped to ``max_per_pool`` leaves to bound the per-head cost (see
    :data:`_DEFAULT_MAX_REJECT_LEAVES_PER_POOL`).

    **Cross-domain gate.** When ``cross_domain_leaves`` is given and the head is
    shallow (``current_node.depth <= cross_domain_max_depth``), those non-virus
    leaves are appended to ``far`` — so the head also learns "not a virus at all".
    For the whole-tree head (e.g. the root) there is no intra-tree "outside", so
    the cross-domain pool becomes its *only* negative source; without it the root
    head has no reject signal at all.

    Args:
        current_node: bigtree node of the head whose reject negatives are sampled.
        max_per_pool: Maximum leaves kept per pool; 0 disables the cap.
        rng: Random source for the cap (deterministic ``Random(0)`` when None).
        cross_domain_leaves: Non-virus sequence leaves (the P4 domain gate); each
            needs ``fasta_path`` / ``header_id`` / ``rank == "sequence"``. ``None``
            or empty keeps the intra-clade-only behaviour.
        cross_domain_max_depth: Max ``node.depth`` (bigtree, root = 1) that
            receives the cross-domain pool; deeper heads keep intra-clade
            negatives only (a non-virus is a trivial negative there).

    Returns:
        Two-tuple ``(near_leaves, far_leaves)``, each capped. ``near`` is empty
        for the whole-tree head; ``far`` is empty too unless the cross-domain gate
        supplies it.
    """
    rng = rng if rng is not None else random.Random(0)
    own_leaves = set(current_node.leaves)
    # The tree's sequence-leaf set is invariant during scheduling — bucketing only
    # re-parents subtrees within the tree and reject nodes are detached, so no
    # sequence leaf is added or removed. Cache the list on the root: recomputing
    # root.leaves (a whole-tree DFS) for every head would make the per-node binary
    # path O(nodes x tree_size).
    root = current_node.root
    all_seq_leaves = getattr(root, "_reject_seq_leaves_cache", None)
    if all_seq_leaves is None:
        all_seq_leaves = [
            leaf for leaf in root.leaves
            if getattr(leaf, "rank", "") == _SEQUENCE_RANK
        ]
        root._reject_seq_leaves_cache = all_seq_leaves
    external = [leaf for leaf in all_seq_leaves if leaf not in own_leaves]

    near: list = []
    far: list = []
    if external:
        ancestor = current_node.parent
        while ancestor is not None:
            if ancestor is root:
                # The root's external seq leaves are exactly `external` (already
                # computed) — reuse it rather than re-scanning the whole tree.
                candidate = external
            else:
                candidate = [
                    leaf for leaf in ancestor.leaves
                    if getattr(leaf, "rank", "") == _SEQUENCE_RANK
                    and leaf not in own_leaves
                ]
            if candidate:
                near = candidate
                break
            ancestor = ancestor.parent

        near_set = set(near)
        far = [leaf for leaf in external if leaf not in near_set]

    # Cross-domain (non-virus) gate: shallow heads also reject "not a virus at
    # all". For the whole-tree head (no intra-tree external) this is the head's
    # only negative source. Appended to `far` so the near/far budget split treats
    # it as a distant clade (and gives the root head its entire reject budget).
    if cross_domain_leaves and current_node.depth <= cross_domain_max_depth:
        far = far + list(cross_domain_leaves)

    return _cap_pool(near, max_per_pool, rng), _cap_pool(far, max_per_pool, rng)


def build_reject_tasks(
    near_leaves: list,
    far_leaves: list,
    n_reject: int,
    near_far_ratio: float,
    min_subseq_len: int,
) -> list[dict]:
    """Build per-leaf extraction tasks for the reject class.

    Splits the ``n_reject`` window budget between the ``near`` and ``far`` leaf
    pools by ``near_far_ratio`` and allocates each split across its leaves with
    :func:`_allocate_n_across_leaves` (the same allocator used for real classes),
    so a reject window is produced exactly like any other. When only one pool is
    non-empty it receives the full budget.

    Args:
        near_leaves: Sibling/cousin sequence leaves (see :func:`sample_reject_leaves`).
        far_leaves: Remaining external sequence leaves.
        n_reject: Total windows to allocate to the reject class.
        near_far_ratio: Fraction of the budget drawn from ``near`` (the rest from
            ``far``). Ignored when one pool is empty.
        min_subseq_len: Minimum subseq length, for per-leaf share weighting.

    Returns:
        List of per-leaf task dicts (``fasta_path``, ``header_id``, ``n``). Empty
        when ``n_reject <= 0`` or both pools are empty.
    """
    if n_reject <= 0 or (not near_leaves and not far_leaves):
        return []

    if near_leaves and far_leaves:
        # Clamp to [0, n_reject] so a mis-set ratio (e.g. --reject-near-far-end > 1,
        # which the CLI does not bound) cannot drive n_far negative and feed a
        # negative sample count into extraction.
        n_near = max(0, min(n_reject, round(n_reject * near_far_ratio)))
        n_far = n_reject - n_near
    elif near_leaves:
        n_near, n_far = n_reject, 0
    else:
        n_near, n_far = 0, n_reject

    tasks: list[dict] = []
    if n_near and near_leaves:
        tasks.extend(_allocate_n_across_leaves(near_leaves, n_near, min_subseq_len))
    if n_far and far_leaves:
        tasks.extend(_allocate_n_across_leaves(far_leaves, n_far, min_subseq_len))
    return tasks
