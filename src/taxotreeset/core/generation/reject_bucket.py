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

Scope: **intra-virus** — negatives are sampled from the same vault. A non-virus
"domain gate" for shallow heads (negatives drawn from other domains of life) is a
separate, future capability.
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
) -> tuple[list, list]:
    """Partition the tree's sequence leaves into ``near`` and ``far`` negatives.

    ``near`` = sequence leaves under the nearest ancestor of ``current_node`` that
    has any leaf outside ``current_node`` (i.e. the closest sibling/cousin clades);
    ``far`` = all other external sequence leaves. Leaves under ``current_node``
    itself (the head's genuine members) are excluded from both. Each pool is
    randomly capped to ``max_per_pool`` leaves to bound the per-head cost (see
    :data:`_DEFAULT_MAX_REJECT_LEAVES_PER_POOL`).

    Args:
        current_node: bigtree node of the head whose reject negatives are sampled.
        max_per_pool: Maximum leaves kept per pool; 0 disables the cap.
        rng: Random source for the cap (deterministic ``Random(0)`` when None).

    Returns:
        Two-tuple ``(near_leaves, far_leaves)``. Both are empty when the head is
        the whole tree (e.g. the root), since there is then no intra-tree
        "outside" — that case needs a cross-domain (non-virus) source instead.
    """
    rng = rng if rng is not None else random.Random(0)
    own_leaves = set(current_node.leaves)
    all_seq_leaves = [
        leaf for leaf in current_node.root.leaves
        if getattr(leaf, "rank", "") == _SEQUENCE_RANK
    ]
    external = [leaf for leaf in all_seq_leaves if leaf not in own_leaves]
    if not external:
        return [], []

    near: list = []
    ancestor = current_node.parent
    while ancestor is not None:
        candidate = [
            leaf for leaf in ancestor.leaves
            if getattr(leaf, "rank", "") == _SEQUENCE_RANK and leaf not in own_leaves
        ]
        if candidate:
            near = candidate
            break
        ancestor = ancestor.parent

    near_set = set(near)
    far = [leaf for leaf in external if leaf not in near_set]
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
        n_near = max(0, round(n_reject * near_far_ratio))
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
