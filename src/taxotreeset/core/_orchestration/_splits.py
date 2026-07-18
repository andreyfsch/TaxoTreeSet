"""Leaf-level train/val/test splitting for the generation orchestrator.

Pure helpers extracted from ``generation_orchestrator.py``: they turn a child's
per-leaf extraction tasks into balanced train/val/test partitions with the
>=1-per-split guarantees, and provide the index/count boundaries those splits
rely on. There is no orchestrator state here — every caller passes what it needs.
"""

import random

_STRATIFIED_TRAIN_RATIO: float = 0.70
_STRATIFIED_VAL_RATIO: float = 0.15
_SPLITS: tuple[str, ...] = ("train", "val", "test")


def _stratified_cuts(leaf_count: int) -> tuple[int, int]:
    """Return ``(train_cut, val_cut)`` index boundaries for a leaf-level split.

    The naive ``max(1, int(L * ratio))`` boundaries leave **test empty at exactly
    three leaves** (train=2, val=1, test=0), because the two ``max(1, ...)`` floors
    consume all three leaves before test gets one. That produced degenerate
    classes — present in the label set and trained, but with zero test support, so
    their per-class metrics are undefined and they drag down macro-F1. This clamp
    pulls ``val_cut`` back so test always receives at least one leaf (and then
    protects train from being emptied), guaranteeing every split gets >= 1 leaf
    whenever ``leaf_count >= 3``.

    Args:
        leaf_count: Number of sequence leaves in the class (assumed >= 3; the
            scarcity path handles 1-2 leaves via disjoint within-sequence regions).

    Returns:
        ``(train_cut, val_cut)`` such that train = ``[0, train_cut)``,
        val = ``[train_cut, val_cut)``, test = ``[val_cut, leaf_count)``.
    """
    train_cut = max(1, int(leaf_count * _STRATIFIED_TRAIN_RATIO))
    val_cut = train_cut + max(1, int(leaf_count * _STRATIFIED_VAL_RATIO))
    if val_cut >= leaf_count:
        val_cut = leaf_count - 1
        if train_cut >= val_cut:
            train_cut = val_cut - 1
    return train_cut, val_cut


def _stratified_counts(n_total: int) -> tuple[int, int, int]:
    """Split one genome's ``n_total`` subseqs into ``(n_train, n_val, n_test)``.

    The window-slicing scarcity path (< 3 genomes) samples ``n`` subsequences
    from each of a genome's three positional regions. Naive ``int(n * ratio)``
    floors leave **train empty at n_total <= 1** and **val empty up to
    n_total == 6** (``int(6 * 0.15) == 0``), so a data-poor class could be
    present in the label set yet have no training (or validation) support — the
    same degeneracy ``_stratified_cuts`` prevents at the leaf level. This clamp
    guarantees train receives >= 1 whenever ``n_total >= 1`` and every split
    receives >= 1 whenever ``n_total >= 3``. For ``n_total`` of 1 or 2 a full
    three-way split is impossible, so it fills train first, then test, then val
    (an untrained class is worse than an unevaluated one).

    Args:
        n_total: Subsequences allocated to this genome (>= 0).

    Returns:
        ``(n_train, n_val, n_test)`` summing to ``n_total``.
    """
    if n_total <= 0:
        return 0, 0, 0
    if n_total == 1:
        return 1, 0, 0
    if n_total == 2:
        return 1, 0, 1
    n_train = max(1, int(n_total * _STRATIFIED_TRAIN_RATIO))
    n_val = max(1, int(n_total * _STRATIFIED_VAL_RATIO))
    n_test = n_total - n_train - n_val
    if n_test < 1:
        # Reclaim one for test from train; n_total >= 3 keeps train >= 1.
        n_test = 1
        n_train = n_total - n_val - n_test
    return n_train, n_val, n_test


def _enrich_task(
    task: dict,
    class_index: int,
    start_pct: float,
    end_pct: float,
) -> dict:
    """Add slicing and class index fields to a per-leaf task.

    Args:
        task: Per-leaf task with 'fasta_path', 'header_id', 'n'.
        class_index: Numeric label for this child.
        start_pct: Sequence slicing start as fraction.
        end_pct: Sequence slicing end as fraction.

    Returns:
        Worker-ready task dict.
    """
    return {
        "fasta_path": task["fasta_path"],
        "header_id": task["header_id"],
        "n": task["n"],
        "class_idx": class_index,
        "start_pct": start_pct,
        "end_pct": end_pct,
    }


def _materialize_leaf_split(
    leaf_tasks: list[dict],
    class_index: int,
    rng: random.Random,
    min_genomes_for_genome_split: int = 3,
) -> dict[str, list[dict]]:
    """Split a single child's per-leaf tasks into train/val/test.

    With ``>= min_genomes_for_genome_split`` genomes the split is by genome
    (leakage-safe: a genome's windows never straddle splits); below it, the
    window-slicing fallback splits each genome positionally so a data-poor
    class still yields non-empty train/val/test. Binary heads pass 4 (rather
    than 3), because a 3-genome genome-level split leaves the test empty.

    Args:
        leaf_tasks: Per-leaf task dicts from
            ``distribute_n_per_class_across_leaves``.
        class_index: Numeric label index for this child.
        rng: Random instance for deterministic shuffling.
        min_genomes_for_genome_split: Threshold below which to use the
            window-slicing fallback (default 3 = the multi-class behaviour).

    Returns:
        Dictionary with 'train', 'val', 'test' keys; each value
        is a list of worker-ready task dicts with class_idx and
        split fractions filled in.
    """
    result: dict[str, list[dict]] = {split: [] for split in _SPLITS}

    if not leaf_tasks:
        return result

    shuffled = list(leaf_tasks)
    rng.shuffle(shuffled)

    if len(shuffled) >= min_genomes_for_genome_split:
        train_cut, val_cut = _stratified_cuts(len(shuffled))

        for index, task in enumerate(shuffled):
            enriched = _enrich_task(task, class_index, 0.0, 1.0)
            if index < train_cut:
                result["train"].append(enriched)
            elif index < val_cut:
                result["val"].append(enriched)
            else:
                result["test"].append(enriched)
    else:
        for task in shuffled:
            n_train, n_val, n_test = _stratified_counts(task["n"])
            if n_train > 0:
                result["train"].append(
                    _enrich_task(
                        {**task, "n": n_train},
                        class_index,
                        0.0,
                        0.70,
                    )
                )
            if n_val > 0:
                result["val"].append(
                    _enrich_task(
                        {**task, "n": n_val},
                        class_index,
                        0.70,
                        0.85,
                    )
                )
            if n_test > 0:
                result["test"].append(
                    _enrich_task(
                        {**task, "n": n_test},
                        class_index,
                        0.85,
                        1.0,
                    )
                )

    return result
