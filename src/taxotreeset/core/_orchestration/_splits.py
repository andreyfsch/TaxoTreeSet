"""Leaf-level train/val/test splitting for the generation orchestrator.

Pure helpers extracted from ``generation_orchestrator.py``: they turn a child's
per-leaf extraction tasks into balanced train/val/test partitions with the
>=1-per-split guarantees, and provide the index/count boundaries those splits
rely on. There is no orchestrator state here — every caller passes what it needs.
"""

import random

from taxotreeset.core._orchestration._cluster import ClusterParams, cluster_genomes
from taxotreeset.dataset.utils import _read_single_sequence

_STRATIFIED_TRAIN_RATIO: float = 0.70
_STRATIFIED_VAL_RATIO: float = 0.15
_SPLITS: tuple[str, ...] = ("train", "val", "test")
# Optional 4th split: a whole MinHash cluster held out of training entirely, for
# an honest novel-sub-lineage generalization measurement (opt-in, only when the
# class has enough clusters to spare one — see ``_cluster_stratified_split``).
_NOVEL_SPLIT: str = "test_novel"
_ALL_SPLITS: tuple[str, ...] = (*_SPLITS, _NOVEL_SPLIT)
# Need at least this many splittable clusters to carve off a novel holdout: hold
# out one and still keep >= 2 to fill train/val/test.
_MIN_CLUSTERS_FOR_NOVEL_HOLDOUT: int = 3

# Cluster-aware window-slicing (few-genome classes): a repeating block -> split
# pattern (~5:1:1 = 71/14/14) that INTERLEAVES all three splits, so val/test
# blocks sit among train blocks (representative) instead of the genome's ends.
_WINDOW_SPLIT_PATTERN: tuple[str, ...] = (
    "train", "train", "val", "train", "train", "test", "train",
)
# Need at least this many blocks (each >= max_subseq_len, so windows keep full
# length) for the pattern to place every split; below it, keep the contiguous cut.
_MIN_BLOCKS_FOR_STRATIFY: int = 6


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


def _assign_stratified(
    tasks: list[dict],
    class_index: int,
    result: dict[str, list[dict]],
) -> None:
    """Assign whole genomes to train/val/test by ``_stratified_cuts`` ranges."""
    train_cut, val_cut = _stratified_cuts(len(tasks))
    for index, task in enumerate(tasks):
        enriched = _enrich_task(task, class_index, 0.0, 1.0)
        if index < train_cut:
            result["train"].append(enriched)
        elif index < val_cut:
            result["val"].append(enriched)
        else:
            result["test"].append(enriched)


def _cluster_stratified_split(
    tasks: list[dict],
    class_index: int,
    min_genomes: int,
    cluster_params: ClusterParams | None = None,
    novel_holdout: bool = False,
) -> dict[str, list[dict]] | None:
    """Spread each MinHash cluster of ``tasks`` across train/val/test.

    Clusters the genomes (see :func:`cluster_genomes`); when there is actionable
    sub-lineage structure, each cluster with enough genomes is stratified across
    the splits (small clusters go to train), so every split spans every
    sub-lineage — the fix for the non-i.i.d.-split instability.

    When ``novel_holdout`` is set and there are ``>= _MIN_CLUSTERS_FOR_NOVEL_HOLDOUT``
    splittable clusters, the smallest such cluster is instead held out **whole**
    into the :data:`_NOVEL_SPLIT` split — a disjoint sub-lineage the model never
    trains on, so ``test_novel`` measures novel-lineage generalization (vs the
    in-distribution ``test``). Holding out the smallest keeps training-data loss
    minimal and still leaves ``>= 2`` clusters to fill train/val/test.

    Args:
        tasks: Per-leaf tasks for the class (one per genome).
        class_index: Numeric label index for this child.
        min_genomes: Cluster size at/above which a cluster is split three ways.
        cluster_params: MinHash tuning knobs; ``None`` uses the defaults.
        novel_holdout: When True, carve off a disjoint ``test_novel`` cluster if
            there are enough clusters to spare one.

    Returns:
        The splits dict (with an optional ``test_novel`` key), or ``None`` when
        there is no structure OR the cluster split would leave a required split
        empty (the caller then keeps the whole-class random split, preserving the
        >= 1-genome-per-split guarantee).
    """
    cp = cluster_params or ClusterParams()
    clusters = cluster_genomes(
        tasks,
        k=cp.k,
        sketch_size=cp.sketch_size,
        threshold=cp.jaccard_threshold,
        min_cluster_genomes=cp.min_cluster_genomes,
        min_cluster_frac=cp.min_cluster_frac,
        max_genomes=cp.max_genomes,
    )
    if clusters is None:
        return None
    splittable = [c for c in clusters if len(c) >= min_genomes]
    small = [c for c in clusters if len(c) < min_genomes]
    holdout: list[dict] | None = None
    if novel_holdout and len(splittable) >= _MIN_CLUSTERS_FOR_NOVEL_HOLDOUT:
        splittable = sorted(splittable, key=len)
        holdout, splittable = splittable[0], splittable[1:]
    candidate: dict[str, list[dict]] = {split: [] for split in _SPLITS}
    for cluster in splittable:
        _assign_stratified(cluster, class_index, candidate)
    for cluster in small:
        # too few genomes to split three ways; keep as training signal
        for task in cluster:
            candidate["train"].append(_enrich_task(task, class_index, 0.0, 1.0))
    if not all(candidate[split] for split in _SPLITS):
        return None
    if holdout is not None:
        candidate[_NOVEL_SPLIT] = [
            _enrich_task(task, class_index, 0.0, 1.0) for task in holdout
        ]
    return candidate


def _even_split(budget: int, n_blocks: int) -> list[int]:
    """Split ``budget`` into ``n_blocks`` non-negative ints summing to it."""
    base, remainder = divmod(budget, n_blocks)
    return [base + (1 if i < remainder else 0) for i in range(n_blocks)]


def _block_stratified_windows(
    task: dict,
    class_index: int,
    max_subseq_len: int,
    result: dict[str, list[dict]],
) -> bool:
    """Spread one genome's windows across interleaved positional blocks.

    A single/few-genome class is window-sliced, and the contiguous cut
    (train 0-70% / val 70-85% / test 85-100%) puts compositionally-distinct genome
    regions in different splits — so val (a genome end) can diverge from train even
    on the same genome. Here the genome is cut into ``L // max_subseq_len`` blocks
    (each >= ``max_subseq_len`` so windows keep full length) and the blocks are
    assigned by :data:`_WINDOW_SPLIT_PATTERN`, which INTERLEAVES the splits so
    val/test blocks sit among train blocks (representative composition). Windows
    stay confined to their block, so no cross-split window overlaps (leakage-safe).

    Args:
        task: One genome's per-leaf task (``fasta_path`` / ``header_id`` / ``n``).
        class_index: Numeric label index.
        max_subseq_len: Upper window length — also the minimum block size.
        result: Splits dict to append to (mutated).

    Returns:
        True when it emitted a block-stratified split; False (do nothing) when the
        genome is unreadable or too short to hold enough blocks — the caller then
        keeps the contiguous window-slicing cut.
    """
    length = len(_read_single_sequence(task.get("fasta_path", ""),
                                       task.get("header_id", "")))
    if length <= 0 or max_subseq_len <= 0:
        return False
    n_blocks = length // max_subseq_len
    if n_blocks < _MIN_BLOCKS_FOR_STRATIFY:
        return False

    blocks: dict[str, list[int]] = {split: [] for split in _SPLITS}
    for i in range(n_blocks):
        blocks[_WINDOW_SPLIT_PATTERN[i % len(_WINDOW_SPLIT_PATTERN)]].append(i)

    n_train, n_val, n_test = _stratified_counts(task["n"])
    budgets = {"train": n_train, "val": n_val, "test": n_test}
    emitted = False
    for split, block_indices in blocks.items():
        budget = budgets[split]
        if budget <= 0 or not block_indices:
            continue
        for idx, n_block in zip(block_indices, _even_split(budget, len(block_indices))):
            if n_block <= 0:
                continue
            result[split].append(
                _enrich_task(
                    {**task, "n": n_block}, class_index, idx / n_blocks,
                    (idx + 1) / n_blocks,
                )
            )
            emitted = True
    return emitted


def _materialize_leaf_split(
    leaf_tasks: list[dict],
    class_index: int,
    rng: random.Random,
    min_genomes_for_genome_split: int = 3,
    cluster_aware: bool = False,
    max_subseq_len: int = 2000,
    cluster_params: ClusterParams | None = None,
    cluster_novel_holdout: bool = False,
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
        cluster_aware: When True, make the split representative: the genome-level
            path MinHash-clusters the genomes and spreads each sub-lineage across
            the splits, and the window-slicing path (few-genome classes) spreads
            each genome's windows across interleaved positional blocks instead of
            contiguous 0-70/70-85/85-100 regions. Both fall back to the current
            behaviour when there is no structure / the genome is too short. Off by
            default — the split is byte-identical to before.
        max_subseq_len: Upper window length; also the minimum block size for the
            cluster-aware window-slicing path (so blocked windows keep full length).
        cluster_params: MinHash tuning knobs for the genome-level cluster-aware
            path; ``None`` uses the defaults.
        cluster_novel_holdout: When True (and ``cluster_aware``), the genome-level
            path may carve off a whole cluster into a disjoint ``test_novel`` split
            (see :func:`_cluster_stratified_split`). The returned dict then has a
            4th ``test_novel`` key; otherwise it has only train/val/test.

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
        assigned = (
            _cluster_stratified_split(
                shuffled, class_index, min_genomes_for_genome_split,
                cluster_params, novel_holdout=cluster_novel_holdout,
            )
            if cluster_aware else None
        )
        if assigned is None:
            _assign_stratified(shuffled, class_index, result)
        else:
            # assigned always has the required 3 splits and MAY add test_novel.
            for split, tasks in assigned.items():
                result[split] = tasks
    else:
        for task in shuffled:
            if cluster_aware and _block_stratified_windows(
                task, class_index, max_subseq_len, result
            ):
                continue
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
