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

# Cluster-aware window-slicing (few-genome classes): a repeating block -> split
# pattern (~5:1:1 = 71/14/14) that INTERLEAVES all three splits, so val/test
# blocks sit among train blocks (representative) instead of the genome's ends.
_WINDOW_SPLIT_PATTERN: tuple[str, ...] = (
    "train", "train", "val", "train", "train", "test", "train",
)
# Need at least this many blocks (each >= max_subseq_len, so windows keep full
# length) for the pattern to place every split; below it, keep the contiguous cut.
_MIN_BLOCKS_FOR_STRATIFY: int = 6


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
    """Assign whole genomes to train/val/test balancing WINDOW VOLUME.

    Each task carries ``n`` = its window budget, and per-genome ``n`` is highly
    unequal — ``_allocate_n_across_leaves`` weights it by genome size — so
    splitting by genome COUNT 70/15/15 lets a few large genomes dominate one
    split's volume. For a class whose windows come from a handful of big genomes
    (most acutely a binary head's *negatives*, drawn from a few large external
    genomes) the realized volume split then swings wildly with the shuffle (val
    anywhere from ~0% to ~45%), which inverts the class priors between train and
    val and makes the head untrainable. This instead places each whole genome —
    leakage-safe, a genome never straddles splits — into the split currently
    furthest below its target volume (``0.70 / 0.15 / 0.15`` of the class's total
    windows), largest genome first. Every split is guaranteed >= 1 genome when
    ``len(tasks) >= 3``.
    """
    total = sum(task["n"] for task in tasks) or 1
    targets = {
        "train": _STRATIFIED_TRAIN_RATIO * total,
        "val": _STRATIFIED_VAL_RATIO * total,
        "test": (1.0 - _STRATIFIED_TRAIN_RATIO - _STRATIFIED_VAL_RATIO) * total,
    }
    volume = {split: 0.0 for split in _SPLITS}
    buckets: dict[str, list[dict]] = {split: [] for split in _SPLITS}
    for task in sorted(tasks, key=lambda t: t["n"], reverse=True):
        # Largest remaining deficit first; ties resolve to _SPLITS order (train).
        split = max(_SPLITS, key=lambda s: targets[s] - volume[s])
        buckets[split].append(task)
        volume[split] += task["n"]

    _ensure_each_split_nonempty(buckets, volume)
    for split in _SPLITS:
        for task in buckets[split]:
            result[split].append(_enrich_task(task, class_index, 0.0, 1.0))


def _ensure_each_split_nonempty(
    buckets: dict[str, list[dict]],
    volume: dict[str, float],
) -> None:
    """Guarantee every split holds >= 1 genome (when >= 3 exist in total).

    Volume-greedy packing can leave a split empty — e.g. three equal genomes all
    pulled toward train/val, or one genome dwarfing the rest. For each empty
    split, move the smallest genome out of the split that has the most volume
    among those holding >= 2, restoring the >= 1-per-split invariant the old
    count-based cut guaranteed. With >= 3 genomes a donor always exists (pigeonhole),
    and moving only from a >= 2 split can never empty the donor.
    """
    for split in _SPLITS:
        if buckets[split]:
            continue
        donor = max(
            (s for s in _SPLITS if len(buckets[s]) >= 2),
            key=lambda s: volume[s], default=None,
        )
        if donor is None:
            continue  # fewer than 3 genomes total; cannot fill every split
        task = min(buckets[donor], key=lambda t: t["n"])
        buckets[donor].remove(task)
        volume[donor] -= task["n"]
        buckets[split].append(task)
        volume[split] += task["n"]


def _cluster_stratified_split(
    tasks: list[dict],
    class_index: int,
    min_genomes: int,
    cluster_params: ClusterParams | None = None,
) -> dict[str, list[dict]] | None:
    """Spread each MinHash cluster of ``tasks`` across train/val/test.

    Clusters the genomes (see :func:`cluster_genomes`); when there is actionable
    sub-lineage structure, each cluster with enough genomes is stratified across
    the splits (small clusters go to train), so every split spans every
    sub-lineage — the fix for the non-i.i.d.-split instability.

    Args:
        tasks: Per-leaf tasks for the class (one per genome).
        class_index: Numeric label index for this child.
        min_genomes: Cluster size at/above which a cluster is split three ways.
        cluster_params: MinHash tuning knobs; ``None`` uses the defaults.

    Returns:
        The splits dict, or ``None`` when there is no structure OR the cluster
        split would leave a split empty (the caller then keeps the whole-class
        random split, preserving the >= 1-genome-per-split guarantee).
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
    candidate: dict[str, list[dict]] = {split: [] for split in _SPLITS}
    for cluster in clusters:
        if len(cluster) >= min_genomes:
            _assign_stratified(cluster, class_index, candidate)
        else:
            # too few genomes to split three ways; keep as training signal
            for task in cluster:
                candidate["train"].append(_enrich_task(task, class_index, 0.0, 1.0))
    if all(candidate[split] for split in _SPLITS):
        return candidate
    return None


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
        task: One genome's per-leaf task (``fasta_path`` / ``header_id`` / ``n``,
            and optionally ``length`` — the genome length, precomputed during task
            distribution so this path need not re-read the genome).
        class_index: Numeric label index.
        max_subseq_len: Upper window length — also the minimum block size.
        result: Splits dict to append to (mutated).

    Returns:
        True when it emitted a block-stratified split; False (do nothing) when the
        genome is unreadable or too short to hold enough blocks — the caller then
        keeps the contiguous window-slicing cut.
    """
    # Prefer the length precomputed during task distribution (no I/O); fall back
    # to reading the genome only when it is absent (e.g. reject-bucket tasks).
    length = task.get("length")
    if length is None:
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


def _assign_stratified_hybrid(
    tasks: list[dict],
    class_index: int,
    result: dict[str, list[dict]],
    max_subseq_len: int,
) -> None:
    """Block-stratify dominant genomes across splits; whole-assign the rest.

    Volume-aware whole-genome assignment (:func:`_assign_stratified`) can still
    skew a split when a **single** genome's window volume exceeds a val-sized
    share (``> _STRATIFIED_VAL_RATIO`` of the class total): it cannot sit whole in
    val/test without overflowing, so it lands lopsided (the P13 residual, seen on
    binary-head *negatives* dominated by one large external genome — e.g. SARS-CoV
    at neg 86/7/7). Such genomes are instead **block-stratified** — their windows
    spread across interleaved positional blocks, leakage-safe — so each contributes
    ~70/15/15 on its own; the remaining (small) genomes are volume-bin-packed by
    :func:`_assign_stratified`. Both subsets are independently balanced, so their
    sum tracks 70/15/15. A dominant genome too short to block (``< _MIN_BLOCKS``)
    falls back to whole assignment.

    Only used for negatives (a negative genome may appear in several splits — it is
    a non-member everywhere), never positives (whole-genome splitting is their
    stricter leakage guarantee).
    """
    total = sum(task["n"] for task in tasks) or 1
    threshold = _STRATIFIED_VAL_RATIO * total
    regular: list[dict] = []
    for task in tasks:
        if task["n"] > threshold and _block_stratified_windows(
            task, class_index, max_subseq_len, result
        ):
            continue  # dominant genome spread across splits
        regular.append(task)
    if regular:
        _assign_stratified(regular, class_index, result)


def _materialize_leaf_split(
    leaf_tasks: list[dict],
    class_index: int,
    rng: random.Random,
    min_genomes_for_genome_split: int = 3,
    cluster_aware: bool = False,
    max_subseq_len: int = 2000,
    cluster_params: ClusterParams | None = None,
    block_stratify_large: bool = False,
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
            contiguous 0-70/70-85/85-100 regions. Both fall back to the plain
            split when there is no structure / the genome is too short. The
            generation pipeline enables this **by default** (``--no-cluster-aware-
            split`` opts out); this helper's parameter defaults to False so a bare
            call still gets the plain split.
        max_subseq_len: Upper window length; also the minimum block size for the
            cluster-aware window-slicing path (so blocked windows keep full length).
        cluster_params: MinHash tuning knobs for the genome-level cluster-aware
            path; ``None`` uses the defaults.
        block_stratify_large: When True (cluster-aware only), a genome whose window
            volume would dominate a split is block-stratified across splits instead
            of assigned whole — see :func:`_assign_stratified_hybrid`. Set for
            *negatives*, whose pool can be dominated by one large external genome.

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
        if cluster_aware and block_stratify_large:
            _assign_stratified_hybrid(shuffled, class_index, result, max_subseq_len)
        elif cluster_aware and (
            assigned := _cluster_stratified_split(
                shuffled, class_index, min_genomes_for_genome_split, cluster_params,
            )
        ) is not None:
            for split in _SPLITS:
                result[split] = assigned[split]
        else:
            _assign_stratified(shuffled, class_index, result)
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
