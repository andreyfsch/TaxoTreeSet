"""Tool-free MinHash clustering of a class's genomes for cluster-aware splits.

A head's genomes are often phylogenetically clustered (non-i.i.d.), so a random
genome-level split can segregate a whole sub-lineage into val/test: the model
never trains on that sub-lineage, so val (a distinct cluster) tanks while test
(genomes resembling train) looks great — unstable, misleading metrics.

``cluster_genomes`` sketches each genome with a bottom-``sketch_size`` MinHash
over its k-mers (stdlib ``zlib.crc32`` as the hash — no external tool) and
single-linkage-clusters them by the bottom-k MinHash Jaccard estimate. It returns
clusters ONLY when there is *actionable* structure (>= 2 clusters, the two
largest each big enough), so the split step can spread each cluster across
train/val/test; otherwise it returns ``None`` and the caller keeps its current
random split. The clustering thus self-verifies the need — homogeneous heads pay
nothing and keep the old behaviour.
"""

import math
import zlib
from collections import defaultdict
from dataclasses import dataclass

from taxotreeset.dataset.utils import _read_single_sequence

_KMER_K = 21
_SKETCH_SIZE = 200
_JACCARD_THRESHOLD = 0.30
_MIN_CLUSTER_GENOMES = 2
# The two largest clusters must EACH cover at least this fraction of the genomes
# for the structure to be actionable. Without it, a diverse head (RefSeq is ~1
# genome/species, so most genomes are singletons) would pass on a couple of tiny
# near-clone pairs, then the stratified split would starve val/test and fall back
# anyway — so require substantial, segregable sub-lineages instead.
_MIN_CLUSTER_FRAC = 0.10
# Pairwise clustering is O(n^2); above this genome count, skip it (caller falls
# back to the random split) rather than stall a wide head.
_MAX_GENOMES = 300


@dataclass(frozen=True)
class ClusterParams:
    """Tunable MinHash-clustering knobs for the cluster-aware split.

    Defaults mirror the module constants. The clustering rarely fires on RefSeq
    (~1 genome/species, so genomes are diverse), so a dataset with denser
    sub-lineages (e.g. a GenBank strain collection) can lower ``threshold`` /
    ``min_cluster_frac`` to make it engage. ``jaccard_threshold``,
    ``min_cluster_genomes`` and ``min_cluster_frac`` are the decision knobs the
    CLI exposes (``--cluster-*``); ``k`` / ``sketch_size`` / ``max_genomes`` are
    cost/quality constants, overridable here in code if ever needed.
    """

    k: int = _KMER_K
    sketch_size: int = _SKETCH_SIZE
    jaccard_threshold: float = _JACCARD_THRESHOLD
    min_cluster_genomes: int = _MIN_CLUSTER_GENOMES
    min_cluster_frac: float = _MIN_CLUSTER_FRAC
    max_genomes: int = _MAX_GENOMES


def _genome_sketch(seq: str, k: int, sketch_size: int) -> frozenset[int]:
    """Return the bottom-``sketch_size`` MinHash sketch (crc32 of each k-mer)."""
    if len(seq) < k:
        return frozenset()
    hashes = {
        zlib.crc32(seq[i:i + k].encode("ascii")) for i in range(len(seq) - k + 1)
    }
    return frozenset(sorted(hashes)[:sketch_size])


def _jaccard(a: frozenset[int], b: frozenset[int], sketch_size: int) -> float:
    """Bottom-k (KMV) MinHash Jaccard estimate between two sketches."""
    if not a or not b:
        return 0.0
    merged = sorted(a | b)[:sketch_size]
    return sum(1 for h in merged if h in a and h in b) / len(merged)


def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Single-linkage clusters (union-find) over the given similarity edges."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def cluster_genomes(
    tasks: list[dict],
    *,
    k: int = _KMER_K,
    sketch_size: int = _SKETCH_SIZE,
    threshold: float = _JACCARD_THRESHOLD,
    min_cluster_genomes: int = _MIN_CLUSTER_GENOMES,
    min_cluster_frac: float = _MIN_CLUSTER_FRAC,
    max_genomes: int = _MAX_GENOMES,
) -> list[list[dict]] | None:
    """Cluster a class's genomes by MinHash similarity, if there is structure.

    Reads each genome (``task['fasta_path']`` / ``task['header_id']``), sketches
    it, and single-linkage-clusters by MinHash Jaccard >= ``threshold``.

    Args:
        tasks: Per-genome task dicts (each references a vault sequence).
        k: k-mer size for the sketch.
        sketch_size: Bottom-k MinHash sketch size per genome.
        threshold: MinHash Jaccard above which two genomes join a cluster.
        min_cluster_genomes: Absolute floor on a cluster's size to count.
        min_cluster_frac: A cluster must also cover at least this fraction of the
            genomes to count; there must be >= 2 such clusters. This rejects
            diverse heads (mostly singletons + a few near-clone pairs) where the
            stratified split would gain nothing.
        max_genomes: Skip clustering above this count (the pairwise pass is
            O(n^2)); the caller then keeps the random split.

    Returns:
        A list of clusters (each a list of the input task dicts) when there is
        actionable structure; otherwise ``None`` (too large, or homogeneous /
        only singletons), signalling the caller to keep its current split.
    """
    n = len(tasks)
    if n < 2 or n > max_genomes:
        return None
    sketches = [
        _genome_sketch(
            _read_single_sequence(t.get("fasta_path", ""), t.get("header_id", "")),
            k, sketch_size,
        )
        for t in tasks
    ]
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _jaccard(sketches[i], sketches[j], sketch_size) >= threshold
    ]
    clusters_idx = _connected_components(n, edges)
    # Actionable only with >= 2 substantial clusters (each >= min_cluster_genomes
    # AND >= min_cluster_frac of the genomes) — a couple of tiny near-clone pairs
    # in an otherwise-diverse head is not segregable structure worth splitting on.
    min_size = max(min_cluster_genomes, math.ceil(min_cluster_frac * n))
    if sum(1 for cluster in clusters_idx if len(cluster) >= min_size) < 2:
        return None
    return [[tasks[i] for i in cluster] for cluster in clusters_idx]
