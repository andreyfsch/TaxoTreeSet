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


# Threshold relaxations tried (after the default) when the default zeroes out.
# A diverse clade (e.g. coronaviruses: every pairwise MinHash Jaccard sits below
# the 0.30 default) yields NO clusters at the default, so the split falls back to
# a volume-only assignment that can strand a whole sub-lineage in val/test (val
# looks great, test collapses to ~chance). Relaxing the Jaccard threshold recovers
# the segregable sub-lineages so each is spread across the folds.
_RELAX_SCHEDULE: tuple[float, ...] = (0.20, 0.15, 0.10, 0.07, 0.05)


def _cluster_at(sketches, n, threshold, sketch_size, min_size):
    """Cluster pre-computed sketches at ``threshold``; return the components only
    when there are >= 2 substantial clusters, else ``None``."""
    edges = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _jaccard(sketches[i], sketches[j], sketch_size) >= threshold
    ]
    clusters_idx = _connected_components(n, edges)
    if sum(1 for cluster in clusters_idx if len(cluster) >= min_size) < 2:
        return None
    return clusters_idx


def cluster_genomes_adaptive(
    tasks: list[dict],
    *,
    k: int = _KMER_K,
    sketch_size: int = _SKETCH_SIZE,
    threshold: float = _JACCARD_THRESHOLD,
    min_cluster_genomes: int = _MIN_CLUSTER_GENOMES,
    min_cluster_frac: float = _MIN_CLUSTER_FRAC,
    max_genomes: int = _MAX_GENOMES,
) -> list[list[dict]] | None:
    """Like :func:`cluster_genomes`, but sketch ONCE and relax on an empty result.

    Tries the default ``threshold`` first (unchanged behaviour for heads that
    already cluster); if it finds no actionable structure, it retries at the
    progressively lower thresholds in :data:`_RELAX_SCHEDULE` (reusing the same
    sketches), returning the first that yields >= 2 substantial clusters. This
    keeps diverse clades — whose every pairwise Jaccard is below the default —
    from falling through to a non-representative volume split. Returns ``None``
    only when even the loosest threshold finds no segregable sub-lineages.
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
    min_size = max(min_cluster_genomes, math.ceil(min_cluster_frac * n))
    schedule = [threshold] + [t for t in _RELAX_SCHEDULE if t < threshold]
    for t in schedule:
        clusters_idx = _cluster_at(sketches, n, t, sketch_size, min_size)
        if clusters_idx is not None:
            return [[tasks[i] for i in cluster] for cluster in clusters_idx]
    return None


# --------------------------------------------------------------------------- #
# Dereplicacao
# --------------------------------------------------------------------------- #
# A clusterizacao acima resolve o SPLIT: ela mantem todos os genomas e espalha as
# sub-linhagens por train/val/test. Isso nao ajuda contra redundancia de FONTE.
#
# Medido em 2026-08-22: o GenBank viral tem 268.312 genomas contra 15.091 do
# RefSeq, mas Influenza A sozinha e 147.147 deles -- 54,8%. Um head treinado sobre
# isso ve majoritariamente influenza, e redundancia de clade e o preditor de falha
# mais forte ja medido neste projeto (r=0,692). A expansao util do GenBank, sem
# Influenza A e SARS-CoV-2, e 7,2x e nao 17,8x -- mas so depois de deduplicar.
#
# Por que guloso e nao clusterizacao: `cluster_genomes` e O(n^2) e desiste acima de
# 300 genomas. O guloso compara cada genoma so contra os REPRESENTANTES ja aceitos,
# entao o custo e O(n * r) com r = quantos sobrevivem. Se a deduplicacao morde --
# que e o caso que motiva isto -- r fica pequeno e 147k entradas sao viaveis.
_DEREP_JACCARD = 0.95


def dereplicate_units(
    units: list[list[dict]],
    *,
    threshold: float = _DEREP_JACCARD,
    k: int = _KMER_K,
    sketch_size: int = _SKETCH_SIZE,
) -> list[list[dict]]:
    """Colapsa genomas quase identicos, guardando um representante de cada grupo.

    A unidade e o GENOMA, nao a sequencia: ``units`` vem de ``_group_by_genome``,
    entao um virus segmentado (influenza tem 8 segmentos) e sketchado inteiro e
    mantido ou descartado inteiro. Sketchar por segmento faria os 8 segmentos
    parecerem 8 genomas distintos e a deduplicacao nao morderia justamente onde
    ela mais importa.

    A ordem de entrada e preservada e o primeiro de cada grupo e o representante,
    entao o resultado e deterministico para uma mesma ordem de entrada.

    Args:
        units: Genomas, cada um a lista de tasks daquele acesso.
        threshold: Jaccard MinHash a partir do qual dois genomas sao replicas.
            1.0 desliga na pratica (so identicos colapsam).
        k: tamanho do k-mer.
        sketch_size: tamanho do sketch bottom-k.

    Returns:
        Os genomas mantidos, na ordem de entrada.
    """
    if threshold >= 1.0 or len(units) < 2:
        return units

    mantidos: list[list[dict]] = []
    sketches_mantidos: list[frozenset[int]] = []
    for unit in units:
        seq = "".join(
            _read_single_sequence(t.get("fasta_path", ""), t.get("header_id", ""))
            for t in unit
        )
        sk = _genome_sketch(seq, k, sketch_size)
        if not sk:
            mantidos.append(unit)          # sem sketch nao ha como julgar; mantem
            continue
        if any(_jaccard(sk, outro, sketch_size) >= threshold
               for outro in sketches_mantidos):
            continue
        mantidos.append(unit)
        sketches_mantidos.append(sk)
    return mantidos
