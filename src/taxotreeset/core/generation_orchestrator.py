import hashlib
from typing import Any
from collections import Counter, defaultdict
from src.taxotreeset.dataset.utils import _get_fasta_sequence_length
from src.taxotreeset.dataset.builder import DatasetBuilder
from src.taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from src.taxotreeset.io.downloader import NCBIDownloader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from bigtree import Node, find_attrs
import numpy as np
import os
import sys
import gc
import json
import logging
import collections
import itertools
import random
from typing import Any, Dict, List

# 🚀 ESTABILIZAÇÃO: Bloqueia threads em C para evitar Core Dumped e conflito de memória
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")

PROTECTED_RANKS = frozenset({
    # fallbacks semânticos curados no mapping.json (999000-003)
    "realm_group",
    "virtual_cluster",   # buckets criados pelo k-means
    # buckets criados pelo Op3 (esta defesa torna a função idempotente)
    "virtual_bucket",
    "virtual_low_data",
})

# Defaults da camada de balanceamento. Todos sobrescritíveis via CLI.
DEFAULT_MIN_NUM_SEQS = 1000
DEFAULT_CUTOFF_PERCENTAGE = 98.0
DEFAULT_USE_EXACT_CAPACITY = True
DEFAULT_MAX_N_PER_CLASS = 20_000

# Bloom filter para aproximação de capacidade
BLOOM_FALSE_POSITIVE_RATE = 0.01
# ~10M unique seqs por nó é teto razoável
BLOOM_EXPECTED_INSERTIONS = 10_000_000

# Cache global de sequências lidas do LMDB. Reduz leituras redundantes
# durante o cálculo de capacidade (uma sequência aparece como descendente
# de múltiplos ancestrais, e cada chamada de compute_node_capacity tentaria
# relê-la). Cache é por-processo; workers spawnados não compartilham.
_SEQUENCE_CACHE = {}
# ~300MB teto pra cache (15000 seqs × 2x margin)
_SEQUENCE_CACHE_MAX_ENTRIES = 30_000


def _read_sequence_cached(fasta_path: str, header_id: str) -> str:
    """Lê sequência do LMDB com cache em memória do processo atual."""
    from src.taxotreeset.dataset.utils import _read_single_sequence
    key = (fasta_path, header_id)
    if key in _SEQUENCE_CACHE:
        return _SEQUENCE_CACHE[key]

    sequence = _read_single_sequence(fasta_path, header_id) or ""

    # Defesa contra explosão: se cache crescer demais, descarta entradas
    # antigas (não LRU sofisticado, só limpa em lote pra simplicidade).
    if len(_SEQUENCE_CACHE) >= _SEQUENCE_CACHE_MAX_ENTRIES:
        # Mantém só as últimas 50% das entradas inseridas (FIFO simples)
        keys = list(_SEQUENCE_CACHE.keys())
        to_remove = keys[:len(keys) // 2]
        for k in to_remove:
            del _SEQUENCE_CACHE[k]

    _SEQUENCE_CACHE[key] = sequence
    return sequence


def make_virtual_id(parent_taxid: str, purpose: str) -> str:
    """Gera ID virtual determinístico de 9 dígitos. Prefixo '9' identifica virtuais."""
    key = f"{parent_taxid}:{purpose}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    suffix = int(h[:8], 16) % 100_000_000
    return f"9{suffix:08d}"


def compute_node_capacity(
    node,
    min_len: int,
    leaf_cache: dict,
    mode: str = "exact",
    max_useful: int = None,
) -> int:
    """
    Computa a capacidade biológica de um nó: o número de subseqs únicas de
    `min_len` bases extraíveis dos genomas das folhas sob ele via sliding window.

    Esta é a métrica "saco real" do mestrado (extract_max_subseqs_set adaptada
    para o pipeline cascateado).

    Modos:
        "exact": constrói set Python completo de subseqs únicas. Memória pode
                 chegar a centenas de GB para heads gigantes. Use em servidor
                 de alta RAM.
        "approximate": Bloom filter com fpr ~1%. Memória limitada a ~12MB
                       independente do tamanho do nó. Use em desktop/WSL.

    Args:
        node: nó bigtree (Node) cuja capacidade será calculada
        min_len: tamanho da janela deslizante (típico 100bp)
        leaf_cache: dict {taxid_str: List[sequence_leaf]} já populado pelo
                    pipeline antes desta chamada
        mode: "exact" ou "approximate"

    Returns:
        int: número estimado de subseqs únicas de `min_len` sob o nó
    """
    import logging
    logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")

    # Coleta TODAS as folhas de sequência descendentes (não filhos diretos)
    all_seq_leaves = []
    if hasattr(node, "descendants"):
        for d in node.descendants:
            if getattr(d, "rank", "") == "sequence":
                all_seq_leaves.append(d)
    else:
        # Fallback: nó pode ser virtual recém-criado
        for child in node.children:
            child_taxid = str(child.name)
            for seq_leaf in leaf_cache.get(child_taxid, []):
                all_seq_leaves.append(seq_leaf)

    if not all_seq_leaves:
        return 0

    if mode == "exact":
        return _capacity_exact(all_seq_leaves, min_len, max_useful=max_useful)
    elif mode == "approximate":
        return _capacity_approximate(all_seq_leaves, min_len, max_useful=max_useful)
    else:
        raise ValueError(f"Modo de capacidade desconhecido: {mode}")


def compute_balanced_extraction_plan(
    parent_node,
    children: list,
    leaf_cache: dict,
    min_len: int = 100,
    min_num_seqs: int = DEFAULT_MIN_NUM_SEQS,
    cutoff_percentage: float = DEFAULT_CUTOFF_PERCENTAGE,
    use_exact_capacity: bool = DEFAULT_USE_EXACT_CAPACITY,
    max_n_per_class: int = DEFAULT_MAX_N_PER_CLASS,
) -> dict:
    """
    Aplica a lógica de balanceamento do mestrado (write_csvs) ao contexto
    cascateado do TaxoTreeSet.
    
    Para cada filho (classe da head), computa sua capacidade real. Decide:
    
    Cenário 1: capacidade mínima >= min_num_seqs
        Todos os filhos são treináveis. n_per_class = capacidade mínima
        (nivelamento por baixo). Nenhuma exclusão.
    
    Cenário 2: capacidade mínima < min_num_seqs
        Calcula cutoff pelo percentil que mantém `cutoff_percentage`% dos
        filhos. n_per_class = cutoff. Filhos abaixo do cutoff são marcados
        para virar bucket virtual_low_data.

    Args:
        parent_node: nó pai (cuja head está sendo planejada)
        children: lista de filhos efetivos (já passou pelo Op3)
        leaf_cache: dict {taxid: List[sequence_leaf]}
        min_len: tamanho mínimo da subseq (típico 100)
        min_num_seqs: threshold para nivelamento sem cutoff
        cutoff_percentage: percentual de filhos mantidos quando cutoff necessário
        use_exact_capacity: True usa modo exact, False usa approximate

    Returns:
        dict com chaves:
            "n_per_class": int — subseqs por classe (uniforme para todos)
            "eligible_children": List[Node] — filhos treináveis na head
            "low_data_children": List[Node] — filhos absorvidos no bucket low_data
            "capacities": Dict[str, int] — taxid -> capacidade calculada
            "scenario": str — "level_all" ou "cutoff_applied"
            "decision_log": str — explicação textual da decisão
    """
    import logging
    logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")

    parent_taxid = str(parent_node.name)
    capacity_mode = "exact" if use_exact_capacity else "approximate"

    # Passo 1: computar capacidade de cada filho
    capacities = {}
    for child in children:
        child_taxid = str(child.name)
        cap = compute_node_capacity(
            child, min_len, leaf_cache, mode=capacity_mode, max_useful=max_n_per_class)
        capacities[child_taxid] = cap

    if not capacities:
        return {
            "n_per_class": 0,
            "eligible_children": [],
            "low_data_children": [],
            "capacities": {},
            "scenario": "empty",
            "decision_log": f"Pai {parent_taxid}: nenhum filho com capacidade > 0",
        }

    min_capacity = min(capacities.values())

    # Passo 2: decisão por cenário
    if min_capacity >= min_num_seqs:
        # CENÁRIO 1: todos podem fornecer >= min_num_seqs
        # Nivela pelo menor; nenhuma classe descartada
        n_per_class = min_capacity
        eligible = list(children)
        low_data = []
        scenario = "level_all"
        decision = (
            f"CENÁRIO 1 (nivelamento global): min_cap={min_capacity} >= "
            f"min_num_seqs={min_num_seqs}. Todos os {len(children)} filhos "
            f"contribuem com n={n_per_class} subseqs"
        )
    else:
        # CENÁRIO 2: cutoff por percentil
        sorted_caps = sorted(capacities.values())
        # Mantém `cutoff_percentage`% dos filhos -> descartamos os (100-p)% piores
        cutoff_idx = max(
            0, int(len(sorted_caps) * (100.0 - cutoff_percentage) / 100.0))
        cutoff_value = sorted_caps[cutoff_idx]

        eligible = []
        low_data = []
        for child in children:
            child_cap = capacities[str(child.name)]
            if child_cap >= cutoff_value:
                eligible.append(child)
            else:
                low_data.append(child)

        n_per_class = cutoff_value
        scenario = "cutoff_applied"
        decision = (
            f"CENÁRIO 2 (cutoff percentil {cutoff_percentage}%): "
            f"min_cap={min_capacity} < min_num_seqs={min_num_seqs}. "
            f"Cutoff={cutoff_value}; eligible={len(eligible)}, "
            f"low_data={len(low_data)}, n_per_class={n_per_class}"
        )

    logger.debug(f"[BALANCE] node={parent_taxid} {decision}")

    original_n = n_per_class
    if max_n_per_class > 0 and n_per_class > max_n_per_class:
        n_per_class = max_n_per_class
        decision += (
            f" | CAPPED a {max_n_per_class} (era {original_n}, "
            f"redução {original_n / max_n_per_class:.1f}x)"
        )
        scenario = scenario + "_capped"

    logger.debug(f"[BALANCE] node={parent_taxid} {decision}")

    return {
        "n_per_class": n_per_class,
        "n_per_class_uncapped": original_n,
        "eligible_children": eligible,
        "low_data_children": low_data,
        "capacities": capacities,
        "scenario": scenario,
        "decision_log": decision,
    }


def _capacity_exact(seq_leaves: list, min_len: int, max_useful: int = None) -> int:
    """
    Modo exato: union de sets de subseqs únicas. Garante precisão biológica
    no custo de O(total_bases) em memória.
    """
    union_set = set()
    early_stop_threshold = (max_useful * 5) if max_useful else None
    for leaf in seq_leaves:
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            continue

        sequence = _read_sequence_cached(fasta_path, header_id)
        if not sequence or len(sequence) < min_len:
            continue

        # Sliding window de min_len bases
        seq_len = len(sequence)
        for i in range(seq_len - min_len + 1):
            union_set.add(sequence[i:i + min_len])

        # Early stop
        if early_stop_threshold and len(union_set) >= early_stop_threshold:
            break

    return len(union_set)


def _capacity_approximate(seq_leaves: list, min_len: int, max_useful: int = None) -> int:
    """
    Modo aproximado: Bloom filter conta subseqs únicas com fpr ~1%.
    Memória constante (~12MB) independente do tamanho do nó.
    Para heads de Caudoviricetes (potencial 100M+ subseqs), o exato seria
    proibitivo em WSL; aqui o approximate é a única opção viável.
    """
    import hashlib
    import math
    import logging
    logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")
    

    # Parâmetros do Bloom filter
    n = BLOOM_EXPECTED_INSERTIONS
    p = BLOOM_FALSE_POSITIVE_RATE
    m = int(-n * math.log(p) / (math.log(2) ** 2))  # bits no array
    k = max(1, int((m / n) * math.log(2)))           # número de hash functions

    # Implementação simples via bytearray + hashes derivados
    bit_array = bytearray((m + 7) // 8)
    seen_estimate = 0

    # Early stop threshold: 5× max_useful dá margem pra erro do Bloom
    # e ainda evita trabalho perdido em nós saturados.
    early_stop_threshold = (max_useful * 5) if max_useful else None

    def _set_bit(idx):
        bit_array[idx // 8] |= (1 << (idx % 8))

    def _get_bit(idx):
        return (bit_array[idx // 8] >> (idx % 8)) & 1

    def _hashes(item_bytes):
        # Gera k hashes via double hashing (Kirsch-Mitzenmacher)
        # Substitui md5/sha1 por hashes mais rápidos baseados em bytes diretos
        # ~10x mais rápido que hashlib em strings curtas
        h1 = int.from_bytes(item_bytes[:8].ljust(8, b'\x00'), "little") & 0x7FFFFFFFFFFFFFFF
        h2 = int.from_bytes(item_bytes[-8:].ljust(8, b'\x00'), "little") & 0x7FFFFFFFFFFFFFFF
        for i in range(k):
            yield (h1 + i * h2) % m

    total_leaves = len(seq_leaves)
    processed = 0

    for leaf in seq_leaves:
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            continue

        sequence = _read_sequence_cached(fasta_path, header_id)
        if not sequence or len(sequence) < min_len:
            processed += 1
            continue

        seq_len = len(sequence)
        for i in range(seq_len - min_len + 1):
            subseq = sequence[i:i + min_len].encode("ascii")

            # Check if all k bits are already set
            indices = list(_hashes(subseq))
            already_present = all(_get_bit(idx) for idx in indices)

            if not already_present:
                seen_estimate += 1
                for idx in indices:
                    _set_bit(idx)
        
        processed += 1
        if processed % 200 == 0:
            logger.info(
                f"  [CAPACITY-PROGRESS] {processed}/{total_leaves} folhas, "
                f"unique count atual: {seen_estimate:,}"
            )

        # EARLY STOP
        if early_stop_threshold and seen_estimate >= early_stop_threshold:
            logger.info(
                f"  [CAPACITY-EARLY-STOP] após {processed}/{total_leaves} folhas: "
                f"count={seen_estimate:,} >= {early_stop_threshold:,} threshold "
                f"(cap final aplicará de qualquer forma)"
            )
            break

    return seen_estimate


def classify_children_by_rank(
    parent_node: Any,
    children: list,
    min_subclades_per_bucket: int = 5,
    canonical_rank_override: str | None = None,
    virtual_id_registry: dict | None = None,
) -> tuple[list, dict]:
    """
    Aplica a estratégia Op3: separa filhos canônicos dos anômalos, e
    decide quais ranks anômalos viram bucket próprio versus quais se
    mesclam num bucket genérico.

    Retorna:
        - filhos_efetivos: lista de filhos a usar como classes da head
          (mistura nós reais + objetos virtuais com taxid/name/rank simulados)
        - new_virtual_ids: dict acumulado de virtual_id → metadata
    """
    parent_taxid = str(parent_node.name)

    if not children:
        return [], virtual_id_registry if virtual_id_registry is not None else {}

    # Separa filhos protegidos dos demais ANTES de qualquer classificação.
    # Protegidos sempre vão pros canônicos efetivos sem participar do voto
    # de rank canônico nem da bucketização. Isso preserva fallbacks semânticos
    # (realm_group de mapping.json) e impede classify recursivo em buckets
    # virtuais já criados.
    protected = [c for c in children
                 if getattr(c, "rank", "") in PROTECTED_RANKS]
    classifiable = [c for c in children
                    if getattr(c, "rank", "") not in PROTECTED_RANKS]

    if not classifiable:
        # Tudo é protegido — não há nada a classificar. Retorna todos como canônicos.
        return list(protected), virtual_id_registry if virtual_id_registry is not None else {}

    # 1. Determinar rank canônico
    if canonical_rank_override:
        canonical_rank = canonical_rank_override
    else:
        rank_counts = Counter(getattr(c, "rank", "unknown")
                              for c in classifiable)
        canonical_rank, _ = rank_counts.most_common(1)[0]

    # 2. Particionar filhos
    canonicos = []
    anomalos_por_rank = defaultdict(list)

    for c in classifiable:
        c_rank = getattr(c, "rank", "unknown")
        if c_rank == canonical_rank:
            canonicos.append(c)
        else:
            anomalos_por_rank[c_rank].append(c)

    # 3. Decidir destino de cada grupo anômalo
    new_virtual_ids = virtual_id_registry if virtual_id_registry is not None else {}
    filhos_efetivos = list(protected) + list(canonicos)
    mescla_residual = []

    for rank, grupo in anomalos_por_rank.items():
        if len(grupo) >= min_subclades_per_bucket:
            # Vira bucket próprio com ID virtual
            vid = make_virtual_id(parent_taxid, rank)

            if vid in new_virtual_ids:
                existing = new_virtual_ids[vid]
                if existing["parent_taxid"] != parent_taxid or existing["purpose"] != rank:
                    raise RuntimeError(
                        f"Colisão de virtual ID: {vid} já existe como "
                        f"(parent={existing['parent_taxid']}, purpose={existing['purpose']}), "
                        f"tentando atribuir (parent={parent_taxid}, purpose={rank}). "
                        f"Pode ser colisão estatística ou bug em make_virtual_id()."
                    )

            virtual_node = _make_virtual_bucket_node(
                vid, parent_taxid, getattr(
                    parent_node, "scientific_name", parent_taxid),
                rank, grupo, parent_node
            )
            filhos_efetivos.append(virtual_node)
            new_virtual_ids[vid] = {
                "parent_taxid": parent_taxid,
                "parent_name": getattr(parent_node, "scientific_name", parent_taxid),
                "purpose": rank,
                "description": (
                    f"Bucket de {rank}s floating sob {parent_taxid} "
                    f"({len(grupo)} subclados; canonical rank do pai = {canonical_rank})"
                ),
                "subclade_count": len(grupo),
            }
            logger.debug(
                f"[BUCKET-RANK] {parent_taxid} → virtual {vid} "
                f"contém {len(grupo)} {rank}s anômalos"
            )
        else:
            # Vai pra mescla residual
            mescla_residual.extend(grupo)
            logger.debug(
                f"[BUCKET-MERGE] {parent_taxid} rank={rank} tem {len(grupo)} "
                f"subclados (< {min_subclades_per_bucket}); mesclando em 'misc'"
            )

    # 4. Bucket residual (se houver)
    if mescla_residual:
        vid = make_virtual_id(parent_taxid, "misc")

        if vid in new_virtual_ids:
            existing = new_virtual_ids[vid]
            if existing["parent_taxid"] != parent_taxid or existing["purpose"] != rank:
                raise RuntimeError(
                    f"Colisão de virtual ID: {vid} já existe como "
                    f"(parent={existing['parent_taxid']}, purpose={existing['purpose']}), "
                    f"tentando atribuir (parent={parent_taxid}, purpose={rank}). "
                    f"Pode ser colisão estatística ou bug em make_virtual_id()."
                )

        virtual_node = _make_virtual_bucket_node(
            vid, parent_taxid, getattr(
                parent_node, "scientific_name", parent_taxid),
            "misc", mescla_residual, parent_node
        )
        filhos_efetivos.append(virtual_node)
        ranks_in_misc = Counter(getattr(c, "rank", "unknown")
                                for c in mescla_residual)
        new_virtual_ids[vid] = {
            "parent_taxid": parent_taxid,
            "parent_name": getattr(parent_node, "scientific_name", parent_taxid),
            "purpose": "misc",
            "description": (
                f"Bucket genérico de filhos anômalos com cardinalidade < "
                f"{min_subclades_per_bucket} por rank. "
                f"Composição: {dict(ranks_in_misc)}"
            ),
            "subclade_count": len(mescla_residual),
            "rank_composition": dict(ranks_in_misc),
        }

    return filhos_efetivos, new_virtual_ids


def _make_low_data_bucket_node(
    parent_node,
    low_data_children: list,
    virtual_id_registry: dict,
    capacities: dict,
) -> "Node":
    """
    Cria um bucket virtual_low_data sob parent_node, reparentando os filhos
    pobres para dentro dele. Adiciona entrada no virtual_id_registry com
    metadados ricos (lista de taxids absorvidos, capacidades individuais).

    Retorna o virtual node criado, já anexado a parent_node.
    """
    from bigtree import Node

    parent_taxid = str(parent_node.name)
    parent_name = getattr(parent_node, "scientific_name", parent_taxid)
    vid = make_virtual_id(parent_taxid, "low_data")

    # Check de colisão (mesma defesa do _make_virtual_bucket_node)
    if vid in virtual_id_registry:
        existing = virtual_id_registry[vid]
        if existing["parent_taxid"] != parent_taxid or existing["purpose"] != "low_data":
            raise RuntimeError(
                f"Colisão de virtual ID: {vid} já existe como "
                f"(parent={existing['parent_taxid']}, purpose={existing['purpose']}), "
                f"tentando atribuir (parent={parent_taxid}, purpose=low_data). "
                f"Pode ser colisão estatística ou bug em make_virtual_id()."
            )

    # Cria virtual e reparenta filhos pobres
    virtual = Node(
        vid,
        parent=parent_node,
        rank="virtual_low_data",
        scientific_name=f"{parent_name}_low_data",
        bucket_purpose="low_data",
        bucket_size=len(low_data_children),
    )

    excluded_taxids = []
    excluded_capacities = {}
    for child in low_data_children:
        child_taxid = str(child.name)
        child.parent = virtual
        excluded_taxids.append(child_taxid)
        excluded_capacities[child_taxid] = capacities.get(child_taxid, 0)

    # Registra com metadados ricos pra rastreabilidade
    virtual_id_registry[vid] = {
        "parent_taxid": parent_taxid,
        "parent_name": parent_name,
        "purpose": "low_data",
        "description": (
            f"Bucket de classes com material genético insuficiente "
            f"({len(low_data_children)} classes absorvidas via cutoff "
            f"percentil). Cada classe individual abaixo do threshold "
            f"min_num_seqs."
        ),
        "subclade_count": len(low_data_children),
        "excluded_taxids": excluded_taxids,
        "excluded_capacities": excluded_capacities,
    }

    return virtual


def _distribute_n_per_class_across_leaves(
    child_node,
    n_per_class: int,
    leaf_cache: dict,
) -> dict:
    """
    Distribui o orçamento n_per_class entre as folhas de sequência sob
    child_node, proporcionalmente ao tamanho do genoma de cada folha
    (adaptação de get_subseqs_from_final_node do mestrado).

    Folhas com genomas maiores recebem mais subseqs. Compensação iterativa
    de erros de arredondamento garante que sum(n_distribuídos) == n_per_class.

    Returns:
        dict {header_id: int} mapeando cada folha ao número de subseqs
        que ela deve contribuir.
    """

    # Coleta folhas sob este child
    seq_leaves = []
    if hasattr(child_node, "descendants"):
        for d in child_node.descendants:
            if getattr(d, "rank", "") == "sequence":
                seq_leaves.append(d)

    if not seq_leaves:
        return {}

    # Caso especial: 1 folha só recebe todo o orçamento
    if len(seq_leaves) == 1:
        leaf = seq_leaves[0]
        return {getattr(leaf, "header_id", ""): n_per_class}

    # Coleta comprimentos dos genomas
    leaf_lengths = {}
    for leaf in seq_leaves:
        header_id = getattr(leaf, "header_id", "")
        fasta_path = getattr(leaf, "fasta_path", "")
        if not header_id or not fasta_path:
            continue
        sequence = _read_sequence_cached(fasta_path, header_id)
        if sequence:
            leaf_lengths[header_id] = len(sequence)

    if not leaf_lengths:
        return {}

    total_len = sum(leaf_lengths.values())
    if total_len == 0:
        return {}

    # Distribui proporcionalmente
    n_per_leaf = {}
    for header_id, leaf_len in leaf_lengths.items():
        fraction = leaf_len / total_len
        n_per_leaf[header_id] = int(round(fraction * n_per_class))

    # Compensação iterativa: se sum < n_per_class, adiciona +1 ao top;
    # se sum > n_per_class, remove -1 do bottom. Garante igualdade exata.
    current_total = sum(n_per_leaf.values())
    diff = n_per_class - current_total

    if diff != 0:
        # Ordena folhas por tamanho (maiores primeiro pra distribuir o erro
        # nelas, evitando que folhas pequenas fiquem com 0 ou negativo)
        sorted_leaves = sorted(leaf_lengths.items(), key=lambda x: -x[1])

        if diff > 0:
            # Falta: adiciona +1 aos N maiores
            for header_id, _ in sorted_leaves[:diff]:
                n_per_leaf[header_id] += 1
        else:
            # Sobra: remove -1 dos N maiores (que podem absorver sem zerar)
            for header_id, _ in sorted_leaves[:abs(diff)]:
                if n_per_leaf[header_id] > 1:
                    n_per_leaf[header_id] -= 1

    return n_per_leaf


def _make_virtual_bucket_node(virtual_id: str, parent_taxid: str, parent_name: str,
                              purpose: str, contained_children: list, parent_node: Node) -> Node:
    """
    Cria um nó bigtree virtual que substitui um conjunto de filhos anômalos.
    O nó virtual tem como descendentes (via a propriedade leaves) todas as
    sequências dos filhos contidos, mas aparece como UM filho do pai original.
    """
    from bigtree import Node

    # Re-parenteia os filhos contidos sob o novo virtual node
    virtual = Node(
        virtual_id,
        parent=parent_node,
        rank="virtual_bucket",
        scientific_name=f"{parent_name}_unclassified_{purpose}",
        bucket_purpose=purpose,
        bucket_size=len(contained_children),
    )

    for child in contained_children:
        child.parent = virtual

    return virtual


def _compute_kmer_vector(sequence: str, vocab: list, k: int = 4) -> list:
    """Projeta a sequência no espaço vetorial completo de frequências de K-mers."""
    if not sequence:
        return [0.0] * len(vocab)
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    counts = collections.Counter(kmers)
    total = sum(counts.values()) or 1
    v_dict = {km: counts[km] / total for km in kmers}
    return [v_dict.get(km, 0.0) for km in vocab]


class GenerationOrchestrator:
    def __init__(
        self,
        registry,
        config_path="configs/mapping.json",
        vault_path="data/vault",
        output_dir="data/datasets",
        max_subseq_len=2000,
        seed=42,
        output_format="parquet",
        min_subclades_per_bucket=5,
        min_num_seqs=DEFAULT_MIN_NUM_SEQS,
        cutoff_percentage=DEFAULT_CUTOFF_PERCENTAGE,
        use_exact_capacity=DEFAULT_USE_EXACT_CAPACITY,
        max_n_per_class=DEFAULT_MAX_N_PER_CLASS,
    ):
        self.registry = registry
        self.vault_path = vault_path
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_subseq_len = max_subseq_len
        self.seed = seed
        self.output_format = output_format
        self.config_path = config_path
        self.min_subclades_per_bucket = min_subclades_per_bucket
        self.min_num_seqs = min_num_seqs
        self.cutoff_percentage = cutoff_percentage
        self.use_exact_capacity = use_exact_capacity
        self.max_n_per_class = max_n_per_class
        self.virtual_id_registry = {}
        self._sequence_cache = {}

        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
        else:
            self.mapping = {}

        self.downloader = NCBIDownloader(
            registry=self.registry, vault_path=self.vault_path)
        self.builder = DatasetBuilder(
            output_dir=self.output_dir,
            max_subseq_len=self.max_subseq_len,
            seed=self.seed,
            output_format=self.output_format
        )


    def run_pipeline(self, target_group: str, min_num_seqs: int = 100, percentage: int = 10, abundance_threshold: int = 2, max_budget: int = 50000) -> None:
        """
        Executes the hierarchical nested dataset extraction phase.
        """
        self.downloader.download_all_pending()

        group_to_id = {"viruses": "10239", "bacteria": "2",
                       "archaea": "2157", "eukaryotes": "2759"}
        target_group_clean = target_group.lower().strip()
        domain_taxid = group_to_id.get(target_group_clean)

        scope_config = self.mapping.get("scopes", {}).get(domain_taxid, {})
        fallback_ids = {str(k) for k in scope_config.get(
            "virtual_id_labels", {}).keys()}
        # Apenas o default_id do escopo é um fallback "verdadeiro" — destino de táxons
        # sem regra de redirect, portanto sem hierarquia NCBI abaixo. K-means só faz
        # sentido aqui, onde é necessário inventar uma estrutura de classes.
        # Os demais virtual_id_labels (999001-003 no caso viral) são agrupadores
        # semânticos que preservam a taxonomia NCBI abaixo deles e não devem ser
        # reorganizados por similaridade de k-mers.
        default_fallback_id = str(scope_config.get("default_id", ""))
        clustering_eligible = {
            default_fallback_id} if default_fallback_id else set()

        logger.info(
            f"Fallback IDs ativos para '{target_group_clean}': {sorted(fallback_ids)}")
        logger.info(
            f"K-means elegível apenas em: {sorted(clustering_eligible)}")
        if not fallback_ids:
            logger.warning(
                f"Escopo '{domain_taxid}' não declara virtual_id_labels. "
                f"K-Means clustering não será aplicado neste domínio."
            )
        logger.info(
            f"Fallback IDs ativos para '{target_group_clean}': {sorted(fallback_ids)}")

        taxon_tree = generate_seqs_by_taxon_tree(
            registry_path=self.registry.registry_path,
            vault_path=self.vault_path,
            domain_taxid=domain_taxid,
            mapping_path=self.config_path,
            noise_patterns_path=os.path.join(  # <-- novo
                os.path.dirname(self.config_path),
                "noise_patterns.json"
            ),
        )

        domain_nodes = [node for node in taxon_tree.descendants if str(
            node.name) == domain_taxid]
        if not domain_nodes:
            logger.error(f"Domain root TaxID '{domain_taxid}' not found.")
            return
        superkingdom_node = domain_nodes[0]

        leaf_cache = {}
        for leaf in find_attrs(superkingdom_node, "rank", "sequence"):
            for ancestor in leaf.ancestors:
                if ancestor.name != "root":
                    leaf_cache.setdefault(str(ancestor.name), []).append(leaf)
            leaf_cache.setdefault(str(leaf.name), []).append(leaf)

        vocab = ["".join(p) for p in itertools.product(
            ["A", "T", "C", "G"], repeat=4)]

        macro_extraction_jobs = []
        master_manifest = {}
        # parent_taxid → single_child_taxid (redirect automático)
        passthrough_map = {}

        kmeans_stats = {"checked": 0, "fired": 0}

        def process_node_cascaded(current_node: Node, accumulated_path: str):
            parent_taxid = str(current_node.name)
            children = [c for c in current_node.children if getattr(
                c, "rank", "") != "sequence"]

            # K-Means restrito ao Fallback ID dinâmico do domínio atual
            path_components = set(accumulated_path.split(os.sep))
            is_under_default_fallback = (
                bool(clustering_eligible &
                     path_components) or parent_taxid in clustering_eligible
            )

            if len(children) > 15 and is_under_default_fallback:
                kmeans_stats["fired"] += 1
                logger.debug(
                    f"[KMEANS-FIRE] node={parent_taxid} children={len(children)} "
                    f"(sob default_fallback {default_fallback_id})"
                )
                vectors = []
                valid_children = []

                for child in children:
                    child_leaves = leaf_cache.get(str(child.name), [])
                    if child_leaves:
                        seq = _read_sequence_cached(
                            getattr(child_leaves[0], "fasta_path", ""),
                            getattr(child_leaves[0], "header_id", "")
                        )
                        if seq:
                            seq_len = len(seq)
                            if seq_len <= 4000:
                                sample_seq = seq
                            else:
                                mid_point = seq_len // 2
                                sample_seq = seq[mid_point -
                                                 2000: mid_point + 2000]
                            vectors.append(
                                _compute_kmer_vector(sample_seq, vocab))
                            valid_children.append(child)

                if vectors:
                    n_cl = min(15, len(vectors))
                    kmeans = MiniBatchKMeans(
                        n_clusters=n_cl, random_state=42,
                        batch_size=64, max_no_improvement=10
                    )
                    labels = kmeans.fit_predict(
                        np.array(vectors, dtype=np.float32))

                    bucket_nodes = {}
                    for child, lbl in zip(valid_children, labels):
                        b_id = f"{parent_taxid}{lbl:02d}"
                        if b_id not in bucket_nodes:
                            b_node = Node(
                                b_id, rank="virtual_cluster",
                                scientific_name=f"Molecular_Subclade_{b_id}"
                            )
                            bucket_nodes[b_id] = b_node
                        child.parent = bucket_nodes[b_id]

                    for b_node in bucket_nodes.values():
                        b_node.parent = current_node

                    children = list(bucket_nodes.values())
                    del vectors
                    gc.collect()
            elif is_under_default_fallback:
                kmeans_stats["checked"] += 1
                logger.debug(
                    f"[KMEANS-SKIP] node={parent_taxid} children={len(children)} (≤15)")

            self._scheduled_count = getattr(self, "_scheduled_count", 0) + 1
            if self._scheduled_count % 100 == 0:
                logger.info(f"[PROGRESS] {self._scheduled_count} nós agendados, "
                            f"jobs acumulados: {len(macro_extraction_jobs)}")
            self._schedule_decision_point(
                current_node, children, accumulated_path,
                abundance_threshold, max_budget,
                macro_extraction_jobs, master_manifest, passthrough_map, leaf_cache
            )

            # Re-coleta filhos APÓS a possível mutação. Agora current_node.children
            # reflete: filhos canônicos + buckets virtuais (no lugar dos anômalos
            # absorvidos). A recursão deve descer por essa nova lista.
            children_post_classify = [
                c for c in current_node.children
                if getattr(c, "rank", "") != "sequence"
            ]

            for child in children_post_classify:
                # Virtual buckets (Op3 ou k-means) são folhas lógicas da hierarquia
                # de heads: a head do pai já classifica suas sequências. Recursar
                # dentro re-classificaria conjuntos já agrupados, gerando buckets
                # aninhados sem sentido.
                if getattr(child, "rank", "").startswith("virtual_"):
                    logger.debug(
                        f"[CASCADE-STOP] node={parent_taxid} → virtual {child.name}: "
                        f"cascata para (head do pai já cobre classes do bucket)"
                    )
                    continue

                process_node_cascaded(
                    child,
                    os.path.join(accumulated_path, str(child.name))
                )

        process_node_cascaded(superkingdom_node, str(superkingdom_node.name))

        logger.info(
            f"Pre-building directory tree for {len(macro_extraction_jobs)} nodes...")
        for job in tqdm(macro_extraction_jobs, desc="Creating output dirs", unit=" dir"):
            os.makedirs(job[1], exist_ok=True)

        # CLEANUP CRÍTICO ANTES DO POOL:
        # Os workers serão forked (ou spawned) a partir daqui. Tudo que está vivo no
        # escopo do pai será copiado/inicializado nos workers. Como a árvore bigtree
        # completa pode chegar a centenas de milhares de nós e nada do que vai abaixo
        # precisa dela (os jobs já trazem todo dado necessário em forma de tuplas),
        # largamos a árvore agora e forçamos GC para evitar swap nos workers.
        logger.info(
            "Liberando árvore taxonômica antes do pool (jobs já carregam dados necessários)...")
        del taxon_tree
        del superkingdom_node
        del leaf_cache
        del domain_nodes
        gc.collect()

        with open(os.path.join(self.output_dir, f"manifest_{target_group_clean}.json"), "w", encoding="utf-8") as f:
            json.dump(master_manifest, f, indent=2)

        # Distribuição de cardinalidade das heads
        from collections import Counter
        card_dist = Counter(len(v["labels"]) for v in master_manifest.values())
        logger.info(
            f"Distribuição de cardinalidade das heads: {dict(sorted(card_dist.items()))}")

        with open(os.path.join(self.output_dir, f"passthroughs_{target_group_clean}.json"), "w", encoding="utf-8") as f:
            json.dump(passthrough_map, f, indent=2)

        with open(os.path.join(self.output_dir, f"virtual_id_registry_{target_group_clean}.json"),
                  "w", encoding="utf-8") as f:
            json.dump({
                "_meta": {
                    "domain": target_group_clean,
                    "scheme": "sha256(parent_taxid:purpose)[:8] % 1e8, prefixed with '9'",
                    "min_subclades_per_bucket": self.min_subclades_per_bucket,
                },
                "virtual_ids": self.virtual_id_registry,
            }, f, indent=2)

        logger.info(
            f"Manifest: {len(master_manifest)} nós com head treinável, "
            f"{len(passthrough_map)} passthroughs (redirect automático), "
            f"Virtual ID registry: {len(self.virtual_id_registry)} buckets criados"
        )

        if macro_extraction_jobs:
            import multiprocessing as mp
            logger.info("Executing PARALLEL disk extraction...")
            results = self.builder.build_node_dataset(
                macro_extraction_jobs, parallel=True)

        logger.info(
            f"K-Means: disparou em {kmeans_stats['fired']} nós, "
            f"pulou {kmeans_stats['checked']} nós sob fallback com ≤15 filhos"
        )
        logger.info(f"Total de jobs agendados: {len(macro_extraction_jobs)}")


    def _prepare_stratified_split(self, leaves, rng):
        """
        Estratifica folhas em train/val/test (70/15/15).
        
        Duas estratégias dependendo do número de folhas:
        - >= 3 folhas: split POR FOLHA. Cada folha vai INTEIRA pra um split.
                    Garante diversidade real entre splits (acessões distintas).
        - < 3 folhas: split POR SEQUÊNCIA. Cada folha é fatiada percentualmente
                    em 70/15/15. Garante 3 splits não-vazios mesmo com 1-2 folhas.
        
        Returns:
            dict {"train": [(fasta_path, header_id, start_pct, end_pct), ...], ...}
        """
        splits = {"train": [], "val": [], "test": []}
        if not leaves:
            return splits

        n = len(leaves)

        if n >= 3:
            # Split por folha — cada folha vai inteira pra UM split
            shuffled = list(leaves)
            rng.shuffle(shuffled)

            train_end = max(1, int(n * 0.70))
            val_end = train_end + max(1, int(n * 0.15))
            # Garante pelo menos 1 folha em test
            if val_end >= n:
                val_end = max(train_end + 1, n - 1)

            for leaf in shuffled[:train_end]:
                f_path = getattr(leaf, "fasta_path", "")
                h_id = getattr(leaf, "header_id", "")
                splits["train"].append((f_path, h_id, 0.0, 1.0))

            for leaf in shuffled[train_end:val_end]:
                f_path = getattr(leaf, "fasta_path", "")
                h_id = getattr(leaf, "header_id", "")
                splits["val"].append((f_path, h_id, 0.0, 1.0))

            for leaf in shuffled[val_end:]:
                f_path = getattr(leaf, "fasta_path", "")
                h_id = getattr(leaf, "header_id", "")
                splits["test"].append((f_path, h_id, 0.0, 1.0))
        else:
            # Split por sequência — cada folha é fatiada em 70/15/15
            for leaf in leaves:
                f_path = getattr(leaf, "fasta_path", "")
                h_id = getattr(leaf, "header_id", "")
                splits["train"].append((f_path, h_id, 0.0, 0.70))
                splits["val"].append((f_path, h_id, 0.70, 0.85))
                splits["test"].append((f_path, h_id, 0.85, 1.0))

        return splits


    def _schedule_decision_point(
        self, current_node, children_list, current_path,
        abundance_threshold, extraction_budget_per_node,
        macro_extraction_jobs, master_manifest,
        passthrough_map, leaf_cache,
    ):
        """
        Agenda jobs de extração pra uma head decisora (current_node), aplicando:
        1. Op3: classify_children_by_rank (separa canônicos + rank-anômalos)
        2. Balanceamento: compute_balanced_extraction_plan (capacidade real + cutoff)
        3. Criação do bucket virtual_low_data (se há filhos descartados pelo cutoff)
        4. Distribuição proporcional do n_per_class entre folhas (mestrado adaptado)
        5. Criação de tasks com n variável por accession
        """
        import logging
        logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")

        parent_taxid = str(current_node.name)

        # Coleta folhas via árvore ao vivo
        all_leaves = [
            lf for lf in current_node.descendants
            if getattr(lf, "rank", "") == "sequence"
        ]
        if not all_leaves:
            logger.debug(
                f"[SCHEDULE-SKIP] node={parent_taxid} "
                f"rank={getattr(current_node, 'rank', '?')} (sem folhas)"
            )
            return

        # ============================================================
        # PASSO 1: Op3 (classify por rank)
        # ============================================================
        children_efetivos, new_virt_ids = classify_children_by_rank(
            current_node, children_list,
            min_subclades_per_bucket=self.min_subclades_per_bucket,
            canonical_rank_override=None,  # futuro: vem de clade_policies
            virtual_id_registry=self.virtual_id_registry,
        )

        # ============================================================
        # PASSO 2: Balanceamento (compute_balanced_extraction_plan)
        # ============================================================
        if not children_efetivos:
            # Nó sem filhos taxonômicos válidos após Op3. Pode acontecer em folhas
            # terminais que tinham só sequências; tratar como SKIP/PASSTHROUGH.
            logger.debug(
                f"[SCHEDULE-SKIP] node={parent_taxid} (nenhum filho após Op3)"
            )
            return

        plan = compute_balanced_extraction_plan(
            parent_node=current_node,
            children=children_efetivos,
            leaf_cache=leaf_cache,
            min_len=100,
            min_num_seqs=self.min_num_seqs,
            cutoff_percentage=self.cutoff_percentage,
            use_exact_capacity=self.use_exact_capacity,
            max_n_per_class=self.max_n_per_class,
        )

        # ============================================================
        # PASSO 3: Criar bucket low_data se necessário
        # ============================================================
        if plan["low_data_children"]:
            low_data_node = _make_low_data_bucket_node(
                parent_node=current_node,
                low_data_children=plan["low_data_children"],
                virtual_id_registry=self.virtual_id_registry,
                capacities=plan["capacities"],
            )
            # Adiciona o bucket ao conjunto de filhos efetivos da head
            eligible_plus_lowdata = list(
                plan["eligible_children"]) + [low_data_node]
            logger.debug(
                f"[BUCKET-LOWDATA] node={parent_taxid} → virtual {low_data_node.name} "
                f"absorveu {len(plan['low_data_children'])} classes pobres"
            )
        else:
            eligible_plus_lowdata = list(plan["eligible_children"])

        if not eligible_plus_lowdata:
            logger.debug(
                f"[SCHEDULE-SKIP] node={parent_taxid} (nenhum filho elegível após "
                f"balanceamento)"
            )
            return

        # ============================================================
        # PASSO 4: Verifica se há sub-taxonomias suficientes pra head
        # ============================================================
        n_per_class = plan["n_per_class"]

        if len(eligible_plus_lowdata) < 2:
            # Head trivial (uma classe só). Aplica passthrough pro único filho.
            only_child = eligible_plus_lowdata[0]
            only_child_taxid = str(only_child.name)
            only_child_name = getattr(
                only_child, "scientific_name", only_child_taxid)
            passthrough_map[parent_taxid] = {
                "redirect_to": only_child_taxid,
                "redirect_name": only_child_name,
            }
            logger.debug(
                f"[SCHEDULE-PASSTHROUGH] node={parent_taxid} → {only_child_taxid} "
                f"({only_child_name})"
            )
            return

        # ============================================================
        # PASSO 5: Constrói local_labels a partir dos filhos efetivos
        # ============================================================
        local_labels = {}
        for idx, child in enumerate(eligible_plus_lowdata):
            child_taxid = str(child.name)
            child_name = getattr(child, "scientific_name", child_taxid)
            child_rank = getattr(child, "rank", "unknown")
            is_fallback = child_rank.startswith("virtual_")
            local_labels[child_taxid] = {
                "class_idx": idx,
                "taxid": child_taxid,
                "name": child_name,
                "rank": child_rank,
                "fallback": is_fallback,
                "capacity": plan["capacities"].get(child_taxid, None),
            }

        # ============================================================
        # PASSO 6: Para cada folha, mapeia pra qual classe e calcula n
        # ============================================================
        # Indexação rápida: leaf → child taxonômico ao qual pertence
        leaf_to_child = {}
        for child in eligible_plus_lowdata:
            if hasattr(child, "descendants"):
                for d in child.descendants:
                    if getattr(d, "rank", "") == "sequence":
                        leaf_to_child[id(d)] = str(child.name)

        # ============================================================
        # PASSO 7: Estratificação train/val/test E distribuição balanceada
        # ============================================================
        #
        # Decisão arquitetural importante:
        # n_per_class é o orçamento TOTAL da classe. Ele é dividido nos 3 splits
        # com proporção 70/15/15, e SOMENTE DENTRO de cada split a distribuição
        # entre folhas é feita proporcionalmente ao comprimento de cada entry
        # (que pode ser uma folha inteira ou uma fatia percentual do genoma).
        #
        # Isso evita o bug onde uma mesma folha taskada em 3 splits (caso n<3)
        # contaria seu orçamento 3 vezes.
        # ============================================================
        rng = random.Random(self.seed)

        # Agrupa folhas por filho (classe)
        leaves_by_child = defaultdict(list)
        for leaf in all_leaves:
            child_taxid = leaf_to_child.get(id(leaf))
            if child_taxid is not None:
                leaves_by_child[child_taxid].append(leaf)

        # Constrói tasks por split, classe a classe
        tasks = {"train": [], "val": [], "test": []}
        target_dir = os.path.join(self.output_dir, current_path)

        # Divide n_per_class nos 3 splits proporção 70/15/15
        n_train = int(n_per_class * 0.70)
        n_val = int(n_per_class * 0.15)
        n_test = n_per_class - n_train - n_val  # restante (garante soma exata)
        n_by_split = {"train": n_train, "val": n_val, "test": n_test}

        for child_taxid, child_leaves in leaves_by_child.items():
            class_idx = local_labels[child_taxid]["class_idx"]

            # Estratifica folhas (por folha ou por sequência)
            split_layout = self._prepare_stratified_split(child_leaves, rng)

            # Para CADA split, distribui o orçamento daquele split entre suas entries
            for split_name in ["train", "val", "test"]:
                layout_entries = split_layout[split_name]
                n_for_split = n_by_split[split_name]
                if not layout_entries or n_for_split <= 0:
                    continue

                # Pesos: tamanho da fatia genômica de cada entry
                # (len(genome) * (end_pct - start_pct))
                weights = []
                for f_path, h_id, sp, ep in layout_entries:
                    seq = _read_sequence_cached(f_path, h_id)
                    if seq:
                        slice_len = max(0, int(len(seq) * (ep - sp)))
                    else:
                        slice_len = 0
                    weights.append(slice_len)

                total_weight = sum(weights)
                if total_weight == 0:
                    continue

                # Distribui n_for_split proporcionalmente
                n_per_entry = [
                    int(round(w / total_weight * n_for_split)) for w in weights
                ]

                # Compensação iterativa para garantir soma exata
                diff = n_for_split - sum(n_per_entry)
                if diff != 0:
                    sorted_idx = sorted(
                        range(len(weights)), key=lambda i: -weights[i]
                    )
                    sign = 1 if diff > 0 else -1
                    for idx in sorted_idx[:abs(diff)]:
                        if sign > 0 or n_per_entry[idx] > 1:
                            n_per_entry[idx] += sign

                # Cria as tasks finais do split
                for (f_path, h_id, sp, ep), n in zip(layout_entries, n_per_entry):
                    if n <= 0:
                        continue
                    tasks[split_name].append({
                        "fasta_path": f_path,
                        "header_id": h_id,
                        "n": n,
                        "start_pct": sp,
                        "end_pct": ep,
                        "class_idx": class_idx,
                    })

        # ============================================================
        # PASSO 8: Registra no manifest e cria job para o pool paralelo
        # ============================================================
        total_tasks = sum(len(t) for t in tasks.values())
        if total_tasks == 0:
            logger.debug(
                f"[SCHEDULE-SKIP] node={parent_taxid} (zero tasks após distribuição)"
            )
            return

        master_manifest[parent_taxid] = {
            "directory_path": target_dir,
            "labels": local_labels,
            "scenario": plan["scenario"],
            "n_per_class": n_per_class,
            "num_leaves": len(all_leaves),
        }

        macro_extraction_jobs.append(
            (parent_taxid, target_dir, tasks, self.max_subseq_len, self.seed,
            self.output_format)
        )

        logger.debug(
            f"[SCHEDULE] node={parent_taxid} rank={getattr(current_node, 'rank', '?')} "
            f"leaves={len(all_leaves)} labels={len(local_labels)} "
            f"path={current_path} n_per_class={n_per_class} "
            f"scenario={plan['scenario']}"
        )

        self._scheduled_count = getattr(self, "_scheduled_count", 0) + 1
        if self._scheduled_count % 100 == 0:
            logger.info(
                f"[PROGRESS] {self._scheduled_count} nós agendados, "
                f"jobs acumulados: {len(macro_extraction_jobs)}"
            )
