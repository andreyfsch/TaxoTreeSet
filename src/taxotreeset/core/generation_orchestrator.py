import os
import sys
import gc
import json
import logging
import collections
import itertools
from typing import Any, Dict, List

# 🚀 ESTABILIZAÇÃO: Bloqueia threads em C para evitar Core Dumped e conflito de memória
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from bigtree import Node, find_attrs
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from src.taxotreeset.io.downloader import NCBIDownloader
from src.taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from src.taxotreeset.dataset.builder import DatasetBuilder
from src.taxotreeset.dataset.utils import _get_fasta_sequence_length

from collections import Counter, defaultdict
from typing import Any
import hashlib
import logging

logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")

PROTECTED_RANKS = frozenset({
    "realm_group",       # fallbacks semânticos curados no mapping.json (999000-003)
    "virtual_cluster",   # buckets criados pelo k-means
    "virtual_bucket",    # buckets criados pelo Op3 (esta defesa torna a função idempotente)
    "virtual_low_data",
})

def make_virtual_id(parent_taxid: str, purpose: str) -> str:
    """Gera ID virtual determinístico de 9 dígitos. Prefixo '9' identifica virtuais."""
    key = f"{parent_taxid}:{purpose}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    suffix = int(h[:8], 16) % 100_000_000
    return f"9{suffix:08d}"

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
        rank_counts = Counter(getattr(c, "rank", "unknown") for c in classifiable)
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
                vid, parent_taxid, getattr(parent_node, "scientific_name", parent_taxid),
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
            vid, parent_taxid, getattr(parent_node, "scientific_name", parent_taxid),
            "misc", mescla_residual, parent_node
        )
        filhos_efetivos.append(virtual_node)
        ranks_in_misc = Counter(getattr(c, "rank", "unknown") for c in mescla_residual)
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
    def __init__(self, registry: Any, config_path: str = "configs/mapping.json", 
                 vault_path: str = "data/vault", output_dir: str = "data/datasets",
                 max_subseq_len: int = 2000, seed: int = 42, output_format: str = "parquet",
                 min_subclades_per_bucket: int = 5):
        self.registry = registry
        self.vault_path = vault_path
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_subseq_len = max_subseq_len
        self.seed = seed
        self.output_format = output_format.lower()
        self.config_path = config_path
        self.min_subclades_per_bucket = min_subclades_per_bucket
        self.virtual_id_registry = {}

        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.mapping = json.load(f)
        else:
            self.mapping = {}

        self.downloader = NCBIDownloader(registry=self.registry, vault_path=self.vault_path)
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
        
        group_to_id = {"viruses": "10239", "bacteria": "2", "archaea": "2157", "eukaryotes": "2759"}
        target_group_clean = target_group.lower().strip()
        domain_taxid = group_to_id.get(target_group_clean)
        
        scope_config = self.mapping.get("scopes", {}).get(domain_taxid, {})
        fallback_ids = {str(k) for k in scope_config.get("virtual_id_labels", {}).keys()}
        # Apenas o default_id do escopo é um fallback "verdadeiro" — destino de táxons
        # sem regra de redirect, portanto sem hierarquia NCBI abaixo. K-means só faz
        # sentido aqui, onde é necessário inventar uma estrutura de classes.
        # Os demais virtual_id_labels (999001-003 no caso viral) são agrupadores
        # semânticos que preservam a taxonomia NCBI abaixo deles e não devem ser
        # reorganizados por similaridade de k-mers.
        default_fallback_id = str(scope_config.get("default_id", ""))
        clustering_eligible = {default_fallback_id} if default_fallback_id else set()

        logger.info(f"Fallback IDs ativos para '{target_group_clean}': {sorted(fallback_ids)}")
        logger.info(f"K-means elegível apenas em: {sorted(clustering_eligible)}")
        if not fallback_ids:
            logger.warning(
                f"Escopo '{domain_taxid}' não declara virtual_id_labels. "
                f"K-Means clustering não será aplicado neste domínio."
            )
        logger.info(f"Fallback IDs ativos para '{target_group_clean}': {sorted(fallback_ids)}")
        
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
        
        domain_nodes = [node for node in taxon_tree.descendants if str(node.name) == domain_taxid]
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

        vocab = ["".join(p) for p in itertools.product(["A", "T", "C", "G"], repeat=4)]
        
        macro_extraction_jobs = []
        master_manifest = {}
        passthrough_map = {}  # parent_taxid → single_child_taxid (redirect automático)
        
        kmeans_stats = {"checked": 0, "fired": 0}

        def process_node_cascaded(current_node: Node, accumulated_path: str):
            parent_taxid = str(current_node.name)
            children = [c for c in current_node.children if getattr(c, "rank", "") != "sequence"]
            
            # K-Means restrito ao Fallback ID dinâmico do domínio atual
            path_components = set(accumulated_path.split(os.sep))
            is_under_default_fallback = (
                bool(clustering_eligible & path_components) or parent_taxid in clustering_eligible
            )

            if len(children) > 15 and is_under_default_fallback:
                kmeans_stats["fired"] += 1
                logger.debug(
                    f"[KMEANS-FIRE] node={parent_taxid} children={len(children)} "
                    f"(sob default_fallback {default_fallback_id})"
                )
                from src.taxotreeset.dataset.utils import _read_single_sequence
                vectors = []
                valid_children = []
                
                for child in children:
                    child_leaves = leaf_cache.get(str(child.name), [])
                    if child_leaves:
                        seq = _read_single_sequence(
                            getattr(child_leaves[0], "fasta_path", ""),
                            getattr(child_leaves[0], "header_id", "")
                        )
                        if seq:
                            seq_len = len(seq)
                            if seq_len <= 4000:
                                sample_seq = seq
                            else:
                                mid_point = seq_len // 2
                                sample_seq = seq[mid_point - 2000 : mid_point + 2000]
                            vectors.append(_compute_kmer_vector(sample_seq, vocab))
                            valid_children.append(child)
                
                if vectors:
                    n_cl = min(15, len(vectors))
                    kmeans = MiniBatchKMeans(
                        n_clusters=n_cl, random_state=42,
                        batch_size=64, max_no_improvement=10
                    )
                    labels = kmeans.fit_predict(np.array(vectors, dtype=np.float32))
                    
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
                logger.debug(f"[KMEANS-SKIP] node={parent_taxid} children={len(children)} (≤15)")

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

        logger.info(f"Pre-building directory tree for {len(macro_extraction_jobs)} nodes...")
        for job in tqdm(macro_extraction_jobs, desc="Creating output dirs", unit=" dir"):
            os.makedirs(job[1], exist_ok=True)
        
        # CLEANUP CRÍTICO ANTES DO POOL:
        # Os workers serão forked (ou spawned) a partir daqui. Tudo que está vivo no
        # escopo do pai será copiado/inicializado nos workers. Como a árvore bigtree
        # completa pode chegar a centenas de milhares de nós e nada do que vai abaixo
        # precisa dela (os jobs já trazem todo dado necessário em forma de tuplas),
        # largamos a árvore agora e forçamos GC para evitar swap nos workers.
        logger.info("Liberando árvore taxonômica antes do pool (jobs já carregam dados necessários)...")
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
        logger.info(f"Distribuição de cardinalidade das heads: {dict(sorted(card_dist.items()))}")
        
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
            results = self.builder.build_node_dataset(macro_extraction_jobs, parallel=True)
        
        logger.info(
            f"K-Means: disparou em {kmeans_stats['fired']} nós, "
            f"pulou {kmeans_stats['checked']} nós sob fallback com ≤15 filhos"
        )
        logger.info(f"Total de jobs agendados: {len(macro_extraction_jobs)}")
        

    def _schedule_decision_point(self, current_node: Node, children_list: list, current_path: str, abundance_threshold: int, 
                             extraction_budget_per_node: int, macro_extraction_jobs: list, master_manifest: dict, passthrough_map: dict, leaf_cache: dict):
        parent_taxid = str(current_node.name)

        # Coleta as folhas dinamicamente a partir da árvore atual (pós K-means)
        # em vez de confiar no leaf_cache, que pode estar dessincronizado com clusters virtuais.
        all_leaves = [lf for lf in current_node.descendants if getattr(lf, "rank", "") == "sequence"]
        if not all_leaves:
            return
        
        # APLICA Op3: classifica filhos em canônicos + bucketizados
        children_efetivos, new_virt_ids = classify_children_by_rank(
            current_node,
            children_list,
            min_subclades_per_bucket=self.min_subclades_per_bucket,
            canonical_rank_override=None,  # futuro: vem do clade_policies
            virtual_id_registry=self.virtual_id_registry,
        )

        parent_target_dir = os.path.abspath(os.path.join(self.output_dir, current_path))

        # Conjunto de filhos válidos do current_node (taxonômicos + clusters virtuais)
        valid_children_ids = {str(c.name) for c in children_efetivos}
        child_nodes_by_id = {str(c.name): c for c in children_efetivos}

        local_labels = {}
        idx = 0

        for leaf in all_leaves:
            # Caminho da RAIZ até a folha: ancestors está em ordem inversa, então damos reverse
            # e adicionamos a própria folha no fim.
            path_root_to_leaf = [str(n.name) for n in reversed(list(leaf.ancestors))] + [str(leaf.name)]

            # Localiza o nó atual no caminho e pega o IMEDIATAMENTE seguinte (filho direto no rumo da folha)
            try:
                p_idx = path_root_to_leaf.index(parent_taxid)
            except ValueError:
                # parent_taxid não está no caminho desta folha — não deveria acontecer,
                # mas se acontecer, esta folha não pertence a este nó. Pula.
                continue

            if p_idx + 1 >= len(path_root_to_leaf):
                # parent_taxid é a própria folha (caso degenerado). Pula.
                continue

            child_id = path_root_to_leaf[p_idx + 1]

            # Sanity check: o filho-direto encontrado precisa estar entre os filhos atuais
            # do current_node. Se não estiver, é sinal de inconsistência (ex.: árvore mutada
            # depois do leaf_cache ser construído). Pula a folha em vez de criar um rótulo fantasma.
            if child_id not in valid_children_ids:
                continue

            if child_id not in local_labels:
                matching_node = child_nodes_by_id[child_id]
                local_labels[child_id] = {
                    "label_idx": idx,
                    "name": getattr(matching_node, "scientific_name", child_id),
                    "is_fallback": getattr(matching_node, "rank", "").startswith("virtual_"),
                }
                idx += 1

            leaf._local_idx = local_labels[child_id]["label_idx"]

        # Se nenhuma folha gerou rótulo válido (todas puladas), não agenda job nenhum.
        if not local_labels:
            # Caso comum em nós-folha taxonômicos (subspecies/strain): só existem
            # sequências diretamente abaixo, sem subtaxa para classificar. Não é erro.
            logger.debug(
                f"[SCHEDULE-SKIP] node={parent_taxid} "
                f"rank={getattr(current_node, 'rank', '?')} "
                f"(nenhum sub-taxon para classificar; {len(all_leaves)} sequências abaixo)"
            )
            return
        
        # Heads triviais (1 classe) não agregam valor à cascata: a inferência seria
        # determinística e a head só consumiria parâmetros. Redirect automático é
        # resolvido pelo próprio manifest do nó pai (que aponta para este nó).
        if len(local_labels) < 2:
            only_child = next(iter(local_labels.items()))
            child_taxid, child_info = only_child
            passthrough_map[parent_taxid] = {
                "redirect_to": child_taxid,
                "name": child_info["name"],
                "reason": "single_child"
            }
            logger.debug(
                f"[SCHEDULE-PASSTHROUGH] node={parent_taxid} → {child_taxid} ({child_info['name']})"
            )
            return

        master_manifest[parent_taxid] = {
            "parent_name": getattr(current_node, "scientific_name", parent_taxid),
            "parent_rank": getattr(current_node, "rank", "unknown"),
            "relative_path": current_path,
            "labels": {
                str(v["label_idx"]): {"taxid": k, "name": v["name"], "fallback": v["is_fallback"]}
                for k, v in local_labels.items()
            },
        }

        tasks = {"train": [], "val": [], "test": []}
        split_layout = self.builder.prepare_stratified_split([current_node])

        # Mapa header_id -> _local_idx para evitar O(N) por split task
        leaf_by_header = {getattr(lf, "header_id", ""): lf for lf in all_leaves}

        for s in ["train", "val", "test"]:
            for f_path, h_id, sp, ep in split_layout[s]:
                lf = leaf_by_header.get(h_id)
                if lf is None or not hasattr(lf, "_local_idx"):
                    # Folha foi pulada por inconsistência ou não tem rótulo. Não inclui.
                    continue
                tasks[s].append({
                    "fasta_path": f_path,
                    "header_id": h_id,
                    "n": 100,
                    "start_pct": sp,
                    "end_pct": ep,
                    "class_idx": lf._local_idx,
                })

        # Só agenda se ao menos um split tem dados.
        if any(tasks[s] for s in ("train", "val", "test")):
            logger.debug(
                f"[SCHEDULE] node={parent_taxid} "
                f"rank={getattr(current_node, 'rank', '?')} "
                f"leaves={len(all_leaves)} labels={len(local_labels)} "
                f"path={current_path}"
            )
            macro_extraction_jobs.append(
                (parent_taxid, parent_target_dir, tasks, self.max_subseq_len, self.seed, self.output_format)
            )