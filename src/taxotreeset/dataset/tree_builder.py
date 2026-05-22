import os
import json
import logging
from bigtree import Node
import taxoniq
from tqdm import tqdm

from src.taxotreeset.io.noise_filter import NoiseFilter

logger = logging.getLogger("TaxoTreeSet.Dataset.TreeBuilder")

def generate_seqs_by_taxon_tree(
    registry_path: str,
    vault_path: str = "data/vault",
    domain_taxid: str = None,
    mapping_path: str = "configs/mapping.json",
    noise_patterns_path: str = "configs/noise_patterns.json",  # <-- novo
) -> Node:
    """
    Builds the base taxonomic tree hierarchy routing explicit redirections.
    Nós administrativos (definidos em noise_patterns.json) são pulados na
    construção da árvore: suas sequências escalam para o próximo ancestor válido.
    """
    with open(registry_path, "r", encoding="utf-8") as f:
        registry_data = json.load(f)

    mapping_data = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

    noise_filter = NoiseFilter(noise_patterns_path)  # <-- novo

    scopes = mapping_data.get("scopes", {})
    group_config = scopes.get(str(domain_taxid), {})
    default_id = group_config.get("default_id", "999000")
    redirections = group_config.get("redirections", {})
    virtual_labels = group_config.get("virtual_id_labels", {})
        
    root = Node("root", rank="root")
    taxons_dict = registry_data.get("taxons", {})
    accessions_dict = registry_data.get("accessions", {})
    
    tasks = [(t, acc) for t, accs in taxons_dict.items() for acc in accs]
    
    logger.info(f"Spawning phylogenetic tree workers for {len(tasks)} metadata entries.")
    
    for taxid_str, acc_id in tqdm(tasks, desc="Resolving Lineage Vectors"):
        acc_info = accessions_dict.get(acc_id, {})
        if not acc_info:
            continue
            
        target_taxid = acc_info.get("taxid") or taxid_str
        
        try:
            taxon = taxoniq.Taxon(int(target_taxid))
            path_ids = [str(t.tax_id) for t in taxon.ranked_lineage][::-1]
        except Exception:
            path_ids = []
            
        # FILTRAGEM DE NOISE TAXA
        # Remove da lineage qualquer ancestor que seja um contêiner administrativo.
        # A accession "escala" para o próximo ancestor válido. Por exemplo, uma
        # espécie sob 'unclassified Caudoviricetes' é vinculada diretamente a
        # Caudoviricetes em vez de ao bolso 'unclassified'.
        filtered_path_ids = []
        for tid in path_ids:
            try:
                t = taxoniq.Taxon(int(tid))
                name = t.scientific_name
                rank = t.rank.name if hasattr(t.rank, "name") else str(t.rank)
            except Exception:
                # Sem nome/rank confiável, mantém na dúvida
                filtered_path_ids.append(tid)
                continue

            if noise_filter.is_noise(name, rank):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[NOISE-SKIP] taxid={tid} name='{name}' "
                        f"rank={rank} reason={noise_filter.explain(name, rank)}"
                    )
                continue
            filtered_path_ids.append(tid)
        path_ids = filtered_path_ids

        if not path_ids:
            # Toda a lineage foi filtrada — accession órfã, descartada
            logger.debug(f"[NOISE-ORPHAN] acc={acc_id} lineage completa filtrada")
            continue
            
        if str(target_taxid) not in path_ids:
            path_ids.append(str(target_taxid))
            
        if domain_taxid and str(domain_taxid) in path_ids:
            idx_anchor = path_ids.index(str(domain_taxid))
            path_ids = path_ids[idx_anchor:]
        else:
            if domain_taxid:
                path_ids = [str(domain_taxid)] + path_ids

        if domain_taxid and len(path_ids) > 1 and path_ids[0] == str(domain_taxid):
            next_level_id = path_ids[1]
            if next_level_id in redirections:
                target_id = str(redirections[next_level_id].get("target_id"))
                # Self-redirect (ex: Orthornavirae → Orthornavirae): lineage canônica preservada
                # Redirect virtual (ex: Pospiviroidae → 999002): insere o grupo virtual
                # acima do táxon original, mantendo a família como nó intermediário
                if target_id != next_level_id:
                    logger.debug(f"[VIRTUAL-INSERT] taxid={next_level_id} → grupo virtual {target_id}")
                    path_ids = [path_ids[0], target_id] + path_ids[1:]
            else:
                logger.debug(f"[FALLBACK-DEFAULT] taxid={next_level_id} sem regra → {default_id}")
                path_ids = [str(domain_taxid), str(default_id), str(target_taxid)]
                
        current = root
        for idx, tid_str in enumerate(path_ids):
            found_child = None
            for child in current.children:
                if child.name == tid_str:
                    found_child = child
                    break
                    
            if found_child is None:
                new_node = Node(tid_str, parent=current)
                if tid_str in virtual_labels:
                    new_node.rank = "realm_group"
                    new_node.scientific_name = virtual_labels[tid_str]
                else:
                    try:
                        t_info = taxoniq.Taxon(int(tid_str))
                        rank_val = t_info.rank
                        new_node.rank = (rank_val.name if hasattr(rank_val, "name") else str(rank_val)).lower().strip()
                        new_node.scientific_name = t_info.scientific_name
                    except Exception:
                        new_node.rank = "unknown"
                        new_node.scientific_name = tid_str
                current = new_node
            else:
                current = found_child
                
        headers_list = acc_info.get("headers", [])
        for header_entry in headers_list:
            if not isinstance(header_entry, dict) or not header_entry.get("id"):
                continue
            h_id = header_entry["id"]
            
            found_seq = None
            for child in current.children:
                if child.name == h_id:
                    found_seq = child
                    break
            if found_seq is None:
                seq_node = Node(h_id, parent=current)
                seq_node.rank = "sequence"
                seq_node.header_id = str(h_id)
                seq_node.fasta_path = os.path.join(vault_path, "sequences.lmdb")
                seq_node.scientific_name = acc_info.get("organism") or ""
    
    stats = noise_filter.stats()
    logger.info(
        f"NoiseFilter: avaliou {stats['evaluated']} nós, "
        f"filtrou {stats['name_hits']} por nome + {stats['rank_hits']} por rank "
        f"= {stats['name_hits'] + stats['rank_hits']} total "
        f"({100 * (stats['name_hits'] + stats['rank_hits']) / max(stats['evaluated'], 1):.1f}%)"
    )
                
    return root