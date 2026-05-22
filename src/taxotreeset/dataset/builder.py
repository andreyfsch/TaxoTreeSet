import os
import tqdm
import pandas as pd
import numpy as np
from src.taxotreeset.dataset.sequence_utils import extract_subseqs
from src.taxotreeset.dataset.utils import _read_single_sequence, _pool_worker_initializer

def extract_parent_node_worker(job):
    """Worker paralelo para extrair e escrever sequências em Parquet."""
    parent_taxid, target_dir, parent_tasks, max_subseq_len, seed, output_format = job
    
    for split in ["train", "val", "test"]:
        tasks = parent_tasks[split]
        if not tasks:
            continue
            
        data = []
        for t in tasks:
            # 1. Carrega a string de DNA real a partir do disco/LMDB
            full_seq = _read_single_sequence(t['fasta_path'], t['header_id'])
            if not full_seq:
                continue
                
            # 2. Aplica o fatiamento da sequência (será 100% se houver diversidade no clado)
            start_idx = int(len(full_seq) * t['start_pct'])
            end_idx = int(len(full_seq) * t['end_pct'])
            sub_seq = full_seq[start_idx:end_idx]
            
            # 3. Repassa a string biológica real e os parâmetros corretos para a função geradora
            seqs = extract_subseqs(
                seq=sub_seq,
                n=t['n'],
                min_len=100,  # Limite inferior fixado por segurança de auditoria
                max_len=max_subseq_len
            )
            
            for s in seqs:
                # Inserção obrigatória da coluna de classe
                data.append({"seq": s, "class_idx": int(t['class_idx'])})
        
        if data:
            df = pd.DataFrame(data)
            df['class_idx'] = df['class_idx'].astype(int)
            out_path = os.path.join(target_dir, f"{split}.{output_format}")
            df.to_parquet(out_path, index=False)
            
    return True


class DatasetBuilder:
    def __init__(self, output_dir, max_subseq_len, seed, output_format):
        self.output_dir = output_dir
        self.max_subseq_len = max_subseq_len
        self.seed = seed
        self.output_format = output_format
        
    def prepare_stratified_split(self, nodes):
        """
        Divide as folhas em train/val/test de forma estratificada.
        Se houver folhas suficientes (>= 3), divide as folhas inteiras.
        Se houver escassez (< 3), fatia a própria sequência (vazamento aceitável por sobrevivência).
        """
        splits = {"train": [], "val": [], "test": []}
        all_leaves = []
        
        for node in nodes:
            all_leaves.extend([l for l in node.leaves if getattr(l, "rank", "") == "sequence"])
        
        if not all_leaves:
            return splits

        # Fixa o gerador de sementes para garantir consistência de split entre execuções
        np.random.seed(self.seed)
        np.random.shuffle(all_leaves)
        
        n = len(all_leaves)
        
        # 🚀 CENÁRIO 1: DIVERSIDADE SUFICIENTE (Sem vazamento de dados intra-sequência)
        if n >= 3:
            train_idx = max(1, int(n * 0.70))
            val_idx = train_idx + max(1, int(n * 0.15))
            
            for i, leaf in enumerate(all_leaves):
                f_path = getattr(leaf, "fasta_path", "")
                h_id = getattr(leaf, "header_id", "")
                
                # 0.0 a 1.0 significa que 100% da sequência vai para o split sorteado
                task = (f_path, h_id, 0.0, 1.0) 
                
                if i < train_idx:
                    splits["train"].append(task)
                elif i < val_idx:
                    splits["val"].append(task)
                else:
                    splits["test"].append(task)
                    
        # 🚀 CENÁRIO 2: ESCASSEZ EXTREMA (Fatiamento de sobrevivência)
        else:
            for leaf in all_leaves:
                f_path = getattr(leaf, "fasta_path", "")
                h_id = getattr(leaf, "header_id", "")
                
                # A mesma sequência é fatiada entre os splits
                splits["train"].append((f_path, h_id, 0.0, 0.70))
                splits["val"].append((f_path, h_id, 0.70, 0.85))
                splits["test"].append((f_path, h_id, 0.85, 1.0))
                
        return splits
        
    def build_node_dataset(self, jobs, parallel=False):
        """Executa a extração, usando paralelismo na I/O de disco."""
        if parallel:
            import multiprocessing as mp
            import psutil
            mem_gb = psutil.virtual_memory().total / (1024**3)
            if mem_gb < 12:
                num_workers = 2
            else:
                num_workers = max(1, mp.cpu_count() - 2)
            print(f"[BUILDER] Worker pool: {num_workers} processos")
            # spawn em vez de fork: workers começam do zero, sem herdar a memória
            # do pai (árvore bigtree, LMDB env, etc). Custa ~1-2s de import por
            # worker, mas elimina swap quando o pai está grande.
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=num_workers, initializer=_pool_worker_initializer) as pool:
                results = []
                with tqdm(total=len(jobs), desc="Building parquets", unit="job") as pbar:
                    for result in pool.imap_unordered(extract_parent_node_worker, jobs, chunksize=1):
                        results.append(result)
                        pbar.update(1)
                return results
        else:
            return [extract_parent_node_worker(j) for j in jobs]