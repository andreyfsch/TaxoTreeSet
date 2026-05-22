import os
import json
import logging
import subprocess
import zipfile
import tempfile
import lmdb
import zlib
from typing import Any, List, Dict
from tqdm import tqdm

# Local module logger setup
logger = logging.getLogger("TaxoTreeSet.IO.Downloader")

class NCBIDownloader:
    """
    Manages high-performance batch downloads of genomic assemblies, leveraging the
    NCBI Datasets CLI multi-accession capabilities to eliminate HTTP handshake overhead.
    """
    def __init__(self, registry: Any, vault_path: str = "data/vault", chunk_size: int = 100):
        self.registry = registry
        self.vault_path = vault_path
        self.chunk_size = chunk_size  
        
        os.makedirs(self.vault_path, exist_ok=True)
        self.lmdb_path = os.path.join(self.vault_path, "sequences.lmdb")
        
        # Guard clean reference pointer without opening the resource pipeline on init
        self.env = None

    def download_batch(self, accessions: List[str]) -> Dict[str, List[dict]]:
        """
        Executes a single bulk CLI command to download an array of accessions simultaneously,
        unpacks the composite payload, and parses records into the target structures.
        """
        logger.debug(f"Spawning bulk NCBI fetch call for {len(accessions)} accessions.")
        batch_results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_output_path = os.path.join(temp_dir, "batch_package.zip")
            cmd = [
                "datasets", "download", "genome", "accession"
            ] + accessions + [
                "--include", "genome",
                "--filename", zip_output_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, env=os.environ.copy())
                
                if not os.path.exists(zip_output_path) or os.path.getsize(zip_output_path) == 0:
                    logger.error(f"Bulk download failed. Package archive is empty or missing.")
                    return batch_results
                
                extract_path = os.path.join(temp_dir, "unpacked")
                with zipfile.ZipFile(zip_output_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                dataset_data_root = os.path.join(extract_path, "ncbi_dataset", "data")
                if not os.path.exists(dataset_data_root):
                    return batch_results

                for current_acc in accessions:
                    acc_dir = os.path.join(dataset_data_root, current_acc)
                    if not os.path.exists(acc_dir):
                        continue
                        
                    fna_file = None
                    for file in os.listdir(acc_dir):
                        if file.endswith((".fna", ".fasta", ".fa")):
                            fna_file = os.path.join(acc_dir, file)
                            break
                            
                    if not fna_file:
                        continue
                        
                    current_header = None
                    current_seq_lines = []
                    headers_metadata = []
                    local_seq_map = {}

                    with open(fna_file, "r", encoding="utf-8") as f:
                        for line in f:
                            stripped = line.strip()
                            if not stripped:
                                continue
                            if stripped.startswith(">"):
                                if current_header:
                                    local_seq_map[current_header] = "".join(current_seq_lines)
                                
                                parts = stripped[1:].split(" ", 1)
                                current_header = parts[0]
                                seq_name = parts[1] if len(parts) > 1 else current_header
                                headers_metadata.append({"id": current_header, "name": seq_name})
                                current_seq_lines = []
                            else:
                                current_seq_lines.append(stripped)
                                
                        if current_header:
                            local_seq_map[current_header] = "".join(current_seq_lines)
                    
                    if local_seq_map:
                        with self.env.begin(write=True) as txn:
                            for h_id, seq_str in local_seq_map.items():
                                compressed = zlib.compress(seq_str.encode('utf-8'))
                                txn.put(h_id.encode('utf-8'), compressed)
                                
                        batch_results[current_acc] = headers_metadata

                return batch_results
                
            except subprocess.CalledProcessError as e:
                logger.error(f"NCBI bulk dataset CLI call triggered an execution failure: {e.stderr.strip()}")
                return batch_results
            except Exception as e:
                logger.error(f"Unexpected fault processing current database ingestion batch: {e}")
                return batch_results

    def download_all_pending(self, checkpoint_frequency: int = 10) -> None:
        """Slices pending targets into balanced chunks and processes them via bulk pipeline iterations."""
        accessions_dict = self.registry.registry.get("accessions", {})
        pending_accessions = [acc for acc, info in accessions_dict.items() if not info.get("downloaded")]
        
        total_accessions = len(accessions_dict)
        total_pending = len(pending_accessions)
        already_downloaded = total_accessions - total_pending
        
        # Safe early return: If there are no pending tasks, env is never opened, avoiding locks
        if total_pending == 0:
            logger.info("All registered accessions are already archived inside LMDB database.")
            return
            
        chunks = [
            pending_accessions[i:i + self.chunk_size] 
            for i in range(0, total_pending, self.chunk_size)
        ]
        
        logger.info(f"Grouped {total_pending} pending accessions into {len(chunks)} batch downloads (Size: {self.chunk_size}).")
        
        # Open write environment ONLY when network streams are strictly required
        self.env = lmdb.open(self.lmdb_path, map_size=1099511627776, max_dbs=0)
        
        try:
            with tqdm(total=total_accessions, initial=already_downloaded, desc="Ingesting Genomes to LMDB", unit=" genome") as pbar:
                for chunk in chunks:
                    completed_batch_data = self.download_batch(chunk)
                    
                    for acc in chunk:
                        if acc in completed_batch_data:
                            self.registry.registry["accessions"][acc]["downloaded"] = True
                            self.registry.registry["accessions"][acc]["local_path"] = self.lmdb_path
                            self.registry.registry["accessions"][acc]["headers"] = completed_batch_data[acc]
                        pbar.update(1)
                    
                    self.registry.save()
        finally:
            # Absolute engineering guarantee: closes the DB layout even if Ctrl+C or crashes occur
            self.env.close()
            self.env = None
            
        logger.info("Bulk download pipeline operations closed successfully.")