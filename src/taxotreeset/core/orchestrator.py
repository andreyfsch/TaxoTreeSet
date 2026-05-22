import json
import os
import subprocess
import logging
from typing import List, Dict, Any
import taxoniq
from bigtree import Node, add_path_to_tree
from tqdm import tqdm

# Local module logger setup
logger = logging.getLogger("TaxoTreeSet.Core.Orchestrator")

class DiscoveryOrchestrator:
    """
    Orchestrates the evolutionary traversal and metadata discovery 
    from a biological root TaxID down to individual terminal species nodes,
    populating both the physical bigtree hierarchy and the local NCBI metadata registry.
    """
    def __init__(self, registry: Any, mapping_config: dict):
        self.registry = registry
        self.mapping = mapping_config
        self.tree_root = Node("root")
        self.logger = logging.getLogger("TaxoTreeSet.Core.Orchestrator")

    def discover_from_root(self, root_taxid: int, assembly_levels: str = "complete,chromosome", checkpoint_interval: int = 500) -> None:
        """
        Queries the NCBI Datasets API from the given root taxon ID,
        streams the genomic reports, resolves their phylogenetic lineages,
        and dynamically persists checkpoints to avoid data loss during massive runs.
        """
        root_id_str = str(root_taxid)
        env = os.environ.copy()
        
        if env.get("NCBI_API_KEY"):
            self.logger.info("NCBI_API_KEY environment variable is active for this subprocess query loop.")

        cmd = [
            "datasets", "summary", "genome", "taxon", root_id_str,
            "--assembly-source", "RefSeq",
            "--assembly-level", assembly_levels,
            "--as-json-lines"
        ]
        
        try:
            self.logger.info(f"Spawning NCBI Datasets CLI subprocess streaming command for TaxID: {root_taxid}")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                env=env,
                bufsize=1
            )
            
            temp_data: Dict[str, List[Dict[str, Any]]] = {}
            
            # Stream genome summary reports line-by-line from the subprocess pipe stream
            with tqdm(desc="Streaming NCBI Genome Reports", unit=" seqs") as pbar:
                for line in process.stdout:
                    if not line.strip(): 
                        continue
                    try:
                        report = json.loads(line)
                        taxid = report.get("organism", {}).get("tax_id")
                        if not taxid: 
                            continue
                        
                        taxid_str = str(taxid)
                        if taxid_str not in temp_data:
                            temp_data[taxid_str] = []
                        
                        temp_data[taxid_str].append(report)
                        pbar.update(1)
                        
                    except json.JSONDecodeError:
                        continue

            process.wait()
            
            if not temp_data:
                stderr_output = process.stderr.read()
                self.logger.error(f"NCBI streaming process returned empty or failed: {stderr_output}")
                return

            self.logger.info(f"Successfully streamed {len(temp_data)} unique species taxa nodes. Commencing hierarchy building.")
            
            # Process each unique species taxon node and build hierarchy with incremental batch checkpoints
            processed_count = 0
            for taxid_str, reports in tqdm(temp_data.items(), desc="Processing Lineage Hierarchy"):
                try:
                    species_taxon = taxoniq.Taxon(int(taxid_str))
                    path_parts = self._resolve_mapped_path(species_taxon, root_id_str)
                    full_path = "root/" + "/".join(path_parts)
                    
                    # Update the local metadata registry data layout
                    for report in reports:
                        self.registry._update_taxon_entry(taxid_str, {"reports": [report]})
                    
                    # Stitch node structural attributes onto the bigtree skeleton layout
                    add_path_to_tree(
                        self.tree_root, 
                        full_path, 
                        node_attrs={
                            "taxid": taxid_str,
                            "rank": "species",
                            "scientific_name": species_taxon.scientific_name
                        }
                    )
                    
                    processed_count += 1
                    
                    # Incremental Checkpoint Flush: saves progress periodically to prevent disk I/O thrashing
                    if processed_count % checkpoint_interval == 0:
                        self.logger.info(f"Reached checkpoint milestone ({processed_count} taxa). Flushing changes to disk.")
                        self.registry.save()
                        
                except Exception as e:
                    self.logger.debug(f"Skipping line resolution exception for TaxID {taxid_str}: {e}")
                    continue

            # Final flush to guarantee last block items are fully written
            self.registry.save()
            self.logger.info("Metadata registration and tree construction workflow executed successfully.")

        except Exception as e:
            self.logger.error(f"Critical execution error detected during orchestrator query discovery: {e}")
            raise

    def _resolve_mapped_path(self, species_taxon: taxoniq.Taxon, root_id_str: str) -> List[str]:
        """
        Traces the complete ancestry lineage upward and applies redirect mappings
        defined inside the mapping configuration scopes to handle virtual fallback zones.
        """
        lineage = list(species_taxon.ranked_lineage)
        path_parts = []
        scope = self.mapping.get("scopes", {}).get(root_id_str, {})
        redirections = scope.get("redirections", {})
        virtual_labels = scope.get("virtual_id_labels", {})
        
        for taxon in lineage:
            tid = str(taxon.tax_id)
            if tid in redirections:
                target_id = redirections[tid]["target_id"]
                name = virtual_labels.get(target_id, redirections[tid]["label"])
            else:
                # Sanitize structural scientific names against illegal directory tokens
                name = taxon.scientific_name.replace(" ", "_").replace("/", "_")
            path_parts.append(name)
            
        return path_parts