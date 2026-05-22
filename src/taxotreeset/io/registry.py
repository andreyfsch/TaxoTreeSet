import json
import os
import subprocess
import logging

# Local module logger setup
logger = logging.getLogger("TaxoTreeSet.IO.Registry")

class NCBIRegistry:
    """
    Manages the local metadata registry database, handles incremental updates,
    and interfaces with the NCBI Datasets CLI to discover reference genomes.
    """
    def __init__(self, config_path: str = "configs/mapping.json", registry_path: str = "data/registry.json"):
        self.config_path = config_path
        self.registry_path = registry_path
        self.registry = self._load_registry()
        self.mapping = self._load_mapping()

    def _load_registry(self) -> dict:
        """Loads an existing registry file or initializes a new one to support resume features."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"last_update": None, "taxons": {}, "accessions": {}}

    def _load_mapping(self) -> dict:
        """Loads scope and redirection rules from the configuration path."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def discover_taxon_metadata(self, taxon_id: int | str) -> None:
        """
        Queries the NCBI Datasets CLI to fetch representative and reference assembly metadata.
        """
        logger.info(f"Discovering genomic metadata for TaxID: {taxon_id}...")
        
        # Command executing the official NCBI Datasets summary API
        cmd = [
            "datasets", "summary", "genome", "taxon", str(taxon_id),
            "--reference", "--as-json-lines"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                self._update_taxon_entry(taxon_id, data)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip() if e.stderr else str(e)
            logger.error(f"Failed to query TaxID {taxon_id} via NCBI CLI: {error_msg}")

    def _update_taxon_entry(self, taxon_id: int | str, data: dict) -> None:
        """Parses and organizes incoming NCBI assembly reports into the local registry structure."""
        reports = data.get("reports", [])
        taxon_key = str(taxon_id)
        
        if taxon_key not in self.registry["taxons"]:
            self.registry["taxons"][taxon_key] = []

        for report in reports:
            accession = report.get("accession")
            if not accession:
                continue
            
            # Map the connection between the Taxon Node and the specific Accession ID
            if accession not in self.registry["taxons"][taxon_key]:
                self.registry["taxons"][taxon_key].append(accession)
            
            # Populate absolute technical metadata fields into the registry if new
            if accession not in self.registry["accessions"]:
                assembly_level = report.get("assembly_info", {}).get("assembly_level", "")
                
                # Resilient check: includes both complete bacterial plasmids/chromosomes 
                # and high-quality eukaryotic chromosome assemblies.
                is_reference = assembly_level in ["Complete Genome", "Chromosome"]
                
                self.registry["accessions"][accession] = {
                    "taxid": taxon_key,
                    "organism": report.get("organism", {}).get("organism_name"),
                    "is_reference": is_reference,
                    "downloaded": False,
                    "local_path": None
                }

    def save(self) -> None:
        """Persists the updated metadata tree registry structure onto the storage disk."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2)
            
        # Changed from logger.info to logger.debug to prevent progress bar distortion
        logger.debug(f"Registry log successfully persisted to: {self.registry_path}")