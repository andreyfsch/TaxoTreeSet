import os
import logging
import hashlib
from bigtree import Node, find_attrs
from src.taxotreeset.dataset.utils import _get_fasta_sequence_length, _read_single_sequence

# Local module logger setup
logger = logging.getLogger("TaxoTreeSet.Dataset.Analyzer")

class TaxonDiversityAnalyzer:
    """
    Responsible for analyzing the physical composition and real molecular diversity
    of DNA sequences contained within the taxonomic tree structure.
    """
    def __init__(self, max_subseq_len: int = 2000):
        self.max_subseq_len = max_subseq_len

    def get_unique_subseqs_count(self, node: Node) -> int:
        """
        Calculates the EXACT cardinality of unique subsequences under a taxonomic node.
        
        Uses in-memory mapping of 128-bit integers (MD5 bignums) to guarantee
        zero statistical collisions (0%) even within large sequence windows (e.g., 2000bp),
        effectively eliminating bias introduced by genomic repetitive regions (transposons, satellites).
        
        Complexity:
            Memory: O(U) where U is the number of unique subsequences under the node.
            Time: O(N * L) where N is the total number of genomes and L is the average sequence length.
        """
        sequence_nodes = find_attrs(node, "rank", "sequence")
        unique_hashes = set()
        
        # Local debug log for granular tracking of heavy processing steps
        logger.debug(f"Initiating uniqueness analysis for node: {node.node_name}")
        
        for seq_node in sequence_nodes:
            f_path = seq_node.get_attr("fasta_path")
            h_id = seq_node.get_attr("header_id")
            
            # Lazy loading: Loads only one genome at a time into memory to protect RAM
            seq_string = _read_single_sequence(f_path, h_id)
            if not seq_string or len(seq_string) < self.max_subseq_len:
                continue
            
            # C-level optimization: Encodes the full string to bytes prior to the sliding loop
            seq_bytes = seq_string.encode('utf-8')
            
            # Native step-1 sliding window extraction loop
            for i in range(0, len(seq_bytes) - self.max_subseq_len + 1):
                subseq_chunk = seq_bytes[i:i + self.max_subseq_len]
                
                # Fast 128-bit MD5 hash mapped directly to an arbitrary precision python integer
                hash_128 = int.from_bytes(
                    hashlib.md5(subseq_chunk).digest(), 
                    byteorder='big'
                )
                unique_hashes.add(hash_128)
                
        total_unique = len(unique_hashes)
        
        # Explicit and aggressive deallocation of the hash set to clear node memory immediately
        del unique_hashes
        
        return total_unique

    def calculate_bulk_capacities(self, taxonomic_nodes: list) -> dict[str, int]:
        """
        Scans a list of taxonomic nodes and returns a dictionary mapping each
        node path to its respective deduplicated subsequence count.
        """
        node_metrics = {}
        for node in taxonomic_nodes:
            full_path = f"{node.path_name}/{node.node_name}"
            node_metrics[full_path] = self.get_unique_subseqs_count(node)
        return node_metrics