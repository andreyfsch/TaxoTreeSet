"""
taxonomy_utils.py

Summary
-------
This module provides core utilities for extracting and organizing viral
reference genome sequences into a hierarchical taxonomic tree using
Kraken2 and NCBI taxonomy.

Extended Summary
----------------
Key functionalities include:
    - Mapping RefSeq genome identifiers to NCBI taxonomy IDs using
      Kraken2 outputs.
    - Loading genome sequences from FASTA files organized by Kraken2.
    - Building an in-memory taxonomic tree (with the `bigtree` library)
      where each node represents a taxonomic rank, and leaf nodes contain
      actual genome sequences.
    - Traversing and extracting data from the taxonomic tree, including:
        - Extracting representative subsequences from final nodes for
          downstream tasks.
        - Generating sets of all unique subsequences of a given window size
          for deduplication or sequence selection.
    - Robust error handling and logging throughout the genome loading and
      taxonomy resolution process.

The resulting data structures enable downstream dataset creation,
sequence sampling, and taxonomic label assignment for machine learning
workflows (e.g., foundation model fine-tuning).

Dependencies
------------
- Kraken2 database with standard folder structure and map files.
- `taxoniq` for NCBI taxonomy queries.
- `bigtree` for tree representation.
- Standard Python libraries.

Examples
--------
Typical usage:

    >>> from taxonomy_utils import generate_seqs_by_taxon_tree
    >>> tree = generate_seqs_by_taxon_tree()
    >>> # tree is a bigtree.Node. Use traversal or extraction utilities
    >>> # as needed.

Notes
-----
Designed for robust, large-scale, reproducible dataset generation in
computational genomics.

"""
# Standard library imports
import os
import random
import logging
import concurrent.futures
from typing import Dict, Set

# Third-party imports
import tqdm
from bigtree import Node, add_path_to_tree, find_attrs
from progress.bar import Bar

# Local application imports
from dataset import sequence_utils
from config import KRAKEN_PATH, KRAKEN_DATABASE
import math
import zlib
import hashlib

def get_unique_subseqs_count(node: Node, window_size: int) -> int:
    """
    Decide entre contagem exata ou HLL com base no tamanho do clado.
    Garante que o CSV volte a ter o volume correto de dados.
    """
    seq_nodes = find_attrs(node, "rank", "sequence")
    
    # Para níveis com muitos nós pequenos (Species, Genus, Family, Order),
    # usamos a contagem exata. Agora é seguro pois usamos Lazy Loading.
    if node.rank in ["genus", "family", "order"]:
        unique_kmers = set()
        for s_node in seq_nodes:
            f_path = s_node.get_attr("fasta_path")
            h_id = s_node.get_attr("header_id")
            seq = _read_single_sequence(f_path, h_id)
            
            for i in range(0, len(seq) - window_size + 1):
                # Usamos o hash nativo do Python (64-bit) que é rápido e 
                # consome muito menos RAM que a string original.
                unique_kmers.add(hash(seq[i:i+window_size]))
            del seq
        
        count = len(unique_kmers)
        unique_kmers.clear()
        del unique_kmers
        return count
    
    # Para níveis gigantes (Class, Phylum, Kingdom), usamos HLL com 
    # um hash de alta qualidade para evitar subestimação.
    else:
        return _estimate_hll_high_precision(seq_nodes, window_size)

def _estimate_hll_high_precision(seq_nodes, window_size, p=14):
    """HLL de alta precisão com SHA1 para evitar o erro de 286."""
    m = 1 << p
    registers = [0] * m
    for s_node in seq_nodes:
        seq = _read_single_sequence(s_node.get_attr("fasta_path"), s_node.get_attr("header_id"))
        for i in range(0, len(seq) - window_size + 1):
            kmer = seq[i:i+window_size]
            # Usamos os primeiros 8 bytes do SHA1 como um inteiro de 64 bits
            x = int(hashlib.sha1(kmer.encode('utf-8')).hexdigest()[:16], 16)
            idx = x & (m - 1)
            w = x >> p
            rho = (bin(w).split('1')[-1].count('0') + 1) if w > 0 else (64 - p)
            registers[idx] = max(registers[idx], rho)
        del seq
    
    # Cálculo HLL padrão...
    alpha_m = 0.7213 / (1 + 1.079 / m)
    estimate = alpha_m * (m ** 2) / sum(math.pow(2, -r) for r in registers)
    return int(estimate)

def estimate_unique_subseqs_hll(final_taxon_node: Node, window_size: int, p: int = 12) -> int:
    m = 1 << p
    registers = [0] * m
    seq_nodes = find_attrs(final_taxon_node, "rank", "sequence")
    
    for node in seq_nodes:
        f_path = node.get_attr("fasta_path")
        h_id = node.get_attr("header_id")
        sequence = _read_single_sequence(f_path, h_id)
        if not sequence: continue

        # Otimização: Usamos uma visualização de memória (memoryview) ou slices
        # E trocamos o SHA1 por adler32 (muito mais rápido em Python)
        for i in range(0, len(sequence) - window_size + 1):
            kmer = sequence[i:i+window_size]
            
            # adler32 retorna um inteiro de 32 bits rapidamente
            x = zlib.adler32(kmer.encode('utf-8'))
            
            idx = x & (m - 1)
            w = x >> p
            rho = (bin(w).split('1')[-1].count('0') + 1) if w > 0 else (32 - p)
            registers[idx] = max(registers[idx], rho)
        
        del sequence

    # 5. Cálculo da estimativa bruta (Raw Estimate)
    alpha_m = 0.7213 / (1 + 1.079 / m) if m >= 64 else 0.673 # Constante de correção
    sum_inv = sum(math.pow(2, -r) for r in registers)
    estimate = alpha_m * (m ** 2) / sum_inv
    
    # 6. Correções para range pequeno (Linear Counting)
    if estimate <= 2.5 * m:
        v = registers.count(0)
        if v > 0:
            estimate = m * math.log(m / v)
            
    return int(estimate)




def get_tax_ids() -> Dict[str, int]:
    """
    Load mapping from RefSeq IDs to NCBI Taxonomy IDs using the
    Kraken2 map file.

    Returns
    -------
    dict of str to int
        Dictionary where keys are RefSeq sequence names (e.g., "NC_002195.1")
        and values are corresponding NCBI Taxonomy IDs.

    Raises
    ------
    FileNotFoundError
        If the Kraken2 seqid2taxid.map file does not exist.

    Notes
    -----
    This function expects the Kraken2 map file to exist at:
    '{KRAKEN_PATH}/{KRAKEN_DATABASE}/seqid2taxid.map'

    Example
    -------
    >>> tax_ids = get_tax_ids()
    >>> tax_ids["NC_002195.1"]
    687377
    """
    tax_ids = {}
    with open(f"{ KRAKEN_PATH }/{ KRAKEN_DATABASE }/seqid2taxid.map") as f:
        for line in f.readlines():
            # Line example:
            # kraken:taxid|687377|NC_002195.1 687377
            ref_seq = line.rstrip().split("|")[-1].split()[0]
            tax_id = int(line.rstrip().split("|")[-1].split()[-1])
            tax_ids[ref_seq] = tax_id

    return tax_ids


def get_num_refseq_files() -> int:
    """
    Return the number of RefSeq files (genome folders) in the Kraken2
    database structure.

    Returns
    -------
    int
        Number of reference sequence genome directories found under
        '{KRAKEN_PATH}/{KRAKEN_DATABASE}/genomes'.

    Notes
    -----
    This function does not check the contents of each folder;
    only their existence.
    """
    logger = logging.getLogger(__name__)
    gen_dir = f"{KRAKEN_PATH}/{KRAKEN_DATABASE}/genomes"
    if not os.path.isdir(gen_dir):
        logger.error(f"Genome directory {gen_dir} does not exist.")
        raise FileNotFoundError(f"Genome directory {gen_dir} does not exist.")
    return len(
        [
            name
            for name in os.listdir(gen_dir)
            if os.path.isdir(os.path.join(gen_dir, name))
        ]
    )


def parse_refseq_folder(args):
    """
    Helper for multiprocessing: parses genome and taxonomy for one RefSeq.

    Parameters
    ----------
    args : tuple
        (ref_seq, tax_ids, KRAKEN_PATH, KRAKEN_DATABASE)
    Returns
    -------
    dict
        {
            "ref_seq": ...,
            "tax_id": ...,
            "sequences": ...,
            "ranked_taxons": ...,
            "error": None or str
        }
    """
    ref_seq, tax_ids, KRAKEN_PATH, KRAKEN_DATABASE = args
    try:
        import taxoniq
        tax_id = tax_ids[ref_seq]
        
        fasta_path = os.path.join(KRAKEN_PATH, KRAKEN_DATABASE, "genomes", ref_seq, "genome.fna")
        
        if not os.path.exists(fasta_path):
            return {"ref_seq": ref_seq, "error": f"Arquivo não encontrado: {fasta_path}"}

        # Extraímos apenas os headers para saber quais sequências existem dentro do arquivo
        # sem carregar o corpo da sequência.
        sequence_headers = []
        with open(fasta_path, "r") as f:
            for line in f:
                if line.startswith(">"):
                    # Ex: ">NC_001422.1 description..." -> "NC_001422.1"
                    header_id = line[1:].split()[0]
                    description = line[1:].strip()
                    sequence_headers.append({"id": header_id, "name": description})

        t = taxoniq.Taxon(tax_id)
        ranked_taxons = t.ranked_lineage
        
        return {
            "ref_seq": ref_seq,
            "tax_id": tax_id,
            "fasta_path": fasta_path, # Guardamos o caminho
            "sequence_headers": sequence_headers,
            "ranked_taxons": ranked_taxons,
            "error": None
        }
    except Exception as e:
        return {"ref_seq": ref_seq, "error": str(e)}


def generate_seqs_by_taxon_tree() -> Node:
    """
    Build a hierarchical taxonomic tree populated with sequences from
    Kraken2 RefSeq genome files.

    This function:
        - Scans the Kraken2 genomes directory (as specified in config) for
          all reference sequence folders.
        - For each reference sequence:
            - Loads all FASTA sequences for that genome.
            - Resolves its full NCBI taxonomy lineage using the
              `taxoniq` library.
            - Builds a hierarchical tree (using bigtree's `Node`)
              representing the taxonomy, with each taxonomic rank as a node
              and sequences as leaf nodes.
            - Attaches each sequence under the appropriate taxon in the tree,
              branching down to the reference sequence and sequence leaves.
        - Avoids duplicate nodes and sequence entries via internal tracking.
        - Reports progress using a terminal progress bar.

    Returns
    -------
    Node
        The root node of the constructed taxonomic tree.
        Each node includes attributes for taxonomic rank, scientific name,
        and (for sequence nodes) sequence data and description.

    Raises
    ------
    FileNotFoundError
        If the Kraken2 reference files or required taxonomy maps are missing.
    KeyError
        If a reference sequence cannot be mapped to a taxonomy ID.
    Exception
        For other errors in taxonomy lookup or file reading.

    See Also
    --------
    get_tax_ids : Loads RefSeq-to-taxonomy mapping.
    get_genome_sequences : Loads genome FASTA files into dicts.
    bigtree.Node : Tree structure used for the hierarchy.

    Notes
    -----
    - Taxonomy is resolved using the `taxoniq` package.
    - The function depends on the configuration of `KRAKEN_PATH` and
      `KRAKEN_DATABASE`.
    - Sequences are attached to leaves with node attributes: 'seq_name',
      'seq'.
    - Uses a progress bar to report status during tree construction.
    - Suitable for viral RefSeq collections, but can be adapted for other
      taxonomic databases.
    - Designed for robust, large-scale, reproducible dataset generation in
      computational genomics.
    - Duplicates and missing data are logged and skipped, so the function is
      robust to partial failures in the underlying data structure.

    Examples
    --------
    >>> from taxonomy_utils import generate_seqs_by_taxon_tree
    >>> tree = generate_seqs_by_taxon_tree()
    >>> # tree is a bigtree.Node object. Use bigtree utilities to traverse
    >>> # or extract data.
    """
    logger = logging.getLogger(__name__)
    taxon_tree = Node("root")
    gen_dir = f"{KRAKEN_PATH}/{KRAKEN_DATABASE}/genomes"
    try:
        tax_ids = get_tax_ids()
    except FileNotFoundError as e:
        logger.error(
            f"Kraken2 seqid2taxid.map file not found at expected location: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading tax IDs: {e}")
        raise

    if not os.path.isdir(gen_dir):
        logger.error(f"Genome directory {gen_dir} does not exist.")
        raise FileNotFoundError(f"Genome directory {gen_dir} does not exist.")

    refseq_dirs = [
        name for name in os.listdir(gen_dir)
        if os.path.isdir(os.path.join(gen_dir, name))
    ]

    num_refseq_files = len(refseq_dirs)
    visited_nodes = set()
    registered_subseq_refs = set()

    # Multiprocessing: Use all available CPUs
    max_workers = os.cpu_count() or 1

    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        # Prepare job args (pass all dependencies explicitly to avoid
        # fork/import bugs)
        jobs = [
            (ref_seq, tax_ids, KRAKEN_PATH, KRAKEN_DATABASE)
            for ref_seq in refseq_dirs
        ]
        # Show progress bar for parallel work
        for result in tqdm.tqdm(
            executor.map(parse_refseq_folder, jobs),
            total=len(jobs),
            desc="Loading genomes+taxonomy in parallel"
        ):
            results.append(result)

    bar = Bar("Building taxon tree", max=num_refseq_files)
    for item in results:
        if item.get("error"):
            logger.warning(f"Error for {item['ref_seq']}: {item['error']}")
            bar.next()
            continue
        
        ref_seq = item["ref_seq"]
        fasta_path = item["fasta_path"]
        sequence_headers = item["sequence_headers"]
        ranked_taxons = item["ranked_taxons"]
        path_parent_rank = ""
        for idx, ranked_taxon in enumerate(reversed(ranked_taxons)):
            scientific_name_sanitized = ranked_taxon.scientific_name.replace(
                " ", "_"
            )
            scientific_name_sanitized = scientific_name_sanitized.replace(
                "/", "_")
            slash = "" if path_parent_rank == "" else "/"
            path_parent_rank += slash + scientific_name_sanitized
            try:
                if path_parent_rank in visited_nodes:
                    already_added = True
                    # Verificamos se as sequências deste arquivo já foram registradas
                    for seq_info in sequence_headers:
                        if seq_info["id"] not in registered_subseq_refs:
                            already_added = False

                    if idx == len(ranked_taxons) - 1 and not already_added:
                        add_path_to_tree(
                            taxon_tree,
                            f"{path_parent_rank}/{ref_seq}",
                            node_attrs={"rank": "ref_seq", "taxid": str(ranked_taxon.tax_id)},
                        )
                        # AQUI: Adicionamos os nós de sequência APENAS com metadados
                        for seq_info in sequence_headers:
                            add_path_to_tree(
                                taxon_tree,
                                f"{path_parent_rank}/{ref_seq}/{seq_info['id']}",
                                node_attrs={
                                    "rank": "sequence",
                                    "seq_name": seq_info["name"],
                                    "fasta_path": fasta_path,  # Caminho para leitura posterior
                                    "header_id": seq_info["id"], # ID para busca no arquivo
                                    "taxid": str(ranked_taxon.tax_id)  # ID taxonômico
                                },
                            )
                            registered_subseq_refs.add(seq_info["id"])
                else:
                    if ranked_taxon.rank.name == "superkingdom":
                        visited_nodes |= {path_parent_rank}
                        taxon_tree = Node.from_dict({
                            "name": scientific_name_sanitized,
                            "rank": ranked_taxon.rank.name,
                            "taxid": str(ranked_taxon.tax_id)
                        })
                    else:
                        visited_nodes |= {path_parent_rank}
                        add_path_to_tree(
                            taxon_tree,
                            path_parent_rank,
                            node_attrs={"rank": ranked_taxon.rank.name, "taxid": str(ranked_taxon.tax_id)},
                        )
                        
                        already_added = True
                        for seq_info in sequence_headers:
                            if seq_info["id"] not in registered_subseq_refs:
                                already_added = False
                        
                        if idx == len(ranked_taxons) - 1 and not already_added:
                            add_path_to_tree(
                                taxon_tree,
                                f"{path_parent_rank}/{ref_seq}",
                                node_attrs={"rank": "ref_seq", "taxid": str(ranked_taxon.tax_id)},
                            )
                            # AQUI: Repetimos a lógica para o caso de novo nó taxonômico
                            for seq_info in sequence_headers:
                                add_path_to_tree(
                                    taxon_tree,
                                    f"{path_parent_rank}/{ref_seq}/{seq_info['id']}",
                                    node_attrs={
                                        "rank": "sequence",
                                        "seq_name": seq_info["name"],
                                        "fasta_path": fasta_path,
                                        "header_id": seq_info["id"],
                                        "taxid": str(ranked_taxon.tax_id)
                                    },
                                )
                                registered_subseq_refs.add(seq_info["id"])
            except Exception as e:
                logger.error(f"Error updating taxon tree at {path_parent_rank}: {e}")
                continue
        bar.next()
    bar.finish()
    return taxon_tree

from bigtree import preorder_iter

def estimar_datasets_hierarquicos(tree):
    # Incluindo root e superkingdom para capturar os datasets de nível superior
    main_ranks = ["root", "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus"]
    
    # Inicializa o dicionário com todas as chaves da lista acima
    breakdown = {rank: 0 for rank in main_ranks}
    total_datasets = 0
    
    print(f"\n{'Rank':<15} | {'Nó':<30} | {'Filhos (Classes)':<15}")
    print("-" * 65)

    for node in preorder_iter(tree):
        rank_name = getattr(node, "rank", "").lower()
        
        if rank_name in main_ranks:
            n_filhos = len(node.children)
            
            if n_filhos >= 2:
                total_datasets += 1
                breakdown[rank_name] += 1
                
                # Mostra os primeiros de cada rank para conferência
                if breakdown[rank_name] <= 3:
                    print(f"{rank_name:<15} | {node.node_name[:30]:<30} | {n_filhos:<15}")

    print("-" * 65)
    print("\nRESUMO DE DATASETS POR NÍVEL:")
    # Itera sobre a lista oficial para manter a ordem no print
    for rank in main_ranks:
        print(f" - {rank.capitalize()}: {breakdown[rank]} datasets")
    
    print(f"\nTOTAL GERAL DE DATASETS: {total_datasets}")

def exportar_analise_filhos(tree, output_file="levantamento_taxids.txt"):
    # 1. Localiza o nó Viruses
    viruses_node = None
    for node in preorder_iter(tree):
        if node.node_name.lower() == "viruses":
            viruses_node = node
            break
    
    if not viruses_node:
        print("Erro: Nó 'Viruses' não encontrado.")
        return

    # 2. Coleta os dados
    stats = []
    for child in viruses_node.children:
        tid = getattr(child, "taxid", "N/A")
        # Conta as folhas (rank sequence)
        n_leaves = len([n for n in preorder_iter(child) if getattr(n, 'rank', '') == 'sequence'])
        stats.append((tid, child.node_name, n_leaves))

    # 3. Ordena por volume de sequências
    stats.sort(key=lambda x: x[2], reverse=True)

    # 4. Escreve no arquivo
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{'TAXID':<15} | {'NOME_CIENTIFICO':<50} | {'SEQUENCIAS':<10}\n")
        f.write("-" * 80 + "\n")
        
        for tid, name, count in stats:
            f.write(f"{str(tid):<15} | {name:<50} | {count:<10}\n")

    print(f"Relatório gerado com sucesso: {output_file}")
    print(f"Total de grupos analisados: {len(stats)}")

tree = generate_seqs_by_taxon_tree()

exportar_analise_filhos(tree)


def get_genome_sequences(ref_seq: str) -> Dict[str, Dict[str, str]]:
    """
    Load all sequences for a specific RefSeq identifier from its genome FASTA
    file.

    Parameters
    ----------
    ref_seq : str
        Reference sequence identifier (e.g., "NC_019947.1").

    Returns
    -------
    dict of str to dict
        Dictionary mapping subsequence IDs to dictionaries with keys:
        - 'seq_name': str, the full description from the FASTA header.
        - 'seq': str, the DNA sequence.

    Raises
    ------
    FileNotFoundError
        If the genome FASTA file does not exist.

    Notes
    -----
    This function expects FASTA files named 'genome.fna' within each
    genome directory:
    '{KRAKEN_PATH}/{KRAKEN_DATABASE}/genomes/{ref_seq}/genome.fna'

    Example
    -------
    >>> genome = get_genome_sequences("NC_019947.1")
    >>> list(genome.keys())
    ['NC_019947.1']
    >>> genome['NC_019947.1']['seq_name']
    'Tomato yellow mottle virus segment DNA-B, complete sequence'
    """
    sequences = {}
    # Start of fasta sequence example:
    # >NC_019947.1 Tomato yellow mottle virus segment DNA-B, complete sequence
    with open(
        f"{ KRAKEN_PATH }/{ KRAKEN_DATABASE }/genomes/{ ref_seq }/genome.fna"
    ) as f:
        lines = f.readlines()
        subseq = None
        subseq_ref = ""
        subseq_name = ""
        for i in range(len(lines)):
            if lines[i].startswith(">"):
                if subseq:
                    sequences[subseq_ref] = {"seq_name": subseq_name,
                                             "seq": subseq}
                line_split = lines[i].split()
                subseq_ref = line_split[0][1:]
                subseq_name = " ".join(line_split[1:])
                subseq = ""
            elif i == len(lines) - 1:
                subseq += lines[i].rstrip()
                sequences[subseq_ref] = {"seq_name": subseq_name,
                                         "seq": subseq}
            else:
                subseq += lines[i].rstrip()

    return sequences

def _get_fasta_sequence_length(fasta_path, header_target):
    """Lê o comprimento da sequência sem guardá-la."""
    length = 0
    found = False
    with open(fasta_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                if line[1:].split()[0] == header_target:
                    found = True
                    continue
                elif found: break
            elif found:
                length += len(line.strip())
    return length

def get_subseqs_from_final_node(
        final_node: Node, n: int, min_len: int = 100,
        max_len: int = 512, rng: random.Random | None = None,
        parallel: bool = True, max_workers: int | None = None) -> list[str]:
    
    sequence_nodes = find_attrs(final_node, "rank", "sequence")
    """
    Extract representative subsequences from a taxon node, in parallel if
    requested.

    For each sequence leaf in the final node, determines the proportional
    number of subsequences to sample based on sequence length, and samples
    subsequences (using a random window) for each.

    When `parallel=True`, the extraction for each sequence is performed in
    parallel across available CPU cores, using
    `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    final_node : Node
        The final taxonomic node from which to extract sequences.
    n : int
        Number of subsequences to extract (total across all sequences).
    min_len : int, optional
        Minimum subsequence length (default is 100).
    max_len : int, optional
        Maximum subsequence length (default is 512).
    rng : random.Random, optional
        Random number generator instance for reproducibility. If None,
        uses global randomness (note: for parallel mode, each worker
        receives its own default RNG unless you provide a seed per
        sequence).
    parallel : bool, optional
        If True, extract subsequences from each sequence in parallel using
        all available CPU cores.
        If False, extraction is performed serially (default is True).
    max_workers : int, optional
        The maximum number of parallel workers to use (default: all CPUs).

    Returns
    -------
    list of str
        List of extracted DNA subsequences from all sequences under
        `final_node`.

    Notes
    -----
    - The total number of subsequences is always exactly `n`, distributed
      across all sequence leaves proportionally to their length. Counts are
      rounded and compensated as needed.
    - When `parallel=True`, each sequence is processed independently in a
      separate process.
    - If the number of sequence leaves is small, parallelization may
      provide little speedup.
    - Reproducibility: For strict reproducibility in parallel mode, you
      must manage random seeds yourself per sequence. Otherwise, worker
      RNGs are independent.
    - Suitable for large taxonomic nodes with many constituent sequences.

    Examples
    --------
    Extract 10 subsequences from all sequences under `node`, in parallel:

    >>> subseqs = get_subseqs_from_final_node(
    ...     node, 10, min_len=150, max_len=400, parallel=True)
    >>> print(subseqs[0])
    'AGGCTT...'

    Or extract serially (for debugging):

    >>> subseqs = get_subseqs_from_final_node(
    ...     node, 10, min_len=150, max_len=400, parallel=False)
    >>> print(subseqs[-1])
    'TTGGCA...'
    """
    # Precisamos dos comprimentos para sua lógica de proporção (fraction * n)
    # mas não das sequências em si.
    node_metadata = {}
    sequence_lengths = {}
    total_len = 0
    
    for node in sequence_nodes:
        ref_seq = node.node_name
        f_path = node.get_attr("fasta_path")
        h_id = node.get_attr("header_id")
        
        # Obtemos apenas o tamanho (muito leve para a RAM)
        length = _get_fasta_sequence_length(f_path, h_id)
        
        node_metadata[ref_seq] = (f_path, h_id)
        sequence_lengths[ref_seq] = length
        total_len += length

    if total_len == 0: return []

    # --- SUA LÓGICA DE PROPORÇÃO E ARREDONDAMENTO PERMANECE IDÊNTICA ---
    n_seqs = {}
    for ref_seq, length in sequence_lengths.items():
        fraction = length / total_len
        n_seqs[ref_seq] = int(round(fraction * n, 0))

    if sum(n_seqs.values()) != n:
        while sum(n_seqs.values()) != n:
            if sum(n_seqs.values()) < n:
                n_seqs[min(n_seqs, key=n_seqs.get)] += 1
            elif sum(n_seqs.values()) > n:
                n_seqs[max(n_seqs, key=n_seqs.get)] -= 1
    # ... (restante da sua lógica de n_seqs[ref_seq] == 0)

    # Prepara os jobs para o Executor
    if rng is not None and hasattr(rng, 'randrange'):
        base_seed = rng.randrange(1 << 30)
    else:
        base_seed = random.randrange(1 << 30)

    jobs = []
    for i, (ref_seq, count) in enumerate(n_seqs.items()):
        if count > 0:
            f_path, h_id = node_metadata[ref_seq]
            # Enviamos apenas metadados (Tuplas leves)
            jobs.append((f_path, h_id, count, min_len, max_len, base_seed + i))

    subseqs = []
    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # O executor agora mapeia os caminhos, não o DNA
            results = list(executor.map(extract_subseqs_worker, jobs))
        for res in results:
            subseqs.extend(res)
    else:
        for job in jobs:
            subseqs.extend(extract_subseqs_worker(job))

    return subseqs

def _read_single_sequence(fasta_path: str, header_target: str) -> str:
    """Lê uma sequência específica de um arquivo FASTA sem carregar o arquivo todo na RAM."""
    seq = []
    found = False
    try:
        with open(fasta_path, "r") as f:
            for line in f:
                if line.startswith(">"):
                    if line[1:].split()[0] == header_target:
                        found = True
                        continue
                    elif found:
                        break  # Encontrou o próximo header, interrompe a leitura
                elif found:
                    seq.append(line.strip())
    except FileNotFoundError:
        logging.getLogger(__name__).error(f"Arquivo não encontrado: {fasta_path}")
        return ""
    return "".join(seq)

def extract_subseqs_worker(args: tuple) -> list[str]:
    """
    Parallel worker function for extracting random DNA subsequences
    from a sequence.

    This function is designed to be used as a worker target for
    `concurrent.futures.ProcessPoolExecutor` or similar parallelization
    frameworks. It unpacks arguments and calls `extract_subseqs`, returning
    a list of subsequences extracted from the input DNA sequence.

    Parameters
    ----------
    args : tuple
        A tuple containing:
            - sequence (str): The full DNA sequence to sample from.
            - n (int): Number of subsequences to extract.
            - min_len (int): Minimum length for each subsequence.
            - rng (random.Random or None): Optional random number generator
              instance for reproducibility. If None, uses the global RNG.
              for reproducibility. If None, uses the global RNG.

    Returns
    -------
    list of str
        List of extracted DNA subsequences.

    Notes
    - This function is required for compatibility with Python's
      multiprocessing, which cannot pickle lambda functions or
      locally-defined functions.
      which cannot pickle lambda functions or locally-defined functions.
    - Each worker receives its own arguments tuple; random seed reproducibility
      is only guaranteed if a seeded RNG or seed is passed per job.
    - Intended for use with `executor.map(extract_subseqs_worker, jobs)` where
      `jobs` is a list of argument tuples as described above.
    - Handles exceptions by propagating them up to the parent process;
      any errors will terminate the corresponding parallel job.

    Examples
    --------
    >>> args = ("ACGTACGTACGT", 5, 3, 6, None)
    >>> result = extract_subseqs_worker(args)
    >>> print(result)
    ['CGTACG', 'TACGTA', 'ACGTAC', 'GTACGT', 'TACGTA']
    """
    fasta_path, header_id, n, min_len, max_len, seed = args
    
    # 1. Leitura local: a string 'seq' nasce aqui, dentro do processo filho.
    # Isso evita que o processo pai tenha que carregar e 'picklar' o DNA.
    from dataset.taxonomy_utils import _read_single_sequence 
    seq_string = _read_single_sequence(fasta_path, header_id)
    
    if not seq_string:
        return []

    # 2. Reconstrução do RNG para garantir a reprodutibilidade que você implementou
    import random
    rng = random.Random(seed)
    
    # 3. CHAMADA DA SUA LÓGICA ORIGINAL
    # Importamos o seu módulo original e passamos os argumentos exatamente como antes
    from dataset import sequence_utils
    return sequence_utils.extract_subseqs(
        seq=seq_string, 
        n=n, 
        min_len=min_len, 
        max_len=max_len, 
        rng=rng
    )


def extract_max_subseqs_set(
    final_taxon_node: Node, window_size: int
) -> Set[str]:
    """
    Generate a set of all unique possible subsequences of a given window size
    from all sequences under the specified taxonomic node.

    Parameters
    ----------
    final_taxon_node : Node
        Taxonomic node containing sequence leaves.
    window_size : int
        Length of the sliding window for subsequence extraction.

    Returns
    -------
    set of str
        Set of unique subsequences of the specified length.

    Notes
    -----
    This is used to deduplicate and analyze sequence content for a taxonomic
    group.

    Example
    -------
    >>> unique_seqs = extract_max_subseqs_set(node, 100)
    >>> len(unique_seqs)
    12345
    """
    max_subseqs = set()
    seqs = find_attrs(final_taxon_node, "rank", "sequence")

    for sequence_node in seqs:
        # CORREÇÃO: Acessar metadados em vez da string bruta
        f_path = sequence_node.get_attr("fasta_path")
        h_id = sequence_node.get_attr("header_id")
        
        # Lê a sequência apenas durante o processamento deste nó
        sequence = _read_single_sequence(f_path, h_id)
        
        if sequence:
            for i in range(0, len(sequence) - window_size + 1):
                max_subseqs.add(sequence[i:i+window_size])
            
            # Força a liberação da string da memória
            del sequence

    return max_subseqs
