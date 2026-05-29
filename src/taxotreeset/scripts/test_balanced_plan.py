"""
Sanity test das funções de balanceamento sem rodar o pipeline.
Carrega um tree_builder real e testa em 3-5 nós conhecidos.
"""
from collections import defaultdict

import os
import sys
_PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.taxotreeset.io.registry import NCBIRegistry
from src.taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from src.taxotreeset.core.generation_orchestrator import (
    compute_node_capacity,
    compute_balanced_extraction_plan,
)
import logging


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")


# Carrega registry e árvore
print("Carregando registry e construindo árvore...")
registry = NCBIRegistry(
    config_path="configs/mapping.json",
    registry_path="data/registry.json",
)

# Path do registry (depende da sua implementação interna)
taxon_tree = generate_seqs_by_taxon_tree(
    registry_path="data/registry.json",
    vault_path="data/vault",
    domain_taxid="10239",
    mapping_path="configs/mapping.json",
    noise_patterns_path="configs/noise_patterns.json",
)

# Constrói leaf_cache (taxid -> list of seq_leaf)
print("Indexando leaf_cache...")
leaf_cache = defaultdict(list)
for d in taxon_tree.descendants:
    if getattr(d, "rank", "") == "sequence":
        # Sobe pra encontrar o taxon-pai do qual essa sequência é folha
        parent = d.parent
        while parent and getattr(parent, "rank", "") in ("sequence", ""):
            parent = parent.parent
        if parent:
            leaf_cache[str(parent.name)].append(d)

print(f"Leaf cache: {len(leaf_cache)} taxids com sequências")
print()

# Testa nós em ordem crescente de cardinalidade:
test_taxids = [
    "2842319",   # Fiersviridae - 178 gêneros, ~178 folhas. ~1-2 min.
    "10663",     # Tequatrovirus - ~80 espécies. ~30s.
    "1542744",   # Gemycircularvirus - ~92 espécies. ~30s.
    # "2731619",  # Caudoviricetes - PULAR pra teste rápido. ~10 min.
    # "10239",    # Viruses - PULAR. ~30+ min.
]

for taxid in test_taxids:
    print("=" * 70)
    nodes_match = [n for n in taxon_tree.descendants if str(n.name) == taxid]
    if not nodes_match:
        print(f"taxid={taxid} não encontrado")
        continue

    node = nodes_match[0]
    children = [c for c in node.children if getattr(
        c, "rank", "") != "sequence"]

    print(f"Testando nó {taxid} ({getattr(node, 'scientific_name', '?')})")
    print(f"  rank={getattr(node, 'rank', '?')}")
    print(f"  filhos={len(children)}")

    if not children:
        print(f"  SKIP: nó terminal taxonômico")
        continue

    # Modo APPROXIMATE (sempre — para teste no WSL)
    print(f"  Computando plano de balanceamento (APPROXIMATE)...")
    plan = compute_balanced_extraction_plan(
        parent_node=node,
        children=children,
        leaf_cache=leaf_cache,
        min_len=100,
        min_num_seqs=1000,
        cutoff_percentage=98.0,
        use_exact_capacity=False,  # forçando approximate
    )

    print(f"\n  RESULTADO:")
    print(f"    Cenário: {plan['scenario']}")
    print(f"    n_per_class: {plan['n_per_class']:,}")
    print(f"    Eligible children: {len(plan['eligible_children'])}")
    print(f"    Low capacity children: {len(plan['low_capacity_children'])}")
    print(f"    Decisão: {plan['decision_log']}")

    # Mostra capacidades top 5 e bottom 5
    caps_sorted = sorted(plan['capacities'].items(), key=lambda x: -x[1])
    print(f"\n  Top 5 capacidades:")
    for tid, cap in caps_sorted[:5]:
        print(f"    {tid}: {cap:,}")
    print(f"  Bottom 5 capacidades:")
    for tid, cap in caps_sorted[-5:]:
        print(f"    {tid}: {cap:,}")
    print()

print("=" * 70)
print("Teste com min_num_seqs=5000 (forçando Cenário 2 em Gemycircularvirus)")
node = next(n for n in taxon_tree.descendants if str(n.name) == "1542744")
children = [c for c in node.children if getattr(c, "rank", "") != "sequence"]
plan = compute_balanced_extraction_plan(
    parent_node=node,
    children=children,
    leaf_cache=leaf_cache,
    min_len=100,
    min_num_seqs=5000,  # forçando cutoff
    cutoff_percentage=98.0,
    use_exact_capacity=False,
)
print(f"  Cenário: {plan['scenario']}")
print(f"  n_per_class: {plan['n_per_class']:,}")
print(f"  Eligible: {len(plan['eligible_children'])}")
print(f"  Low_capacity: {len(plan['low_capacity_children'])}")

print("Teste concluído.")
