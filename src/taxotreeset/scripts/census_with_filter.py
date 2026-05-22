# scripts/census_with_filter.py
import sys
sys.path.insert(0, "src")

import json
import tarfile
import os
from collections import Counter, defaultdict
import urllib.request

from taxotreeset.io.noise_filter import NoiseFilter

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
TAXDUMP_DIR = "/tmp/ncbi_taxdump"

# === Parsers idênticos ao script anterior ===
# Cole o bloco de parsing de nodes.dmp e names.dmp aqui
# ...
# Baixa e extrai uma vez
if not os.path.exists(f"{TAXDUMP_DIR}/nodes.dmp"):
    os.makedirs(TAXDUMP_DIR, exist_ok=True)
    tarball = f"{TAXDUMP_DIR}/taxdump.tar.gz"
    if not os.path.exists(tarball):
        print(f"Baixando taxdump.tar.gz ...")
        urllib.request.urlretrieve(TAXDUMP_URL, tarball)
    with tarfile.open(tarball) as tf:
        tf.extractall(TAXDUMP_DIR)

# Parser do nodes.dmp: cada linha é "taxid | parent_taxid | rank | ..."
print("Parsing nodes.dmp ...")
parent_of = {}     # taxid -> parent_taxid
rank_of = {}       # taxid -> rank string
with open(f"{TAXDUMP_DIR}/nodes.dmp") as f:
    for line in f:
        parts = [p.strip() for p in line.split("|")]
        taxid, parent, rank = parts[0], parts[1], parts[2]
        parent_of[taxid] = parent
        rank_of[taxid] = rank

# Parser do names.dmp: pega só scientific_name
print("Parsing names.dmp ...")
name_of = {}
with open(f"{TAXDUMP_DIR}/names.dmp") as f:
    for line in f:
        parts = [p.strip() for p in line.split("|")]
        taxid, name, _, name_type = parts[0], parts[1], parts[2], parts[3]
        if name_type == "scientific name":
            name_of[taxid] = name

# === Carrega o filtro ===
nf = NoiseFilter("configs/noise_patterns.json")

# === Carrega registry ===
print("\nCarregando registry...")
reg = json.load(open("data/registry.json"))
accession_taxids = set()
for acc_info in reg["accessions"].values():
    tid = acc_info.get("taxid")
    if tid:
        accession_taxids.add(str(tid))
print(f"Accessions: {len(reg['accessions'])}, taxids únicos: {len(accession_taxids)}")

# === Computa lineages aplicando o filtro ===
print("Computando lineages com filtro de noise ativo...")
reachable_taxids = set()
filtered_out = set()
for tid in accession_taxids:
    cur = tid
    while cur and cur != "1":
        name = name_of.get(cur, "")
        rank = rank_of.get(cur, "")
        if nf.is_noise(name, rank):
            filtered_out.add(cur)
            cur = parent_of.get(cur)
            continue
        reachable_taxids.add(cur)
        cur = parent_of.get(cur)

print(f"Táxons alcançáveis (após filtro): {len(reachable_taxids):,}")
print(f"Táxons filtrados (noise): {len(filtered_out):,}")
print(f"\nNoiseFilter stats: {nf.stats()}")

# === Re-computa children apenas com táxons alcançáveis ===
children_alcancaveis = defaultdict(set)
for tid in reachable_taxids:
    parent = parent_of.get(tid)
    # Sobe ancestrais até achar um parent que também esteja em reachable
    while parent and parent != "1" and parent not in reachable_taxids:
        parent = parent_of.get(parent)
    if parent in reachable_taxids:
        children_alcancaveis[parent].add((tid, rank_of[tid]))

# === Análise rank-mixed ===
problematic = []
for parent_tid, kids in children_alcancaveis.items():
    if len(kids) < 5:
        continue
    rank_dist = Counter(r for _, r in kids)
    pname = name_of.get(parent_tid, "?")
    prank = rank_of.get(parent_tid, "?")
    problematic.append((len(kids), parent_tid, pname, prank, dict(rank_dist)))

problematic.sort(reverse=True)
total_rank_mixed = sum(1 for _, _, _, _, dist in problematic if len(dist) > 1)
print(f"\nClados com >= 5 filhos (pós-filtro): {len(problematic)}")
print(f"  Rank-mixed: {total_rank_mixed}\n")

print("Clados rank-mixed remanescentes (Camada B + C):")
for n, ptid, pname, prank, dist in problematic:
    if len(dist) > 1:
        print(f"  {n:4d} | {ptid:>8} {prank:>15} {pname}")
        print(f"          ranks: {dist}")

print(f"\nTop 30 por cardinalidade:")
for n, ptid, pname, prank, dist in problematic[:30]:
    flag = " ⚠️" if len(dist) > 1 else ""
    print(f"  {n:4d} | {ptid:>8} {prank:>15} {pname}{flag}")