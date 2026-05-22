"""Census taxonômico universal: percorre a árvore NCBI inteira
sob um domínio escolhido, identifica clados com cardinalidade
problemática e rank-mixing."""
import urllib.request
import tarfile
import os
from collections import Counter, defaultdict

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
TAXDUMP_DIR = "/tmp/ncbi_taxdump"

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

"""Census restrito ao registry: identifica clados problemáticos
que efetivamente têm accessions no seu dataset."""
import json
import tarfile
import os
from collections import Counter, defaultdict
import urllib.request

TAXDUMP_DIR = "/tmp/ncbi_taxdump"
# (mesmo parser de nodes.dmp e names.dmp do census anterior)
# ... cole o bloco de parser aqui ...

# Carrega o registry e descobre quais taxids têm accessions
print("\nCarregando registry...")
reg = json.load(open("data/registry.json"))
accession_taxids = set()
for acc_info in reg["accessions"].values():
    tid = acc_info.get("taxid")
    if tid:
        accession_taxids.add(str(tid))
print(f"Accessions no registry: {len(reg['accessions'])}, taxids únicos: {len(accession_taxids)}")

# Computa a lineage completa de cada taxid do registry
print("Computando lineages...")
reachable_taxids = set()
for tid in accession_taxids:
    cur = tid
    while cur and cur != "1":  # 1 = root
        reachable_taxids.add(cur)
        cur = parent_of.get(cur)
print(f"Total de táxons alcançáveis pelo registry: {len(reachable_taxids):,}")

# Para cada nó alcançável, conta SOMENTE filhos alcançáveis (não todo o NCBI)
children_alcancaveis = defaultdict(set)
for tid in reachable_taxids:
    parent = parent_of.get(tid)
    if parent in reachable_taxids:
        children_alcancaveis[parent].add((tid, rank_of[tid]))

# Análise igual ao census anterior
print(f"\nClados alcançáveis com >= 5 filhos diretos:\n")
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
print(f"Total de clados com >= 5 filhos no registry: {len(problematic)}")
print(f"  Destes, rank-mixed: {total_rank_mixed}\n")

print(f"Todos os clados rank-mixed (são esses que precisam de política):")
for n, ptid, pname, prank, dist in problematic:
    if len(dist) > 1:
        print(f"  {n:4d} filhos | {ptid:>8} {prank:>15} {pname}")
        print(f"            ranks: {dist}")

print(f"\nTop 30 por cardinalidade absoluta:")
for n, ptid, pname, prank, dist in problematic[:30]:
    flag = " ⚠️" if len(dist) > 1 else ""
    print(f"  {n:4d} filhos | {ptid:>8} {prank:>15} {pname}{flag}")