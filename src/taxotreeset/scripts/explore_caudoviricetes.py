import taxoniq
from collections import Counter

# Caudoviricetes = 2731619
caud = taxoniq.Taxon(2731619)

# Pega todos os descendentes em qualquer rank até family/genus
# (taxoniq não dá descendentes diretamente, vamos pelo registry)
import json
reg = json.load(open("data/registry.json"))

# Pra cada accession, verificar a lineage e categorizar
rank_counter = Counter()
caud_children = set()  # filhos diretos de Caudoviricetes em qualquer rank

for acc_id, info in reg["accessions"].items():
    taxid = info.get("taxid")
    if not taxid:
        continue
    try:
        t = taxoniq.Taxon(int(taxid))
        lineage_ids = [str(x.tax_id) for x in t.ranked_lineage]
        if "2731619" not in lineage_ids:
            continue
        # acha o nó IMEDIATAMENTE abaixo de Caudoviricetes na lineage
        idx = lineage_ids.index("2731619")
        if idx == 0:
            continue
        child_taxid = lineage_ids[idx - 1]  # imediatamente sob Caud
        child_taxon = taxoniq.Taxon(int(child_taxid))
        rank = child_taxon.rank.name if hasattr(child_taxon.rank, "name") else str(child_taxon.rank)
        caud_children.add((child_taxid, rank))
    except Exception:
        continue

print(f"Filhos diretos de Caudoviricetes: {len(caud_children)}")
print("Distribuição por rank:")
for rank, count in Counter(r for _, r in caud_children).most_common():
    print(f"  {rank}: {count}")