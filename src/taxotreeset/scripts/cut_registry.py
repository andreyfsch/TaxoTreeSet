#!/usr/bin/env python3
"""Recorta data/registry.json para um subconjunto que exercita todos os caminhos
de redirect do mapping.json (self-redirect, virtual fallback, default fallback)."""
import json
import sys
from pathlib import Path

try:
    import taxoniq
except ImportError:
    print("taxoniq não está instalado neste ambiente.")
    sys.exit(1)

REGISTRY_IN = Path("data/registry.json")
REGISTRY_OUT = Path("data/registry_test.json")

# Anchors selecionados pra cobrir cada caminho do mapping.json (escopo 10239)
# Formato: taxid_ancestral → (label_descritivo, max_accessions_a_pegar)
ANCHORS = {
    "2732396": ("Orthornavirae (self-redirect)", 12),
    "2731360": ("Heunggongvirae (self-redirect)", 8),
    "185751":  ("Pospiviroidae → 999002", 8),
    "1458186": ("Alphasatellitidae → 999002", 6),
    "1993640": ("Tolecusatellitidae → 999002", 6),
    "2842321": ("Kolmioviridae → 999002", 6),
    "687329":  ("Anelloviridae → 999003", 25),   # força >15 sob 999003 quando combinado com 2946196
    "2946196": ("Polydnaviriformidae → 999003", 20),
    "2946196": ("Polydnaviriformidae → 999003", 4),
    "10474":   ("Fuselloviridae → 999001", 6),
    "2732090": ("Loebvirae → 999001", 4),
}

def lineage_taxids(taxid_str: str) -> set[str]:
    try:
        t = taxoniq.Taxon(int(taxid_str))
        return {str(x.tax_id) for x in t.ranked_lineage}
    except Exception:
        return set()

def main():
    with REGISTRY_IN.open() as f:
        reg = json.load(f)
    
    print(f"Registry original: {len(reg['accessions'])} accessions, "
          f"{len(reg['taxons'])} taxons únicos")
    
    selected_acc, selected_tax = {}, {}
    counts = {a: 0 for a in ANCHORS}
    
    # Itera accessions e classifica pelo primeiro anchor que casa no lineage
    for acc_id, acc_info in reg["accessions"].items():
        taxid = acc_info.get("taxid")
        if not taxid:
            continue
        lin = lineage_taxids(taxid)
        for anchor, (_, cap) in ANCHORS.items():
            if anchor in lin and counts[anchor] < cap:
                selected_acc[acc_id] = acc_info
                selected_tax.setdefault(taxid, []).append(acc_id)
                counts[anchor] += 1
                break
    
    print("\nDistribuição por anchor:")
    for a, (label, cap) in ANCHORS.items():
        c = counts[a]
        status = "OK" if c == cap else ("PARCIAL" if c > 0 else "VAZIO")
        print(f"  [{status:7}] {label}: {c}/{cap}")
    
    print(f"\nTotal selecionado: {len(selected_acc)} accessions, "
          f"{len(selected_tax)} taxons únicos")
    
    out = {
        "last_update": reg.get("last_update"),
        "taxons": selected_tax,
        "accessions": selected_acc,
    }
    
    REGISTRY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_OUT.open("w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\nEscrito em {REGISTRY_OUT}")

if __name__ == "__main__":
    main()