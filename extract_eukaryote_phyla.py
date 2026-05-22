import json
import subprocess
import os
import taxoniq
from tqdm import tqdm

def analyze_eukaryote_phyla():
    env = os.environ.copy()
    
    # CORREÇÃO: Adicionado 'chromosome' ao assembly-level
    cmd = [
        "datasets", "summary", "genome", "taxon", "2759",
        "--assembly-source", "RefSeq",
        "--assembly-level", "complete,chromosome", 
        "--as-json-lines"
    ]
    
    print("🛰️  Consultando o NCBI RefSeq (Eukaryota)...")
    print("Buscando genomas nos níveis 'Complete' e 'Chromosome'.")
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        
        groups_metrics = {}
        unclassified_count = 0
        
        with tqdm(desc="Analisando espécies de Eucariotos", unit=" seqs") as pbar:
            for line in process.stdout:
                if not line.strip(): continue
                try:
                    report = json.loads(line)
                    taxid = report.get("organism", {}).get("tax_id")
                    if not taxid: continue
                    
                    species_taxon = taxoniq.Taxon(int(taxid))
                    group_id = None
                    group_name = None
                    
                    for parent in species_taxon.ranked_lineage:
                        if parent.rank and parent.rank.name == "phylum":
                            group_id = str(parent.tax_id)
                            group_name = parent.scientific_name
                            break
                        if parent.scientific_name in ["Metazoa", "Viridiplantae", "Fungi"] or (parent.rank and parent.rank.name == "kingdom"):
                            group_id = str(parent.tax_id)
                            group_name = parent.scientific_name
                    
                    if group_id:
                        if group_id not in groups_metrics:
                            groups_metrics[group_id] = {"name": group_name, "count": 0}
                        groups_metrics[group_id]["count"] += 1
                    else:
                        unclassified_count += 1
                        
                    pbar.update(1)
                    
                except Exception:
                    continue

        process.wait()
        
        if not groups_metrics:
            stderr_output = process.stderr.read()
            print(f"❌ Erro na execução do CLI: {stderr_output}")
            return

        print("\n📊 Grupos/Filos de Eukaryota encontrados no seu Dataset Real:")
        print(f"{'TaxID':<12} | {'Nome do Grupo':<30} | {'Genomas de Alta Qualidade':<15}")
        print("-" * 65)
        
        sorted_groups = sorted(groups_metrics.items(), key=lambda x: x[1]["count"], reverse=True)
        
        for tid, info in sorted_groups:
            print(f"{tid:<12} | {info['name']:<30} | {info['count']:<15}")
            
        print("-" * 65)
        print(f"Sequências em nós órfãos ou não resolvidos: {unclassified_count}")
        
        print("\n🔧 Esqueleto de configuração para o seu 'mapping.json':")
        print('"redirections": {')
        for tid, info in sorted_groups:
            # Mantemos o corte adaptativo de >= 2 genomas para virar nó na cascata LoRA
            target = tid if info['count'] >= 2 else "999301"
            clean_name = info['name'].replace(" ", "_").replace("/", "_")
            print(f'  "{tid}": {{ "target_id": "{target}", "label": "{clean_name}" }},')
        print('}')

    except Exception as e:
        print(f"❌ Erro crítico no script: {e}")

if __name__ == "__main__":
    analyze_eukaryote_phyla()