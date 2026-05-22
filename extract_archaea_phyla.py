import json
import subprocess
import os
import taxoniq
from tqdm import tqdm

def analyze_archaea_phyla():
    # Garante a herança das variáveis de ambiente (incluindo a NCBI_API_KEY)
    env = os.environ.copy()
    
    # Comando para buscar todas as assembleias completas de Archaea (TaxID 2157)
    cmd = [
        "datasets", "summary", "genome", "taxon", "2157",
        "--assembly-source", "RefSeq",
        "--assembly-level", "complete",
        "--as-json-lines"
    ]
    
    print("🛰️  Consultando o NCBI RefSeq (Archaea)... Isto deve levar poucos segundos.")
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        
        phyla_metrics = {}
        unclassified_count = 0
        
        with tqdm(desc="Analisando espécies de Archaea", unit=" seqs") as pbar:
            for line in process.stdout:
                if not line.strip(): continue
                try:
                    report = json.loads(line)
                    taxid = report.get("organism", {}).get("tax_id")
                    if not taxid: continue
                    
                    # Resolução da linhagem ascendente via taxoniq
                    species_taxon = taxoniq.Taxon(int(taxid))
                    
                    phylum_id = None
                    phylum_name = None
                    
                    for parent in species_taxon.ranked_lineage:
                        if parent.rank and parent.rank.name == "phylum":
                            phylum_id = str(parent.tax_id)
                            phylum_name = parent.scientific_name
                            break
                    
                    if phylum_id:
                        if phylum_id not in phyla_metrics:
                            phyla_metrics[phylum_id] = {"name": phylum_name, "count": 0}
                        phyla_metrics[phylum_id]["count"] += 1
                    else:
                        unclassified_count += 1
                        
                    pbar.update(1)
                    
                except Exception:
                    continue

        process.wait()
        
        if not phyla_metrics:
            stderr_output = process.stderr.read()
            print(f"❌ Erro na execução do CLI: {stderr_output}")
            return

        print("\n📊 Relação de Filos de Archaea encontrados no seu Dataset Real:")
        print(f"{'TaxID':<12} | {'Nome do Filo':<30} | {'Genomas Completos':<15}")
        print("-" * 65)
        
        sorted_phyla = sorted(phyla_metrics.items(), key=lambda x: x[1]["count"], reverse=True)
        
        for tid, info in sorted_phyla:
            print(f"{tid:<12} | {info['name']:<30} | {info['count']:<15}")
            
        print("-" * 65)
        print(f"Sequências sem filo definido: {unclassified_count}")
        
        print("\n🔧 Esqueleto de configuração para o seu 'mapping.json':")
        print('"redirections": {')
        for tid, info in sorted_phyla:
            # Dado o tamanho reduzido de Archaea, um corte de >= 5 genomas completos 
            # costuma ser ideal para justificar um adaptador LoRA dedicado.
            target = tid if info['count'] >= 5 else "999201"
            print(f'  "{tid}": {{ "target_id": "{target}", "label": "{info["name"].replace(" ", "_")}" }},')
        print('}')

    except Exception as e:
        print(f"❌ Erro crítico no script: {e}")

if __name__ == "__main__":
    analyze_archaea_phyla()