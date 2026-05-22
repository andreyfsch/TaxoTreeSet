import json
import subprocess
import os
import taxoniq
from tqdm import tqdm

def analyze_bacteria_phyla():
    # Garante que o ambiente herda a API Key do terminal
    env = os.environ.copy()
    
    # Comando para buscar todas as assembleias completas de Bacteria (TaxID 2)
    cmd = [
        "datasets", "summary", "genome", "taxon", "2",
        "--assembly-source", "RefSeq",
        "--assembly-level", "complete",
        "--as-json-lines"
    ]
    
    print("🛰️  Consultando o NCBI RefSeq (Bacteria)... Isto pode levar alguns segundos para iniciar.")
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        
        phyla_metrics = {}
        unclassified_count = 0
        
        with tqdm(desc="Analisando espécies bacterianas", unit=" seqs") as pbar:
            for line in process.stdout:
                if not line.strip(): continue
                try:
                    report = json.loads(line)
                    taxid = report.get("organism", {}).get("tax_id")
                    if not taxid: continue
                    
                    # Resolvemos a linhagem "para cima" usando o taxoniq (que é 100% confiável nisso)
                    species_taxon = taxoniq.Taxon(int(taxid))
                    
                    # Buscamos o nó de rank 'phylum' na linhagem desta espécie
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
                    # Ignora falhas pontuais de busca no banco local do taxoniq
                    continue

        process.wait()
        
        if not phyla_metrics:
            stderr_output = process.stderr.read()
            print(f"❌ Erro na execução do CLI: {stderr_output}")
            return

        # Exibe os resultados ordenados por abundância (maior para menor)
        print("\n📊 Relação de Filos encontrados no seu Dataset Real:")
        print(f"{'TaxID':<12} | {'Nome do Filo':<30} | {'Genomas Completos':<15}")
        print("-" * 65)
        
        # Ordena do filo com mais genomas para o com menos
        sorted_phyla = sorted(phyla_metrics.items(), key=lambda x: x[1]["count"], reverse=True)
        
        for tid, info in sorted_phyla:
            print(f"{tid:<12} | {info['name']:<30} | {info['count']:<15}")
            
        print("-" * 65)
        print(f"Sequências sem filo definido: {unclassified_count}")
        
        # Gera o esqueleto do JSON para você copiar e colar no mapping.json
        print("\n🔧 Esqueleto de configuração para o seu 'mapping.json':")
        print('"redirections": {')
        for tid, info in sorted_phyla:
            # Aqui você decide o corte (ex: se tiver mais de 50 genomas, mantém individual, se não, joga pro virtual_id)
            target = tid if info['count'] > 50 else "999101"
            print(f'  "{tid}": {{ "target_id": "{target}", "label": "{info["name"]}" }},')
        print('}')

    except Exception as e:
        print(f"❌ Erro crítico no script: {e}")

if __name__ == "__main__":
    analyze_bacteria_phyla()