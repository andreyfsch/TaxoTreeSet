import os
import json
import pandas as pd

def verify_dataset(manifest_path, datasets_root):
    if not os.path.exists(manifest_path):
        print(f"Erro: Arquivo {manifest_path} não encontrado.")
        return

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"Auditando {len(manifest)} nós estruturais...")
    
    for taxid, info in manifest.items():
        # Busca pela chave correta (path ou relative_path)
        rel_path = info.get('path') or info.get('relative_path')
        
        if not rel_path:
            print(f"[ERRO] ID {taxid} no manifesto não possui caminho (path/relative_path)")
            continue
            
        full_path = os.path.join(datasets_root, rel_path)
        
        # Valida se os arquivos esperados existem
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(full_path, f"{split}.parquet")
            if not os.path.exists(file_path):
                continue
                
            try:
                df = pd.read_parquet(file_path)
                
                if 'header_id' not in df.columns:
                    print(f"[ERRO] {split}.parquet em {rel_path} não possui coluna 'header_id'")
                    continue
                
                # Valida se os índices estão dentro do limite do manifesto local
                # labels é um dicionário, então pegamos as chaves numéricas
                max_label = max(map(int, info['labels'].keys()))
                
                actual_max = df['header_id'].max()
                if actual_max > max_label:
                    print(f"[ERRO] {split}.parquet em {rel_path} possui header_id={actual_max}, mas o manifesto só vai até {max_label}")
            
            except Exception as e:
                print(f"[ERRO] Falha ao ler {file_path}: {e}")

    print("Auditoria concluída.")

if __name__ == "__main__":
    verify_dataset('data/datasets/manifest_viruses.json', 'data/datasets')