import argparse
import json
import logging
import os
import sys
from src.taxotreeset.io.registry import NCBIRegistry
from src.taxotreeset.core.orchestrator import DiscoveryOrchestrator


def setup_logging():
    """Configura a telemetria gravando em arquivo e exibindo no terminal."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("discovery.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    setup_logging()
    logger = logging.getLogger("TaxoTreeSet.CLI")

    # 1. Configuração do Parser de Argumentos de Linha de Comando (CLI)
    parser = argparse.ArgumentParser(
        description="TaxoTreeSet - Motor de Mapeamento Taxonômico e Geração de Registros para ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--taxon-id", "-t",
        type=int,
        default=10239,
        help="NCBI TaxID da raiz biológica para iniciar o mapeamento (ex: 10239 para Vírus, 2 para Bacteria)"
    )
    parser.add_argument(
        "--mapping", "-m",
        type=str,
        default="configs/mapping.json",
        help="Caminho para o arquivo JSON de mapeamento de escopos e redirecionamentos"
    )
    parser.add_argument(
        "--registry", "-r",
        type=str,
        default="data/registry.json",
        help="Caminho de destino para o arquivo de inventário/registro"
    )
    parser.add_argument(
        "--reset", "-f",
        action="store_true",
        help="Se definido, força a exclusão do registry.json antigo antes de iniciar a nova descoberta"
    )

    args = parser.parse_args()

    # 2. Tratamento da Idempotência (Gerenciamento do arquivo antigo)
    if args.reset and os.path.exists(args.registry):
        try:
            os.remove(args.registry)
            logger.info(
                f"🧹 Flag --reset ativada. Arquivo antigo removido com sucesso: {args.registry}")
        except Exception as e:
            logger.error(
                f"Não foi possível remover o registro antigo em {args.registry}: {e}")
            sys.exit(1)
    elif os.path.exists(args.registry):
        logger.info(
            f"🔄 Arquivo de registro localizado em {args.registry}. Modo de adição dinâmica/incremental ativo.")

    # 3. Validação do arquivo de mapeamento de escopos
    if not os.path.exists(args.mapping):
        logger.error(
            f"❌ Erro Crítico: Arquivo de mapeamento ausente em {args.mapping}")
        sys.exit(1)

    try:
        with open(args.mapping, "r", encoding="utf-8") as f:
            mapping_config = json.load(f)
    except Exception as e:
        logger.error(f"❌ Erro ao ler o arquivo JSON de mapeamento: {e}")
        sys.exit(1)

    # 4. Inicialização Segura dos Componentes Estruturais
    try:
        # O NCBIRegistry internamente deve carregar o arquivo se ele existir (modo incremental)
        # ou instanciar um dicionário vazio caso o arquivo não exista
        registry = NCBIRegistry(
            config_path=args.mapping,
            registry_path=args.registry
        )

        orchestrator = DiscoveryOrchestrator(
            registry=registry,
            mapping_config=mapping_config
        )

        # 5. Execução do Fluxo de Trabalho (Workflow)
        logger.info(
            f"🚀 Iniciando varredura taxonômica para o TaxID: {args.taxon_id}")
        orchestrator.discover_from_root(args.taxon_id)

        logger.info("🎉 Processo de descoberta finalizado com sucesso.")
        print("\n" + "="*50)
        print("   🧬 Mapeamento Taxonômico Concluído!")
        print(f"   Raiz Processada : {args.taxon_id}")
        print(f"   Registro Atualizado : {args.registry}")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(
            f"💥 Falha crítica durante a execução do pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
