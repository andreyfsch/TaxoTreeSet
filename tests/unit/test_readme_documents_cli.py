"""O README documenta toda a CLI publica.

POR QUE ISTO E UM TESTE E NAO UMA REVISAO. Em 2026-08-22 uma auditoria automatica
achou 19 flags e um subcomando inteiro (`composition`) ausentes do README, alem da
descricao de topo dizendo "from NCBI RefSeq" depois que o GenBank passou a ser
suportado. Nada disso apareceu em leitura por impressao -- o README parecia
completo. O repositorio e publico, entao a deriva entre codigo e documentacao
precisa quebrar o CI em vez de esperar alguem reparar.

O teste falha na direcao util: acrescentar uma flag sem documenta-la quebra; remover
uma flag documentada nao quebra (texto orfao e barato, promessa nao cumprida nao e).
"""
import argparse
from pathlib import Path

import pytest

from taxotreeset.cli import benchmark, composition, discover, generate, separability

README = Path(__file__).resolve().parents[2] / "README.md"

SUBCOMANDOS = {
    "discover": discover,
    "generate": generate,
    "separability": separability,
    "composition": composition,
    "benchmark": benchmark,
}

# Flags genericas de argparse/logging, iguais em todo subcomando: documenta-las
# cinco vezes seria ruido. Qualquer OUTRA isencao precisa de justificativa aqui.
ISENTAS = {"--help", "--log-level"}


def _flags(modulo) -> set[str]:
    """As opcoes longas que o subcomando registra."""
    parser = argparse.ArgumentParser(add_help=False)
    modulo.add_arguments(parser)
    return {
        opcao
        for acao in parser._actions
        for opcao in acao.option_strings
        if opcao.startswith("--")
    }


@pytest.fixture(scope="module")
def texto() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("nome", sorted(SUBCOMANDOS))
def test_subcomando_e_mencionado(nome: str, texto: str) -> None:
    assert f"taxotreeset {nome}" in texto, (
        f"O subcomando `{nome}` nao aparece no README. Cada subcomando publico "
        f"precisa de pelo menos uma secao que diga para que ele serve."
    )


@pytest.mark.parametrize("nome", sorted(SUBCOMANDOS))
def test_flags_documentadas(nome: str, texto: str) -> None:
    ausentes = sorted(f for f in _flags(SUBCOMANDOS[nome]) - ISENTAS if f not in texto)
    assert not ausentes, (
        f"Flags de `{nome}` ausentes do README: {ausentes}. Documente-as (ou, se "
        f"forem mesmo genericas, acrescente a ISENTAS com uma justificativa)."
    )


def test_descricao_de_topo_nao_promete_so_refseq(texto: str) -> None:
    """A fonte deixou de ser RefSeq-apenas; a primeira frase tem de refletir isso."""
    abertura = texto[: texto.index("## Overview")]
    assert "GenBank" in abertura, (
        "A descricao de topo nao menciona GenBank, mas --assembly-source o suporta. "
        "Foi exatamente esse tipo de deriva que motivou este teste."
    )
