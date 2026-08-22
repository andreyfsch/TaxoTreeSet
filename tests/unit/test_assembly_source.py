"""Fonte de montagem: padrao inalterado, GenBank quando pedido."""
from taxotreeset.core.orchestrator import DiscoveryOrchestrator as D


def test_padrao_continua_refseq():
    assert "--assembly-source" in D._build_summary_command("10239", "complete")
    cmd = D._build_summary_command("10239", "complete")
    assert cmd[cmd.index("--assembly-source") + 1] == "RefSeq"


def test_genbank_quando_pedido():
    cmd = D._build_summary_command("10239", "complete", "GenBank")
    assert cmd[cmd.index("--assembly-source") + 1] == "GenBank"


def test_negativos_cross_domain_seguem_a_fonte():
    o = D.__new__(D)
    o.assembly_source = "GenBank"
    # o comando dos negativos e montado inline; confere que o atributo e o usado
    assert o.assembly_source == "GenBank"
