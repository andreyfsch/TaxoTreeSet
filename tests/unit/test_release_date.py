"""release_date entra no registro da accession, para o corte temporal."""
from taxotreeset.io.registry import NCBIRegistry


def _rec(**info):
    return NCBIRegistry._build_accession_entry(
        "10239", {"assembly_info": info, "organism": {"organism_name": "X"},
                  "assembly_stats": {"total_sequence_length": 100}})


def test_captura_a_data():
    assert _rec(assembly_level="Complete Genome",
                release_date="2024-03-15")["release_date"] == "2024-03-15"


def test_ausente_vira_none():
    assert _rec(assembly_level="Complete Genome")["release_date"] is None


def test_nao_quebra_os_outros_campos():
    r = _rec(assembly_level="Complete Genome", release_date="2020-01-01")
    assert r["taxid"] == "10239" and r["is_reference"] is True
    assert r["total_sequence_length"] == 100 and r["downloaded"] is False
