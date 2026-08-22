"""Deduplicacao: desligada por padrao, colapsa replicas quando ligada."""
from unittest.mock import patch

from taxotreeset.core._orchestration._cluster import dereplicate_units

MOD = "taxotreeset.core._orchestration._cluster._read_single_sequence"


def _unit(nome, seq):
    return [{"fasta_path": nome, "header_id": nome, "_seq": seq}]


def _falso(seqs):
    return lambda caminho, header: seqs[header]


def test_threshold_1_e_no_op():
    seqs = {"a": "ACGT" * 40, "b": "ACGT" * 40}
    units = [_unit("a", seqs["a"]), _unit("b", seqs["b"])]
    with patch(MOD, side_effect=_falso(seqs)):
        assert dereplicate_units(units, threshold=1.0) == units


def test_colapsa_identicos():
    seqs = {"a": "ACGTTGCA" * 30, "b": "ACGTTGCA" * 30}
    units = [_unit("a", seqs["a"]), _unit("b", seqs["b"])]
    with patch(MOD, side_effect=_falso(seqs)):
        mantidos = dereplicate_units(units, threshold=0.95)
    assert len(mantidos) == 1
    assert mantidos[0][0]["header_id"] == "a"      # o primeiro e o representante


def test_mantem_divergentes():
    seqs = {"a": "ACGT" * 60, "b": "TTAGGCAT" * 30}
    units = [_unit("a", seqs["a"]), _unit("b", seqs["b"])]
    with patch(MOD, side_effect=_falso(seqs)):
        assert len(dereplicate_units(units, threshold=0.95)) == 2


def test_genoma_segmentado_e_uma_unidade():
    """Os 8 segmentos de um acesso sao sketchados juntos, nao como 8 genomas."""
    seqs = {"s1": "ACGT" * 30, "s2": "TTGG" * 30,
            "r1": "ACGT" * 30, "r2": "TTGG" * 30}
    a = [{"fasta_path": "x", "header_id": "s1"}, {"fasta_path": "x", "header_id": "s2"}]
    b = [{"fasta_path": "y", "header_id": "r1"}, {"fasta_path": "y", "header_id": "r2"}]
    with patch(MOD, side_effect=_falso(seqs)):
        mantidos = dereplicate_units([a, b], threshold=0.95)
    assert len(mantidos) == 1        # replicas segmento a segmento -> um genoma so
