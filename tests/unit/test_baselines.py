"""Tests for the retained-only baseline glue (benchmark, P11-P5)."""

from unittest.mock import patch

from taxotreeset.benchmark.baselines import (
    export_retained_reference,
    parse_centrifuge_output,
    parse_kaiju_output,
    parse_kraken2_output,
    taxid_rank_map,
)

_READ = "taxotreeset.benchmark.baselines._read_single_sequence"

# T1 under genus GA (family F); T2 under genus GB (family F)
_LIN = {
    "T1": [{"taxid": "T1", "rank": "species"},
           {"taxid": "GA", "rank": "genus"},
           {"taxid": "F", "rank": "family"}],
    "T2": [{"taxid": "T2", "rank": "species"},
           {"taxid": "GB", "rank": "genus"},
           {"taxid": "F", "rank": "family"}],
}
_ACC = {
    "a1": {"taxid": "T1", "local_path": "/v.lmdb", "headers": [{"id": "H1"}]},
    "a2": {"taxid": "T2", "local_path": "/v.lmdb", "headers": [{"id": "H2"}]},
}


class TestTaxidRankMap:
    def test_maps_every_lineage_node(self):
        m = taxid_rank_map(_LIN)
        assert m["F"] == "family"
        assert m["GA"] == "genus"
        assert m["T1"] == "species"


class TestExportRetainedReference:
    def test_excludes_genomes_under_a_held_out_clade(self, tmp_path):
        fasta, seqmap = tmp_path / "ref.fasta", tmp_path / "map.tsv"
        with patch(_READ, return_value="ACGTACGTAC"):
            n = export_retained_reference(
                {"GA"}, _ACC, _LIN, str(fasta), str(seqmap))
        assert n == 1  # only T2/H2 retained (T1 lives under held-out GA)
        text = fasta.read_text()
        assert ">H2|kraken:taxid|T2" in text
        assert "H1" not in text
        assert seqmap.read_text().strip() == "H2\tT2"

    def test_retains_everything_without_holdout(self, tmp_path):
        fasta, seqmap = tmp_path / "r.fa", tmp_path / "m.tsv"
        with patch(_READ, return_value="ACGT" * 5):
            n = export_retained_reference(
                set(), _ACC, _LIN, str(fasta), str(seqmap))
        assert n == 2

    def test_skips_unreadable_genomes(self, tmp_path):
        fasta, seqmap = tmp_path / "r.fa", tmp_path / "m.tsv"
        with patch(_READ, return_value=""):
            n = export_retained_reference(
                set(), _ACC, _LIN, str(fasta), str(seqmap))
        assert n == 0


class TestParseKraken2Output:
    def test_classified_map_to_taxon_rank_unclassified_abstain(self):
        ranks = taxid_rank_map(_LIN)
        lines = [
            "C\tr1\tT2\t150\tmap",
            "U\tr2\t0\t150\t",       # unclassified
            "C\tr3\t0\t150\t",       # classified but taxid 0
            "C\tr4\tF\t150\tmap",    # LCA back-off to family
        ]
        preds = parse_kraken2_output(lines, ranks)
        assert preds["r1"] == ("T2", "species")
        assert preds["r2"] == (None, None)
        assert preds["r3"] == (None, None)
        assert preds["r4"] == ("F", "family")

    def test_ignores_malformed_lines(self):
        assert parse_kraken2_output(["garbage", ""], {}) == {}


class TestParseKaijuOutput:
    def test_reuses_kraken2_format(self):
        # Kaiju emits the same C|U <read> <taxid> shape.
        ranks = taxid_rank_map(_LIN)
        preds = parse_kaiju_output(["C\tr1\tT1", "U\tr2\t0"], ranks)
        assert preds["r1"] == ("T1", "species")
        assert preds["r2"] == (None, None)


class TestParseCentrifugeOutput:
    def test_keeps_best_score_per_read_and_abstains_on_zero(self):
        ranks = taxid_rank_map(_LIN)
        lines = [
            "readID\tseqID\ttaxID\tscore\t2ndBestScore\thitLength\tqueryLength\tnumMatches",
            "r1\tNC_1\tGA\t900\t0\t150\t150\t2",     # r1: lower score
            "r1\tNC_2\tT1\t1800\t0\t150\t150\t2",    # r1: higher score -> wins
            "r2\tNC_3\t0\t0\t0\t0\t150\t0",          # unclassified -> abstain
            "r3\tNC_4\tF\t500\t0\t150\t150\t1",
        ]
        preds = parse_centrifuge_output(lines, ranks)
        assert preds["r1"] == ("T1", "species")   # best-scoring assignment
        assert preds["r2"] == (None, None)
        assert preds["r3"] == ("F", "family")

    def test_skips_header_and_malformed(self):
        assert parse_centrifuge_output(["readID\tseqID\ttaxID", "junk"], {}) == {}
