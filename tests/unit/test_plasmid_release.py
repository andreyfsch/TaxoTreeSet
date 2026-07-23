"""Tests for the RefSeq plasmid release GBFF parser (P9)."""

from taxotreeset.dataset.utils import _read_single_sequence
from taxotreeset.io.plasmid_release import (
    PlasmidRecord,
    ingest_records_to_vault,
    parse_gbff_records,
    record_to_report,
)

# A minimal two-record GBFF stream in the RefSeq plasmid release format. The
# first record is well-formed; the second omits the host db_xref (unplaceable).
_GBFF = """\
LOCUS       NZ_CP012345          40 bp    DNA     circular BCT 01-JAN-2020
DEFINITION  Escherichia coli strain K-12 plasmid pXYZ, complete sequence.
ACCESSION   NZ_CP012345
VERSION     NZ_CP012345.1
SOURCE      Escherichia coli
  ORGANISM  Escherichia coli
            Bacteria; Pseudomonadota; Gammaproteobacteria.
FEATURES             Location/Qualifiers
     source          1..40
                     /organism="Escherichia coli"
                     /mol_type="genomic DNA"
                     /strain="K-12"
                     /db_xref="taxon:562"
                     /plasmid="pXYZ"
ORIGIN
        1 atgcatgcat gcatgcatgc atgcatgcat gcatgcatgc
//
LOCUS       NZ_NOHOST            12 bp    DNA     circular BCT 01-JAN-2020
DEFINITION  Unplaceable plasmid, complete sequence.
ACCESSION   NZ_NOHOST
VERSION     NZ_NOHOST.1
FEATURES             Location/Qualifiers
     source          1..12
                     /organism="unknown"
ORIGIN
        1 acgtacgtacgt
//
"""


class TestParseGbff:
    def _records(self):
        return list(parse_gbff_records(_GBFF.splitlines(keepends=True)))

    def test_yields_only_placeable_records(self):
        # The host-less second record is dropped, not raised.
        recs = self._records()
        assert len(recs) == 1
        assert recs[0].accession == "NZ_CP012345.1"

    def test_extracts_host_taxid_and_organism(self):
        rec = self._records()[0]
        assert rec.host_taxid == "562"
        assert rec.organism == "Escherichia coli"

    def test_reconstructs_uppercased_sequence(self):
        rec = self._records()[0]
        assert rec.sequence == "ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"
        assert rec.length == 40  # from the LOCUS bp count
        assert len(rec.sequence) == 40

    def test_version_preferred_over_accession(self):
        # accession.version is the vault key; bare ACCESSION would collide.
        assert self._records()[0].accession.endswith(".1")

    def test_empty_stream_yields_nothing(self):
        assert list(parse_gbff_records([])) == []

    def test_trailing_record_without_terminator(self):
        text = "\n".join(_GBFF.splitlines()[:17])  # drop the closing //
        recs = list(parse_gbff_records(text.splitlines(keepends=True)))
        assert len(recs) == 1 and recs[0].host_taxid == "562"


class TestRecordToReport:
    def test_report_shape_matches_registry_contract(self):
        rec = PlasmidRecord(
            accession="NZ_CP012345.1", host_taxid="562",
            organism="Escherichia coli", length=40, sequence="A" * 40)
        report = record_to_report(rec)
        assert report["accession"] == "NZ_CP012345.1"
        assert report["organism"]["tax_id"] == 562        # int, groups by host
        assert report["organism"]["organism_name"] == "Escherichia coli"
        assert report["assembly_stats"]["total_sequence_length"] == 40
        assert report["assembly_info"]["assembly_level"] == "Complete Genome"


class TestIngestToVault:
    def test_sequences_round_trip_through_the_vault(self, tmp_path):
        lmdb_path = str(tmp_path / "sequences.lmdb")
        records = [
            PlasmidRecord("NZ_A.1", "562", "E. coli", 8, "ACGTACGT"),
            PlasmidRecord("NZ_B.1", "1280", "S. aureus", 4, "TTGG"),
        ]
        reports = ingest_records_to_vault(records, lmdb_path)

        assert [r["accession"] for r in reports] == ["NZ_A.1", "NZ_B.1"]
        # Read back exactly like a downloaded genome (uppercased on read).
        assert _read_single_sequence(lmdb_path, "NZ_A.1") == "ACGTACGT"
        assert _read_single_sequence(lmdb_path, "NZ_B.1") == "TTGG"

    def test_empty_stream_creates_vault_without_error(self, tmp_path):
        lmdb_path = str(tmp_path / "sequences.lmdb")
        assert ingest_records_to_vault([], lmdb_path) == []

    def test_ingest_from_a_parsed_gbff_stream(self, tmp_path):
        lmdb_path = str(tmp_path / "sequences.lmdb")
        reports = ingest_records_to_vault(
            parse_gbff_records(_GBFF.splitlines(keepends=True)), lmdb_path)
        assert len(reports) == 1
        seq = _read_single_sequence(lmdb_path, "NZ_CP012345.1")
        assert seq.startswith("ATGCATGC") and len(seq) == 40
