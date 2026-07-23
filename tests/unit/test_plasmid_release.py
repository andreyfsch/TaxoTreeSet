"""Tests for the RefSeq plasmid release GBFF parser (P9)."""

import gzip
import hashlib
import io
import os
from unittest.mock import patch

import pytest

from taxotreeset.dataset.utils import _read_single_sequence
from taxotreeset.io import plasmid_release
from taxotreeset.io.plasmid_release import (
    PlasmidRecord,
    _parse_html_listing,
    _parse_manifest,
    fetch_release,
    ingest_records_to_vault,
    iter_release_records,
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


class TestIterReleaseDir:
    def test_reads_plain_and_gzip_files(self, tmp_path):
        (tmp_path / "plasmid.1.genomic.gbff").write_text(_GBFF, encoding="utf-8")
        with gzip.open(tmp_path / "plasmid.2.genomic.gbff.gz", "wt") as f:
            f.write(_GBFF)
        # One placeable record per file (the host-less second is dropped).
        recs = list(iter_release_records(str(tmp_path)))
        assert len(recs) == 2
        assert {r.accession for r in recs} == {"NZ_CP012345.1"}

    def test_empty_dir_yields_nothing(self, tmp_path):
        assert list(iter_release_records(str(tmp_path))) == []


class TestManifestParsing:
    def test_parses_md5_first_column_order(self):
        text = (
            "d41d8cd98f00b204e9800998ecf8427e  plasmid.1.genomic.gbff.gz\n"
            "d41d8cd98f00b204e9800998ecf8427e  plasmid.1.protein.faa.gz\n"  # dropped
        )
        assert _parse_manifest(text) == [
            ("plasmid.1.genomic.gbff.gz", "d41d8cd98f00b204e9800998ecf8427e")]

    def test_parses_filename_first_column_order(self):
        text = "plasmid.2.genomic.gbff.gz\td41d8cd98f00b204e9800998ecf8427e\n"
        assert _parse_manifest(text) == [
            ("plasmid.2.genomic.gbff.gz", "d41d8cd98f00b204e9800998ecf8427e")]

    def test_manifest_without_checksum_yields_none_md5(self):
        assert _parse_manifest("plasmid.3.genomic.gbff.gz\n") == [
            ("plasmid.3.genomic.gbff.gz", None)]

    def test_html_listing_extracts_gbff_links(self):
        html = (
            '<a href="plasmid.1.genomic.gbff.gz">x</a>'
            '<a href="plasmid.1.protein.faa.gz">y</a>'
            '<a href="plasmid.2.genomic.gbff.gz">z</a>'
        )
        assert _parse_html_listing(html) == [
            "plasmid.1.genomic.gbff.gz", "plasmid.2.genomic.gbff.gz"]


class TestFetchRelease:
    def _fake_urlopen(self, manifest, gbff_name, gbff_bytes):
        def opener(url, timeout=None):
            if url.endswith("plasmid.files.installed"):
                return io.BytesIO(manifest.encode("utf-8"))
            if url.endswith(gbff_name):
                return io.BytesIO(gbff_bytes)
            raise plasmid_release.urllib.error.URLError("404")
        return opener

    def test_downloads_verifies_and_is_resumable(self, tmp_path):
        gbff_name = "plasmid.1.genomic.gbff.gz"
        gbff_bytes = gzip.compress(_GBFF.encode("utf-8"))
        md5 = hashlib.md5(gbff_bytes).hexdigest()
        manifest = f"{md5}  {gbff_name}\n"
        dest = str(tmp_path / "release")

        opener = self._fake_urlopen(manifest, gbff_name, gbff_bytes)
        with patch.object(plasmid_release.urllib.request, "urlopen", opener):
            fetch_release(dest, base_url="http://x/")
            # md5 verified + renamed into place; parses end-to-end.
            recs = list(iter_release_records(dest))
            assert len(recs) == 1 and recs[0].host_taxid == "562"

            # Rerun is a no-op: the file is up to date (md5 matches), no .part left.
            fetch_release(dest, base_url="http://x/")
        assert not os.path.exists(os.path.join(dest, gbff_name + ".part"))

    def test_md5_mismatch_is_retried_then_raises(self, tmp_path):
        gbff_name = "plasmid.1.genomic.gbff.gz"
        gbff_bytes = b"corrupt"
        manifest = f"{'0' * 32}  {gbff_name}\n"  # wrong md5 -> never verifies
        dest = str(tmp_path / "release")

        opener = self._fake_urlopen(manifest, gbff_name, gbff_bytes)
        with patch.object(plasmid_release.urllib.request, "urlopen", opener):
            with pytest.raises(RuntimeError):
                fetch_release(dest, base_url="http://x/", retries=2)
