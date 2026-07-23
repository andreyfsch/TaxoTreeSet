"""Parse the RefSeq *plasmid* release into host-labeled plasmid records.

"Plasmid" is not a taxon — every RefSeq plasmid record is taxonomically assigned
to its **host** organism — so TaxoTreeSet cannot acquire plasmids by walking a
root TaxID the way it walks Viruses or Bacteria. Instead it reads the curated,
standalone RefSeq plasmid release
(``https://ftp.ncbi.nlm.nih.gov/refseq/release/plasmid/``), which ships the whole
plasmid collection as GenBank flat files (``plasmid.N.genomic.gbff.gz``) — no
full-kingdom Bacteria crawl. Each GBFF record self-contains everything the
pipeline needs: the accession (``VERSION``), the host TaxID (the ``source``
feature's ``/db_xref="taxon:NNN"``), the organism name, and the sequence
(``ORIGIN``).

This module is the tool-free acquisition primitive (P9): it parses a GBFF stream
into :class:`PlasmidRecord`\\ s and adapts each into the synthetic *assembly
report* shape the discovery registration path already consumes, so the plasmid
branch reuses the existing lineage-resolution + tree-building cascade unchanged
(``core/orchestrator.py``). Fetching the release files and ingesting the
sequences into the vault live alongside it; see ``docs/BACKLOG.md`` P9.
"""

import glob
import gzip
import logging
import os
import re
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import lmdb

from taxotreeset.io.downloader import LMDB_MAP_SIZE_BYTES

logger = logging.getLogger("TaxoTreeSet.IO.PlasmidRelease")

# Commit the vault write transaction every N records so a large release is not
# buffered as one giant uncommitted transaction (mirrors the downloader, which
# commits per accession batch rather than per run).
_INGEST_COMMIT_EVERY = 1000

# The host TaxID lives in the source feature's /db_xref="taxon:NNN"; it is the
# first taxon db_xref in a record (source is the first feature), so a plain
# first-match search across the record is reliable for RefSeq flat files.
_TAXON_RE = re.compile(r'/db_xref="taxon:(\d+)"')
_ORGANISM_QUAL_RE = re.compile(r'/organism="([^"]+)"')
_VERSION_RE = re.compile(r"^VERSION\s+(\S+)")
_ACCESSION_RE = re.compile(r"^ACCESSION\s+(\S+)")
_LOCUS_LEN_RE = re.compile(r"^LOCUS\s+\S+\s+(\d+)\s+bp", re.IGNORECASE)
_ORGANISM_LINE_RE = re.compile(r"^\s*ORGANISM\s+(.+)")

# A complete plasmid sequence is reference-quality; this matches
# NCBIRegistry._REFERENCE_ASSEMBLY_LEVELS so the record survives scoping.
_PLASMID_ASSEMBLY_LEVEL = "Complete Genome"


@dataclass(frozen=True)
class PlasmidRecord:
    """One plasmid sequence from the RefSeq plasmid release.

    Attributes:
        accession: The record's ``accession.version`` (the LMDB vault key and
            the registry accession id).
        host_taxid: NCBI TaxID of the host organism the plasmid is assigned to.
        organism: Host organism scientific name (may be empty if absent).
        length: Sequence length in bp.
        sequence: The uppercased nucleotide sequence.
    """

    accession: str
    host_taxid: str
    organism: str
    length: int
    sequence: str


def iter_release_records(release_dir: str) -> Iterator[PlasmidRecord]:
    """Stream records from every GBFF file in a RefSeq plasmid release directory.

    Reads the release's GenBank flat files (``*.gbff`` / ``*.gbff.gz``, e.g.
    ``plasmid.1.genomic.gbff.gz``) in sorted order, decompressing gzip
    transparently. Files are opened and closed one at a time so the whole release
    is never held open or in memory. The caller is expected to have fetched the
    release directory already (a bulk FTP mirror of
    ``refseq/release/plasmid/``).

    Args:
        release_dir: Directory containing the release's GBFF files.

    Yields:
        Every :class:`PlasmidRecord` across all files, in file-then-record order.
    """
    paths = sorted({
        p
        for pattern in ("*.gbff", "*.gbff.gz")
        for p in glob.glob(os.path.join(release_dir, pattern))
    })
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from parse_gbff_records(handle)


def parse_gbff_records(lines: Iterable[str]) -> Iterator[PlasmidRecord]:
    """Stream GenBank flat-file records into :class:`PlasmidRecord`\\ s.

    Records are delimited by a ``//`` line. A record that lacks an accession or
    a host TaxID (nothing to key the vault on, or no host to place it under) is
    skipped rather than raised, mirroring the discovery path's tolerance of
    unresolvable taxa. Accepts any iterable of lines (an open text handle, a
    decompressed byte stream decoded to text, or a list — the caller owns I/O).

    Args:
        lines: Iterable of GBFF text lines (newline-terminated or not).

    Yields:
        One :class:`PlasmidRecord` per parseable record, in file order.
    """
    buffer: list[str] = []
    for line in lines:
        if line.startswith("//"):
            record = _parse_record(buffer)
            if record is not None:
                yield record
            buffer = []
            continue
        buffer.append(line.rstrip("\n"))
    # A trailing record with no closing `//` (truncated file) is still parsed.
    if buffer:
        record = _parse_record(buffer)
        if record is not None:
            yield record


def _parse_record(buffer: list[str]) -> PlasmidRecord | None:
    """Extract one record's fields from its accumulated lines.

    Returns ``None`` when the record has no accession or no host TaxID.
    """
    accession = _find_accession(buffer)
    host_taxid = _find_first(buffer, _TAXON_RE.search)
    if not accession or not host_taxid:
        return None

    sequence = _extract_sequence(buffer)
    locus_length = _find_first(buffer, _LOCUS_LEN_RE.match)
    return PlasmidRecord(
        accession=accession,
        host_taxid=host_taxid,
        organism=_find_organism(buffer),
        length=int(locus_length) if locus_length else len(sequence),
        sequence=sequence,
    )


def _find_first(buffer: list[str], matcher) -> str:
    """First capture group produced by ``matcher`` over the lines, else ""."""
    for line in buffer:
        m = matcher(line)
        if m:
            return m.group(1)
    return ""


def _find_accession(buffer: list[str]) -> str:
    """The record's accession.version — VERSION preferred over ACCESSION.

    ``ACCESSION`` (bare id) precedes ``VERSION`` (id.version) in the flat file,
    so a bare first-match would drop the version — but the version is the vault
    key, and two records of the same accession differ only by it.
    """
    return _find_first(buffer, _VERSION_RE.match) or _find_first(
        buffer, _ACCESSION_RE.match)


def _find_organism(buffer: list[str]) -> str:
    """The host organism name, from the /organism qualifier or ORGANISM line."""
    return _find_first(buffer, _ORGANISM_QUAL_RE.search) or _find_first(
        buffer, _ORGANISM_LINE_RE.match).strip()


def _extract_sequence(buffer: list[str]) -> str:
    """Concatenate the ORIGIN block into an uppercased nucleotide string.

    GenBank ORIGIN lines interleave a running base offset with 10-base blocks
    (``  1 atgcat gcatgc ...``); keeping only alphabetic characters drops the
    offsets and spacing without a positional parse.
    """
    seq_parts: list[str] = []
    in_origin = False
    for line in buffer:
        if line.startswith("ORIGIN"):
            in_origin = True
            continue
        if in_origin:
            seq_parts.append("".join(ch for ch in line if ch.isalpha()))
    return "".join(seq_parts).upper()


def record_to_report(record: PlasmidRecord) -> dict[str, Any]:
    """Adapt a plasmid record to the synthetic assembly-report shape.

    The discovery registration path (``_register_taxon`` →
    ``NCBIRegistry._update_taxon_entry`` / ``_build_accession_entry``) consumes
    NCBI assembly reports; emitting the same shape lets the plasmid branch reuse
    that path unchanged. The host TaxID is carried in ``organism.tax_id`` so the
    caller groups records by host exactly as ``_stream_ncbi_summaries`` groups
    assemblies by organism.

    Returns:
        A report dict with ``accession``, ``assembly_info.assembly_level``,
        ``organism.{organism_name, tax_id}``, and
        ``assembly_stats.total_sequence_length``.
    """
    return {
        "accession": record.accession,
        "assembly_info": {"assembly_level": _PLASMID_ASSEMBLY_LEVEL},
        "organism": {
            "organism_name": record.organism,
            "tax_id": int(record.host_taxid),
        },
        "assembly_stats": {"total_sequence_length": record.length},
    }


def ingest_records_to_vault(
    records: Iterable[PlasmidRecord],
    lmdb_path: str,
) -> list[dict[str, Any]]:
    """Write each record's sequence to the LMDB vault and return its report.

    Uses the same vault contract as the downloader — key ``accession.version``,
    value ``zlib.compress(sequence.encode("utf-8"))`` — so plasmid sequences read
    back through ``dataset/utils._read_single_sequence`` exactly like downloaded
    genomes (plasmid records are standalone nucleotide accessions and cannot go
    through the assembly-oriented ``datasets download genome accession`` path, so
    they are ingested here directly).

    The records are streamed: each sequence is written and dropped, so the whole
    release is never held in memory. Only the small per-record reports (no
    sequence) are accumulated and returned — they are the input to the discovery
    registration step. The write transaction is committed every
    ``_INGEST_COMMIT_EVERY`` records.

    Args:
        records: Iterable of :class:`PlasmidRecord` (typically a
            ``parse_gbff_records`` stream).
        lmdb_path: Filesystem path of the vault LMDB directory (created if
            absent, like the downloader's ``sequences.lmdb``).

    Returns:
        One synthetic assembly report per ingested record, in stream order.
    """
    parent = os.path.dirname(lmdb_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    reports: list[dict[str, Any]] = []
    env = lmdb.open(lmdb_path, map_size=LMDB_MAP_SIZE_BYTES, max_dbs=0)
    try:
        txn = env.begin(write=True)
        try:
            for record in records:
                txn.put(
                    record.accession.encode("utf-8"),
                    zlib.compress(record.sequence.encode("utf-8")),
                )
                reports.append(record_to_report(record))
                if len(reports) % _INGEST_COMMIT_EVERY == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        except BaseException:
            txn.abort()
            raise
    finally:
        env.close()

    logger.info("Ingested %d plasmid sequence(s) into %s", len(reports), lmdb_path)
    return reports
