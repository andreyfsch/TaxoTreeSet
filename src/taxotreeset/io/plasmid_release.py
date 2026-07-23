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

This module is the tool-free acquisition primitive (P9). It covers the whole
release path: :func:`fetch_release` downloads the release's GBFF files (md5-
verified, resumable), :func:`iter_release_records` / :func:`parse_gbff_records`
stream them into :class:`PlasmidRecord`\\ s, :func:`ingest_records_to_vault`
writes the sequences into the LMDB vault, and :func:`record_to_report` adapts
each record into the synthetic *assembly report* shape the discovery
registration path already consumes — so the plasmid branch reuses the existing
lineage-resolution + tree-building cascade unchanged (``core/orchestrator.py``).
See ``docs/BACKLOG.md`` P9.
"""

import glob
import gzip
import hashlib
import logging
import os
import re
import urllib.error
import urllib.request
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

# RefSeq plasmid release location + layout. The release ships a *.files.installed
# manifest (``<md5>  <path>`` pairs) alongside the data files; the directory HTML
# listing is the fallback when the manifest cannot be read.
_RELEASE_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/release/plasmid/"
_MANIFEST_NAME = "plasmid.files.installed"
_GBFF_SUFFIX = ".genomic.gbff.gz"
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_HREF_RE = re.compile(r'href="(plasmid[^"?/]*\.genomic\.gbff\.gz)"')
_DOWNLOAD_CHUNK = 1 << 20  # 1 MiB streaming chunk

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


def fetch_release(
    dest_dir: str,
    base_url: str = _RELEASE_BASE_URL,
    retries: int = 3,
    timeout: int = 60,
) -> str:
    """Download the RefSeq plasmid release GBFF files into ``dest_dir``.

    Idempotent and resumable: each file is checksummed against the release
    manifest and skipped when already present and valid, so an interrupted sync
    resumes on the next run without re-downloading complete files. Downloads
    stream to a ``.part`` sibling and are renamed only after a full read (and md5
    match when known), so an interrupted transfer never leaves a truncated file
    the parser would trust.

    Args:
        dest_dir: Directory to sync the release into (created if absent).
        base_url: Release directory URL (default the NCBI RefSeq plasmid release).
        retries: Per-file download attempts before giving up.
        timeout: Per-request socket timeout in seconds.

    Returns:
        ``dest_dir`` (for chaining into :func:`iter_release_records`).

    Raises:
        RuntimeError: If a file fails to download after ``retries`` attempts.
    """
    if not base_url.endswith("/"):
        base_url += "/"
    os.makedirs(dest_dir, exist_ok=True)

    entries = _list_release_gbff(base_url, timeout)
    if not entries:
        logger.warning(
            "No %s files found at %s — nothing to fetch.", _GBFF_SUFFIX, base_url)
        return dest_dir

    logger.info(
        "RefSeq plasmid release: syncing %d GBFF file(s) into %s",
        len(entries), dest_dir,
    )
    for filename, md5 in entries:
        target = os.path.join(dest_dir, filename)
        if _is_up_to_date(target, md5):
            logger.info("Up to date, skipping %s", filename)
            continue
        _download_file(base_url + filename, target, md5, retries, timeout)
    return dest_dir


def _list_release_gbff(base_url: str, timeout: int) -> list[tuple[str, str | None]]:
    """Enumerate the release's GBFF files as ``(filename, md5-or-None)`` pairs.

    Prefers the ``*.files.installed`` manifest (gives checksums for verified,
    resumable downloads); falls back to scraping the directory HTML listing when
    the manifest is unreadable or lists no GBFF files.
    """
    manifest = _fetch_text(base_url + _MANIFEST_NAME, timeout)
    if manifest is not None:
        entries = _parse_manifest(manifest)
        if entries:
            return entries
    listing = _fetch_text(base_url, timeout)
    if listing is not None:
        return [(name, None) for name in _parse_html_listing(listing)]
    return []


def _parse_manifest(text: str) -> list[tuple[str, str | None]]:
    """Parse a RefSeq ``*.files.installed`` manifest into GBFF (name, md5) pairs.

    The two columns are a 32-hex-char md5 and a path; their order varies across
    releases, so the md5 is identified by shape rather than position. Only
    ``*.genomic.gbff.gz`` entries are kept.
    """
    entries: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        tokens = line.split()
        md5 = next((t for t in tokens if _MD5_RE.match(t)), None)
        name = next(
            (os.path.basename(t) for t in tokens if not _MD5_RE.match(t)), None)
        if name and name.endswith(_GBFF_SUFFIX):
            entries.append((name, md5))
    return entries


def _parse_html_listing(html: str) -> list[str]:
    """GBFF filenames linked in an FTP directory's autoindex HTML listing."""
    return sorted(set(_HREF_RE.findall(html)))


def _fetch_text(url: str, timeout: int) -> str | None:
    """Fetch a URL as text, or None when it cannot be retrieved."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        logger.debug("Could not fetch %s: %s", url, exc)
        return None


def _is_up_to_date(target: str, md5: str | None) -> bool:
    """True when ``target`` exists and matches ``md5`` (or has no checksum)."""
    if not os.path.exists(target):
        return False
    if md5 is None:
        return True  # present but no checksum to verify against — trust it
    return _file_md5(target) == md5


def _file_md5(path: str) -> str:
    # md5 here is an integrity check against the release manifest, not security.
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str, target: str, md5: str | None, retries: int, timeout: int,
) -> None:
    """Stream ``url`` to ``target`` with retries and md5 verification."""
    part = target + ".part"
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _stream_to_file(url, part, timeout)
            if md5 is not None and _file_md5(part) != md5:
                raise OSError(f"md5 mismatch for {os.path.basename(target)}")
            os.replace(part, target)
            logger.info("Downloaded %s", os.path.basename(target))
            return
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            logger.warning(
                "Download attempt %d/%d for %s failed: %s",
                attempt, retries, os.path.basename(target), exc,
            )
    if os.path.exists(part):
        os.remove(part)
    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}")


def _stream_to_file(url: str, part: str, timeout: int) -> None:
    """Stream a URL to a file in chunks (never holds the whole file in memory)."""
    with urllib.request.urlopen(url, timeout=timeout) as response, open(
        part, "wb"
    ) as out:
        for chunk in iter(lambda: response.read(_DOWNLOAD_CHUNK), b""):
            out.write(chunk)


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
