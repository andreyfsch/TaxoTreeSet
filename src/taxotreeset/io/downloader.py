"""Genome assembly downloader using the NCBI Datasets CLI.

This module provides the NCBIDownloader class, which manages batched
downloads of genome assemblies via the official NCBI Datasets CLI and
persists sequences into an LMDB vault for efficient random access during
dataset generation.

The downloader uses three design choices to keep transfers efficient:

1. **Multi-accession batch fetching**: instead of one CLI call per
   accession (which would incur an HTTP handshake each time), the
   downloader passes all accessions in a chunk to a single CLI command.
   This trades some memory for substantial wall-clock savings.

2. **LMDB storage with zlib compression**: each sequence is stored as a
   compressed value keyed by its header ID. LMDB provides O(log n)
   random access and memory-mapped reads from worker processes during
   the parallel dataset generation phase.

3. **Idempotency via the registry**: the downloader consults the
   registry's downloaded flag before requesting an accession, so
   interrupted runs resume from where they left off. A self-healing
   check at startup detects inconsistencies between the registry state
   and the physical LMDB vault.

Typical usage::

    from taxotreeset.io.registry import NCBIRegistry
    from taxotreeset.io.downloader import NCBIDownloader

    registry = NCBIRegistry(registry_path="data/registry.json")
    downloader = NCBIDownloader(registry=registry, vault_path="data/vault")
    downloader.download_all_pending()
"""

import logging
import os
import subprocess
import tempfile
import zipfile
import zlib
from typing import Any

import lmdb
from tqdm import tqdm

logger = logging.getLogger("TaxoTreeSet.IO.Downloader")

LMDB_MAP_SIZE_BYTES = 1_099_511_627_776  # 1 TiB virtual address space
LMDB_DATA_FILE_NAME = "data.mdb"
LMDB_MIN_VALID_SIZE_BYTES = 1024
FASTA_EXTENSIONS = (".fna", ".fasta", ".fa")


class NCBIDownloader:
    """Batch downloader for genome assemblies via the NCBI Datasets CLI.

    Maintains an internal LMDB environment that is opened lazily, only
    when there is actual work to do. Sequences are stored compressed
    with zlib under their header IDs as keys; the registry's accession
    entries track download status and link back to the LMDB location.

    The downloader is resilient to migrations between machines and to
    interrupted runs: if the registry claims accessions are downloaded
    but the physical LMDB vault does not exist (or is empty), the
    download state is automatically reset so the next run re-fetches
    everything.

    Attributes:
        registry: The NCBIRegistry instance holding accession metadata.
        vault_path: Filesystem directory hosting the LMDB vault.
        chunk_size: Number of accessions per CLI invocation.
        lmdb_path: Full path to the LMDB environment within vault_path.
    """

    _DEFAULT_CHUNK_SIZE = 100
    # Cap the total sequence bytes per CLI invocation so a chunk of many large
    # assemblies does not become a single 60+ GiB request the NCBI server
    # rejects. A single accession larger than the cap cannot be split further and
    # is requested alone in its own chunk (warned by _split_into_chunks).
    _DEFAULT_MAX_BYTES_PER_CHUNK = 5 * 1024 ** 3  # 5 GiB
    # Defline keywords marking sequences to drop when exclude_plasmids is set.
    # Substring, case-insensitive (so "megaplasmid" is also caught). Extend this
    # set to also drop organelles (e.g. mitochondrion, chloroplast) in future.
    _EXCLUDED_MOLECULE_KEYWORDS: tuple[str, ...] = ("plasmid",)
    # An accession the CLI returns but whose FASTA cannot be ingested (missing
    # directory, unparsable file, withdrawn record) is retried up to this many
    # times, then given up on — so one permanently-bad accession does not force a
    # re-download of its chunk on every run / refinement round. A whole-CLI
    # failure (transient network) does NOT count against this limit.
    _MAX_DOWNLOAD_ATTEMPTS: int = 3

    def __init__(
        self,
        registry: Any,
        vault_path: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        max_bytes_per_chunk: int = _DEFAULT_MAX_BYTES_PER_CHUNK,
        tmp_dir: str | None = None,
        exclude_plasmids: bool = False,
    ) -> None:
        """Initialize the downloader without opening the LMDB environment.

        The LMDB environment is opened lazily by ``download_all_pending``
        only when there is work to do. This avoids holding write locks
        unnecessarily and allows multiple downloader instances to
        coexist as long as they do not concurrently write.

        Args:
            registry: NCBIRegistry instance to consult and update.
            vault_path: Directory where the LMDB vault will be stored.
                Created automatically if it does not exist.
            chunk_size: Number of accessions per bulk CLI invocation.
                Larger values reduce overhead but increase memory and
                temporary disk usage during each batch.
            tmp_dir: Directory for temporary download archives and
                extracted files. Defaults to the OS temp dir when None.
                Set to a path on a large drive to avoid inflating the
                WSL VHDX on the system drive.
            exclude_plasmids: When True, sequences whose FASTA defline marks
                them as plasmids are dropped at ingestion (never stored in the
                vault nor recorded as headers). Off by default; intended for
                Bacteria, where plasmids (horizontally transferred) add little
                reliable phylogenetic signal.
        """
        self.registry: Any = registry
        self.vault_path: str = vault_path
        self.chunk_size: int = chunk_size
        self.max_bytes_per_chunk: int = max_bytes_per_chunk
        self.tmp_dir: str | None = tmp_dir
        self.exclude_plasmids: bool = exclude_plasmids

        os.makedirs(self.vault_path, exist_ok=True)
        self.lmdb_path: str = os.path.join(self.vault_path, "sequences.lmdb")
        self._env: lmdb.Environment | None = None

    def download_all_pending(self) -> None:
        """Download all accessions marked as pending in the registry.

        Workflow:
            1. Detect and self-heal any inconsistency between the
               registry's download state and the physical LMDB vault.
            2. Identify accessions still pending download.
            3. Open the LMDB environment in write mode.
            4. Iterate over chunks of pending accessions; for each
               chunk, invoke the NCBI Datasets CLI, parse the resulting
               FASTA files, and persist sequences to LMDB.
            5. Update the registry after each chunk to enable resume.

        On any unhandled exception during the chunk loop, the LMDB
        environment is still closed gracefully via try/finally.
        """
        self._reset_state_if_lmdb_missing()

        pending = self._collect_pending_accessions()
        accessions = self.registry.registry.get("accessions", {})
        total_pending = len(pending)
        # Count actually-downloaded accessions rather than (total - pending),
        # which would inflate the bar by the deferred (not-processed) accessions.
        already_downloaded = sum(
            1 for info in accessions.values() if info.get("downloaded")
        )

        if total_pending == 0:
            logger.info("All registered accessions are already archived in LMDB.")
            return

        chunks = self._split_into_chunks(pending)
        logger.info(
            f"Grouped {total_pending} pending accessions into "
            f"{len(chunks)} batch downloads "
            f"(chunk_size: {self.chunk_size}, "
            f"max_bytes_per_chunk: {self.max_bytes_per_chunk / 1024**3:.1f} GiB)."
        )

        self._env = lmdb.open(
            self.lmdb_path,
            map_size=LMDB_MAP_SIZE_BYTES,
            max_dbs=0,
        )

        try:
            self._process_chunks(
                chunks=chunks,
                bar_total=already_downloaded + total_pending,
                bar_initial=already_downloaded,
            )
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None

        logger.info("Bulk download pipeline finished successfully.")

    def _reset_state_if_lmdb_missing(self) -> None:
        """Self-healing check for inconsistent registry-vs-vault state.

        When a project is migrated between machines, the registry JSON
        may be transferred while the LMDB vault is not (it is typically
        large and excluded from version control). The result is that
        the registry claims accessions are downloaded, but the physical
        sequences are missing.

        This method detects that situation and resets the downloaded
        flag on all accessions, forcing a re-fetch. The check runs in
        constant time and only triggers when the registry has at least
        one accession marked as downloaded.
        """
        accessions = self.registry.registry.get("accessions", {})
        any_marked_downloaded = any(
            info.get("downloaded") for info in accessions.values()
        )
        if not any_marked_downloaded:
            return

        lmdb_data_file = os.path.join(self.lmdb_path, LMDB_DATA_FILE_NAME)
        is_missing = not os.path.exists(lmdb_data_file)
        is_too_small = (
            os.path.exists(lmdb_data_file)
            and os.path.getsize(lmdb_data_file) < LMDB_MIN_VALID_SIZE_BYTES
        )

        if not (is_missing or is_too_small):
            return

        reason = "is missing" if is_missing else "is too small to be valid"
        logger.warning(
            f"Inconsistency detected: registry claims accessions are "
            f"downloaded, but LMDB vault at {self.lmdb_path} {reason}. "
            "Resetting download state to force re-fetch."
        )

        for info in accessions.values():
            info["downloaded"] = False
            info.pop("local_path", None)
            info.pop("headers", None)
            info.pop("download_attempts", None)  # fresh vault -> fresh retries
        self.registry.save()

    def reconcile_with_vault(self) -> int:
        """Reset accessions whose recorded headers are missing from LMDB.

        Detects vault degradation (Case 1): for each accession marked
        downloaded that has recorded headers, verifies every header ID is
        present as an LMDB key. Accessions with any missing header are
        reset to pending (downloaded=False, headers and local_path
        cleared) so the next download re-fetches them.

        Returns:
            Number of accessions reset to pending.
        """
        lmdb_data_file = os.path.join(self.lmdb_path, LMDB_DATA_FILE_NAME)
        if not os.path.exists(lmdb_data_file):
            return 0

        accessions = self.registry.registry.get("accessions", {})
        reset_count = 0
        env = lmdb.open(
            self.lmdb_path,
            map_size=LMDB_MAP_SIZE_BYTES,
            max_dbs=0,
            readonly=True,
            lock=False,
        )
        try:
            with env.begin() as txn:
                for accession, info in accessions.items():
                    if not info.get("downloaded"):
                        continue
                    headers = info.get("headers")
                    if not headers:
                        continue
                    missing = any(
                        txn.get(header["id"].encode("utf-8")) is None
                        for header in headers
                    )
                    if missing:
                        info["downloaded"] = False
                        info["local_path"] = None
                        info.pop("headers", None)
                        info.pop("download_attempts", None)
                        reset_count += 1
        finally:
            env.close()

        if reset_count:
            logger.warning(
                "Vault reconciliation reset %d accession(s) to pending "
                "(headers missing from LMDB).", reset_count
            )
            self.registry.save()
        return reset_count

    def _collect_pending_accessions(self) -> list[str]:
        """Return the list of accession IDs that still need downloading.

        Excludes accessions flagged as ``download_deferred=True`` by the
        selective download selection pass (reserved for a later refinement
        batch) and accessions that have exhausted ``_MAX_DOWNLOAD_ATTEMPTS``
        ingest failures (permanently bad — retrying them only re-downloads their
        chunk each run).

        Returns:
            List of accession strings eligible for download. Order
            matches the registry's insertion order.
        """
        accessions = self.registry.registry.get("accessions", {})
        return [
            accession
            for accession, info in accessions.items()
            if not info.get("downloaded")
            and not info.get("download_deferred")
            and info.get("download_attempts", 0) < self._MAX_DOWNLOAD_ATTEMPTS
        ]

    def _split_into_chunks(self, accessions: list[str]) -> list[list[str]]:
        """Split accessions into volume-capped chunks for CLI batching.

        Each chunk is bounded by both ``chunk_size`` (max accessions) and
        ``max_bytes_per_chunk`` (max total sequence bytes). For accessions
        whose ``total_sequence_length`` is unknown the volume guard is
        skipped and only ``chunk_size`` applies.

        Args:
            accessions: Flat list of accession IDs to split.

        Returns:
            List of lists, each within the configured size/volume bounds.
        """
        registry_accessions = self.registry.registry.get("accessions", {})
        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_bytes = 0

        for accession in accessions:
            vol = (registry_accessions.get(accession) or {}).get(
                "total_sequence_length"
            ) or 0
            if vol > self.max_bytes_per_chunk:
                logger.warning(
                    "Accession %s (%.1f GiB) exceeds max_bytes_per_chunk "
                    "(%.1f GiB) and cannot be split; requested alone.",
                    accession, vol / 1024**3,
                    self.max_bytes_per_chunk / 1024**3,
                )
            at_count_limit = len(current_chunk) >= self.chunk_size
            at_volume_limit = (
                vol > 0
                and current_bytes > 0
                and current_bytes + vol > self.max_bytes_per_chunk
            )
            if current_chunk and (at_count_limit or at_volume_limit):
                chunks.append(current_chunk)
                current_chunk = []
                current_bytes = 0
            current_chunk.append(accession)
            current_bytes += vol

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _process_chunks(
        self,
        chunks: list[list[str]],
        bar_total: int,
        bar_initial: int,
    ) -> None:
        """Execute the per-chunk download loop with progress reporting.

        Args:
            chunks: List of accession chunks to process sequentially.
            bar_total: Progress-bar total (already-downloaded + pending; deferred
                accessions are excluded since they are not processed this run).
            bar_initial: Accessions already downloaded, the bar's start position.
        """
        with tqdm(
            total=bar_total,
            initial=bar_initial,
            desc="Ingesting genomes to LMDB",
            unit=" genome",
        ) as progress_bar:
            for chunk in chunks:
                batch_results, attempted = self.download_batch(chunk)
                self._update_registry_for_batch(chunk, batch_results, attempted)
                progress_bar.update(len(chunk))
                self.registry.save()

    def _update_registry_for_batch(
        self,
        chunk: list[str],
        batch_results: dict[str, list[dict[str, str]]],
        attempted: list[str],
    ) -> None:
        """Mark successes and count per-accession ingest failures after a batch.

        A successful accession is marked downloaded and its retry counter
        cleared. An accession the CLI returned but that failed to ingest (in
        ``attempted`` but not ``batch_results``) has its ``download_attempts``
        counter bumped; once it reaches ``_MAX_DOWNLOAD_ATTEMPTS`` it is given up
        on (thereafter skipped by ``_collect_pending_accessions``). Accessions the
        CLI never returned (a whole-batch/transient failure, absent from
        ``attempted``) are left untouched so a network blip does not count.

        Args:
            chunk: Accession IDs requested in the batch.
            batch_results: Successfully ingested accessions -> header metadata.
            attempted: Accessions the CLI returned and tried to ingest.
        """
        accessions = self.registry.registry["accessions"]
        attempted_set = set(attempted)
        for accession in chunk:
            info = accessions[accession]
            if accession in batch_results:
                info["downloaded"] = True
                info["local_path"] = self.lmdb_path
                info["headers"] = batch_results[accession]
                info.pop("download_attempts", None)
            elif accession in attempted_set:
                info["download_attempts"] = info.get("download_attempts", 0) + 1
                if info["download_attempts"] >= self._MAX_DOWNLOAD_ATTEMPTS:
                    logger.warning(
                        "Accession %s failed to ingest %d times; giving up "
                        "(skipped on future runs).",
                        accession, info["download_attempts"],
                    )

    def download_batch(
        self,
        accessions: list[str],
    ) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
        """Download a single batch of accessions via the NCBI CLI.

        Spawns one ``datasets download`` subprocess that fetches all
        requested accessions in a ZIP archive, unpacks it, parses the
        FASTA files, and persists each sequence into LMDB.

        Each accession is ingested independently: a failure in one (missing
        directory, unparsable FASTA, LMDB error) is logged and skipped rather
        than allowed to abort the whole batch — otherwise a single bad accession
        would abort the run and re-block its chunk on every subsequent run.

        Args:
            accessions: List of accession IDs to download together in a
                single CLI invocation.

        Returns:
            ``(batch_results, attempted)``. ``batch_results`` maps each
            successfully ingested accession to its header metadata (each a list
            of ``{'id', 'name'}`` dicts, possibly empty when every sequence was
            filtered). ``attempted`` lists the accessions the CLI returned and
            tried to ingest; entries in ``attempted`` but not ``batch_results``
            are per-accession ingest failures (counted toward the retry limit).
            ``attempted`` is empty when the CLI call failed entirely (a transient
            failure that must not count against any accession).
        """
        logger.debug(f"Spawning bulk NCBI fetch for {len(accessions)} accessions.")
        batch_results: dict[str, list[dict[str, str]]] = {}

        with tempfile.TemporaryDirectory(dir=self.tmp_dir) as temp_dir:
            archive_path = os.path.join(temp_dir, "batch_package.zip")

            if not self._invoke_ncbi_datasets_cli(accessions, archive_path):
                return batch_results, []

            extracted_root = self._extract_assembly_archive(archive_path, temp_dir)
            if extracted_root is None:
                return batch_results, []

            for accession in accessions:
                try:
                    headers = self._ingest_accession_fasta(accession, extracted_root)
                except Exception as exc:
                    # Isolate one bad accession so it cannot abort the batch (and
                    # every later chunk). It stays pending; the retry counter in
                    # _update_registry_for_batch eventually gives up on it.
                    logger.error(
                        "Ingest failed for accession %s: %s — skipping.",
                        accession, exc,
                    )
                    headers = None
                # None = genuine failure (missing/empty/raised) -> omit so it
                # stays pending. A list (even empty, when every sequence was
                # filtered) means the accession was processed.
                if headers is not None:
                    batch_results[accession] = headers

        return batch_results, list(accessions)

    def _invoke_ncbi_datasets_cli(
        self,
        accessions: list[str],
        archive_path: str,
    ) -> bool:
        """Run the NCBI Datasets CLI to download genomes into a ZIP archive.

        Args:
            accessions: List of accession IDs to fetch.
            archive_path: Filesystem path where the resulting ZIP file
                will be written.

        Returns:
            True if the archive was produced and is non-empty. False on
            any CLI failure or empty output.
        """
        command = [
            "datasets",
            "download",
            "genome",
            "accession",
            *accessions,
            "--include",
            "genome",
            "--filename",
            archive_path,
        ]

        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=os.environ.copy(),
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            stdout = exc.stdout.strip() if exc.stdout else ""
            detail = "\n".join(filter(None, [stderr, stdout])) or str(exc)
            logger.error(f"NCBI Datasets CLI invocation failed: {detail}")
            return False

        if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
            logger.error("Bulk download failed: package archive is empty or missing.")
            return False

        return True

    def _extract_assembly_archive(
        self,
        archive_path: str,
        temp_dir: str,
    ) -> str | None:
        """Extract the NCBI Datasets ZIP archive into the temporary dir.

        Args:
            archive_path: Path to the ZIP file produced by the CLI.
            temp_dir: Directory under which the archive will be extracted.

        Returns:
            Path to the ``ncbi_dataset/data`` subdirectory inside the
            extracted tree, or None if the expected layout is absent.
        """
        extract_path = os.path.join(temp_dir, "unpacked")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_path)

        dataset_root = os.path.join(extract_path, "ncbi_dataset", "data")
        if not os.path.exists(dataset_root):
            logger.error(
                "Unexpected archive layout: "
                f"'ncbi_dataset/data' not found in {archive_path}."
            )
            return None
        return dataset_root

    def _ingest_accession_fasta(
        self,
        accession: str,
        dataset_root: str,
    ) -> list[dict[str, str]] | None:
        """Parse a single accession's FASTA file and persist to LMDB.

        Args:
            accession: Accession ID being processed.
            dataset_root: Path to the ``ncbi_dataset/data`` directory
                containing per-accession subdirectories.

        Returns:
            The header metadata for the sequences written to LMDB -- possibly an
            empty list when ``exclude_plasmids`` filtered every sequence out --
            or ``None`` when the accession failed (its directory, FASTA, or
            content is missing). The None-vs-list distinction lets the caller
            retry genuine failures while marking a present-but-all-filtered
            accession as processed (no retry loop).
        """
        accession_dir = os.path.join(dataset_root, accession)
        if not os.path.exists(accession_dir):
            return None

        fasta_path = self._find_fasta_in_directory(accession_dir)
        if not fasta_path:
            return None

        sequences, headers_metadata = self._parse_fasta_file(fasta_path)
        if not sequences:
            return None

        if self.exclude_plasmids:
            sequences, headers_metadata = self._drop_excluded_molecules(
                accession, sequences, headers_metadata
            )

        self._persist_sequences_to_lmdb(sequences)
        return headers_metadata

    @classmethod
    def _is_excluded_molecule(cls, name: str) -> bool:
        """Return True if a FASTA defline name marks an excluded molecule type.

        Heuristic substring match against ``_EXCLUDED_MOLECULE_KEYWORDS`` (e.g.
        a defline like "... plasmid pXYZ, complete sequence"). Reliable for
        RefSeq deflines; the authoritative alternative is the NCBI sequence
        report's ``assigned_molecule_location_type``.
        """
        lowered = name.lower()
        return any(keyword in lowered for keyword in cls._EXCLUDED_MOLECULE_KEYWORDS)

    def _drop_excluded_molecules(
        self,
        accession: str,
        sequences: dict[str, str],
        headers_metadata: list[dict[str, str]],
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """Drop excluded-molecule sequences (e.g. plasmids) from a parse result.

        Filters the sequence map and the header metadata in lock-step so the
        vault and the recorded headers stay consistent, and logs how many were
        skipped.

        Args:
            accession: Accession being processed (for logging).
            sequences: Header-ID -> sequence map from ``_parse_fasta_file``.
            headers_metadata: Ordered ``{'id', 'name'}`` records.

        Returns:
            The filtered ``(sequences, headers_metadata)`` pair.
        """
        kept_headers = [
            header
            for header in headers_metadata
            if not self._is_excluded_molecule(header["name"])
        ]
        kept_ids = {header["id"] for header in kept_headers}
        kept_sequences = {
            seq_id: seq for seq_id, seq in sequences.items() if seq_id in kept_ids
        }
        skipped = len(headers_metadata) - len(kept_headers)
        if skipped:
            logger.info(
                "Accession %s: skipped %d plasmid/excluded sequence(s) at "
                "ingestion.", accession, skipped,
            )
        return kept_sequences, kept_headers

    @staticmethod
    def _find_fasta_in_directory(directory: str) -> str | None:
        """Locate the first FASTA file in a directory.

        Args:
            directory: Directory to scan.

        Returns:
            Full path to the FASTA file, or None if none is found.
        """
        for file_name in os.listdir(directory):
            if file_name.endswith(FASTA_EXTENSIONS):
                return os.path.join(directory, file_name)
        return None

    @staticmethod
    def _parse_fasta_file(
        fasta_path: str,
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        """Parse a FASTA file into a sequence map and header metadata.

        The FASTA file is read line by line. The first whitespace-
        separated token on each header line is used as the sequence ID;
        the remainder of the line becomes the human-readable name.
        Sequence lines are concatenated until the next header or EOF.

        Args:
            fasta_path: Filesystem path to the FASTA file.

        Returns:
            Two-tuple ``(sequences, headers)``:
                - sequences: dict mapping header ID to the full sequence
                  string.
                - headers: list of {'id': ..., 'name': ...} dicts in the
                  order the sequences appear in the file.
        """
        sequences: dict[str, str] = {}
        headers_metadata: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        current_header: str | None = None
        current_seq_lines: list[str] = []
        skip_current = False  # reading a duplicate-id record to be dropped

        with open(fasta_path, encoding="utf-8") as fasta_file:
            for line in fasta_file:
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith(">"):
                    if current_header is not None and not skip_current:
                        sequences[current_header] = "".join(current_seq_lines)

                    parts = stripped[1:].split(" ", 1)
                    current_header = parts[0]
                    sequence_name = parts[1] if len(parts) > 1 else current_header
                    if current_header in seen_ids:
                        # A record id repeated within one FASTA is malformed; keep
                        # the first occurrence so `sequences` and `headers_metadata`
                        # stay one-to-one (the LMDB key is the id).
                        logger.warning(
                            "Duplicate FASTA record id %r in %s; keeping the first.",
                            current_header, fasta_path,
                        )
                        skip_current = True
                    else:
                        seen_ids.add(current_header)
                        headers_metadata.append(
                            {"id": current_header, "name": sequence_name}
                        )
                        skip_current = False
                    current_seq_lines = []
                else:
                    current_seq_lines.append(stripped)

            if current_header is not None and not skip_current:
                sequences[current_header] = "".join(current_seq_lines)

        return sequences, headers_metadata

    def _persist_sequences_to_lmdb(self, sequences: dict[str, str]) -> None:
        """Write parsed sequences to LMDB with zlib compression.

        The LMDB key is the sequence record id (its FASTA accession.version),
        which is a **single global namespace** across every accession in the
        vault. This is safe because NCBI sequence ids are globally unique, so two
        different assemblies never emit the same id; the same id reappears only
        when the *same* accession is re-ingested (an idempotent overwrite of its
        own value). A collision across distinct assemblies would silently
        overwrite — it cannot happen with NCBI ids, and is called out here so the
        invariant is explicit if the id source ever changes.

        Args:
            sequences: Dictionary mapping record id to the raw sequence
                string. All entries are written within a single LMDB
                transaction for atomicity.

        Raises:
            RuntimeError: If the LMDB environment has not been opened.
        """
        if self._env is None:
            raise RuntimeError(
                "LMDB environment is not open. "
                "Call download_all_pending() instead of download_batch() directly."
            )

        with self._env.begin(write=True) as txn:
            for header_id, sequence in sequences.items():
                compressed = zlib.compress(sequence.encode("utf-8"))
                txn.put(header_id.encode("utf-8"), compressed)
