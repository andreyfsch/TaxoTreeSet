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

    from src.taxotreeset.io.registry import NCBIRegistry
    from src.taxotreeset.io.downloader import NCBIDownloader

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

    _DEFAULT_VAULT_PATH = "data/vault"
    _DEFAULT_CHUNK_SIZE = 100

    def __init__(
        self,
        registry: Any,
        vault_path: str = _DEFAULT_VAULT_PATH,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
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
        """
        self.registry: Any = registry
        self.vault_path: str = vault_path
        self.chunk_size: int = chunk_size

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
        total_accessions = len(self.registry.registry.get("accessions", {}))
        total_pending = len(pending)
        already_downloaded = total_accessions - total_pending

        if total_pending == 0:
            logger.info("All registered accessions are already archived in LMDB.")
            return

        chunks = self._split_into_chunks(pending)
        logger.info(
            f"Grouped {total_pending} pending accessions into "
            f"{len(chunks)} batch downloads (chunk size: {self.chunk_size})."
        )

        self._env = lmdb.open(
            self.lmdb_path,
            map_size=LMDB_MAP_SIZE_BYTES,
            max_dbs=0,
        )

        try:
            self._process_chunks(
                chunks=chunks,
                total_accessions=total_accessions,
                already_downloaded=already_downloaded,
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
        self.registry.save()

    def _collect_pending_accessions(self) -> list[str]:
        """Return the list of accession IDs that still need downloading.

        Returns:
            List of accession strings with downloaded=False in the
            registry. Order matches the registry's insertion order.
        """
        accessions = self.registry.registry.get("accessions", {})
        return [
            accession
            for accession, info in accessions.items()
            if not info.get("downloaded")
        ]

    def _split_into_chunks(self, accessions: list[str]) -> list[list[str]]:
        """Split a flat list of accessions into chunks of chunk_size.

        Args:
            accessions: Flat list of accession IDs to split.

        Returns:
            List of lists, each containing up to chunk_size accessions.
            The final chunk may be smaller.
        """
        return [
            accessions[i : i + self.chunk_size]
            for i in range(0, len(accessions), self.chunk_size)
        ]

    def _process_chunks(
        self,
        chunks: list[list[str]],
        total_accessions: int,
        already_downloaded: int,
    ) -> None:
        """Execute the per-chunk download loop with progress reporting.

        Args:
            chunks: List of accession chunks to process sequentially.
            total_accessions: Total accessions in the registry, used to
                size the progress bar.
            already_downloaded: Accessions completed in prior runs, used
                as the progress bar's starting position.
        """
        with tqdm(
            total=total_accessions,
            initial=already_downloaded,
            desc="Ingesting genomes to LMDB",
            unit=" genome",
        ) as progress_bar:
            for chunk in chunks:
                batch_results = self.download_batch(chunk)
                self._update_registry_for_batch(chunk, batch_results)
                progress_bar.update(len(chunk))
                self.registry.save()

    def _update_registry_for_batch(
        self,
        chunk: list[str],
        batch_results: dict[str, list[dict[str, str]]],
    ) -> None:
        """Update registry entries for successfully downloaded accessions.

        Args:
            chunk: List of accession IDs that were requested in the batch.
            batch_results: Dictionary mapping accession ID to its parsed
                FASTA headers metadata. Accessions absent from this dict
                are considered failed and left untouched.
        """
        accessions = self.registry.registry["accessions"]
        for accession in chunk:
            if accession not in batch_results:
                continue
            accessions[accession]["downloaded"] = True
            accessions[accession]["local_path"] = self.lmdb_path
            accessions[accession]["headers"] = batch_results[accession]

    def download_batch(
        self,
        accessions: list[str],
    ) -> dict[str, list[dict[str, str]]]:
        """Download a single batch of accessions via the NCBI CLI.

        Spawns one ``datasets download`` subprocess that fetches all
        requested accessions in a ZIP archive, unpacks it, parses the
        FASTA files, and persists each sequence into LMDB. Returns a
        mapping of successfully processed accessions to their header
        metadata.

        Accessions that fail to download or whose FASTA files are
        missing are silently omitted from the result. Callers can detect
        failures by comparing input and output keys.

        Args:
            accessions: List of accession IDs to download together in a
                single CLI invocation.

        Returns:
            Dictionary mapping accession ID to a list of header
            metadata dictionaries (each with 'id' and 'name' keys). An
            empty dict is returned if the CLI call fails entirely.
        """
        logger.debug(f"Spawning bulk NCBI fetch for {len(accessions)} accessions.")
        batch_results: dict[str, list[dict[str, str]]] = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "batch_package.zip")

            if not self._invoke_ncbi_datasets_cli(accessions, archive_path):
                return batch_results

            extracted_root = self._extract_assembly_archive(archive_path, temp_dir)
            if extracted_root is None:
                return batch_results

            for accession in accessions:
                headers = self._ingest_accession_fasta(accession, extracted_root)
                if headers:
                    batch_results[accession] = headers

        return batch_results

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
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            logger.error(f"NCBI Datasets CLI invocation failed: {stderr}")
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
    ) -> list[dict[str, str]]:
        """Parse a single accession's FASTA file and persist to LMDB.

        Args:
            accession: Accession ID being processed.
            dataset_root: Path to the ``ncbi_dataset/data`` directory
                containing per-accession subdirectories.

        Returns:
            List of header metadata dictionaries for sequences that were
            successfully written to LMDB. Empty list if the accession's
            directory or FASTA file is missing.
        """
        accession_dir = os.path.join(dataset_root, accession)
        if not os.path.exists(accession_dir):
            return []

        fasta_path = self._find_fasta_in_directory(accession_dir)
        if not fasta_path:
            return []

        sequences, headers_metadata = self._parse_fasta_file(fasta_path)
        if not sequences:
            return []

        self._persist_sequences_to_lmdb(sequences)
        return headers_metadata

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
        current_header: str | None = None
        current_seq_lines: list[str] = []

        with open(fasta_path, encoding="utf-8") as fasta_file:
            for line in fasta_file:
                stripped = line.strip()
                if not stripped:
                    continue

                if stripped.startswith(">"):
                    if current_header is not None:
                        sequences[current_header] = "".join(current_seq_lines)

                    parts = stripped[1:].split(" ", 1)
                    current_header = parts[0]
                    sequence_name = parts[1] if len(parts) > 1 else current_header
                    headers_metadata.append(
                        {"id": current_header, "name": sequence_name}
                    )
                    current_seq_lines = []
                else:
                    current_seq_lines.append(stripped)

            if current_header is not None:
                sequences[current_header] = "".join(current_seq_lines)

        return sequences, headers_metadata

    def _persist_sequences_to_lmdb(self, sequences: dict[str, str]) -> None:
        """Write parsed sequences to LMDB with zlib compression.

        Args:
            sequences: Dictionary mapping header ID to the raw sequence
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
