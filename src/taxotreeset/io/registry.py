"""Local metadata registry for NCBI Taxonomy and genome accessions.

This module provides the NCBIRegistry class, which manages the local
JSON-based registry used by the discovery and generation phases of the
TaxoTreeSet pipeline. The registry serves as the authoritative inventory
of:

- Taxon-to-accession associations (which genome accessions belong to
  each NCBI TaxID).
- Per-accession metadata (organism name, assembly level, download
  status, local storage path).
- Scope mapping configuration (loaded from configs/mapping.json).

The registry is designed for incremental updates: existing entries are
preserved across runs, and new metadata is merged in without overwriting
previously discovered information. This enables resuming interrupted
discovery sessions and adding new domains to an existing registry.

Sequence data itself is stored separately in an LMDB vault (see
src/taxotreeset/io/downloader.py); the registry only holds metadata
needed to locate and identify each accession.

Typical usage::

    from taxotreeset.io.registry import NCBIRegistry

    registry = NCBIRegistry(
        config_path="configs/mapping.json",
        registry_path="data/registry.json",
    )
    registry.discover_taxon_metadata(taxon_id=10239)
    registry.save()
"""

import json
import logging
import os
import subprocess
from typing import Any, Union

logger = logging.getLogger("TaxoTreeSet.IO.Registry")

TaxonId = Union[int, str]


class NCBIRegistry:
    """Inventory of NCBI taxa and genome accessions with persistence.

    Maintains an in-memory representation of the registry that is loaded
    from disk at construction time and serialized back via ``save()``.
    The registry structure is::

        {
            "last_update": <ISO timestamp or None>,
            "taxons": {
                "<taxid>": ["<accession_1>", "<accession_2>", ...]
            },
            "accessions": {
                "<accession>": {
                    "taxid": "<taxid>",
                    "organism": "<organism_name>",
                    "is_reference": <bool>,
                    "total_sequence_length": <int or None>,
                    "downloaded": <bool>,
                    "local_path": "<path_to_lmdb or None>"
                }
            },
            "lineages": {
                "<taxid>": [
                    {"taxid": "<id>", "rank": "<rank>", "name": "<name>"},
                    ...
                ]
            }
        }

    The ``lineages`` map caches each species TaxID's resolved ancestry
    (species to root) so generation can scope and place accessions
    without re-resolving lineages, and so entries resolved via the NCBI
    fallback (TaxIDs newer than the taxoniq snapshot) survive into
    generation.

    Attributes:
        config_path: Path to the scope mapping configuration JSON file.
        registry_path: Path to the registry JSON file (loaded and saved
            from this location).
        registry: The in-memory registry dictionary.
        mapping: The in-memory scope mapping configuration.
    """

    _DEFAULT_CONFIG_PATH = "configs/mapping.json"
    _REFERENCE_ASSEMBLY_LEVELS = frozenset({"Complete Genome", "Chromosome"})

    def __init__(
        self,
        registry_path: str,
        config_path: str = _DEFAULT_CONFIG_PATH,
    ) -> None:
        """Initialize the registry by loading existing state and configuration.

        Args:
            config_path: Path to the scope mapping configuration file.
                Defaults to ``configs/mapping.json``.
            registry_path: Path to the registry persistence file.
                If the file exists, it is loaded; otherwise an empty
                registry structure is initialized. Defaults to
                ``data/registry.json``.
        """
        self.config_path: str = config_path
        self.registry_path: str = registry_path
        self.registry: dict[str, Any] = self._load_registry()
        self.mapping: dict[str, Any] = self._load_mapping()

    def _load_registry(self) -> dict[str, Any]:
        """Load an existing registry from disk or initialize a fresh one.

        When the registry file exists, its content is parsed and returned
        as-is. When it does not exist, an empty registry skeleton is
        returned, enabling incremental population on the next save.

        Returns:
            The loaded or freshly initialized registry dictionary.
        """
        if os.path.exists(self.registry_path):
            with open(self.registry_path, encoding="utf-8") as registry_file:
                loaded = json.load(registry_file)
            # Backfill any sections absent in older on-disk schemas so
            # callers can rely on every key being present.
            for key, default in self._empty_registry().items():
                if key not in loaded:
                    loaded[key] = default
            return loaded
        return self._empty_registry()

    @staticmethod
    def _empty_registry() -> dict[str, Any]:
        """Return a fresh, empty registry skeleton.

        Returns:
            Dictionary with the canonical registry structure and empty
            sub-dictionaries.
        """
        return {
            "last_update": None,
            "taxons": {},
            "accessions": {},
            "lineages": {},
        }

    def _load_mapping(self) -> dict[str, Any]:
        """Load scope and redirection rules from the configuration file.

        Returns:
            Parsed mapping configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            json.JSONDecodeError: If the configuration file is malformed.
        """
        with open(self.config_path, encoding="utf-8") as config_file:
            return json.load(config_file)

    def discover_taxon_metadata(self, taxon_id: TaxonId) -> None:
        """Fetch reference genome metadata for a taxon via NCBI Datasets.

        Spawns a subprocess running the ``datasets summary`` command from
        the NCBI Datasets CLI to retrieve assembly summaries for all
        reference genomes under the given taxon. Each returned assembly
        record is parsed and merged into the registry.

        On subprocess failure, an error is logged and the registry is
        left in its current state (no partial updates). The caller may
        retry the discovery on transient failures.

        Args:
            taxon_id: NCBI TaxID to query. Accepts both int and str
                representations; internally normalized to string.

        Example:
            >>> registry = NCBIRegistry()
            >>> registry.discover_taxon_metadata(10239)
            >>> registry.save()
        """
        logger.info(f"Discovering genomic metadata for TaxID: {taxon_id}")

        command = [
            "datasets",
            "summary",
            "genome",
            "taxon",
            str(taxon_id),
            "--reference",
            "--as-json-lines",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            error_message = exc.stderr.strip() if exc.stderr else str(exc)
            logger.error(
                f"Failed to query TaxID {taxon_id} via NCBI CLI: {error_message}"
            )
            return

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            assembly_data = json.loads(line)
            self._update_taxon_entry(taxon_id, assembly_data)

    def store_lineage(
        self,
        taxon_id: TaxonId,
        lineage: list[dict[str, str]],
    ) -> None:
        """Cache a species TaxID's resolved ancestry in the registry.

        Idempotent and overwrite-safe: re-resolving a lineage simply
        refreshes the cached entry. Storing here lets generation scope
        and place accessions without re-resolving, and preserves
        lineages resolved via the NCBI fallback.

        Args:
            taxon_id: Species TaxID the lineage belongs to.
            lineage: Ancestors species-to-root, each a dict with
                ``taxid``, ``rank``, and ``name`` keys.
        """
        self.registry["lineages"][str(taxon_id)] = lineage

    def _update_taxon_entry(
        self,
        taxon_id: TaxonId,
        assembly_data: dict[str, Any],
    ) -> None:
        """Merge a single NCBI assembly report into the registry.

        For each report in the payload, this method:
        1. Associates the accession with the taxon under ``taxons``.
        2. Creates a new accession metadata entry under ``accessions``
           if not already present.

        Existing accession metadata is never overwritten — once an
        accession has been recorded, subsequent discoveries are
        idempotent. This ensures downloaded status and local paths
        survive across discovery passes.

        Args:
            taxon_id: NCBI TaxID owning the assembly records.
            assembly_data: Parsed JSON dictionary from a single line of
                the NCBI Datasets CLI output. Expected to contain a
                ``reports`` array.
        """
        reports = assembly_data.get("reports", [])
        taxon_key = str(taxon_id)

        if taxon_key not in self.registry["taxons"]:
            self.registry["taxons"][taxon_key] = []

        for report in reports:
            accession = report.get("accession")
            if not accession:
                continue

            if accession not in self.registry["taxons"][taxon_key]:
                self.registry["taxons"][taxon_key].append(accession)

            if accession not in self.registry["accessions"]:
                self.registry["accessions"][accession] = self._build_accession_entry(
                    taxon_key, report
                )

    @classmethod
    def _build_accession_entry(
        cls,
        taxon_key: str,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """Construct a new accession metadata entry from an NCBI report.

        The reference status is derived from the assembly level: a genome
        is considered reference-quality when its level is either
        'Complete Genome' or 'Chromosome'. This catches both bacterial
        complete assemblies and high-quality eukaryotic chromosome-level
        assemblies in a single check.

        Args:
            taxon_key: String representation of the owning TaxID.
            report: Single assembly report dictionary from the NCBI
                Datasets CLI output.

        Returns:
            New accession metadata dictionary ready to be inserted into
            the registry.
        """
        assembly_info = report.get("assembly_info", {})
        assembly_level = assembly_info.get("assembly_level", "")
        is_reference = assembly_level in cls._REFERENCE_ASSEMBLY_LEVELS

        organism_info = report.get("organism", {})
        organism_name = organism_info.get("organism_name")
        assembly_stats = report.get("assembly_stats", {})
        total_sequence_length = assembly_stats.get("total_sequence_length")

        return {
            "taxid": taxon_key,
            "organism": organism_name,
            "is_reference": is_reference,
            "total_sequence_length": total_sequence_length,
            "downloaded": False,
            "local_path": None,
        }

    def save(self) -> None:
        """Persist the current registry state to disk as formatted JSON.

        Creates the destination directory if it does not exist. The
        registry file is written in pretty-printed form (indent=2) to
        ease manual inspection and git diff readability.

        This method is intentionally logged at DEBUG level rather than
        INFO because it is called frequently during discovery (after
        each batch) and would otherwise distort progress bar output.
        """
        destination_dir = os.path.dirname(self.registry_path)
        if destination_dir:
            os.makedirs(destination_dir, exist_ok=True)

        with open(self.registry_path, "w", encoding="utf-8") as registry_file:
            json.dump(self.registry, registry_file, indent=2)

        logger.debug(f"Registry persisted to: {self.registry_path}")
