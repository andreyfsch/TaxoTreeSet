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

import datetime
import hashlib
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

    The ``lineages`` map caches each leaf TaxID's resolved ancestry
    (leaf to root) so generation can scope and place accessions
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
            "capacities": {},
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
            try:
                assembly_data = json.loads(line)
            except json.JSONDecodeError:
                # The CLI can emit non-JSON informational lines; skip them
                # rather than aborting the whole discovery.
                continue
            self._update_taxon_entry(taxon_id, assembly_data)

    def store_capacities(self, capacities: dict[str, int], min_len: int) -> None:
        """Persist pre-computed node capacities for a given window size.

        Merges the supplied mapping into the registry's capacity cache.
        Existing entries for other ``min_len`` values are preserved;
        only the entries for the given ``min_len`` are updated.

        Args:
            capacities: Mapping of TaxID string to capacity value, as
                returned by ``compute_all_capacities``.
            min_len: Sliding-window size used to produce these capacities.
                Stored as the dict key so multiple window sizes coexist.
        """
        min_len_key = str(min_len)
        cache = self.registry["capacities"]
        for taxid, value in capacities.items():
            cache.setdefault(taxid, {})[min_len_key] = value

    def load_capacities(self, min_len: int) -> dict[str, int]:
        """Return all cached node capacities for a given window size.

        Args:
            min_len: Sliding-window size to look up.

        Returns:
            Mapping of TaxID string to capacity. Empty when no entry
            exists for this ``min_len``.
        """
        min_len_key = str(min_len)
        return {
            taxid: entries[min_len_key]
            for taxid, entries in self.registry["capacities"].items()
            if min_len_key in entries
        }

    def _invalidate_ancestor_capacities(self, leaf_taxid: str) -> None:
        """Remove cached capacities for all ancestors of a leaf taxon.

        Called when a new accession is added under ``leaf_taxid`` so
        that stale capacity values for every ancestor are evicted. The
        lineage must already be stored in the registry before this method
        is called (``store_lineage`` must precede ``_update_taxon_entry``
        in the discovery flow).

        All ``min_len`` entries for each affected ancestor are removed
        together: adding a new sequence invalidates the capacity for any
        window size, so partial retention would leave stale values.

        Args:
            leaf_taxid: Leaf TaxID (a species or a rank below it) whose
                ancestor capacities should be invalidated.
        """
        lineage = self.registry["lineages"].get(leaf_taxid, [])
        cache = self.registry["capacities"]
        for ancestor in lineage:
            ancestor_taxid = ancestor["taxid"]
            if ancestor_taxid in cache:
                del cache[ancestor_taxid]
                logger.debug(
                    "Capacity cache invalidated for taxid %s "
                    "(new accession under %s).",
                    ancestor_taxid,
                    leaf_taxid,
                )

    def store_lineage(
        self,
        taxon_id: TaxonId,
        lineage: list[dict[str, str]],
    ) -> None:
        """Cache a leaf TaxID's resolved ancestry in the registry.

        Idempotent and overwrite-safe: re-resolving a lineage simply
        refreshes the cached entry. Storing here lets generation scope
        and place accessions without re-resolving, and preserves
        lineages resolved via the NCBI fallback.

        Args:
            taxon_id: Leaf TaxID the lineage belongs to.
            lineage: Ancestors leaf-to-root, each a dict with
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
                self._invalidate_ancestor_capacities(taxon_key)

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
        raw_len = assembly_stats.get("total_sequence_length")
        total_sequence_length = int(raw_len) if raw_len is not None else None

        return {
            "taxid": taxon_key,
            "organism": organism_name,
            "is_reference": is_reference,
            "total_sequence_length": total_sequence_length,
            "downloaded": False,
            "download_deferred": False,
            "local_path": None,
        }

    def get_pending_volume(self, domain_taxid: str | None = None) -> int:
        """Sum total_sequence_length of all pending accessions in scope.

        Counts accessions where ``downloaded`` is False, regardless of
        whether they are deferred. Used before the selection pass to
        decide whether selective download is needed.

        Args:
            domain_taxid: Optional domain anchor. When given, only
                accessions whose lineage contains this TaxID
                are counted. When None, counts across the whole registry.

        Returns:
            Total estimated download volume in bytes.
        """
        lineages = self.registry["lineages"]
        accessions = self.registry["accessions"]
        taxons = self.registry["taxons"]
        domain_str = str(domain_taxid) if domain_taxid else None

        seen: set[str] = set()
        total = 0
        for taxid, acc_list in taxons.items():
            if domain_str is not None:
                stored = lineages.get(taxid)
                if not stored or not any(a["taxid"] == domain_str for a in stored):
                    continue
            for acc_id in acc_list:
                if acc_id in seen:
                    continue
                seen.add(acc_id)
                info = accessions.get(acc_id, {})
                if not info.get("downloaded"):
                    total += int(info.get("total_sequence_length") or 0)
        return total

    def mark_accessions_deferred(self, accession_ids: list[str]) -> None:
        """Set ``download_deferred=True`` for the given accession IDs.

        Called by the selective download selection pass to flag accessions
        that are not needed for the first download batch. The downloader
        skips deferred accessions; they remain available for the
        refinement phase.

        Args:
            accession_ids: Accession IDs to mark as deferred.
        """
        registry_accessions = self.registry["accessions"]
        for acc_id in accession_ids:
            if acc_id in registry_accessions:
                registry_accessions[acc_id]["download_deferred"] = True

    def reset_selection_flags(self, domain_taxid: str | None = None) -> None:
        """Clear all ``download_deferred`` flags for pending accessions in scope.

        Called before a new selection pass so prior-run deferral decisions
        do not persist. Only clears flags for accessions that are not yet
        downloaded; already-downloaded accessions are unaffected.

        Args:
            domain_taxid: Optional domain anchor to restrict the reset.
                When None, clears flags across the entire registry.
        """
        lineages = self.registry["lineages"]
        accessions = self.registry["accessions"]
        taxons = self.registry["taxons"]
        domain_str = str(domain_taxid) if domain_taxid else None

        seen: set[str] = set()
        for taxid, acc_list in taxons.items():
            if domain_str is not None:
                stored = lineages.get(taxid)
                if not stored or not any(a["taxid"] == domain_str for a in stored):
                    continue
            for acc_id in acc_list:
                if acc_id in seen:
                    continue
                seen.add(acc_id)
                info = accessions.get(acc_id)
                if info and info.get("download_deferred"):
                    info["download_deferred"] = False

    def save(self) -> None:
        """Persist the current registry state to disk as formatted JSON.

        Creates the destination directory if it does not exist. The
        registry file is written in pretty-printed form (indent=2) to
        ease manual inspection and git diff readability.

        The write is **atomic**: the JSON is written to a temp sibling and then
        ``os.replace``d into place, so an interrupted write (a WSL crash, a full
        disk) never truncates the authoritative registry into a file the next
        load cannot parse. ``save()`` runs after every download chunk and
        discovery checkpoint, so the crash window is wide and losing the registry
        would forfeit all discovery/download progress.

        This method is intentionally logged at DEBUG level rather than
        INFO because it is called frequently during discovery (after
        each batch) and would otherwise distort progress bar output.
        """
        destination_dir = os.path.dirname(self.registry_path)
        if destination_dir:
            os.makedirs(destination_dir, exist_ok=True)

        tmp_path = f"{self.registry_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as registry_file:
            json.dump(self.registry, registry_file, indent=2)
        os.replace(tmp_path, self.registry_path)

        logger.debug(f"Registry persisted to: {self.registry_path}")

    def mark_updated(self) -> None:
        """Stamp the registry with the current UTC time as its last NCBI update.

        Called when discovery refreshes the registry from NCBI, so the snapshot
        records *when* "the current state of RefSeq" was captured -- part of the
        provenance that makes a run reproducible.
        """
        self.registry["last_update"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()

    def accession_snapshot(self) -> dict[str, Any]:
        """Return a reproducible snapshot of the registered accessions.

        Accession keys carry NCBI version suffixes (e.g. ``GCF_000857325.2``),
        which are immutable, so the sorted accession list and its SHA-256 digest
        uniquely and reproducibly identify the genome set behind a dataset --
        even though "the current state of RefSeq" drifts over time. Recording
        this turns a run into a citable, re-fetchable snapshot.

        Returns:
            Dict with ``n_accessions``, ``sha256`` (digest of the
            newline-joined sorted accession list) and the sorted ``accessions``
            list.
        """
        accessions = sorted(self.registry.get("accessions", {}))
        digest = hashlib.sha256("\n".join(accessions).encode("utf-8")).hexdigest()
        return {
            "n_accessions": len(accessions),
            "sha256": digest,
            "accessions": accessions,
        }
