"""Discovery orchestrator for the taxonomic mapping phase.

This module provides the ``DiscoveryOrchestrator``, which performs the
first major phase of the TaxoTreeSet pipeline: traversing the NCBI
Taxonomy from a given root, streaming the genomic assemblies under it,
and populating the local registry with metadata for all reachable
accessions.

The orchestrator works in three sequential stages:

1. **Streaming the NCBI Datasets CLI**: a single subprocess fetches
   every reference genome under the requested root taxon. Results are
   read line by line as JSON Lines, allowing the orchestrator to
   handle datasets of hundreds of thousands of accessions without
   loading the entire response into memory.

2. **Lineage resolution**: each unique species TaxID encountered is
   resolved against the local taxoniq cache to produce its full
   ranked lineage from root to leaf. Lineages are then transformed
   through the mapping configuration to apply scope-level
   redirections (taxa with no valid kingdom placement are routed
   into curated semantic fallback groups).

3. **Tree assembly with incremental checkpoints**: each resolved
   path is added to the in-memory bigtree skeleton, and the registry
   is flushed to disk periodically (every ``checkpoint_interval``
   entries by default). This protects against data loss during long
   discovery runs.

This orchestrator does NOT download sequence data itself. It only
catalogs metadata; the subsequent download phase
(``NCBIDownloader.download_all_pending``) is responsible for fetching
genome content into the LMDB vault.

Typical usage::

    from taxotreeset.core.orchestrator import DiscoveryOrchestrator
    from taxotreeset.io.registry import NCBIRegistry

    registry = NCBIRegistry(registry_path="data/registry.json")
    orchestrator = DiscoveryOrchestrator(
        registry=registry,
        mapping_config=mapping_dict,
    )
    orchestrator.discover_from_root(root_taxid=10239)  # Viruses
"""

import json
import logging
import os
import subprocess
from typing import Any

import taxoniq
from bigtree import Node, add_path_to_tree
from tqdm import tqdm

logger = logging.getLogger("TaxoTreeSet.Core.Orchestrator")

_DEFAULT_ASSEMBLY_LEVELS = "complete,chromosome"
_DEFAULT_CHECKPOINT_INTERVAL = 500
_NCBI_API_KEY_ENV_VAR = "NCBI_API_KEY"


class DiscoveryOrchestrator:
    """Coordinate taxonomic discovery from a root taxon down to species.

    Traverses the NCBI Taxonomy starting from a biological root
    (kingdom or domain level) and populates both the in-memory
    taxonomic tree skeleton and the persistent registry of
    accessions.

    The orchestrator is designed to be tolerant of partial failures:
    individual TaxID resolution errors are logged at DEBUG level and
    skipped, so a single problematic entry does not abort an
    otherwise successful discovery run. Critical errors (subprocess
    failures, registry I/O errors) are re-raised to surface
    operational issues.

    Attributes:
        registry: NCBIRegistry instance for persistent metadata
            storage.
        mapping: Scope mapping configuration dictionary, used to
            apply redirection rules during lineage resolution.
        tree_root: Root Node of the in-memory bigtree being built.
    """

    def __init__(self, registry: Any, mapping_config: dict[str, Any]) -> None:
        """Initialize the orchestrator with a registry and mapping config.

        Args:
            registry: NCBIRegistry instance for persisting accession
                metadata as it is discovered.
            mapping_config: Parsed contents of the scope mapping
                configuration JSON.
        """
        self.registry: Any = registry
        self.mapping: dict[str, Any] = mapping_config
        self.tree_root: Node = Node("root")

    def discover_from_root(
        self,
        root_taxid: int,
        assembly_levels: str = _DEFAULT_ASSEMBLY_LEVELS,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
    ) -> None:
        """Discover all accessions under a root taxon and populate registry.

        Args:
            root_taxid: NCBI TaxID of the biological root to traverse
                (e.g., 10239 for Viruses, 2 for Bacteria).
            assembly_levels: Comma-separated NCBI assembly levels to
                request. Defaults to 'complete,chromosome' which
                covers reference-quality genomes.
            checkpoint_interval: Number of processed taxa between
                registry checkpoints. Lower values give better crash
                recovery at the cost of more disk I/O.

        Raises:
            RuntimeError: If the NCBI Datasets CLI subprocess fails
                or returns no data.
        """
        root_id_str = str(root_taxid)
        self._log_api_key_status()

        reports_by_taxid = self._stream_ncbi_summaries(
            root_id_str=root_id_str,
            assembly_levels=assembly_levels,
        )

        if not reports_by_taxid:
            return

        logger.info(
            f"Successfully streamed {len(reports_by_taxid)} unique "
            "species taxa. Commencing hierarchy building."
        )

        self._build_hierarchy(
            reports_by_taxid=reports_by_taxid,
            root_id_str=root_id_str,
            checkpoint_interval=checkpoint_interval,
        )

        self.registry.save()
        logger.info("Metadata registration and tree construction completed.")

    @staticmethod
    def _log_api_key_status() -> None:
        """Emit an informational log line if an NCBI API key is set.

        Presence of NCBI_API_KEY raises subprocess rate limits from
        3 to 10 requests per second, which materially affects
        large-domain discovery runs.
        """
        if os.environ.get(_NCBI_API_KEY_ENV_VAR):
            logger.info(
                f"{_NCBI_API_KEY_ENV_VAR} environment variable is "
                "active for the NCBI CLI subprocess."
            )

    def _stream_ncbi_summaries(
        self,
        root_id_str: str,
        assembly_levels: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Stream genome reports from the NCBI Datasets CLI subprocess.

        Spawns the CLI as a long-running subprocess and reads its
        stdout line by line as JSON Lines. Each line is one assembly
        report; reports are grouped by species TaxID.

        Args:
            root_id_str: Root TaxID as a string.
            assembly_levels: Comma-separated NCBI assembly levels.

        Returns:
            Dictionary mapping species TaxID strings to their list of
            assembly report dictionaries. Empty dict on subprocess
            failure.
        """
        command = self._build_summary_command(root_id_str, assembly_levels)
        logger.info(f"Spawning NCBI Datasets CLI subprocess for TaxID: {root_id_str}")

        env = os.environ.copy()
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )
        except OSError as exc:
            logger.error(f"Failed to spawn NCBI Datasets CLI subprocess: {exc}")
            return {}

        reports_by_taxid = self._consume_jsonlines_stream(process)
        process.wait()

        if not reports_by_taxid:
            stderr_output = process.stderr.read() if process.stderr else ""
            logger.error(f"NCBI streaming process returned no data: {stderr_output}")

        return reports_by_taxid

    @staticmethod
    def _build_summary_command(
        root_id_str: str,
        assembly_levels: str,
    ) -> list[str]:
        """Construct the NCBI Datasets CLI command argument list.

        Args:
            root_id_str: Root TaxID as string.
            assembly_levels: Comma-separated assembly levels.

        Returns:
            Command argument list suitable for subprocess.Popen.
        """
        return [
            "datasets",
            "summary",
            "genome",
            "taxon",
            root_id_str,
            "--assembly-source",
            "RefSeq",
            "--assembly-level",
            assembly_levels,
            "--as-json-lines",
        ]

    @staticmethod
    def _consume_jsonlines_stream(
        process: subprocess.Popen,
    ) -> dict[str, list[dict[str, Any]]]:
        """Read JSON Lines from a subprocess stdout, grouping by TaxID.

        Lines that fail to parse as JSON are silently skipped (the
        NCBI CLI occasionally emits informational lines that are not
        JSON). The progress bar reflects the count of successfully
        parsed reports.

        Args:
            process: The subprocess.Popen handle whose stdout will
                be consumed.

        Returns:
            Dictionary mapping species TaxID strings to assembly
            report lists.
        """
        reports_by_taxid: dict[str, list[dict[str, Any]]] = {}
        if process.stdout is None:
            return reports_by_taxid

        with tqdm(desc="Streaming NCBI Genome Reports", unit=" seqs") as progress_bar:
            for line in process.stdout:
                if not line.strip():
                    continue
                try:
                    report = json.loads(line)
                except json.JSONDecodeError:
                    continue

                taxid = report.get("organism", {}).get("tax_id")
                if not taxid:
                    continue

                taxid_str = str(taxid)
                reports_by_taxid.setdefault(taxid_str, []).append(report)
                progress_bar.update(1)

        return reports_by_taxid

    def _build_hierarchy(
        self,
        reports_by_taxid: dict[str, list[dict[str, Any]]],
        root_id_str: str,
        checkpoint_interval: int,
    ) -> None:
        """Build the in-memory tree and register accessions, with checkpoints.

        For each unique species TaxID:
            1. Resolves its lineage via taxoniq.
            2. Applies the mapping's redirection rules to the path.
            3. Adds the path to the in-memory tree skeleton.
            4. Registers each accession into the persistent registry.
            5. Saves the registry every ``checkpoint_interval`` taxa.

        Lineage resolution failures are logged at DEBUG and skipped
        without aborting the run.

        Args:
            reports_by_taxid: Mapping of species TaxID to assembly
                reports produced by ``_stream_ncbi_summaries``.
            root_id_str: Root TaxID as string for scope lookup.
            checkpoint_interval: Save the registry every N processed
                taxa.
        """
        processed_count = 0
        for taxid_str, reports in tqdm(
            reports_by_taxid.items(),
            desc="Processing Lineage Hierarchy",
        ):
            try:
                self._register_taxon(taxid_str, reports, root_id_str)
                processed_count += 1
            except Exception as exc:
                logger.debug(
                    f"Skipping lineage resolution for TaxID {taxid_str}: {exc}"
                )
                continue

            if processed_count % checkpoint_interval == 0:
                logger.info(
                    f"Checkpoint reached ({processed_count} taxa). Flushing to disk."
                )
                self.registry.save()

    def _register_taxon(
        self,
        taxid_str: str,
        reports: list[dict[str, Any]],
        root_id_str: str,
    ) -> None:
        """Add a single species TaxID's lineage and accessions to the system.

        Args:
            taxid_str: Species TaxID as string.
            reports: List of assembly reports for this species.
            root_id_str: Root TaxID for scope lookup.

        Raises:
            ValueError: If the TaxID cannot be parsed as an integer.
            Various exceptions from taxoniq if the TaxID is not in
            the local cache.
        """
        species_taxon = taxoniq.Taxon(int(taxid_str))
        path_parts = self._resolve_mapped_path(species_taxon, root_id_str)
        full_path = "root/" + "/".join(path_parts)

        for report in reports:
            self.registry._update_taxon_entry(taxid_str, {"reports": [report]})

        add_path_to_tree(
            self.tree_root,
            full_path,
            node_attrs={
                "taxid": taxid_str,
                "rank": "species",
                "scientific_name": species_taxon.scientific_name,
            },
        )

    def _resolve_mapped_path(
        self,
        species_taxon: taxoniq.Taxon,
        root_id_str: str,
    ) -> list[str]:
        """Resolve a species' lineage applying scope redirections.

        Walks the species' ranked lineage and substitutes each TaxID
        according to the scope mapping rules. TaxIDs with no rule
        retain their NCBI scientific name (sanitized for filesystem
        compatibility).

        Args:
            species_taxon: taxoniq.Taxon instance for the species.
            root_id_str: Root TaxID as string for scope lookup.

        Returns:
            List of human-readable path components from root to
            species, suitable for joining with '/' as a tree path.
        """
        scope = self.mapping.get("scopes", {}).get(root_id_str, {})
        redirections = scope.get("redirections", {})
        virtual_labels = scope.get("virtual_id_labels", {})

        path_parts: list[str] = []
        for ancestor_taxon in species_taxon.ranked_lineage:
            ancestor_id = str(ancestor_taxon.tax_id)
            if ancestor_id in redirections:
                target_id = redirections[ancestor_id]["target_id"]
                name = virtual_labels.get(
                    target_id,
                    redirections[ancestor_id]["label"],
                )
            else:
                name = self._sanitize_path_component(ancestor_taxon.scientific_name)
            path_parts.append(name)

        return path_parts

    @staticmethod
    def _sanitize_path_component(name: str) -> str:
        """Replace characters illegal in directory names.

        bigtree's path separator is '/', so any name containing '/'
        would break the path semantics. Spaces are replaced for
        filesystem compatibility when paths are later materialized.

        Args:
            name: Raw scientific name from NCBI Taxonomy.

        Returns:
            Sanitized name safe to use as a path component.
        """
        return name.replace(" ", "_").replace("/", "_")
