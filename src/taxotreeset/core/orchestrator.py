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

2. **Lineage resolution**: each unique leaf TaxID encountered (a
   species, or a rank below it such as a no_rank strain) is resolved
   against the local taxoniq cache to produce its full lineage from
   root to leaf. Lineages are then transformed
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
import tempfile
from typing import Any, NamedTuple

import taxoniq
from bigtree import Node, add_path_to_tree

from taxotreeset.ranks import CANONICAL_RANKS_SPECIES_TO_ROOT
from tqdm import tqdm

logger = logging.getLogger("TaxoTreeSet.Core.Orchestrator")

_DEFAULT_ASSEMBLY_LEVELS = "complete,chromosome"
_DEFAULT_CHECKPOINT_INTERVAL = 500

# NCBI Datasets labels the viral top rank "acellular_root"; taxoniq calls
# it superkingdom. Accept either as the lineage's root rank.
_DATASETS_SUPERKINGDOM_KEYS: tuple[str, ...] = (
    "superkingdom",
    "acellular_root",
)


class _Ancestor(NamedTuple):
    """A resolved lineage node: TaxID, canonical rank, and name."""

    tax_id: int
    rank: str
    scientific_name: str
_NCBI_API_KEY_ENV_VAR = "NCBI_API_KEY"
# TaxIDs per bulk `datasets summary taxonomy taxon` call when prefetching the
# NCBI lineage fallback (keeps each command line well under OS arg limits).
_TAXONOMY_BATCH_SIZE = 300


class DiscoveryOrchestrator:
    """Coordinate taxonomic discovery from a root taxon down to its leaf taxa.

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

    def __init__(
        self,
        registry: Any,
        mapping_config: dict[str, Any],
        all_ranks: bool = False,
    ) -> None:
        """Initialize the orchestrator with a registry and mapping config.

        Args:
            registry: NCBIRegistry instance for persisting accession
                metadata as it is discovered.
            mapping_config: Parsed contents of the scope mapping
                configuration JSON.
            all_ranks: When True, resolve each lineage at FULL NCBI
                granularity (subgenus, subfamily, suborder, clade, …) via
                taxoniq's ``lineage`` instead of the 8 canonical ranks from
                ``ranked_lineage``. The extra intermediate taxa become heads
                wherever they branch (single-child sub-ranks are still
                collapsed by passthroughs).
        """
        self.registry: Any = registry
        self.mapping: dict[str, Any] = mapping_config
        self.all_ranks: bool = all_ranks
        self.tree_root: Node = Node("root")
        # taxid -> its NCBI taxonomy json-line ("" = looked up, none found), shared
        # by the lineage + self-node fallbacks and warmed by prefetch_ncbi_taxonomy.
        self._ncbi_taxonomy_cache: dict[str, str] = {}

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

        Failure is tolerant, not fatal: a subprocess failure or an empty CLI
        result is logged and the method returns without modifying the registry;
        per-taxon lineage failures are logged and skipped (see
        ``_build_hierarchy``). The method therefore does not raise on the
        NCBI/streaming path.
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
            "leaf taxa. Commencing hierarchy building."
        )

        self._build_hierarchy(
            reports_by_taxid=reports_by_taxid,
            root_id_str=root_id_str,
            checkpoint_interval=checkpoint_interval,
        )

        self.registry.mark_updated()
        self.registry.save()
        logger.info("Metadata registration and tree construction completed.")

    def discover_from_reports(
        self,
        reports: list[dict[str, Any]],
        root_id_str: str,
        vault_lmdb_path: str | None = None,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
    ) -> None:
        """Register a *pre-acquired* accession set (grouped by host TaxID).

        The bottom-up counterpart to :meth:`discover_from_root`. Some scopes have
        no root taxon to walk top-down — most notably plasmids, which are not a
        taxon: each RefSeq plasmid record is assigned to its **host** organism, so
        acquisition starts from the plasmid accession *set* (parsed from the
        RefSeq plasmid release) and resolves each record's host lineage. This
        method takes the synthetic assembly reports that acquisition produced
        (``io/plasmid_release.record_to_report``), groups them by host TaxID, and
        reuses the same lineage-resolution + tree-build + registration path as the
        top-down walk (:meth:`_build_hierarchy`).

        When ``vault_lmdb_path`` is given, each registered accession is marked
        downloaded against that vault — the sequences were ingested directly from
        the release (standalone nucleotide accessions cannot go through the
        assembly-oriented ``datasets download`` path), so there is nothing left to
        fetch. Reports whose host lineage cannot be resolved are skipped by
        ``_build_hierarchy`` and thus never marked downloaded (a harmless vault
        orphan), consistent with the top-down path's tolerance.

        Args:
            reports: Flat list of synthetic assembly reports, each carrying its
                host TaxID at ``organism.tax_id``.
            root_id_str: Scope key for mapping redirections/labels (the host
                lineage still builds the full path when the scope has no rules).
            vault_lmdb_path: Vault the sequences were ingested into; when set,
                registered accessions are marked downloaded against it.
            checkpoint_interval: Registry save cadence, in processed taxa.
        """
        reports_by_taxid = self._group_reports_by_host(reports)
        if not reports_by_taxid:
            return

        logger.info(
            "Registering %d pre-acquired accession(s) across %d host taxa.",
            len(reports), len(reports_by_taxid),
        )
        self._build_hierarchy(
            reports_by_taxid=reports_by_taxid,
            root_id_str=root_id_str,
            checkpoint_interval=checkpoint_interval,
        )
        if vault_lmdb_path is not None:
            self._mark_reports_downloaded(reports_by_taxid, vault_lmdb_path)

        self.registry.mark_updated()
        self.registry.save()
        logger.info("Pre-acquired accession registration completed.")

    @staticmethod
    def _group_reports_by_host(
        reports: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group flat reports by their host TaxID string.

        Mirrors ``_consume_jsonlines_stream``'s grouping of assembly reports by
        ``organism.tax_id``. Reports with no host TaxID are dropped (nothing to
        place them under).
        """
        grouped: dict[str, list[dict[str, Any]]] = {}
        for report in reports:
            taxid = report.get("organism", {}).get("tax_id")
            if not taxid:
                continue
            grouped.setdefault(str(taxid), []).append(report)
        return grouped

    def _mark_reports_downloaded(
        self,
        reports_by_taxid: dict[str, list[dict[str, Any]]],
        vault_lmdb_path: str,
    ) -> None:
        """Flag each registered accession as already present in the vault.

        The plasmid sequence for an accession is ingested directly (its record id
        is the LMDB key), so the accession is complete the moment it is
        registered: mark it downloaded with a single header whose ``id`` is the
        vault key. Mirrors the downloader's post-batch registry update.
        """
        accessions = self.registry.registry["accessions"]
        marked = 0
        for reports in reports_by_taxid.values():
            for report in reports:
                accession = report.get("accession")
                info = accessions.get(accession)
                if info is None:
                    continue
                organism = report.get("organism", {}).get("organism_name")
                info["downloaded"] = True
                info["local_path"] = vault_lmdb_path
                info["headers"] = [{"id": accession, "name": organism or accession}]
                info.pop("download_attempts", None)
                marked += 1
        logger.info("Marked %d pre-acquired accession(s) as downloaded.", marked)

    def stream_reference_reports(
        self, taxid: str, limit: int,
    ) -> list[dict[str, Any]]:
        """Stream up to ``limit`` RefSeq *reference*-genome reports for a taxon.

        A bounded, reference-only variant of :meth:`_stream_ncbi_summaries` used to
        acquire a small cross-domain negative sample (the P4 non-virus gate)
        without crawling a whole domain: it requests ``--reference`` genomes and
        stops after ``limit`` reports, terminating the subprocess early. Returns an
        empty list on spawn/parse failure (tolerant, like discovery).

        Args:
            taxid: Domain/clade TaxID to sample reference genomes from.
            limit: Maximum reports to return (<= 0 returns nothing).

        Returns:
            Up to ``limit`` assembly report dicts, in stream order.
        """
        if limit <= 0:
            return []
        command = [
            "datasets", "summary", "genome", "taxon", str(taxid),
            "--assembly-source", "RefSeq", "--reference", "--as-json-lines",
        ]
        logger.info(
            "Sampling up to %d reference genome(s) under TaxID %s (cross-domain "
            "negatives).", limit, taxid,
        )
        reports: list[dict[str, Any]] = []
        stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            try:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=stderr_file,
                    text=True, env=os.environ.copy(), bufsize=1,
                )
            except OSError as exc:
                logger.error("Failed to spawn NCBI CLI for cross-domain sample: %s", exc)
                return []
            with process:
                for line in process.stdout or []:
                    if not line.strip():
                        continue
                    try:
                        reports.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(reports) >= limit:
                        process.terminate()  # bounded: stop the stream early
                        break
        finally:
            stderr_file.close()
        return reports

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
        report; reports are grouped by leaf (organism) TaxID.

        Args:
            root_id_str: Root TaxID as a string.
            assembly_levels: Comma-separated NCBI assembly levels.

        Returns:
            Dictionary mapping leaf TaxID strings to their list of
            assembly report dictionaries. Empty dict on subprocess
            failure.
        """
        command = self._build_summary_command(root_id_str, assembly_levels)
        logger.info(f"Spawning NCBI Datasets CLI subprocess for TaxID: {root_id_str}")

        env = os.environ.copy()
        # Drain stderr to a temp file instead of a PIPE. We consume only stdout
        # while the child runs, so a stderr PIPE that the child fills past its
        # ~64 KiB buffer (plausible on a large-domain run's progress output)
        # would block the child while we block reading stdout — a deadlock. A
        # file never blocks the writer; we read it back only on failure.
        stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    text=True,
                    env=env,
                    bufsize=1,
                )
            except OSError as exc:
                logger.error(f"Failed to spawn NCBI Datasets CLI subprocess: {exc}")
                return {}

            # `with process` closes stdout and waits on exit.
            with process:
                reports_by_taxid = self._consume_jsonlines_stream(process)

            if not reports_by_taxid:
                stderr_file.seek(0)
                stderr_output = stderr_file.read().decode("utf-8", errors="replace")
                logger.error(
                    "NCBI streaming process returned no data: %s",
                    stderr_output.strip(),
                )
            return reports_by_taxid
        finally:
            stderr_file.close()

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
            Dictionary mapping leaf TaxID strings to assembly
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

        For each unique leaf TaxID:
            1. Resolves its lineage via taxoniq, falling back to the
               NCBI taxonomy CLI for TaxIDs newer than the snapshot.
            2. Applies the mapping's redirection rules to the path.
            3. Adds the path to the in-memory tree skeleton.
            4. Registers each accession into the persistent registry.
            5. Saves the registry every ``checkpoint_interval`` taxa.

        Lineage resolution failures are logged at DEBUG and skipped
        without aborting the run.

        Args:
            reports_by_taxid: Mapping of leaf TaxID to assembly
                reports produced by ``_stream_ncbi_summaries``.
            root_id_str: Root TaxID as string for scope lookup.
            checkpoint_interval: Save the registry every N processed
                taxa.
        """
        # Warm the NCBI taxonomy cache in bulk for the leaves taxoniq cannot
        # resolve, so the per-taxon fallbacks below become cache hits instead of a
        # subprocess each (the registration bottleneck for host-heavy sets).
        self.prefetch_ncbi_taxonomy(
            [t for t in reports_by_taxid if self._needs_ncbi_fallback(t)]
        )

        processed_count = 0
        skipped_count = 0
        for taxid_str, reports in tqdm(
            reports_by_taxid.items(),
            desc="Processing Lineage Hierarchy",
        ):
            try:
                self._register_taxon(taxid_str, reports, root_id_str)
                processed_count += 1
            except Exception as exc:
                skipped_count += 1
                logger.debug(
                    f"Skipping lineage resolution for TaxID {taxid_str}: {exc}"
                )
                continue

            if processed_count % checkpoint_interval == 0:
                logger.info(
                    f"Checkpoint reached ({processed_count} taxa). Flushing to disk."
                )
                self.registry.save()

        # Surface systematic failures: a few skips are normal (unresolvable taxa),
        # but skips outnumbering successes usually means something is wrong (bad
        # config, taxoniq/CLI unavailable) rather than a per-taxon anomaly.
        if skipped_count:
            summary = (
                "Hierarchy build: %d taxa registered, %d skipped "
                "(lineage resolution failures)."
            )
            if skipped_count >= processed_count:
                logger.warning(summary, processed_count, skipped_count)
            else:
                logger.info(summary, processed_count, skipped_count)

    def _register_taxon(
        self,
        taxid_str: str,
        reports: list[dict[str, Any]],
        root_id_str: str,
    ) -> None:
        """Add a single leaf taxon's lineage and accessions to the system.

        Args:
            taxid_str: Leaf/organism TaxID as string (a species or a rank
                below it, e.g. a no_rank strain).
            reports: List of assembly reports for this taxon.
            root_id_str: Root TaxID for scope lookup.

        Raises:
            ValueError: If the TaxID cannot be parsed as an integer.
            RuntimeError: If the lineage cannot be resolved by either
                taxoniq or the NCBI taxonomy fallback.
        """
        lineage = self._resolve_lineage(int(taxid_str))
        # Record the leaf taxon itself when it is non-canonical (e.g. a
        # no_rank strain) and thus absent from its own ranked lineage, so
        # tree building can label that node instead of leaving it unknown.
        if not lineage or str(lineage[0].tax_id) != taxid_str:
            self_node = self._resolve_self_node(int(taxid_str))
            if self_node is not None and str(self_node.tax_id) == taxid_str:
                lineage = [self_node, *lineage]
        path_parts = self._resolve_mapped_path(lineage, root_id_str)
        full_path = "root/" + "/".join(path_parts)

        self.registry.store_lineage(
            taxid_str,
            [
                {
                    "taxid": str(ancestor.tax_id),
                    "rank": ancestor.rank,
                    "name": ancestor.scientific_name,
                }
                for ancestor in lineage
            ],
        )
        for report in reports:
            self.registry._update_taxon_entry(taxid_str, {"reports": [report]})

        add_path_to_tree(
            self.tree_root,
            full_path,
            node_attrs={
                "taxid": taxid_str,
                "rank": lineage[0].rank,
                "scientific_name": lineage[0].scientific_name,
            },
        )

    def _resolve_lineage(self, taxid: int) -> list[_Ancestor]:
        """Resolve a taxon's canonical lineage, from that taxon to root.

        Tries the local taxoniq cache first; on a cache miss (a TaxID
        newer than taxoniq's snapshot) falls back to a live NCBI Datasets
        CLI taxonomy lookup. Both paths yield the same canonical rank set
        and order so downstream tree paths stay consistent.

        Args:
            taxid: Leaf/organism TaxID to resolve — usually a species,
                often a rank below it (e.g. a no_rank strain).

        Returns:
            Ancestors from the taxon to root.

        Raises:
            RuntimeError: If neither taxoniq nor the NCBI fallback can
                resolve the TaxID.
        """
        try:
            taxon = taxoniq.Taxon(taxid)
            ancestors = (
                taxon.lineage if self.all_ranks
                else taxon.ranked_lineage
            )
            return [
                _Ancestor(
                    int(a.tax_id),
                    a.rank.name,
                    a.scientific_name,
                )
                for a in ancestors
            ]
        except KeyError:
            lineage = self._fetch_lineage_via_ncbi(taxid)
            if not lineage:
                raise RuntimeError(
                    f"Could not resolve lineage for TaxID {taxid} via "
                    "taxoniq or the NCBI taxonomy fallback."
                )
            return lineage

    def _resolve_self_node(self, taxid: int) -> _Ancestor | None:
        """Resolve a taxon's own name and rank (not its ancestors).

        Used to record the leaf taxon itself in its stored lineage when
        it is non-canonical (e.g. a no_rank strain below species), which
        ranked_lineage omits. Lets tree building label that node from the
        registry instead of leaving it unknown.

        Args:
            taxid: TaxID to resolve.

        Returns:
            The taxon as an _Ancestor, or None if it cannot be resolved.
        """
        try:
            taxon = taxoniq.Taxon(taxid)
            return _Ancestor(int(taxon.tax_id), taxon.rank.name, taxon.scientific_name)
        except Exception:
            return self._fetch_self_node_via_ncbi(taxid)

    def _fetch_self_node_via_ncbi(self, taxid: int) -> _Ancestor | None:
        """Resolve a taxon's own name and rank via the NCBI Datasets CLI.

        Args:
            taxid: TaxID to resolve.

        Returns:
            The taxon as an _Ancestor, or None if the lookup yields
            nothing usable.
        """
        line = self._ncbi_taxonomy_line(taxid)
        if not line:
            return None
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        taxonomy = payload.get("taxonomy", {})
        current = taxonomy.get("current_scientific_name", {})
        name = current.get("name")
        rank = taxonomy.get("rank") or "no_rank"
        resolved_id = taxonomy.get("tax_id")
        if name and resolved_id is not None:
            return _Ancestor(int(resolved_id), str(rank).lower(), str(name))
        return None

    @staticmethod
    def _needs_ncbi_fallback(taxid_str: str) -> bool:
        """True if ``taxid_str`` is absent from taxoniq's snapshot (needs the CLI).

        A non-integer key returns False — it is not a taxon the bulk prefetch can
        resolve, and the registration loop handles/skips it.
        """
        try:
            taxoniq.Taxon(int(taxid_str))
            return False
        except KeyError:
            return True
        except (ValueError, TypeError):
            return False

    def prefetch_ncbi_taxonomy(self, taxids: list[str]) -> None:
        """Warm the NCBI taxonomy cache for many taxa in a few bulk calls.

        The per-taxon lineage fallback (``_fetch_lineage_via_ncbi`` /
        ``_fetch_self_node_via_ncbi``) otherwise spawns **one subprocess per
        taxon** — the dominant cost when registering an accession set with many
        hosts newer than taxoniq's snapshot (e.g. the RefSeq-plasmid host tree,
        thousands of recent bacteria). This batches them into
        ``_TAXONOMY_BATCH_SIZE``-sized ``datasets summary taxonomy taxon <ids>``
        calls and caches each reply line by its tax_id, so the registration loop's
        fallbacks become cache hits. Best-effort: a taxon the bulk reply omits
        (e.g. a merged id) simply falls back to its own single call later.
        """
        pending = [
            t for t in dict.fromkeys(taxids) if t not in self._ncbi_taxonomy_cache
        ]
        if not pending:
            return
        logger.info(
            "Prefetching NCBI taxonomy for %s taxa in batches of %d.",
            f"{len(pending):,}", _TAXONOMY_BATCH_SIZE,
        )
        for start in range(0, len(pending), _TAXONOMY_BATCH_SIZE):
            batch = pending[start:start + _TAXONOMY_BATCH_SIZE]
            resolved = self._run_taxonomy_query(batch)
            for key in batch:
                # Cache "" for misses so a repeat query is not re-attempted in bulk.
                self._ncbi_taxonomy_cache[key] = resolved.get(key, "")

    def _ncbi_taxonomy_line(self, taxid: int) -> str | None:
        """The NCBI taxonomy json-line for ``taxid``, cached; single call on miss."""
        key = str(taxid)
        if key not in self._ncbi_taxonomy_cache:
            self._ncbi_taxonomy_cache[key] = self._run_taxonomy_query([key]).get(key, "")
        return self._ncbi_taxonomy_cache[key] or None

    @staticmethod
    def _run_taxonomy_query(taxids: list[str]) -> dict[str, str]:
        """One ``datasets summary taxonomy taxon <ids>`` call -> {tax_id: json-line}."""
        command = [
            "datasets", "summary", "taxonomy", "taxon", *taxids, "--as-json-lines",
        ]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.debug(
                "NCBI taxonomy query failed for %d taxa: %s", len(taxids), exc)
            return {}
        lines: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            tax_id = payload.get("taxonomy", {}).get("tax_id")
            if tax_id is not None:
                lines[str(tax_id)] = line
        return lines

    def _fetch_lineage_via_ncbi(self, taxid: int) -> list[_Ancestor]:
        """Fetch a taxon's lineage from the NCBI Datasets CLI.

        Fallback when taxoniq's static snapshot does not know a recently
        classified TaxID. In ``--all-ranks`` mode the ordered ``taxonomy.parents``
        array is used — it carries the non-canonical (no_rank/clade/sub*)
        ancestors the rank-keyed ``classification`` dict cannot — and those
        ancestors predate the cache miss, so taxoniq resolves them into the same
        all-ranks lineage the primary path produces. Otherwise, or if ``parents``
        cannot be resolved, the canonical ``classification`` ranks are used.

        Args:
            taxid: Leaf/organism TaxID to resolve — usually a species,
                often a rank below it (e.g. a no_rank strain).

        Returns:
            The taxon's ancestors, deepest-first (all ranks when ``all_ranks`` and
            ``parents`` resolves, else canonical species-to-root), or an empty list
            if the CLI returns nothing usable.
        """
        line = self._ncbi_taxonomy_line(taxid)
        if not line:
            return []
        taxonomy = self._parse_taxonomy(line)
        if not taxonomy:
            return []

        # --all-ranks: the canonical `classification` dict is keyed by rank name and
        # cannot carry the non-canonical (no_rank/clade/sub*) intermediates, but the
        # ordered `parents` array can. Resolve those ancestors through taxoniq (they
        # predate the cache miss, so taxoniq knows them) for the same all-ranks
        # lineage the primary path yields.
        if self.all_ranks:
            lineage = self._lineage_from_parents(taxonomy.get("parents", []))
            if lineage:
                return lineage
            logger.warning(
                "NCBI all-ranks fallback for TaxID %s could not resolve `parents`; "
                "using canonical ranks only for this leaf.", taxid,
            )

        classification = taxonomy.get("classification")
        if not classification:
            return []
        lineage = []
        for rank in CANONICAL_RANKS_SPECIES_TO_ROOT:
            node = self._classification_node_for_rank(classification, rank)
            if node is not None:
                lineage.append(_Ancestor(int(node["id"]), rank, str(node["name"])))
        return lineage

    def _lineage_from_parents(self, parents: list) -> list[_Ancestor]:
        """Full all-ranks ancestor lineage from the CLI's ordered ``parents`` array.

        ``parents`` is ordered root -> immediate parent and includes the
        non-canonical ranks (no_rank/clade/sub*) the canonical classification omits.
        The ancestors predate the taxoniq cache miss that triggered this fallback,
        so taxoniq resolves them: the deepest resolvable ancestor's taxoniq
        ``.lineage`` (which is itself + all its ancestors, deepest-first) is exactly
        the shape the primary path produces. The leaf itself is not in ``parents`` —
        ``_register_taxon`` records it via ``_resolve_self_node``.

        Args:
            parents: Ordered ancestor TaxIDs from the CLI taxonomy reply.

        Returns:
            Ancestors from the deepest resolvable parent to root, deepest-first,
            or an empty list if none resolve (e.g. ``parents`` missing).
        """
        for parent_taxid in reversed(parents):
            try:
                ancestor = taxoniq.Taxon(int(parent_taxid))
                return [
                    _Ancestor(int(a.tax_id), a.rank.name, a.scientific_name)
                    for a in ancestor.lineage
                ]
            except Exception:
                continue
        return []

    @staticmethod
    def _parse_taxonomy(stdout: str) -> dict[str, Any] | None:
        """Extract the taxonomy object from a taxonomy JSON-lines reply.

        The taxonomy object carries both the canonical ``classification`` dict and
        the ordered ``parents`` ancestor array the all-ranks fallback needs.

        Args:
            stdout: Raw stdout from the datasets taxonomy command.

        Returns:
            The first non-empty ``taxonomy`` object, or None if the reply has no
            parsable report.
        """
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            # With --as-json-lines each line is a report itself, with
            # taxonomy at the top level (no enclosing "reports" array).
            taxonomy = payload.get("taxonomy")
            if taxonomy:
                return taxonomy
        return None

    @staticmethod
    def _parse_taxonomy_classification(stdout: str) -> dict[str, Any] | None:
        """Extract the classification dict from a taxonomy JSON-lines reply.

        Args:
            stdout: Raw stdout from the datasets taxonomy command.

        Returns:
            The classification mapping (rank name to {id, name}), or None
            if the reply has no parsable report.
        """
        taxonomy = DiscoveryOrchestrator._parse_taxonomy(stdout)
        return taxonomy.get("classification") if taxonomy else None

    @staticmethod
    def _classification_node_for_rank(
        classification: dict[str, Any],
        rank: str,
    ) -> dict[str, Any] | None:
        """Return the {id, name} node for a canonical rank, if present.

        Handles the viral top rank, which the CLI labels
        ``acellular_root`` where taxoniq uses ``superkingdom``.

        Args:
            classification: Rank-to-node mapping from the CLI.
            rank: Canonical rank name to look up.

        Returns:
            The rank's node, or None if absent.
        """
        if rank == "superkingdom":
            for key in _DATASETS_SUPERKINGDOM_KEYS:
                node = classification.get(key)
                if node is not None:
                    return node
            return None
        return classification.get(rank)

    def _resolve_mapped_path(
        self,
        lineage: list[_Ancestor],
        root_id_str: str,
    ) -> list[str]:
        """Resolve a leaf taxon's lineage applying scope redirections.

        Walks the taxon's lineage — species-to-root, or deeper when the leaf
        is a rank below species — and substitutes each TaxID according to the
        scope mapping rules. TaxIDs with no rule retain their NCBI scientific
        name (sanitized for filesystem compatibility).

        Args:
            lineage: Ancestors from the leaf taxon to root, as produced by
                taxoniq or by the NCBI-CLI fallback.
            root_id_str: Root TaxID as string for scope lookup.

        Returns:
            List of human-readable path components from root to the leaf
            taxon, suitable for joining with '/' as a tree path.
        """
        scope = self.mapping.get("scopes", {}).get(root_id_str, {})
        redirections = scope.get("redirections", {})
        virtual_labels = scope.get("virtual_id_labels", {})

        path_parts: list[str] = []
        for ancestor in lineage:
            ancestor_id = str(ancestor.tax_id)
            if ancestor_id in redirections:
                target_id = redirections[ancestor_id]["target_id"]
                name = virtual_labels.get(
                    target_id,
                    redirections[ancestor_id]["label"],
                )
            else:
                name = self._sanitize_path_component(ancestor.scientific_name)
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
