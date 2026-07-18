"""Generation orchestrator: cascaded LoRA dataset materialization.

This module is the orchestration entry point of the TaxoTreeSet
generation phase. It coordinates four major stages in sequence:

1. **Download pending accessions** via the NCBIDownloader, ensuring
   the LMDB vault holds every genome the registry expects.

2. **Build the taxonomic tree** via ``generate_seqs_by_taxon_tree``,
   applying noise filtering and scope redirections from the mapping
   configuration.

3. **Schedule extraction jobs** by walking the tree depth-first and,
   at each decision point, applying rank-aware bucketing, per-class
   balancing, low-capacity bucket materialization, and n-per-class
   distribution across the children's sequence leaves.

4. **Dispatch parallel disk extraction** to a spawn-based worker
   pool that materializes Parquet shards (train/val/test) for every
   trainable head.

The actual algorithms (capacity computation, balancing, bucketing,
distribution) live in the ``generation`` subpackage, which the
orchestrator imports and composes. This module is the "glue" that
holds the pipeline together, owns mutable state (the manifests,
passthrough map, virtual ID registry), and delegates pure
computation to the subpackage.

See ``docs/GLOSSARY.md`` for definitions of architectural terms
used throughout (rank-aware bucketing, low-capacity bucketing,
passthrough, cascade terminator).

Typical usage::

    from taxotreeset.core.generation_orchestrator import (
        GenerationOrchestrator,
    )
    from taxotreeset.io.registry import NCBIRegistry

    registry = NCBIRegistry(registry_path="data/registry.json")
    pipeline = GenerationOrchestrator(
        registry=registry,
        config_path="configs/mapping.json",
        vault_path="data/vault",
        output_dir="data/datasets",
        max_subseq_len=2000,
        seed=42,
        output_format="parquet",
    )
    pipeline.run_pipeline(target_group="viruses", abundance_threshold=2)
"""

import json
import logging
import os
import random
import time
from typing import Any

from bigtree import Node

from taxotreeset.core.generation import (
    build_reject_tasks,
    classify_children_by_rank,
    compute_balanced_extraction_plan,
    distribute_n_per_class_across_leaves,
    make_low_capacity_bucket_node,
    make_rare_taxa_bucket_node,
    make_reject_bucket_node,
    register_virtual_bucket,
    sample_reject_leaves,
)
from taxotreeset.core.generation.capacity import compute_all_capacities
from taxotreeset.core.generation.constants import (
    DEFAULT_CUTOFF_PERCENTAGE,
    DEFAULT_MAX_N_PER_CLASS,
    DEFAULT_MIN_LEAVES_PER_CLASS,
    DEFAULT_MIN_NUM_SEQS,
    DEFAULT_RARE_TAXA_STRATEGY,
    DEFAULT_USE_EXACT_CAPACITY,
    is_recursion_terminator,
)
from taxotreeset.dataset.builder import DatasetBuilder
from taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from tqdm import tqdm

from taxotreeset.logging_utils import get_ui_logger
from taxotreeset.ranks import (
    CANONICAL_RANKS_ROOT_TO_SPECIES,
    is_below_boundary,
    is_canonical_rank,
)
from taxotreeset.taxonomy import resolve_to_taxid
from taxotreeset.core.orchestrator import DiscoveryOrchestrator
from taxotreeset.io.downloader import NCBIDownloader
from taxotreeset.core._orchestration._splits import (
    _SPLITS,
    _materialize_leaf_split as _materialize_leaf_split_fn,
    _stratified_counts as _stratified_counts,
    _stratified_cuts as _stratified_cuts,
)

# Force single-threaded execution in BLAS/MKL backends. Multi-threaded
# C libraries can deadlock or segfault under the spawn-based worker
# pool used by the DatasetBuilder, especially when sklearn and numpy
# are imported in workers. Setting these env vars BEFORE importing
# numpy is essential; doing it later has no effect because the
# libraries read the variables at import time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")
ui_logger = get_ui_logger()

_SELECTIVE_DOWNLOAD_THRESHOLD_BYTES: int = 50 * 1024 ** 3  # 50 GiB
_MAX_REFINEMENT_ROUNDS: int = 5

_DOMAIN_GROUP_TO_TAXID: dict[str, str] = {
    "viruses": "10239",
    "bacteria": "2",
    "archaea": "2157",
    "eukaryotes": "2759",
}

# Special root meaning "every domain": no anchor TaxID -- the whole registry is
# in scope. Resolves to a None domain_taxid throughout the pipeline.
_ALL_DOMAINS = "all"


def _capture_tool_versions() -> dict[str, str]:
    """Capture versions of the external tools that determine the data snapshot.

    Records the NCBI ``datasets`` CLI, the taxoniq taxonomy package, and the
    Python runtime, so a generated dataset can be reproduced against the same
    tooling. Each lookup degrades to ``"unknown"`` on failure (e.g. the CLI is
    absent on a ``--no-sync`` run).

    Returns:
        Mapping of tool name to version string.
    """
    import platform
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        completed = subprocess.run(
            ["datasets", "--version"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        datasets_cli = (completed.stdout or completed.stderr).strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        datasets_cli = "unknown"

    try:
        taxoniq_version = _pkg_version("taxoniq")
    except PackageNotFoundError:
        taxoniq_version = "unknown"

    return {
        "datasets_cli": datasets_cli,
        "taxoniq": taxoniq_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _fmt_elapsed(secs: float) -> str:
    if secs < 90:
        return f"{secs:.0f}s"
    if secs < 5400:
        return f"{secs / 60:.0f}min"
    return f"{secs / 3600:.1f}h"


class GenerationOrchestrator:
    """Coordinate the full dataset generation workflow.

    Owns the registry, the in-memory taxonomic tree, the downloader,
    and the dataset builder. The single public entry point,
    ``run_pipeline``, executes the four-stage workflow described in
    the module docstring.

    Most heavy lifting is delegated to functions in the
    ``generation`` subpackage. This class focuses on orchestration
    state (manifests, passthrough map, virtual ID registry) and
    on the recursive tree walk that schedules extraction jobs.

    Attributes:
        registry: NCBIRegistry instance providing accession metadata.
        config_path: Path to the scope mapping configuration JSON.
        vault_path: Path to the LMDB sequence vault.
        output_dir: Root directory where Parquet shards are written.
        max_subseq_len: Maximum subsequence length in base pairs.
        seed: Random seed for reproducible splits.
        output_format: 'parquet' (production) or 'csv' (debug).
        min_subclades_per_bucket: Threshold for rank-aware bucketing.
        min_num_seqs: Threshold for triggering cutoff scenario.
        cutoff_percentage: Percentile retained when cutoff applies.
        max_n_per_class: Hard ceiling on n_per_class.
        use_exact_capacity: True for set union, False for Bloom.
        downloader: NCBIDownloader instance.
        builder: DatasetBuilder instance.
    """

    def __init__(
        self,
        registry: Any,
        vault_path: str,
        output_dir: str,
        config_path: str = "configs/mapping.json",
        max_subseq_len: int = 2000,
        min_subseq_len: int = 100,
        seed: int = 42,
        output_format: str = "parquet",
        min_subclades_per_bucket: int = 5,
        min_num_seqs: int = DEFAULT_MIN_NUM_SEQS,
        cutoff_percentage: float = DEFAULT_CUTOFF_PERCENTAGE,
        max_n_per_class: int = DEFAULT_MAX_N_PER_CLASS,
        use_exact_capacity: bool = DEFAULT_USE_EXACT_CAPACITY,
        min_leaves_per_class: int = DEFAULT_MIN_LEAVES_PER_CLASS,
        rare_taxa_strategy: str = DEFAULT_RARE_TAXA_STRATEGY,
        selective_download_threshold: int = _SELECTIVE_DOWNLOAD_THRESHOLD_BYTES,
        spill_dir: str | None = None,
        tmp_dir: str | None = None,
        n_workers: int | None = None,
        n_gpu_workers: int | None = None,
        exclude_plasmids: bool = False,
        reject_class: bool = False,
        reject_fraction: float = 1.0,
        reject_near_far_start: float = 0.5,
        reject_near_far_end: float = 0.9,
        binary_only: bool = False,
        binary_budget: int = 30000,
        binary_extract_batch_size: int = 300,
        all_ranks: bool = False,
    ) -> None:
        """Initialize the orchestrator and its collaborating components.

        Args:
            registry: NCBIRegistry from the discovery phase.
            vault_path: Directory hosting the LMDB sequence vault.
            output_dir: Where Parquet shards are written.
            config_path: Path to the scope mapping JSON.
            max_subseq_len: Upper bound on subseq length, in bp.
            min_subseq_len: Lower bound on subseq length and the
                sliding-window size for capacity measurement, in bp.
            seed: Random seed for reproducible splits and sampling.
            output_format: 'parquet' or 'csv'.
            min_subclades_per_bucket: Minimum subclade count for a
                non-canonical rank to receive its own virtual bucket.
            min_num_seqs: Threshold below which the cutoff scenario
                is triggered in the balancing layer.
            cutoff_percentage: Percentile of children retained when
                cutoff applies.
            max_n_per_class: Hard ceiling on n_per_class.
            use_exact_capacity: True for exact set union (precise),
                False for Bloom filter (constant memory).
            min_leaves_per_class: Minimum sequence-leaf count for a
                child to remain a standalone training label. Children
                below this floor are diverted to a rare-taxa bucket
                under the 'fallback' strategy.
            rare_taxa_strategy: 'fallback' to divert low-leaf children
                into a virtual_rare_taxa bucket, or 'keep' to retain
                every child regardless of leaf count.
            selective_download_threshold: Total pending download volume
                in bytes above which the selective download selection
                pass is activated. Below this threshold, all pending
                accessions are downloaded as usual. Defaults to 50 GiB,
                which sits above bacteria-scale (~27 GB) and well below
                eukaryota-scale (~3 TB).
            tmp_dir: Directory for temporary download archives. Passed
                to NCBIDownloader; defaults to the OS temp dir when
                None. Point to a large drive to keep the system drive
                from being inflated by transient ZIP extraction.
            n_workers: CPU worker processes for the parallel leaf phase
                of the bottom-up capacity pass. Defaults to cpu_count - 1
                when None. Pass 1 to disable CPU parallelism.
            n_gpu_workers: GPU worker processes for large leaves (requires
                CuPy). Defaults to auto-detect (all CUDA devices). Pass 0
                to disable GPU acceleration.
            exclude_plasmids: Drop plasmid sequences at ingestion (passed to
                NCBIDownloader).
            reject_class: When True, append a ``virtual_reject`` class to every
                head, trained on sequence leaves sampled from outside the head's
                subtree (near siblings + far clades). Teaches the head to reject
                mis-routed / out-of-distribution inputs instead of forcing them
                into a real class. Opt-in; off leaves generation unchanged.
            reject_fraction: Size of the reject class relative to ``n_per_class``
                (1.0 = balanced with the real classes).
            reject_near_far_start: Fraction of reject windows drawn from the
                nearest ancestor's sibling clades (near; the rest from farther
                clades) at the SHALLOWEST reject-eligible head (the root's
                children). Shallow heads see diverse intruders, so this is low.
            reject_near_far_end: The same fraction at the DEEPEST head. Upstream
                heads prune distant intruders, so a deep head mostly meets near
                (sibling) intruders — this is high (near-heavy). The near fraction
                is linearly interpolated between start and end by the head's depth
                (relative to the tree's depth). Set ``end == start`` for a flat,
                depth-independent ratio.
        """
        self.registry: Any = registry
        self.config_path: str = config_path
        self.vault_path: str = vault_path
        self.output_dir: str = output_dir
        self.max_subseq_len: int = max_subseq_len
        if min_subseq_len > max_subseq_len:
            raise ValueError(
                f"min_subseq_len ({min_subseq_len}) cannot exceed "
                f"max_subseq_len ({max_subseq_len})."
            )
        self.min_subseq_len: int = min_subseq_len
        self.seed: int = seed
        self.output_format: str = output_format
        self.min_subclades_per_bucket: int = min_subclades_per_bucket
        self.min_num_seqs: int = min_num_seqs
        self.cutoff_percentage: float = cutoff_percentage
        self.max_n_per_class: int = max_n_per_class
        self.use_exact_capacity: bool = use_exact_capacity
        self.min_leaves_per_class: int = min_leaves_per_class
        self.rare_taxa_strategy: str = rare_taxa_strategy
        self.reject_class: bool = reject_class
        self.reject_fraction: float = reject_fraction
        self.reject_near_far_start: float = reject_near_far_start
        self.reject_near_far_end: float = reject_near_far_end
        self.binary_only: bool = binary_only
        self.binary_budget: int = binary_budget
        self.binary_extract_batch_size: int = binary_extract_batch_size
        self.all_ranks: bool = all_ranks
        self.selective_download_threshold: int = selective_download_threshold
        self.spill_dir: str | None = spill_dir
        self.tmp_dir: str | None = tmp_dir
        self.n_workers: int | None = n_workers
        self.n_gpu_workers: int | None = n_gpu_workers
        self._schedule_pbar = None
        self._selective_download_active: bool = False
        self._depth_boundary: str | None = None
        self._single_level: bool = False
        self._all_capacities: dict[str, int] | None = None

        self.downloader: NCBIDownloader = NCBIDownloader(
            registry=self.registry,
            vault_path=self.vault_path,
            tmp_dir=self.tmp_dir,
            exclude_plasmids=exclude_plasmids,
        )
        self.builder: DatasetBuilder = DatasetBuilder(
            output_dir=self.output_dir,
            max_subseq_len=self.max_subseq_len,
            seed=self.seed,
            output_format=self.output_format,
            min_subseq_len=self.min_subseq_len,
        )

    def _sync_with_ncbi(self, target_group: str) -> None:
        """Reconcile the registry and vault with NCBI for a scope.

        Re-runs discovery for the target group's domain so that new NCBI
        accessions enter the registry as pending, then reconciles the
        vault: accessions marked downloaded whose recorded headers are
        missing from the LMDB are reset to pending for re-download.

        When the total pending download volume meets or exceeds
        ``selective_download_threshold``, a selection pass is run to
        mark low-priority accessions as deferred so Stage 1 only
        downloads the subset needed to satisfy the balancing targets.

        Args:
            target_group: Domain identifier to synchronize.
        """
        domain_taxid = self._resolve_root_taxid(target_group)
        with open(self.config_path, encoding="utf-8") as handle:
            mapping_config = json.load(handle)
        discovery = DiscoveryOrchestrator(
            registry=self.registry,
            mapping_config=mapping_config,
            all_ranks=self.all_ranks,
        )
        if domain_taxid is None:
            # "all": re-discover every domain already present in the registry,
            # so a single-domain registry is not surprised by an unrelated
            # full-domain crawl. Falls back to all four when empty.
            for dom_taxid in self._domains_to_sync():
                discovery.discover_from_root(int(dom_taxid))
        else:
            discovery.discover_from_root(int(domain_taxid))
        self._reconcile_vault_against_registry()

        pending_volume = self.registry.get_pending_volume(domain_taxid)
        gib = pending_volume / 1024 ** 3
        threshold_gib = self.selective_download_threshold / 1024 ** 3
        if pending_volume >= self.selective_download_threshold:
            ui_logger.info(
                f"Pending volume {gib:.1f} GiB exceeds the "
                f"{threshold_gib:.1f} GiB threshold. "
                "Running selective download selection pass."
            )
            self._run_selective_download(domain_taxid)
        else:
            ui_logger.info(
                f"Pending volume {gib:.1f} GiB is below the "
                f"{threshold_gib:.1f} GiB threshold; "
                "all pending accessions will be downloaded."
            )

    def _domains_to_sync(self) -> list[str]:
        """Return the superkingdom TaxIDs to re-discover for an ``all`` sync.

        Restricts to the domains already represented in the registry's
        stored lineages, so syncing ``all`` over a single-domain registry
        does not trigger an unrelated full-domain crawl. Falls back to all
        four superkingdoms when the registry has no lineages yet.
        """
        lineages = self.registry.registry.get("lineages", {})
        present = [
            taxid
            for taxid in _DOMAIN_GROUP_TO_TAXID.values()
            if any(
                any(a.get("taxid") == taxid for a in stored)
                for stored in lineages.values()
            )
        ]
        return present or list(_DOMAIN_GROUP_TO_TAXID.values())

    def _run_selective_download(self, domain_taxid: str | None) -> None:
        """Run the estimation pass and defer accessions not needed for Phase 1.

        Builds an estimation tree from stored lineages (no vault access
        required), injects total_sequence_length as a capacity proxy,
        determines per-label n_per_class targets via the standard
        balancing layer, and marks all pending accessions that are not
        needed to satisfy those targets as deferred.

        Args:
            domain_taxid: Root TaxID of the scope being processed.
        """
        self._selective_download_active = True
        self.registry.reset_selection_flags(domain_taxid)

        estimation_tree = self._build_target_tree(domain_taxid)
        if estimation_tree is None or not estimation_tree.children:
            ui_logger.warning(
                "Could not build estimation tree for selective download; "
                "all pending accessions will be downloaded."
            )
            return

        estimated_capacities = self._estimate_capacities_from_registry(domain_taxid)
        downloaded_cap, pending_index = self._build_scope_accession_index(domain_taxid)

        domain_node = self._find_domain_node(estimation_tree, domain_taxid)
        if domain_node is None:
            return

        label_targets: dict[str, int] = {}
        self._collect_label_targets(
            node=domain_node,
            children_list=self._collect_real_children(domain_node),
            estimated_capacities=estimated_capacities,
            targets=label_targets,
        )

        selected: set[str] = set()
        for label_taxid, n_per_class in label_targets.items():
            already_have = downloaded_cap.get(label_taxid, 0)
            still_need = max(0, n_per_class - already_have)
            pending_sorted = sorted(
                pending_index.get(label_taxid, []),
                key=lambda x: (not x[1], -x[2]),
            )
            cumulative = 0
            for acc_id, _is_ref, seq_len in pending_sorted:
                if cumulative >= still_need:
                    break
                selected.add(acc_id)
                cumulative += seq_len

        all_pending = self._collect_scope_pending_accessions(domain_taxid)
        deferred = all_pending - selected
        self.registry.mark_accessions_deferred(list(deferred))
        self.registry.save()

        ui_logger.info(
            f"Selective download: {len(selected):,} accessions selected, "
            f"{len(deferred):,} deferred for refinement."
        )

    def _estimate_capacities_from_registry(
        self, domain_taxid: str | None
    ) -> dict[str, int]:
        """Estimate node capacities using total_sequence_length metadata.

        For each leaf taxon in scope, the estimated capacity equals the sum
        of total_sequence_length across all its accessions. That value is
        propagated bottom-up to every ancestor via the stored lineages so
        the balancing layer can consume it as a capacity_override.

        Args:
            domain_taxid: Root TaxID to restrict the computation to.

        Returns:
            Mapping of TaxID string to estimated capacity in base pairs.
        """
        lineages = self.registry.registry["lineages"]
        accessions = self.registry.registry["accessions"]
        taxons = self.registry.registry["taxons"]
        domain_str = str(domain_taxid) if domain_taxid else None

        result: dict[str, int] = {}
        for taxid, acc_list in taxons.items():
            stored = lineages.get(taxid)
            if not stored:
                continue
            if domain_str is not None and not any(
                a["taxid"] == domain_str for a in stored
            ):
                continue
            leaf_cap = sum(
                int(accessions.get(acc, {}).get("total_sequence_length") or 0)
                for acc in acc_list
            )
            if leaf_cap == 0:
                continue
            # Credit the leaf and every ancestor exactly once. The stored
            # lineage already includes the leaf at index 0, so a separate
            # ``result[taxid] += leaf_cap`` would double-count the leaf's own
            # capacity; the set makes it robust whether or not the leaf is
            # present in its stored lineage.
            for node_id in {taxid, *(a["taxid"] for a in stored)}:
                result[node_id] = result.get(node_id, 0) + leaf_cap
        return result

    def _build_scope_accession_index(
        self, domain_taxid: str | None
    ) -> tuple[dict[str, int], dict[str, list[tuple[str, bool, int]]]]:
        """Build per-label capacity and pending accession lists for selection.

        Iterates every leaf taxid in scope, then attributes each
        accession's total_sequence_length to the leaf itself and all
        its ancestors (the set of potential labels in any decision point).

        Args:
            domain_taxid: Root TaxID of the scope.

        Returns:
            Two-tuple of:
            - downloaded_cap: label taxid → summed total_sequence_length
              of already-downloaded accessions, for deducting from the
              target when selecting pending accessions.
            - pending_index: label taxid → list of
              (accession_id, is_reference, total_sequence_length) for
              pending accessions under that label, unsorted.
        """
        lineages = self.registry.registry["lineages"]
        accessions = self.registry.registry["accessions"]
        taxons = self.registry.registry["taxons"]
        domain_str = str(domain_taxid) if domain_taxid else None

        downloaded_cap: dict[str, int] = {}
        pending_index: dict[str, list[tuple[str, bool, int]]] = {}

        for taxid, acc_list in taxons.items():
            stored = lineages.get(taxid)
            if not stored:
                continue
            if domain_str is not None and not any(
                a["taxid"] == domain_str for a in stored
            ):
                continue
            label_taxids = [taxid] + [a["taxid"] for a in stored]
            for acc_id in acc_list:
                info = accessions.get(acc_id, {})
                seq_len = int(info.get("total_sequence_length") or 0)
                is_ref = bool(info.get("is_reference"))
                is_downloaded = bool(info.get("downloaded"))
                for label in label_taxids:
                    if is_downloaded:
                        downloaded_cap[label] = downloaded_cap.get(label, 0) + seq_len
                    else:
                        pending_index.setdefault(label, []).append(
                            (acc_id, is_ref, seq_len)
                        )
        return downloaded_cap, pending_index

    def _collect_scope_pending_accessions(self, domain_taxid: str | None) -> set[str]:
        """Return all pending accession IDs within the given domain scope.

        Args:
            domain_taxid: Root TaxID of the scope.

        Returns:
            Set of accession IDs that are not yet downloaded.
        """
        lineages = self.registry.registry["lineages"]
        accessions = self.registry.registry["accessions"]
        taxons = self.registry.registry["taxons"]
        domain_str = str(domain_taxid) if domain_taxid else None

        result: set[str] = set()
        for taxid, acc_list in taxons.items():
            stored = lineages.get(taxid)
            if not stored:
                continue
            if domain_str is not None and not any(
                a["taxid"] == domain_str for a in stored
            ):
                continue
            for acc_id in acc_list:
                info = accessions.get(acc_id, {})
                if not info.get("downloaded"):
                    result.add(acc_id)
        return result

    def _collect_label_targets(
        self,
        node: Node,
        children_list: list,
        estimated_capacities: dict[str, int],
        targets: dict[str, int],
    ) -> None:
        """Walk the estimation tree and collect per-label n_per_class targets.

        Mirrors ``_schedule_decision_point`` but only runs the balancing
        layer (no extraction scheduling). Uses ``min_leaves_per_class=0``
        and ``rare_taxa_strategy='keep'`` so the leaf-count floor does
        not suppress children that have no sequence leaves in the
        estimation tree. The estimated ``n_per_class`` is recorded for
        every child at each decision point; for children that appear in
        multiple decision points (retained at an ancestor and as a label
        themselves), the maximum target is kept.

        Args:
            node: Current decision point.
            children_list: Direct real children of the node.
            estimated_capacities: Capacity proxy dict from
                ``_estimate_capacities_from_registry``, injected via
                ``capacity_override``.
            targets: Accumulator mapping label taxid to its n_per_class
                target (mutated in-place).
        """
        if self._is_passthrough_case(children_list):
            child = children_list[0]
            self._collect_label_targets(
                node=child,
                children_list=self._collect_real_children(child),
                estimated_capacities=estimated_capacities,
                targets=targets,
            )
            return

        effective_children, _ = classify_children_by_rank(
            node,
            children_list,
            min_subclades_per_bucket=self.min_subclades_per_bucket,
            all_ranks=self.all_ranks,
        )
        if not effective_children:
            return

        plan = compute_balanced_extraction_plan(
            parent_node=node,
            children=effective_children,
            leaf_cache={},
            min_len=self.min_subseq_len,
            min_num_seqs=self.min_num_seqs,
            cutoff_percentage=self.cutoff_percentage,
            use_exact_capacity=self.use_exact_capacity,
            max_n_per_class=self.max_n_per_class,
            min_leaves_per_class=0,
            rare_taxa_strategy="keep",
            progress_callback=None,
            capacity_override=estimated_capacities,
        )

        n_per_class = plan["n_per_class"]
        if n_per_class == 0:
            return

        for child in plan["retained_children"]:
            child_taxid = str(child.name)
            targets[child_taxid] = max(targets.get(child_taxid, 0), n_per_class)

        for child in plan.get("low_capacity_children", []):
            child_taxid = str(child.name)
            child_cap = estimated_capacities.get(child_taxid, 0)
            targets[child_taxid] = max(targets.get(child_taxid, 0), child_cap)

        if self._single_level:
            return

        for child in plan["retained_children"]:
            child_rank = getattr(child, "rank", "")
            if is_recursion_terminator(child_rank):
                continue
            if self._depth_boundary is not None and is_below_boundary(
                child_rank, self._depth_boundary
            ):
                continue
            grand_children = self._collect_real_children(child)
            if grand_children:
                self._collect_label_targets(
                    node=child,
                    children_list=grand_children,
                    estimated_capacities=estimated_capacities,
                    targets=targets,
                )

    def _run_refinement_pass(
        self, domain_taxid: str | None, tree_root: Node
    ) -> bool:
        """Check for capacity shortfalls and undefer accessions for the next round.

        Re-derives per-label n_per_class targets using the same
        estimation pass as Phase 1 (total_sequence_length proxy over all
        accessions, downloaded or deferred). Compares each label's real
        capacity from ``self._all_capacities`` against its estimated
        target. For labels that fell short — meaning the size proxy
        over-estimated capacity due to repetitive genomic content —
        additional deferred accessions are undeferred (reference-assembly
        first, then by decreasing size) until the residual gap is covered
        or the deferred pool for that label is exhausted.

        Args:
            domain_taxid: Root TaxID of the scope being processed.
            tree_root: Taxonomic tree from the current download round.
                Only the node structure is used; sequence leaves are
                ignored because capacity comes from ``self._all_capacities``.

        Returns:
            True if at least one deferred accession was undeferred,
            indicating another download round is warranted. False when
            all labels meet their targets or no deferred accessions
            remain for shortfall labels.
        """
        estimated_capacities = self._estimate_capacities_from_registry(domain_taxid)
        label_targets: dict[str, int] = {}
        domain_node = self._find_domain_node(tree_root, domain_taxid)
        if domain_node is None:
            return False

        self._collect_label_targets(
            node=domain_node,
            children_list=self._collect_real_children(domain_node),
            estimated_capacities=estimated_capacities,
            targets=label_targets,
        )

        shortfall: dict[str, int] = {
            label: target - (self._all_capacities or {}).get(label, 0)
            for label, target in label_targets.items()
            if (self._all_capacities or {}).get(label, 0) < target
        }

        if not shortfall:
            ui_logger.info("Refinement: all labels met their estimated targets.")
            return False

        ui_logger.info(
            f"Refinement: {len(shortfall)} label(s) below estimated target; "
            "undefering additional accessions."
        )

        deferred_index = self._build_deferred_accession_index(domain_taxid)
        newly_undeferred: set[str] = set()

        for label_taxid, still_need in shortfall.items():
            deferred_sorted = sorted(
                deferred_index.get(label_taxid, []),
                key=lambda x: (not x[1], -x[2]),
            )
            cumulative = 0
            for acc_id, _is_ref, seq_len in deferred_sorted:
                if cumulative >= still_need:
                    break
                newly_undeferred.add(acc_id)
                cumulative += seq_len

        if not newly_undeferred:
            ui_logger.info(
                "Refinement: no deferred accessions remain for shortfall labels; "
                "proceeding with available capacity."
            )
            return False

        registry_accessions = self.registry.registry["accessions"]
        for acc_id in newly_undeferred:
            if acc_id in registry_accessions:
                registry_accessions[acc_id]["download_deferred"] = False
        self.registry.save()

        ui_logger.info(
            f"Refinement: {len(newly_undeferred):,} accession(s) undeferred "
            "for next download round."
        )
        return True

    def _build_deferred_accession_index(
        self, domain_taxid: str | None
    ) -> dict[str, list[tuple[str, bool, int]]]:
        """Build per-label deferred accession lists for the refinement pass.

        Same structure as ``_build_scope_accession_index`` but restricted
        to accessions currently marked ``download_deferred=True``. Used
        by ``_run_refinement_pass`` to identify which deferred accessions
        to undefer for each shortfall label.

        Args:
            domain_taxid: Root TaxID of the scope.

        Returns:
            Mapping of label taxid → list of
            (accession_id, is_reference, total_sequence_length) for
            deferred accessions under that label, unsorted.
        """
        lineages = self.registry.registry["lineages"]
        accessions = self.registry.registry["accessions"]
        taxons = self.registry.registry["taxons"]
        domain_str = str(domain_taxid) if domain_taxid else None

        index: dict[str, list[tuple[str, bool, int]]] = {}
        for taxid, acc_list in taxons.items():
            stored = lineages.get(taxid)
            if not stored:
                continue
            if domain_str is not None and not any(
                a["taxid"] == domain_str for a in stored
            ):
                continue
            label_taxids = [taxid] + [a["taxid"] for a in stored]
            for acc_id in acc_list:
                info = accessions.get(acc_id, {})
                if not info.get("download_deferred"):
                    continue
                seq_len = int(info.get("total_sequence_length") or 0)
                is_ref = bool(info.get("is_reference"))
                for label in label_taxids:
                    index.setdefault(label, []).append((acc_id, is_ref, seq_len))
        return index

    def _reconcile_vault_against_registry(self) -> None:
        """Reconcile the vault against the registry (delegates to downloader)."""
        self.downloader.reconcile_with_vault()

    def run_pipeline(
        self,
        target_group: str,
        min_num_seqs: int = 100,
        percentage: int = 10,
        abundance_threshold: int = 2,
        max_budget: int = 50_000,
        sync: bool = True,
        stop_at: str | None = None,
        single_level: bool = False,
    ) -> None:
        """Execute the full generation pipeline for a target group.

        Args:
            target_group: Domain identifier ('viruses', 'bacteria',
                'archaea', 'eukaryotes', or 'all').
            min_num_seqs: Legacy parameter; the active balancing
                layer uses ``self.min_num_seqs`` from the constructor.
            percentage: Legacy parameter retained for compatibility.
            abundance_threshold: Minimum sequence abundance to avoid
                fallback redirection during tree construction.
            max_budget: Legacy upper bound on total extraction budget.
            stop_at: Canonical rank at which the cascade stops creating
                heads (e.g. 'family'). Nodes at or below this rank
                become training labels but not heads of their own.
                None descends to the deepest available rank.
            single_level: When True, generate only the root's head (its
                direct children become labels) with no further
                recursion. Mutually exclusive with stop_at.

        Raises:
            ValueError: If stop_at is not a canonical rank, or if both
                stop_at and single_level are given.
        """
        _ = min_num_seqs, percentage, max_budget  # legacy CLI params
        if single_level and stop_at is not None:
            raise ValueError(
                "stop_at and single_level are mutually exclusive."
            )
        if stop_at is not None and not is_canonical_rank(stop_at):
            raise ValueError(
                f"stop_at must be a canonical rank, got {stop_at!r}. "
                f"Valid ranks: {list(CANONICAL_RANKS_ROOT_TO_SPECIES)}."
            )
        self._depth_boundary = stop_at
        self._single_level = single_level
        self._selective_download_active = False
        t_pipeline_start = time.monotonic()

        from taxotreeset.core.preflight import run_preflight
        run_preflight(
            registry=self.registry,
            vault_path=self.vault_path,
            output_dir=self.output_dir,
            spill_dir=self.spill_dir,
            n_gpu_workers=self.n_gpu_workers,
            sync=sync,
        )

        if sync:
            ui_logger.info("Syncing registry and vault with NCBI.")
            self._sync_with_ncbi(target_group)

        domain_taxid = self._resolve_root_taxid(target_group)
        tree_root: Node | None = None

        for round_num in range(_MAX_REFINEMENT_ROUNDS + 1):
            suffix = f" (refinement round {round_num})" if round_num > 0 else ""

            # Stage 1 — download
            n_pending_before = sum(
                1 for v in self.registry.registry.get("accessions", {}).values()
                if not v.get("downloaded", False)
            )
            ui_logger.info(f"Stage 1/4: Downloading pending accessions{suffix}.")
            t1 = time.monotonic()
            self.downloader.download_all_pending()
            n_pending_after = sum(
                1 for v in self.registry.registry.get("accessions", {}).values()
                if not v.get("downloaded", False)
            )
            n_downloaded = n_pending_before - n_pending_after
            ui_logger.info(
                "✓ Stage 1/4  %s   (%s)",
                _fmt_elapsed(time.monotonic() - t1),
                f"{n_downloaded:,} downloaded" if n_downloaded else "nothing to download",
            )

            # Stage 2 — tree build
            ui_logger.info(f"Stage 2/4: Building taxonomic tree{suffix}.")
            t2 = time.monotonic()
            tree_root = self._build_target_tree(domain_taxid)

            if tree_root is None or not tree_root.children:
                if sync:
                    ui_logger.error(
                        f"No data found for root '{target_group}' "
                        f"(TaxID {domain_taxid or 'all'}) after syncing with NCBI. "
                        "Verify the root exists in NCBI RefSeq."
                    )
                else:
                    ui_logger.error(
                        f"No data found for root '{target_group}' "
                        f"(TaxID {domain_taxid or 'all'}) in the registry. Re-run "
                        "without --no-sync to discover and download it "
                        "from NCBI."
                    )
                return

            # Capacity pass (part of Stage 2)
            if round_num == 0:
                self._all_capacities = self._load_or_compute_capacities(tree_root)
            else:
                ui_logger.info(
                    f"Computing node capacities via bottom-up pass "
                    f"(min_len={self.min_subseq_len})."
                )
                self._all_capacities = compute_all_capacities(
                    tree_root, self.min_subseq_len,
                    spill_dir=self.spill_dir, n_workers=self.n_workers,
                    n_gpu_workers=self.n_gpu_workers,
                )

            n_taxa = sum(1 for _ in tree_root.descendants)
            n_cap = len(self._all_capacities)
            ui_logger.info(
                "✓ Stage 2/4  %s   (%s taxa in tree, %s nodes with capacity)",
                _fmt_elapsed(time.monotonic() - t2),
                f"{n_taxa:,}",
                f"{n_cap:,}",
            )

            if not self._selective_download_active:
                break
            if round_num >= _MAX_REFINEMENT_ROUNDS:
                ui_logger.info(
                    f"Maximum refinement rounds ({_MAX_REFINEMENT_ROUNDS}) "
                    "reached; proceeding with current capacities."
                )
                break
            if not self._run_refinement_pass(domain_taxid, tree_root):
                break

        if tree_root is None:
            return

        # Stage 3 — scheduling
        ui_logger.info("Stage 3/4: Scheduling extraction jobs.")
        t3 = time.monotonic()
        scheduling_artifacts = self._schedule_pipeline_jobs(
            tree_root=tree_root,
            domain_taxid=domain_taxid,
            abundance_threshold=abundance_threshold,
        )
        self.registry.store_capacities(self._all_capacities, self.min_subseq_len)
        self.registry.save()

        self._persist_scheduling_artifacts(
            target_group=target_group,
            scheduling_artifacts=scheduling_artifacts,
        )
        n_heads = len(scheduling_artifacts["master_manifest"])
        ui_logger.info(
            "✓ Stage 3/4  %s   (%s heads scheduled)",
            _fmt_elapsed(time.monotonic() - t3),
            f"{n_heads:,}",
        )

        # Stage 4 — extraction (multi-class dispatches here; --binary-only has
        # already streamed its extraction in batches during Stage 3).
        self._run_extraction_stage(scheduling_artifacts, n_heads)

        self._write_label_maps(scheduling_artifacts)
        self._write_run_metadata(
            target_group=target_group,
            scheduling_artifacts=scheduling_artifacts,
            n_taxa=n_taxa,
            n_cap=n_cap,
            abundance_threshold=abundance_threshold,
            t_pipeline_start=t_pipeline_start,
        )

        ui_logger.info("Pipeline finished successfully.")

    @staticmethod
    def _resolve_root_taxid(target_root: str) -> str | None:
        """Resolve the generation root to an NCBI TaxID string.

        Accepts ``"all"`` (every domain), a domain shortcut (viruses,
        bacteria, archaea, eukaryotes), a numeric TaxID, or a clade
        scientific name. The shortcuts are convenience aliases for the four
        superkingdom TaxIDs; anything else is resolved via taxoniq with an
        NCBI fallback.

        Args:
            target_root: ``"all"``, a domain shortcut, a numeric TaxID, or
                a clade scientific name.

        Returns:
            The resolved NCBI TaxID as a string, or ``None`` for ``"all"``
            -- there is no anchor TaxID, so the whole registry is in scope.

        Raises:
            ValueError: If the reference cannot be resolved.
        """
        if target_root == _ALL_DOMAINS:
            return None
        if target_root in _DOMAIN_GROUP_TO_TAXID:
            return _DOMAIN_GROUP_TO_TAXID[target_root]
        return resolve_to_taxid(target_root)
    def _build_target_tree(self, domain_taxid: str | None) -> Node | None:
        """Construct the taxonomic tree anchored at the domain TaxID.

        Args:
            domain_taxid: NCBI TaxID of the root domain.

        Returns:
            The constructed tree root, or None on construction failure.
        """
        return generate_seqs_by_taxon_tree(
            registry_path=self.registry.registry_path,
            vault_path=self.vault_path,
            domain_taxid=domain_taxid,
            mapping_path=self.config_path,
            all_ranks=self.all_ranks,
        )

    def _load_or_compute_capacities(self, tree_root: Node) -> dict[str, int]:
        """Return node capacities from the registry cache or by computing them.

        Checks whether the registry already holds a complete capacity entry
        for every non-sequence node in ``tree_root`` at ``self.min_subseq_len``.
        A complete hit skips the bottom-up pass entirely. A partial or empty
        cache triggers a fresh bottom-up computation whose result is ready to
        be persisted by the caller.

        The cache is considered complete when every non-sequence descendant of
        ``tree_root`` has a cached value for the requested window size. A
        partial hit (some nodes cached, some missing) is treated as a miss to
        avoid mixing stale and fresh values in the same scheduling run.

        Args:
            tree_root: Root of the taxonomic tree built for this run.

        Returns:
            Mapping of TaxID string to capacity for every non-sequence node
            in the tree, sourced from the cache or freshly computed.
        """
        min_len = self.min_subseq_len
        cached = self.registry.load_capacities(min_len)
        if cached:
            tree_taxids = {
                str(node.name)
                for node in tree_root.descendants
                if getattr(node, "rank", "") != "sequence"
            }
            if tree_taxids.issubset(cached.keys()):
                ui_logger.info(
                    f"Loaded {len(cached):,} cached node capacities "
                    f"(min_len={min_len})."
                )
                return cached

        ui_logger.info(
            f"Computing node capacities via bottom-up pass (min_len={min_len})."
        )
        return compute_all_capacities(
            tree_root, min_len,
            spill_dir=self.spill_dir, n_workers=self.n_workers,
            n_gpu_workers=self.n_gpu_workers,
        )

    def _schedule_pipeline_jobs(
        self,
        tree_root: Node,
        domain_taxid: str | None,
        abundance_threshold: int,
    ) -> dict[str, Any]:
        """Walk the tree and schedule extraction jobs at decision points.

        Initializes bookkeeping structures (master manifest,
        passthrough map, virtual ID registry, leaf cache) and
        triggers recursive scheduling at the domain root.

        Args:
            tree_root: Root of the constructed taxonomic tree.
            domain_taxid: Domain anchor TaxID.
            abundance_threshold: Minimum sequence abundance.

        Returns:
            Dictionary with keys 'extraction_jobs', 'master_manifest',
            'passthrough_map', 'virtual_id_registry'.
        """
        extraction_jobs: list = []
        master_manifest: dict[str, dict] = {}
        passthrough_map: dict[str, str] = {}
        virtual_id_registry: dict[str, dict] = {}
        leaf_cache: dict[str, list] = {}

        domain_node = self._find_domain_node(tree_root, domain_taxid)
        if domain_node is None:
            logger.warning(f"Domain node {domain_taxid} not found in tree.")
            return {
                "extraction_jobs": extraction_jobs,
                "master_manifest": master_manifest,
                "passthrough_map": passthrough_map,
                "virtual_id_registry": virtual_id_registry,
            }

        children_list = self._collect_real_children(domain_node)
        self._schedule_pbar = None
        try:
            if self.binary_only:
                # Binary heads are extracted in batches inside this call to
                # bound peak memory; extraction_jobs stays empty so Stage 4 is
                # a no-op for --binary-only.
                self._schedule_binary_heads(
                    domain_node=domain_node,
                    master_manifest=master_manifest,
                    leaf_cache=leaf_cache,
                    passthrough_map=passthrough_map,
                )
            else:
                self._schedule_pbar = tqdm(
                    desc="Computing node capacities", unit=" nodes"
                )
                self._schedule_decision_point(
                    current_node=domain_node,
                    children_list=children_list,
                    accumulated_path=domain_node.name,
                    abundance_threshold=abundance_threshold,
                    extraction_jobs=extraction_jobs,
                    master_manifest=master_manifest,
                    passthrough_map=passthrough_map,
                    virtual_id_registry=virtual_id_registry,
                    leaf_cache=leaf_cache,
                )
        finally:
            if self._schedule_pbar is not None:
                self._schedule_pbar.close()
            self._schedule_pbar = None

        return {
            "extraction_jobs": extraction_jobs,
            "master_manifest": master_manifest,
            "passthrough_map": passthrough_map,
            "virtual_id_registry": virtual_id_registry,
        }

    @staticmethod
    def _find_domain_node(
        tree_root: Node, domain_taxid: str | None
    ) -> Node | None:
        """Locate the domain anchor node under the tree root.

        Args:
            tree_root: Tree root from tree_builder.
            domain_taxid: NCBI TaxID of the domain, or ``None`` for the
                whole-registry ("all") scope.

        Returns:
            The matching child Node; ``tree_root`` itself when
            ``domain_taxid`` is ``None`` (no single anchor); or ``None`` if
            not found.
        """
        if domain_taxid is None:
            return tree_root
        for child in tree_root.children:
            if child.name == domain_taxid:
                return child
        return None

    @staticmethod
    def _collect_real_children(node: Node) -> list:
        """Return direct children that are taxonomic nodes (not sequences).

        Args:
            node: Parent node.

        Returns:
            List of children whose rank is not 'sequence'.
        """
        return [
            child for child in node.children if getattr(child, "rank", "") != "sequence"
        ]

    def _write_label_maps(self, scheduling_artifacts: dict[str, Any]) -> None:
        """Write label_map.json into every head's output directory.

        Each head directory becomes self-contained: the parquet files
        carry integer class indices, and label_map.json provides the
        id2label / label2id mappings in the HuggingFace-standard format
        so fine-tuning scripts can load the head without touching the
        root-level manifest.

        Args:
            scheduling_artifacts: Output of _schedule_pipeline_jobs.
        """
        master_manifest = scheduling_artifacts.get("master_manifest", {})
        n_written = 0
        for taxid, v in master_manifest.items():
            head_dir = v.get("directory_path", "")
            if not head_dir:
                continue
            labels = v.get("labels", {})
            classes = sorted(
                [
                    {
                        "class_idx": lv.get("class_idx", -1),
                        "taxid": label_taxid,
                        "name": lv.get("name", label_taxid),
                        "rank": lv.get("rank", "unknown"),
                    }
                    for label_taxid, lv in labels.items()
                ],
                key=lambda x: x["class_idx"],
            )
            # Disambiguate duplicate class names so id2label / label2id stay
            # bijective — two distinct taxa can carry the same NCBI scientific
            # name (homonyms) or sanitize to the same string, and a bare
            # ``{name: idx}`` dict would silently drop the earlier class.
            seen_names: set[str] = set()
            for c in classes:
                if c["name"] in seen_names:
                    c["name"] = f"{c['name']} ({c['taxid']})"
                seen_names.add(c["name"])
            id2label = {str(c["class_idx"]): c["name"] for c in classes}
            label2id = {c["name"]: c["class_idx"] for c in classes}
            label_map = {
                "head_taxid": taxid,
                "head_name": v.get("scientific_name", taxid),
                "head_rank": v.get("rank", "unknown"),
                "id2label": id2label,
                "label2id": label2id,
                "classes": classes,
            }
            os.makedirs(head_dir, exist_ok=True)
            label_map_path = os.path.join(head_dir, "label_map.json")
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump(label_map, f, indent=2)
            n_written += 1
        logger.info("Label maps written for %d heads.", n_written)

    def _write_run_metadata(
        self,
        target_group: str,
        scheduling_artifacts: dict[str, Any],
        n_taxa: int,
        n_cap: int,
        abundance_threshold: int,
        t_pipeline_start: float,
    ) -> None:
        """Write run_metadata_{target_group}.json to the output directory.

        The file captures every parameter needed to reproduce the run,
        a summary of what was generated, and a per-head breakdown with
        class lists. Downstream tools (training scripts, inference
        cascade, evaluation harness) can read this file instead of
        reconstructing parameters from CLI history.

        Args:
            target_group: Root group name used in the filename.
            scheduling_artifacts: Output of _schedule_pipeline_jobs.
            n_taxa: Total descendant taxa in the taxonomic tree.
            n_cap: Number of nodes with a computed capacity.
            abundance_threshold: Minimum sequence count per class.
            t_pipeline_start: monotonic timestamp from pipeline start.
        """
        import datetime

        try:
            from importlib.metadata import version as _pkg_version
            pkg_version = _pkg_version("taxotreeset")
        except Exception:
            pkg_version = "unknown"

        tools = _capture_tool_versions()
        snapshot = self.registry.accession_snapshot()

        master_manifest = scheduling_artifacts.get("master_manifest", {})
        # One manifest entry per head, for both the multi-class and the
        # (batch-extracted, so extraction_jobs is empty) binary paths.
        n_heads = len(master_manifest)
        n_classes_total = sum(len(v.get("labels", {})) for v in master_manifest.values())

        heads = []
        for taxid, v in master_manifest.items():
            labels = v.get("labels", {})
            heads.append({
                "taxid": taxid,
                "name": v.get("scientific_name", taxid),
                "rank": v.get("rank", "unknown"),
                "directory": v.get("directory_path", ""),
                "scenario": v.get("scenario", ""),
                "n_per_class": v.get("n_per_class", 0),
                "n_leaves": v.get("num_leaves", 0),
                "n_classes": len(labels),
                "classes": [
                    {
                        "taxid": label_taxid,
                        "name": lv.get("name", label_taxid),
                        "rank": lv.get("rank", "unknown"),
                        "class_idx": lv.get("class_idx", -1),
                    }
                    for label_taxid, lv in labels.items()
                ],
            })

        # Aggregate per-rank statistics for quick dataset inspection.
        import statistics as _stats
        from collections import defaultdict as _dd
        _by_rank: dict[str, list] = _dd(list)
        _caps = self._all_capacities or {}
        for head in heads:
            _by_rank[head["rank"]].append({
                "npc": head["n_per_class"],
                "nc": head["n_classes"],
                "cap": _caps.get(head["taxid"], 0),
            })
        by_rank = {}
        for rank, items in sorted(_by_rank.items()):
            npc_vals = [i["npc"] for i in items]
            cap_vals = [i["cap"] for i in items if i["cap"] > 0]
            by_rank[rank] = {
                "n_heads": len(items),
                "total_classes": sum(i["nc"] for i in items),
                "median_n_per_class": _stats.median(npc_vals) if npc_vals else 0,
                "median_capacity": int(_stats.median(cap_vals)) if cap_vals else 0,
            }

        metadata = {
            "taxotreeset_version": pkg_version,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - t_pipeline_start, 1),
            "provenance": {
                "tools": tools,
                "registry_last_update": self.registry.registry.get("last_update"),
                "accession_snapshot": {
                    "file": f"accession_snapshot_{target_group}.json",
                    "sha256": snapshot["sha256"],
                    "n_accessions": snapshot["n_accessions"],
                },
            },
            "parameters": {
                "root": target_group,
                "min_subseq_len": self.min_subseq_len,
                "max_subseq_len": self.max_subseq_len,
                "max_n_per_class": self.max_n_per_class,
                "min_leaves_per_class": self.min_leaves_per_class,
                "min_num_seqs": self.min_num_seqs,
                "cutoff_percentage": self.cutoff_percentage,
                "rare_taxa_strategy": self.rare_taxa_strategy,
                "seed": self.seed,
                "abundance_threshold": abundance_threshold,
                "stop_at": self._depth_boundary,
                "single_level": self._single_level,
                "output_format": self.output_format,
                "reject_class": self.reject_class,
                "reject_fraction": self.reject_fraction,
                "reject_near_far_start": self.reject_near_far_start,
                "reject_near_far_end": self.reject_near_far_end,
                "binary_only": self.binary_only,
                "binary_budget": self.binary_budget,
                "all_ranks": self.all_ranks,
            },
            "summary": {
                "n_heads": n_heads,
                "n_classes_total": n_classes_total,
                "n_taxa_in_tree": n_taxa,
                "n_nodes_with_capacity": n_cap,
                "n_accessions": len(self.registry.registry.get("accessions", {})),
                "by_rank": by_rank,
            },
            "heads": heads,
        }

        os.makedirs(self.output_dir, exist_ok=True)
        metadata_path = os.path.join(
            self.output_dir, f"run_metadata_{target_group}.json"
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Run metadata written to %s", metadata_path)

        # Full reproducible accession snapshot (versioned accessions + digest),
        # written alongside the metadata so the run can be re-fetched and cited.
        snapshot_path = os.path.join(
            self.output_dir, f"accession_snapshot_{target_group}.json"
        )
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "generated_at": metadata["generated_at"],
                    "tools": tools,
                    "n_accessions": snapshot["n_accessions"],
                    "sha256": snapshot["sha256"],
                    "accessions": snapshot["accessions"],
                },
                f,
                indent=2,
            )
        logger.info("Accession snapshot written to %s", snapshot_path)

    def _persist_scheduling_artifacts(
        self,
        target_group: str,
        scheduling_artifacts: dict[str, Any],
    ) -> None:
        """Write the manifest, passthrough map, and virtual ID registry.

        These sidecar files are critical metadata that allows
        downstream consumers (training scripts, inference cascade)
        to interpret the generated dataset.

        Args:
            target_group: CLI-friendly group name used in filenames.
            scheduling_artifacts: Output of _schedule_pipeline_jobs.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        manifest_path = os.path.join(self.output_dir, f"manifest_{target_group}.json")
        passthroughs_path = os.path.join(
            self.output_dir, f"passthroughs_{target_group}.json"
        )
        virtual_registry_path = os.path.join(
            self.output_dir, f"virtual_id_registry_{target_group}.json"
        )

        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(
                scheduling_artifacts["master_manifest"],
                manifest_file,
                indent=2,
            )

        with open(passthroughs_path, "w", encoding="utf-8") as passthrough_file:
            json.dump(
                scheduling_artifacts["passthrough_map"],
                passthrough_file,
                indent=2,
            )

        with open(virtual_registry_path, "w", encoding="utf-8") as virtual_file:
            json.dump(
                {"virtual_ids": scheduling_artifacts["virtual_id_registry"]},
                virtual_file,
                indent=2,
            )

        logger.info(
            f"Persisted manifest "
            f"({len(scheduling_artifacts['master_manifest'])} heads), "
            f"passthroughs "
            f"({len(scheduling_artifacts['passthrough_map'])}), "
            f"virtual IDs "
            f"({len(scheduling_artifacts['virtual_id_registry'])})."
        )

    def _run_extraction_stage(
        self, scheduling_artifacts: dict, n_heads: int
    ) -> None:
        """Run Stage 4 extraction and log its completion.

        For ``--binary-only`` runs extraction was already streamed in batches
        during Stage 3 (to bound peak memory), so ``extraction_jobs`` is empty
        and nothing is dispatched here; the multi-class path dispatches its
        accumulated jobs now.

        Args:
            scheduling_artifacts: Artifacts from ``_schedule_pipeline_jobs``.
            n_heads: Number of scheduled heads, for the summary line.
        """
        t4 = time.monotonic()
        if not self.binary_only:
            ui_logger.info("Stage 4/4: Dispatching parallel disk extraction.")
            self._execute_extraction(scheduling_artifacts["extraction_jobs"])
        n_classes = sum(
            len(v.get("labels", {}))
            for v in scheduling_artifacts.get("master_manifest", {}).values()
        )
        stage4_note = (
            "streamed during Stage 3" if self.binary_only
            else _fmt_elapsed(time.monotonic() - t4)
        )
        ui_logger.info(
            "✓ Stage 4/4  %s   (%s heads, %s classes)",
            stage4_note,
            f"{n_heads:,}",
            f"{n_classes:,}",
        )

    def _execute_extraction(self, extraction_jobs: list) -> None:
        """Dispatch the extraction jobs to the worker pool.

        Args:
            extraction_jobs: List of job tuples from scheduling.
        """
        if not extraction_jobs:
            logger.warning("No extraction jobs to dispatch. Skipping.")
            return

        logger.info(
            f"Dispatching {len(extraction_jobs)} extraction jobs to the worker pool."
        )
        self.builder.build_node_dataset(extraction_jobs, parallel=True)

    def _on_capacity_computed(self) -> None:
        """Advance the scheduling progress bar by one capacity computation."""
        if self._schedule_pbar is not None:
            self._schedule_pbar.update(1)

    def _schedule_binary_heads(
        self,
        domain_node: Node,
        master_manifest: dict,
        leaf_cache: dict,
        passthrough_map: dict,
    ) -> None:
        """Schedule and extract a binary belongs/not-belongs head per node.

        ``--binary-only`` mode. For each non-root taxonomic node ``N`` (a real
        taxon, not an individual genome), emit a 2-class dataset:

        - **belongs** (label 1): up to ``binary_budget`` windows spread across
          ``N``'s subtree genomes.
        - **not-belongs** (label 0): an equal budget of out-of-subtree windows
          (near siblings + far clades, split by the depth-scaled near/far ratio),
          reusing the same reject sampler the multi-class head uses.

        Train/val/test split is by genome, with the window-slicing fallback
        (``min_genomes_for_genome_split=4``) so data-poor nodes still yield a
        valid non-empty split — no viability gating. Nodes with no external
        negatives (e.g. directly under the domain root) or no extractable
        positives are skipped, since a 2-class head cannot be formed there. The
        learning auditor/monitor certify which trained heads actually learn.

        Extraction is streamed in batches of
        ``self.binary_extract_batch_size`` heads instead of accumulating every
        head's task lists in memory first: at all-ranks scale (tens of
        thousands of heads, each carrying up to a few thousand reject task
        dicts) holding them all at once exhausts RAM. Each batch's jobs are
        built, dispatched to the worker pool, and freed before the next, so
        peak memory is bounded by one batch rather than the whole run. The
        ``master_manifest`` (lightweight per-head metadata) is still populated
        in full for downstream persistence.

        Single-child nodes are passthroughs: a node with exactly one taxonomic
        child covers the identical subtree as that child, so its
        belongs/not-belongs head would be redundant. Such nodes are skipped (no
        head, no parquet) and recorded in ``passthrough_map`` — mirroring the
        multi-class path's ``_is_passthrough_case`` — so only genuine branching
        (decidable) nodes become heads.

        Args:
            domain_node: The domain anchor node (its descendants get heads).
            master_manifest: Manifest to populate (mutated).
            leaf_cache: Per-node sequence-leaf cache.
            passthrough_map: Map of collapsed node -> its single child (mutated).
        """
        caps = self._all_capacities or {}
        nodes = [
            n for n in domain_node.descendants
            if getattr(n, "rank", "") != "sequence"
        ]
        total = len(nodes)
        batch_size = max(1, self.binary_extract_batch_size)
        ui_logger.info(
            "Scheduling + extracting binary heads over %s taxonomic nodes "
            "in batches of %s (one belongs/not-belongs head per node)...",
            f"{total:,}", f"{batch_size:,}")
        batch: list = []
        scheduled = 0
        skipped = 0
        passthrough = 0
        last_log = time.monotonic()

        def flush_batch() -> None:
            nonlocal scheduled
            if not batch:
                return
            self.builder.build_node_dataset(batch, parallel=True)
            scheduled += len(batch)
            ui_logger.info(
                "  extracted binary batch: +%s heads  (%s extracted, "
                "%s skipped)", f"{len(batch):,}", f"{scheduled:,}",
                f"{skipped:,}")
            batch.clear()

        for i, node in enumerate(nodes):
            now = time.monotonic()
            if now - last_log >= 15.0:
                ui_logger.info(
                    "  binary heads: %s/%s nodes scanned  ->  %s extracted, "
                    "%s pending, %s skipped", f"{i:,}", f"{total:,}",
                    f"{scheduled:,}", f"{len(batch):,}", f"{skipped:,}")
                last_log = now
            taxid = str(node.name)
            # Passthrough: a node with a single taxonomic child covers the exact
            # same subtree as that child, so its belongs/not-belongs head is
            # redundant. Skip it (no head/parquet) and record the collapse, so
            # only genuine branching nodes become heads.
            node_children = self._collect_real_children(node)
            if self._is_passthrough_case(node_children):
                passthrough_map[taxid] = str(node_children[0].name)
                passthrough += 1
                continue
            cap = caps.get(taxid, 0)
            budget = min(self.binary_budget, cap) if cap else self.binary_budget
            if budget <= 0:
                skipped += 1
                continue
            name = getattr(node, "scientific_name", taxid)

            pos_tasks = distribute_n_per_class_across_leaves(
                n_per_class=budget, children=[node], parent_taxid=taxid,
                parent_name=name, leaf_cache=leaf_cache,
                min_subseq_len=self.min_subseq_len,
            ).get(taxid, [])
            near, far = sample_reject_leaves(node, rng=random.Random(self.seed))
            neg_tasks = build_reject_tasks(
                near_leaves=near, far_leaves=far, n_reject=budget,
                near_far_ratio=self._reject_near_ratio(node),
                min_subseq_len=self.min_subseq_len,
            )
            if not pos_tasks or not neg_tasks:
                skipped += 1
                continue

            rng = random.Random(self.seed)
            pos_split = self._materialize_leaf_split(
                pos_tasks, 1, rng, min_genomes_for_genome_split=4)
            neg_split = self._materialize_leaf_split(
                neg_tasks, 0, rng, min_genomes_for_genome_split=4)
            parent_tasks = {s: pos_split[s] + neg_split[s] for s in _SPLITS}
            if not any(parent_tasks[s] for s in _SPLITS):
                skipped += 1
                continue

            path_parts = [p for p in node.path_name.split("/") if p]
            target_dir = os.path.join(self.output_dir, *path_parts)
            os.makedirs(target_dir, exist_ok=True)
            num_leaves = sum(
                1 for leaf in node.leaves
                if getattr(leaf, "rank", "") == "sequence"
            )
            master_manifest[taxid] = {
                "directory_path": target_dir,
                "scientific_name": name,
                "rank": getattr(node, "rank", "unknown"),
                "scenario": "binary_belongs",
                "n_per_class": budget,
                "num_leaves": num_leaves,
                "labels": {
                    f"not_belongs_{taxid}": {
                        "class_idx": 0, "taxid": f"not_belongs_{taxid}",
                        "name": f"not_belongs_{name}",
                        "rank": "virtual_not_belongs", "fallback": True,
                        "capacity": 0,
                    },
                    taxid: {
                        "class_idx": 1, "taxid": taxid, "name": name,
                        "rank": getattr(node, "rank", "unknown"),
                        "fallback": False, "capacity": cap,
                    },
                },
            }
            batch.append((
                taxid, target_dir, parent_tasks,
                self.max_subseq_len, self.seed, self.output_format,
            ))
            if len(batch) >= batch_size:
                flush_batch()
        flush_batch()
        ui_logger.info(
            "Scheduled + extracted %s binary heads (%s of %s nodes: %s "
            "passthroughs collapsed, %s skipped for no data/negatives).",
            f"{scheduled:,}", f"{total:,}", f"{total:,}", f"{passthrough:,}",
            f"{skipped:,}")

    def _schedule_decision_point(
        self,
        current_node: Node,
        children_list: list,
        accumulated_path: str,
        abundance_threshold: int,
        extraction_jobs: list,
        master_manifest: dict,
        passthrough_map: dict,
        virtual_id_registry: dict,
        leaf_cache: dict,
    ) -> None:
        """Recursively schedule extraction at one taxonomic decision point.

        Each call:
            1. Detects passthrough (single-child) case and recurses.
            2. Applies rank-aware bucketing to mixed-rank children.
            3. Computes the balanced extraction plan.
            4. Materializes a low-capacity bucket when cutoff applies.
            5. Distributes n_per_class across retained children's
               sequence leaves.
            6. Builds the extraction job for this node.
            7. Recurses into each canonical (non-virtual) child.

        Args:
            current_node: The node being scheduled.
            children_list: Direct real children of the node.
            accumulated_path: '/'-joined TaxID path from root.
            abundance_threshold: Minimum sequence abundance.
            extraction_jobs: List to append new jobs to (mutated).
            master_manifest: Manifest to populate (mutated).
            passthrough_map: Passthrough map (mutated).
            virtual_id_registry: Virtual buckets metadata (mutated).
            leaf_cache: Per-node leaf cache (mutated).
        """
        if self._is_passthrough_case(children_list):
            self._handle_passthrough(
                current_node=current_node,
                children_list=children_list,
                accumulated_path=accumulated_path,
                abundance_threshold=abundance_threshold,
                extraction_jobs=extraction_jobs,
                master_manifest=master_manifest,
                passthrough_map=passthrough_map,
                virtual_id_registry=virtual_id_registry,
                leaf_cache=leaf_cache,
            )
            return

        effective_children, new_virtual_buckets = classify_children_by_rank(
            current_node,
            children_list,
            min_subclades_per_bucket=self.min_subclades_per_bucket,
            all_ranks=self.all_ranks,
        )
        self._register_virtual_buckets(
            new_virtual_buckets=new_virtual_buckets,
            virtual_id_registry=virtual_id_registry,
            parent_taxid=str(current_node.name),
            parent_name=getattr(current_node, "scientific_name", str(current_node.name)),
        )

        if not effective_children:
            return

        plan = compute_balanced_extraction_plan(
            parent_node=current_node,
            children=effective_children,
            leaf_cache=leaf_cache,
            min_len=self.min_subseq_len,
            min_num_seqs=self.min_num_seqs,
            cutoff_percentage=self.cutoff_percentage,
            use_exact_capacity=self.use_exact_capacity,
            max_n_per_class=self.max_n_per_class,
            min_leaves_per_class=self.min_leaves_per_class,
            rare_taxa_strategy=self.rare_taxa_strategy,
            progress_callback=self._on_capacity_computed,
            capacity_override=self._all_capacities,
        )

        retained_children = self._handle_low_capacity_bucket(
            current_node=current_node,
            plan=plan,
            virtual_id_registry=virtual_id_registry,
        )
        retained_children = self._handle_rare_taxa_bucket(
            current_node=current_node,
            plan=plan,
            retained_children=retained_children,
            virtual_id_registry=virtual_id_registry,
        )

        if not retained_children or plan["n_per_class"] == 0:
            return

        per_child_tasks = distribute_n_per_class_across_leaves(
            n_per_class=plan["n_per_class"],
            children=retained_children,
            parent_taxid=str(current_node.name),
            parent_name=getattr(
                current_node, "scientific_name", str(current_node.name)
            ),
            leaf_cache=leaf_cache,
        )

        retained_children = self._maybe_add_reject_class(
            current_node=current_node,
            retained_children=retained_children,
            per_child_tasks=per_child_tasks,
            plan=plan,
            virtual_id_registry=virtual_id_registry,
        )

        job = self._build_extraction_job(
            current_node=current_node,
            retained_children=retained_children,
            per_child_tasks=per_child_tasks,
            plan=plan,
            accumulated_path=accumulated_path,
            master_manifest=master_manifest,
        )

        if job is not None:
            extraction_jobs.append(job)

        self._recurse_into_canonical_children(
            retained_children=retained_children,
            accumulated_path=accumulated_path,
            abundance_threshold=abundance_threshold,
            extraction_jobs=extraction_jobs,
            master_manifest=master_manifest,
            passthrough_map=passthrough_map,
            virtual_id_registry=virtual_id_registry,
            leaf_cache=leaf_cache,
        )

    @staticmethod
    def _is_passthrough_case(children_list: list) -> bool:
        """Detect whether this node should be treated as a passthrough.

        A passthrough is a node with exactly one taxonomic child; the
        node's head is redirected to the child, and no Parquet is
        produced for the parent.

        Args:
            children_list: Direct real children of the node.

        Returns:
            True if this is a single-child node.
        """
        return len(children_list) == 1

    def _handle_passthrough(
        self,
        current_node: Node,
        children_list: list,
        accumulated_path: str,
        abundance_threshold: int,
        extraction_jobs: list,
        master_manifest: dict,
        passthrough_map: dict,
        virtual_id_registry: dict,
        leaf_cache: dict,
    ) -> None:
        """Record a passthrough and recurse into the single child.

        Args:
            current_node: Parent node being passed through.
            children_list: Single-element list with the child.
            accumulated_path: TaxID path from root.
            abundance_threshold: Minimum sequence abundance.
            extraction_jobs: Jobs list (mutated).
            master_manifest: Manifest (mutated).
            passthrough_map: Passthrough map (mutated).
            virtual_id_registry: Virtual IDs (mutated).
            leaf_cache: Leaf cache (mutated).
        """
        child = children_list[0]
        passthrough_map[str(current_node.name)] = str(child.name)
        logger.debug(f"[PASSTHROUGH] {current_node.name} -> {child.name}")

        next_path = f"{accumulated_path}/{child.name}"
        next_children = self._collect_real_children(child)

        self._schedule_decision_point(
            current_node=child,
            children_list=next_children,
            accumulated_path=next_path,
            abundance_threshold=abundance_threshold,
            extraction_jobs=extraction_jobs,
            master_manifest=master_manifest,
            passthrough_map=passthrough_map,
            virtual_id_registry=virtual_id_registry,
            leaf_cache=leaf_cache,
        )

    @staticmethod
    def _register_virtual_buckets(
        new_virtual_buckets: list[dict],
        virtual_id_registry: dict,
        parent_taxid: str,
        parent_name: str,
    ) -> None:
        """Add freshly created virtual buckets to the registry.

        Delegates each bucket's registration to
        ``register_virtual_bucket``, which records the parent context
        and raises on virtual-ID collisions.

        Args:
            new_virtual_buckets: List of bucket metadata dicts.
            virtual_id_registry: The registry to populate (mutated).
            parent_taxid: TaxID of the parent under which all the
                buckets in this batch were created.
            parent_name: Parent's scientific name (human-readable).
        """
        for bucket in new_virtual_buckets:
            register_virtual_bucket(
                virtual_id_registry=virtual_id_registry,
                bucket_metadata=bucket,
                parent_taxid=parent_taxid,
                parent_name=parent_name,
            )

    def _handle_low_capacity_bucket(
        self,
        current_node: Node,
        plan: dict,
        virtual_id_registry: dict,
    ) -> list:
        """Materialize the low-capacity bucket when cutoff applies.

        Args:
            current_node: Parent node being scheduled.
            plan: Balancing plan from compute_balanced_extraction_plan.
            virtual_id_registry: Registry to populate (mutated).

        Returns:
            Final list of retained training labels, including the
            low-capacity bucket if created.
        """
        if not plan["low_capacity_children"]:
            return plan["retained_children"]

        parent_taxid = str(current_node.name)
        parent_name = getattr(current_node, "scientific_name", parent_taxid)

        bucket_node, bucket_meta = make_low_capacity_bucket_node(
            parent_node=current_node,
            low_capacity_children=plan["low_capacity_children"],
            parent_taxid=parent_taxid,
            parent_name=parent_name,
        )
        register_virtual_bucket(
            virtual_id_registry=virtual_id_registry,
            bucket_metadata=bucket_meta,
            parent_taxid=parent_taxid,
            parent_name=parent_name,
        )

        return [*plan["retained_children"], bucket_node]

    def _handle_rare_taxa_bucket(
        self,
        current_node: Node,
        plan: dict,
        retained_children: list,
        virtual_id_registry: dict,
    ) -> list:
        """Materialize the rare-taxa bucket when the leaf-count floor diverts children.

        Runs after the low-capacity bucket so both synthetic labels can
        coexist in the same head. Children diverted by the leaf-count
        floor (plan['rare_taxa_children']) are absorbed into a single
        virtual_rare_taxa node that becomes a fallback label; the model
        learns to route rare or novel inputs here instead of forcing
        them into an under-supported specific class.

        Args:
            current_node: Parent node being scheduled.
            plan: Balancing plan from compute_balanced_extraction_plan.
            retained_children: Current list of training labels (already
                including the low-capacity bucket if one was created).
            virtual_id_registry: Registry to populate (mutated).

        Returns:
            Updated list of retained training labels, with the rare-taxa
            bucket appended when one is created.
        """
        if not plan.get("rare_taxa_children"):
            return retained_children
        parent_taxid = str(current_node.name)
        parent_name = getattr(current_node, "scientific_name", parent_taxid)
        bucket_node, bucket_meta = make_rare_taxa_bucket_node(
            parent_node=current_node,
            rare_taxa_children=plan["rare_taxa_children"],
            parent_taxid=parent_taxid,
            parent_name=parent_name,
        )
        register_virtual_bucket(
            virtual_id_registry=virtual_id_registry,
            bucket_metadata=bucket_meta,
            parent_taxid=parent_taxid,
            parent_name=parent_name,
        )
        return [*retained_children, bucket_node]

    def _reject_near_ratio(self, node: Node) -> float:
        """Depth-scaled near fraction of the reject bucket for ``node``'s head.

        Distant intruders are pruned by upstream HEADS, so the intruders a deep
        head actually faces are near-heavy (siblings/cousins). The relevant
        depth is therefore the number of *decidable* nodes above the head — i.e.
        NON-passthrough ancestors, since a passthrough (single taxonomic child)
        is not a head and prunes nothing. Using the raw ``node.depth`` (which
        counts passthrough nodes in the all-ranks tree) over-states depth for
        heads under long single-child chains (e.g. a strain sitting directly
        under a clade). The near fraction is linearly interpolated from
        ``reject_near_far_start`` at decidable depth 2 (the root's decidable
        children) to ``reject_near_far_end`` at the deepest decidable node.
        ``end == start`` gives a flat, depth-independent ratio.

        Args:
            node: The head (parent) node being scheduled.

        Returns:
            Near fraction in ``[start, end]`` for this head's decidable depth.
        """
        start, end = self.reject_near_far_start, self.reject_near_far_end
        depths = self._decidable_depths(node.root)
        d_min = 2                         # a decidable child of the root
        d_max = max(depths.values()) if depths else d_min
        if d_max <= d_min:
            return start
        d = depths.get(id(node), d_min)
        frac = min(1.0, max(0.0, (d - d_min) / (d_max - d_min)))
        return start + (end - start) * frac

    def _decidable_depths(self, root: Node) -> dict:
        """Map ``id(node)`` -> count of non-passthrough nodes from root to it.

        A passthrough node (exactly one taxonomic child) is not a head and does
        not prune, so it does not add to a node's effective (pruning) depth.
        Computed top-down once per tree and cached, so ``_reject_near_ratio`` is
        O(1) per head.

        Args:
            root: The tree root.

        Returns:
            Dict mapping node identity to its decidable depth (root's decidable
            children are depth 2, matching the old ``node.depth`` convention).
        """
        if getattr(self, "_dd_root_id", None) == id(root):
            return self._dd_map
        depths: dict[int, int] = {}
        stack = [(root, 0)]
        while stack:
            n, parent_d = stack.pop()
            kids = [c for c in n.children if getattr(c, "rank", "") != "sequence"]
            d = parent_d + (0 if len(kids) == 1 else 1)   # passthrough adds 0
            depths[id(n)] = d
            for c in kids:
                stack.append((c, d))
        self._dd_root_id = id(root)
        self._dd_map = depths
        return depths

    def _maybe_add_reject_class(
        self,
        current_node: Node,
        retained_children: list,
        per_child_tasks: dict[str, list[dict]],
        plan: dict,
        virtual_id_registry: dict,
    ) -> list:
        """Append a reject class of out-of-subtree negatives to the head.

        When ``self.reject_class`` is enabled, samples sequence leaves from
        outside ``current_node``'s subtree (near siblings + far clades), builds
        ``round(n_per_class * reject_fraction)`` extraction tasks for them, and
        appends a detached ``virtual_reject`` node as an extra training label.
        The reject windows never re-parent any tree node; they are injected
        directly into ``per_child_tasks`` (mutated). No-op when disabled, at the
        root (no intra-tree "outside"), or when the budget is zero.

        Args:
            current_node: Parent node (the head) being scheduled.
            retained_children: Current list of training labels.
            per_child_tasks: Per-class extraction tasks (mutated to add reject).
            plan: Balancing plan (provides ``n_per_class``).
            virtual_id_registry: Registry to populate (mutated).

        Returns:
            ``retained_children`` with the reject node appended when created,
            otherwise the input list unchanged.
        """
        if not self.reject_class:
            return retained_children

        near_leaves, far_leaves = sample_reject_leaves(
            current_node, rng=random.Random(self.seed)
        )
        n_reject = round(plan["n_per_class"] * self.reject_fraction)
        reject_tasks = build_reject_tasks(
            near_leaves=near_leaves,
            far_leaves=far_leaves,
            n_reject=n_reject,
            near_far_ratio=self._reject_near_ratio(current_node),
            min_subseq_len=self.min_subseq_len,
        )
        if not reject_tasks:
            return retained_children

        parent_taxid = str(current_node.name)
        parent_name = getattr(current_node, "scientific_name", parent_taxid)
        reject_node, reject_meta = make_reject_bucket_node(
            parent_node=current_node,
            parent_taxid=parent_taxid,
            parent_name=parent_name,
        )
        register_virtual_bucket(
            virtual_id_registry=virtual_id_registry,
            bucket_metadata=reject_meta,
            parent_taxid=parent_taxid,
            parent_name=parent_name,
        )
        per_child_tasks[str(reject_node.name)] = reject_tasks
        return [*retained_children, reject_node]

    def _build_extraction_job(
        self,
        current_node: Node,
        retained_children: list,
        per_child_tasks: dict[str, list[dict]],
        plan: dict,
        accumulated_path: str,
        master_manifest: dict,
    ) -> tuple | None:
        """Assemble a worker-ready extraction job for this decision point.

        Splits each child's per-leaf tasks into train/val/test sets
        using deterministic shuffling, then packages the result into
        the tuple format expected by ``extract_parent_node_worker``.

        Also writes the head's metadata into the master manifest.

        Args:
            current_node: Parent node being scheduled.
            retained_children: Final list of training labels.
            per_child_tasks: Output of
                ``distribute_n_per_class_across_leaves``.
            plan: Balancing plan (scenario, capacities).
            accumulated_path: '/'-joined TaxID path.
            master_manifest: Manifest to populate (mutated).

        Returns:
            Job tuple ready for the worker pool, or None if no
            tasks survived the split (an empty head).
        """
        parent_taxid = str(current_node.name)
        target_dir = os.path.join(self.output_dir, *accumulated_path.split("/"))
        os.makedirs(target_dir, exist_ok=True)

        parent_tasks: dict[str, list[dict]] = {split: [] for split in _SPLITS}
        labels_metadata: dict[str, dict] = {}

        rng = random.Random(self.seed)

        for class_index, child in enumerate(retained_children):
            child_taxid = str(child.name)
            leaf_tasks = per_child_tasks.get(child_taxid, [])
            if not leaf_tasks:
                continue

            leaf_split = self._materialize_leaf_split(
                leaf_tasks=leaf_tasks,
                class_index=class_index,
                rng=rng,
            )

            for split_name in _SPLITS:
                parent_tasks[split_name].extend(leaf_split[split_name])

            child_rank = getattr(child, "rank", "unknown")
            labels_metadata[child_taxid] = {
                "class_idx": class_index,
                "taxid": child_taxid,
                "name": getattr(child, "scientific_name", child_taxid),
                "rank": child_rank,
                "fallback": child_rank.startswith("virtual_"),
                "capacity": plan["capacities"].get(child_taxid, 0),
            }

        if not any(parent_tasks[split] for split in _SPLITS):
            return None

        num_leaves = sum(
            1 for leaf in current_node.leaves
            if getattr(leaf, "rank", "") == "sequence"
        )
        master_manifest[parent_taxid] = {
            "directory_path": target_dir,
            "scientific_name": getattr(current_node, "scientific_name", parent_taxid),
            "rank": getattr(current_node, "rank", "unknown"),
            "scenario": plan["scenario"],
            "n_per_class": plan["n_per_class"],
            "num_leaves": num_leaves,
            "labels": labels_metadata,
        }

        return (
            parent_taxid,
            target_dir,
            parent_tasks,
            self.max_subseq_len,
            self.seed,
            self.output_format,
        )

    def _materialize_leaf_split(
        self,
        leaf_tasks: list[dict],
        class_index: int,
        rng: random.Random,
        min_genomes_for_genome_split: int = 3,
    ) -> dict[str, list[dict]]:
        """Split a single child's per-leaf tasks into train/val/test.

        Thin delegator to
        :func:`taxotreeset.core._orchestration._splits._materialize_leaf_split`;
        see that function for the genome-level vs window-slicing split semantics.
        """
        return _materialize_leaf_split_fn(
            leaf_tasks, class_index, rng, min_genomes_for_genome_split
        )

    def _recurse_into_canonical_children(
        self,
        retained_children: list,
        accumulated_path: str,
        abundance_threshold: int,
        extraction_jobs: list,
        master_manifest: dict,
        passthrough_map: dict,
        virtual_id_registry: dict,
        leaf_cache: dict,
    ) -> None:
        """Recurse into each retained child that is not a virtual bucket.

        Virtual buckets are cascade terminators: they exist as training
        labels in their parent's head but do not host their own sub-
        cascades. Only canonical (non-virtual) children are recursed
        into.

        Args:
            retained_children: All retained training labels.
            accumulated_path: TaxID path from root.
            abundance_threshold: Minimum sequence abundance.
            extraction_jobs: Jobs list (mutated).
            master_manifest: Manifest (mutated).
            passthrough_map: Passthrough map (mutated).
            virtual_id_registry: Virtual IDs (mutated).
            leaf_cache: Leaf cache (mutated).
        """
        if self._single_level:
            return
        for child in retained_children:
            child_rank = getattr(child, "rank", "")
            if is_recursion_terminator(child_rank):
                continue
            if self._depth_boundary is not None and is_below_boundary(
                child_rank, self._depth_boundary
            ):
                continue

            grand_children = self._collect_real_children(child)
            if not grand_children:
                continue

            next_path = f"{accumulated_path}/{child.name}"
            self._schedule_decision_point(
                current_node=child,
                children_list=grand_children,
                accumulated_path=next_path,
                abundance_threshold=abundance_threshold,
                extraction_jobs=extraction_jobs,
                master_manifest=master_manifest,
                passthrough_map=passthrough_map,
                virtual_id_registry=virtual_id_registry,
                leaf_cache=leaf_cache,
            )
