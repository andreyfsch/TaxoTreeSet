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

from taxotreeset.core.generation.capacity import compute_all_capacities
from taxotreeset.core.generation.constants import (
    DEFAULT_CUTOFF_PERCENTAGE,
    DEFAULT_MAX_N_PER_CLASS,
    DEFAULT_MIN_LEAVES_PER_CLASS,
    DEFAULT_MIN_NUM_SEQS,
    DEFAULT_RARE_TAXA_STRATEGY,
    DEFAULT_USE_EXACT_CAPACITY,
)
from taxotreeset.dataset.builder import DatasetBuilder
from taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree

from taxotreeset.logging_utils import get_ui_logger
from taxotreeset.ranks import (
    CANONICAL_RANKS_ROOT_TO_SPECIES,
    is_canonical_rank,
)
from taxotreeset.taxonomy import resolve_to_taxid
from taxotreeset.io.downloader import NCBIDownloader
from taxotreeset.core._orchestration._cluster import ClusterParams
from taxotreeset.benchmark.holdout import (
    build_holdout_manifest,
    prune_holdout,
    select_holdout_taxids,
)
from taxotreeset.core._orchestration._splits import (
    _materialize_leaf_split as _materialize_leaf_split_fn,
    _stratified_counts as _stratified_counts,
    _stratified_cuts as _stratified_cuts,
)
from taxotreeset.core._orchestration._manifest import (
    _capture_tool_versions as _capture_tool_versions,
    _persist_scheduling_artifacts as _persist_scheduling_artifacts_fn,
    _write_label_maps as _write_label_maps_fn,
    _write_run_metadata as _write_run_metadata_fn,
)
from taxotreeset.core._orchestration._sync import (
    _DOMAIN_GROUP_TO_TAXID,
    _SyncManager,
)
from taxotreeset.core._orchestration._scheduler import (
    _CascadeScheduler,
    _collect_real_children as _collect_real_children_fn,
    _find_domain_node as _find_domain_node_fn,
    _is_passthrough_case as _is_passthrough_case_fn,
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

# Special root meaning "every domain": no anchor TaxID -- the whole registry is
# in scope. Resolves to a None domain_taxid throughout the pipeline.
_ALL_DOMAINS = "all"


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
        keep_imbalance: bool = False,
        cluster_aware_split: bool = True,
        cluster_params: ClusterParams | None = None,
        holdout_clades: list[str] | None = None,
        holdout_rank: str | None = None,
        holdout_fraction: float | None = None,
        holdout_seed: int = 0,
        holdout_manifest_path: str | None = None,
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
        self.keep_imbalance: bool = keep_imbalance
        self.cluster_aware_split: bool = cluster_aware_split
        self.cluster_params: ClusterParams = cluster_params or ClusterParams()
        # Clade-holdout open-set benchmark (P11-P1): withhold whole clades from
        # training; the eval set + scorer live in later phases.
        self.holdout_clades: list[str] | None = holdout_clades
        self.holdout_rank: str | None = holdout_rank
        self.holdout_fraction: float | None = holdout_fraction
        self.holdout_seed: int = holdout_seed
        self.holdout_manifest_path: str | None = holdout_manifest_path
        self._holdout_requested: bool = bool(holdout_clades or holdout_rank)
        self._holdout_taxids: set[str] | None = None
        self._holdout_manifest: list[dict] | None = None
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
        self._selective_download_active: bool = False
        self._depth_boundary: str | None = None
        self._single_level: bool = False
        self._single_level_taxid: str | None = None
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

        Thin delegator to
        :meth:`taxotreeset.core._orchestration._sync._SyncManager._sync_with_ncbi`.
        """
        _SyncManager(self)._sync_with_ncbi(target_group)

    def _domains_to_sync(self) -> list[str]:
        """Return the superkingdom TaxIDs to re-discover for an ``all`` sync.

        Thin delegator to :class:`_SyncManager`; see it for the scope rules.
        """
        return _SyncManager(self)._domains_to_sync()

    def _estimate_capacities_from_registry(
        self, domain_taxid: str | None
    ) -> dict[str, int]:
        """Estimate node capacities using total_sequence_length metadata.

        Thin delegator to :class:`_SyncManager`; see it for the
        bottom-up size-proxy propagation.
        """
        return _SyncManager(self)._estimate_capacities_from_registry(domain_taxid)

    def _build_scope_accession_index(
        self, domain_taxid: str | None
    ) -> tuple[dict[str, int], dict[str, list[tuple[str, bool, int]]]]:
        """Build per-label capacity and pending accession lists for selection.

        Thin delegator to :class:`_SyncManager`.
        """
        return _SyncManager(self)._build_scope_accession_index(domain_taxid)

    def _collect_scope_pending_accessions(self, domain_taxid: str | None) -> set[str]:
        """Return all pending accession IDs within the given domain scope.

        Thin delegator to :class:`_SyncManager`.
        """
        return _SyncManager(self)._collect_scope_pending_accessions(domain_taxid)

    def _run_refinement_pass(
        self, domain_taxid: str | None, tree_root: Node
    ) -> bool:
        """Check for capacity shortfalls and undefer accessions for another round.

        Thin delegator to
        :meth:`_SyncManager._run_refinement_pass`; see it for the shortfall /
        undefer logic. Returns True when another download round is warranted.
        """
        return _SyncManager(self)._run_refinement_pass(domain_taxid, tree_root)

    def run_pipeline(
        self,
        target_group: str,
        min_num_seqs: int = 100,
        percentage: int = 10,
        abundance_threshold: int = 2,
        max_budget: int = 50_000,
        sync: bool = True,
        stop_at: str | None = None,
        single_level: bool | str = False,
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
            single_level: ``True`` generates only the root's head (its
                direct children become labels) with no further recursion.
                A TaxID string instead generates only the head at that node
                (anywhere in the ``--root`` tree): the full tree is still
                built, so the reject / not-belongs negatives are sampled
                from outside the node's subtree exactly as in a full run —
                pair it with ``--root <ancestor>`` (not ``--root <TaxID>``,
                which would leave no external pool) to regenerate a single
                head as a drop-in replacement. Mutually exclusive with
                stop_at.

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
        self._single_level = bool(single_level)
        self._single_level_taxid = (
            single_level if isinstance(single_level, str) else None
        )
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

        scope_taxids = self._resolve_scope_taxids(target_group)
        # A single domain anchors scheduling at its node (unchanged); the whole
        # registry (None) and multi-root scopes anchor at the empty root.
        domain_taxid = self._scope_anchor(scope_taxids)
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
            tree_root = self._build_target_tree(scope_taxids)

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

            # Clade-holdout (P11-P1): withhold whole clades from training so they
            # appear only in a downstream open-set eval set. Select + record the
            # manifest on the FULL tree, then prune before capacity + scheduling so
            # the label space and balancing reflect only the retained set.
            if self._holdout_requested:
                self._apply_holdout(tree_root, domain_taxid)

            # Capacity pass (part of Stage 2).
            self._all_capacities = self._run_capacity_pass(tree_root, round_num)

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

        self._write_holdout_manifest(target_group)  # no-op unless holdout ran

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

    def _run_capacity_pass(self, tree_root: Node, round_num: int) -> dict[str, int]:
        """Compute node capacities for the (possibly pruned) tree.

        Round 0 of a non-holdout run may reuse cached capacities; a holdout run
        forces a fresh bottom-up pass so retained ancestors are not credited with
        pruned descendants, and refinement rounds always recompute.
        """
        if round_num == 0 and not self._holdout_requested:
            return self._load_or_compute_capacities(tree_root)
        ui_logger.info(
            f"Computing node capacities via bottom-up pass "
            f"(min_len={self.min_subseq_len})."
        )
        return compute_all_capacities(
            tree_root, self.min_subseq_len,
            spill_dir=self.spill_dir, n_workers=self.n_workers,
            n_gpu_workers=self.n_gpu_workers,
        )

    def _apply_holdout(self, tree_root: Node, domain_taxid: str | None) -> None:
        """Select + record + prune the held-out clades (open-set benchmark, P11-P1).

        Selection and the manifest are computed once, on the full tree, then the
        held-out subtrees are pruned in place so the retained tree drives capacity
        and scheduling. On a refinement re-build the stable selection is re-pruned.
        """
        scope_node = self._find_domain_node(tree_root, domain_taxid) or tree_root
        if self._holdout_taxids is None:
            self._holdout_taxids = select_holdout_taxids(
                scope_node,
                explicit=self.holdout_clades,
                rank=self.holdout_rank,
                fraction=self.holdout_fraction,
                seed=self.holdout_seed,
            )
            self._holdout_manifest = build_holdout_manifest(
                scope_node, self._holdout_taxids, seed=self.holdout_seed
            )
            if not self._holdout_taxids:
                ui_logger.warning(
                    "Clade-holdout requested but no eligible clades selected "
                    "(check --holdout-clades / --holdout-rank / --holdout-fraction)."
                )
            else:
                n_genomes = sum(e["n_genomes"] for e in self._holdout_manifest)
                ui_logger.info(
                    "Clade-holdout: withholding %s clades (%s genomes) from training.",
                    f"{len(self._holdout_taxids):,}", f"{n_genomes:,}",
                )
        n_pruned = prune_holdout(scope_node, self._holdout_taxids)
        logger.info("Clade-holdout: pruned %d held-out subtree(s).", n_pruned)

    def _write_holdout_manifest(self, target_group: str) -> None:
        """Persist the held-out-clade manifest to the output directory."""
        if self._holdout_manifest is None:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        path = self.holdout_manifest_path or os.path.join(
            self.output_dir, f"benchmark_manifest_{target_group}.json"
        )
        payload = {
            "scope": target_group,
            "params": {
                "holdout_clades": self.holdout_clades,
                "holdout_rank": self.holdout_rank,
                "holdout_fraction": self.holdout_fraction,
                "holdout_seed": self.holdout_seed,
            },
            "n_holdout_clades": len(self._holdout_taxids or ()),
            "holdout": self._holdout_manifest,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        ui_logger.info("Clade-holdout manifest written: %s", path)

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

    def _resolve_scope_taxids(self, target_root: str) -> frozenset[str] | None:
        """Resolve a possibly comma-separated ``--root`` into domain TaxIDs.

        A single token behaves exactly as before. ``"all"`` (which cannot be
        combined with other roots) resolves to ``None`` — the whole registry, no
        anchor. Several tokens (e.g. ``--root Viruses,Bacteria``) resolve to the
        set of their TaxIDs, which builds an empty-root forest with each as a
        top-level subtree.

        Args:
            target_root: One root, or a comma-separated list of roots (each a
                domain shortcut, numeric TaxID, or clade name).

        Returns:
            ``None`` for ``"all"``, else a frozenset of resolved TaxID strings.

        Raises:
            ValueError: If ``--root`` is empty or mixes ``"all"`` with others.
        """
        tokens = [t.strip() for t in target_root.split(",") if t.strip()]
        if not tokens:
            raise ValueError("--root resolved to no scopes.")
        if any(t == _ALL_DOMAINS for t in tokens):
            if len(tokens) > 1:
                raise ValueError(
                    f"'{_ALL_DOMAINS}' cannot be combined with other roots."
                )
            return None
        return frozenset(self._resolve_root_taxid(t) for t in tokens)

    @staticmethod
    def _scope_anchor(scope_taxids: frozenset[str] | None) -> str | None:
        """Return the single scheduling anchor, or None for the empty-root forest.

        A one-domain scope anchors scheduling at that domain node (unchanged
        behaviour); the whole-registry (None) and multi-root scopes anchor at the
        empty root and schedule over its top-level children.
        """
        if scope_taxids is not None and len(scope_taxids) == 1:
            return next(iter(scope_taxids))
        return None

    def _build_target_tree(
        self, scope: str | frozenset[str] | None
    ) -> Node | None:
        """Construct the taxonomic tree for one or several domain scopes.

        A single TaxID (str), ``None`` (whole registry), or a one-element set
        builds one tree exactly as before. A set of several TaxIDs builds each
        domain's tree independently — each with its own scope config, lineage
        anchoring, and redirections — and grafts their domain anchors under one
        empty ``root``, forming the multi-root forest the empty-root scheduling
        path then walks.

        Args:
            scope: A single domain TaxID, ``None`` for the whole registry, or a
                frozenset of TaxIDs for a multi-root forest.

        Returns:
            The constructed tree root, or None on construction failure.
        """
        if scope is None or isinstance(scope, str):
            return self._generate_domain_tree(scope)
        taxids = sorted(scope)
        if len(taxids) <= 1:
            return self._generate_domain_tree(taxids[0] if taxids else None)
        combined = Node("root", rank="root")
        for taxid in taxids:
            sub = self._generate_domain_tree(taxid)
            if sub is None:
                continue
            for child in list(sub.children):
                child.parent = combined
        return combined

    def _generate_domain_tree(self, domain_taxid: str | None) -> Node | None:
        """Build a single domain's tree (thin wrapper over the tree builder)."""
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

        Thin delegator to :meth:`_CascadeScheduler._schedule_pipeline_jobs`.
        """
        return _CascadeScheduler(self)._schedule_pipeline_jobs(
            tree_root, domain_taxid, abundance_threshold
        )

    @staticmethod
    def _find_domain_node(
        tree_root: Node, domain_taxid: str | None
    ) -> Node | None:
        """Locate the domain anchor node under the tree root.

        Thin delegator to :func:`_scheduler._find_domain_node`.
        """
        return _find_domain_node_fn(tree_root, domain_taxid)

    @staticmethod
    def _collect_real_children(node: Node) -> list:
        """Return direct children that are taxonomic nodes (not sequences).

        Thin delegator to :func:`_scheduler._collect_real_children`.
        """
        return _collect_real_children_fn(node)

    def _write_label_maps(self, scheduling_artifacts: dict[str, Any]) -> None:
        """Write label_map.json into every head's output directory.

        Thin delegator to
        :func:`taxotreeset.core._orchestration._manifest._write_label_maps`.
        """
        _write_label_maps_fn(scheduling_artifacts)

    def _write_run_metadata(
        self,
        target_group: str,
        scheduling_artifacts: dict[str, Any],
        n_taxa: int,
        n_cap: int,
        abundance_threshold: int,
        t_pipeline_start: float,
    ) -> None:
        """Write run_metadata_{target_group}.json + the accession snapshot.

        Thin delegator to
        :func:`taxotreeset.core._orchestration._manifest._write_run_metadata`.
        """
        _write_run_metadata_fn(
            self,
            target_group,
            scheduling_artifacts,
            n_taxa,
            n_cap,
            abundance_threshold,
            t_pipeline_start,
        )

    def _persist_scheduling_artifacts(
        self,
        target_group: str,
        scheduling_artifacts: dict[str, Any],
    ) -> None:
        """Write the manifest, passthrough map, and virtual ID registry.

        Thin delegator to
        :func:`taxotreeset.core._orchestration._manifest._persist_scheduling_artifacts`.
        """
        _persist_scheduling_artifacts_fn(self, target_group, scheduling_artifacts)

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




    @staticmethod
    def _is_passthrough_case(children_list: list) -> bool:
        """Detect whether this node should be treated as a passthrough.

        Thin delegator to :func:`_scheduler._is_passthrough_case`.
        """
        return _is_passthrough_case_fn(children_list)





    def _reject_near_ratio(self, node: Node) -> float:
        """Depth-scaled near fraction of the reject bucket for a head.

        Thin delegator to :meth:`_CascadeScheduler._reject_near_ratio`.
        """
        return _CascadeScheduler(self)._reject_near_ratio(node)


    def _maybe_add_reject_class(
        self,
        current_node: Node,
        retained_children: list,
        per_child_tasks: dict[str, list[dict]],
        plan: dict,
        virtual_id_registry: dict,
    ) -> list:
        """Append a reject class of out-of-subtree negatives to the head.

        Thin delegator to :meth:`_CascadeScheduler._maybe_add_reject_class`.
        """
        return _CascadeScheduler(self)._maybe_add_reject_class(
            current_node,
            retained_children,
            per_child_tasks,
            plan,
            virtual_id_registry,
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
            leaf_tasks, class_index, rng, min_genomes_for_genome_split,
            cluster_aware=self.cluster_aware_split,
            max_subseq_len=self.max_subseq_len,
            cluster_params=self.cluster_params,
        )

