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
from typing import Any

from bigtree import Node

from taxotreeset.core.generation import (
    classify_children_by_rank,
    compute_balanced_extraction_plan,
    distribute_n_per_class_across_leaves,
    make_low_capacity_bucket_node,
    make_rare_taxa_bucket_node,
    register_virtual_bucket,
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

_DOMAIN_GROUP_TO_TAXID: dict[str, str] = {
    "viruses": "10239",
    "bacteria": "2",
    "archaea": "2157",
    "eukaryotes": "2759",
}

_STRATIFIED_TRAIN_RATIO: float = 0.70
_STRATIFIED_VAL_RATIO: float = 0.15
_SPLITS: tuple[str, ...] = ("train", "val", "test")


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
        self._schedule_pbar = None
        self._depth_boundary: str | None = None
        self._single_level: bool = False
        self._all_capacities: dict[str, int] | None = None

        self.downloader: NCBIDownloader = NCBIDownloader(
            registry=self.registry,
            vault_path=self.vault_path,
        )
        self.builder: DatasetBuilder = DatasetBuilder(
            output_dir=self.output_dir,
            max_subseq_len=self.max_subseq_len,
            seed=self.seed,
            output_format=self.output_format,
        )

    def _sync_with_ncbi(self, target_group: str) -> None:
        """Reconcile the registry and vault with NCBI for a scope.

        Re-runs discovery for the target group's domain so that new NCBI
        accessions enter the registry as pending, then reconciles the
        vault: accessions marked downloaded whose recorded headers are
        missing from the LMDB are reset to pending for re-download.

        Args:
            target_group: Domain identifier to synchronize.
        """
        domain_taxid = self._resolve_root_taxid(target_group)
        with open(self.config_path, encoding="utf-8") as handle:
            mapping_config = json.load(handle)
        discovery = DiscoveryOrchestrator(
            registry=self.registry,
            mapping_config=mapping_config,
        )
        discovery.discover_from_root(int(domain_taxid))
        self._reconcile_vault_against_registry()

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
        if sync:
            ui_logger.info("Syncing registry and vault with NCBI.")
            self._sync_with_ncbi(target_group)

        ui_logger.info("Stage 1/4: Downloading pending accessions.")
        self.downloader.download_all_pending()

        ui_logger.info("Stage 2/4: Building taxonomic tree.")
        domain_taxid = self._resolve_root_taxid(target_group)
        tree_root = self._build_target_tree(domain_taxid)

        if tree_root is None or not tree_root.children:
            if sync:
                ui_logger.error(
                    f"No data found for root '{target_group}' "
                    f"(TaxID {domain_taxid}) after syncing with NCBI. "
                    "Verify the root exists in NCBI RefSeq."
                )
            else:
                ui_logger.error(
                    f"No data found for root '{target_group}' "
                    f"(TaxID {domain_taxid}) in the registry. Re-run "
                    "without --no-sync to discover and download it "
                    "from NCBI."
                )
            return

        ui_logger.info("Stage 3/4: Scheduling extraction jobs.")
        self._all_capacities = self._load_or_compute_capacities(tree_root)
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

        ui_logger.info("Stage 4/4: Dispatching parallel disk extraction.")
        self._execute_extraction(scheduling_artifacts["extraction_jobs"])

        ui_logger.info("Pipeline finished successfully.")

    @staticmethod
    def _resolve_root_taxid(target_root: str) -> str:
        """Resolve the generation root to an NCBI TaxID string.

        Accepts a domain shortcut (viruses, bacteria, archaea,
        eukaryotes), a numeric TaxID, or a clade scientific name. The
        shortcuts are convenience aliases for the four superkingdom
        TaxIDs; anything else is resolved via taxoniq with an NCBI
        fallback.

        Args:
            target_root: Domain shortcut, numeric TaxID, or clade name.

        Returns:
            The resolved NCBI TaxID as a string.

        Raises:
            ValueError: If the reference cannot be resolved.
        """
        if target_root in _DOMAIN_GROUP_TO_TAXID:
            return _DOMAIN_GROUP_TO_TAXID[target_root]
        return resolve_to_taxid(target_root)
    def _build_target_tree(self, domain_taxid: str) -> Node | None:
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
        return compute_all_capacities(tree_root, min_len)

    def _schedule_pipeline_jobs(
        self,
        tree_root: Node,
        domain_taxid: str,
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
        self._schedule_pbar = tqdm(
            desc="Computing node capacities", unit=" nodes"
        )
        try:
            self._schedule_decision_point(
                current_node=domain_node,
                children_list=children_list,
                accumulated_path=domain_taxid,
                abundance_threshold=abundance_threshold,
                extraction_jobs=extraction_jobs,
                master_manifest=master_manifest,
                passthrough_map=passthrough_map,
                virtual_id_registry=virtual_id_registry,
                leaf_cache=leaf_cache,
            )
        finally:
            self._schedule_pbar.close()
            self._schedule_pbar = None

        return {
            "extraction_jobs": extraction_jobs,
            "master_manifest": master_manifest,
            "passthrough_map": passthrough_map,
            "virtual_id_registry": virtual_id_registry,
        }

    @staticmethod
    def _find_domain_node(tree_root: Node, domain_taxid: str) -> Node | None:
        """Locate the domain anchor node under the tree root.

        Args:
            tree_root: Tree root from tree_builder.
            domain_taxid: NCBI TaxID of the domain.

        Returns:
            The matching child Node, or None if not found.
        """
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

    def _prepare_stratified_split(
        self,
        leaves: list,
        rng: random.Random,
    ) -> dict[str, list]:
        """Split sequence leaves into train/val/test sets.

        Mirrors the logic in DatasetBuilder.prepare_stratified_split
        but operates on a flat list of leaves. Used during scheduling
        to apportion each child's leaves to splits before computing
        per-leaf extraction tasks.

        Args:
            leaves: Flat list of sequence leaf nodes.
            rng: random.Random instance for deterministic shuffling.

        Returns:
            Dictionary with 'train', 'val', 'test' keys; each value
            is a list of leaf nodes (or fraction tuples in the
            scarcity scenario).
        """
        splits: dict[str, list] = {key: [] for key in _SPLITS}
        if not leaves:
            return splits

        shuffled = list(leaves)
        rng.shuffle(shuffled)

        if len(shuffled) >= 3:
            return self._split_distinct_leaves(shuffled, splits)
        return self._split_fractions_per_leaf(shuffled, splits)

    @staticmethod
    def _split_distinct_leaves(
        shuffled_leaves: list,
        splits: dict[str, list],
    ) -> dict[str, list]:
        """Assign whole leaves to train/val/test by index ranges.

        Args:
            shuffled_leaves: Pre-shuffled list of leaves.
            splits: Empty splits dict to populate.

        Returns:
            The populated splits dict.
        """
        leaf_count = len(shuffled_leaves)
        train_cut = max(1, int(leaf_count * _STRATIFIED_TRAIN_RATIO))
        val_cut = train_cut + max(1, int(leaf_count * _STRATIFIED_VAL_RATIO))

        for index, leaf in enumerate(shuffled_leaves):
            if index < train_cut:
                splits["train"].append(leaf)
            elif index < val_cut:
                splits["val"].append(leaf)
            else:
                splits["test"].append(leaf)
        return splits

    @staticmethod
    def _split_fractions_per_leaf(
        leaves: list,
        splits: dict[str, list],
    ) -> dict[str, list]:
        """Slice each leaf's sequence across all three splits.

        Used in extreme scarcity (< 3 leaves). Each leaf produces
        one entry in every split, distinguished by its (start_pct,
        end_pct) fractions.

        Args:
            leaves: List of leaves (any order).
            splits: Empty splits dict to populate.

        Returns:
            The populated splits dict.
        """
        for leaf in leaves:
            splits["train"].append((leaf, 0.0, 0.70))
            splits["val"].append((leaf, 0.70, 0.85))
            splits["test"].append((leaf, 0.85, 1.0))
        return splits

    def _on_capacity_computed(self) -> None:
        """Advance the scheduling progress bar by one capacity computation."""
        if self._schedule_pbar is not None:
            self._schedule_pbar.update(1)

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
    ) -> dict[str, list[dict]]:
        """Split a single child's per-leaf tasks into train/val/test.

        Args:
            leaf_tasks: Per-leaf task dicts from
                ``distribute_n_per_class_across_leaves``.
            class_index: Numeric label index for this child.
            rng: Random instance for deterministic shuffling.

        Returns:
            Dictionary with 'train', 'val', 'test' keys; each value
            is a list of worker-ready task dicts with class_idx and
            split fractions filled in.
        """
        result: dict[str, list[dict]] = {split: [] for split in _SPLITS}

        if not leaf_tasks:
            return result

        shuffled = list(leaf_tasks)
        rng.shuffle(shuffled)

        if len(shuffled) >= 3:
            train_cut = max(1, int(len(shuffled) * _STRATIFIED_TRAIN_RATIO))
            val_cut = train_cut + max(1, int(len(shuffled) * _STRATIFIED_VAL_RATIO))

            for index, task in enumerate(shuffled):
                enriched = self._enrich_task(task, class_index, 0.0, 1.0)
                if index < train_cut:
                    result["train"].append(enriched)
                elif index < val_cut:
                    result["val"].append(enriched)
                else:
                    result["test"].append(enriched)
        else:
            for task in shuffled:
                n_total = task["n"]
                n_train = int(n_total * _STRATIFIED_TRAIN_RATIO)
                n_val = int(n_total * _STRATIFIED_VAL_RATIO)
                n_test = max(0, n_total - n_train - n_val)
                if n_train > 0:
                    result["train"].append(
                        self._enrich_task(
                            {**task, "n": n_train},
                            class_index,
                            0.0,
                            0.70,
                        )
                    )
                if n_val > 0:
                    result["val"].append(
                        self._enrich_task(
                            {**task, "n": n_val},
                            class_index,
                            0.70,
                            0.85,
                        )
                    )
                if n_test > 0:
                    result["test"].append(
                        self._enrich_task(
                            {**task, "n": n_test},
                            class_index,
                            0.85,
                            1.0,
                        )
                    )

        return result

    @staticmethod
    def _enrich_task(
        task: dict,
        class_index: int,
        start_pct: float,
        end_pct: float,
    ) -> dict:
        """Add slicing and class index fields to a per-leaf task.

        Args:
            task: Per-leaf task with 'fasta_path', 'header_id', 'n'.
            class_index: Numeric label for this child.
            start_pct: Sequence slicing start as fraction.
            end_pct: Sequence slicing end as fraction.

        Returns:
            Worker-ready task dict.
        """
        return {
            "fasta_path": task["fasta_path"],
            "header_id": task["header_id"],
            "n": task["n"],
            "class_idx": class_index,
            "start_pct": start_pct,
            "end_pct": end_pct,
        }

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
