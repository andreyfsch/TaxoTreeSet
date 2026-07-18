"""Recursive cascade scheduler for the generation orchestrator.

Extracted from ``generation_orchestrator.py``: the depth-first traversal that,
at each decision point, applies rank-aware bucketing, per-class balancing,
low-capacity / rare-taxa / reject bucketing, and n-per-class distribution, then
builds the extraction jobs (multi-class) or streams batched binary heads.

``_CascadeScheduler`` wraps the orchestrator (``ctx``) and owns the transient
scheduling progress bar; the per-run bookkeeping (extraction_jobs, master
manifest, passthrough map, virtual-id registry, leaf cache) is created in
``_schedule_pipeline_jobs`` and threaded through the recursion as before. The
pure tree helpers are module-level functions. Config, the builder, the registry,
and ``_all_capacities`` are read through ``ctx``; intra-cluster calls stay
``self._...``.
"""

import logging
import os
import random
import time
from typing import TYPE_CHECKING, Any

from bigtree import Node
from tqdm import tqdm

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
from taxotreeset.core.generation.constants import is_recursion_terminator
from taxotreeset.core._orchestration._splits import _SPLITS
from taxotreeset.logging_utils import get_ui_logger
from taxotreeset.ranks import (
    is_below_boundary,
)

if TYPE_CHECKING:
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator

logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")
ui_logger = get_ui_logger()


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


class _CascadeScheduler:
    """Recursive extraction-job scheduler bound to an orchestrator (``ctx``)."""

    def __init__(self, ctx: "GenerationOrchestrator") -> None:
        self.ctx = ctx
        self._schedule_pbar = None

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

        domain_node = _find_domain_node(tree_root, domain_taxid)
        if domain_node is None:
            logger.warning(f"Domain node {domain_taxid} not found in tree.")
            return {
                "extraction_jobs": extraction_jobs,
                "master_manifest": master_manifest,
                "passthrough_map": passthrough_map,
                "virtual_id_registry": virtual_id_registry,
            }

        children_list = _collect_real_children(domain_node)
        self._schedule_pbar = None
        try:
            if self.ctx.binary_only:
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
        ``self.ctx.binary_extract_batch_size`` heads instead of accumulating every
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
        caps = self.ctx._all_capacities or {}
        nodes = [
            n for n in domain_node.descendants
            if getattr(n, "rank", "") != "sequence"
        ]
        total = len(nodes)
        batch_size = max(1, self.ctx.binary_extract_batch_size)
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
            self.ctx.builder.build_node_dataset(batch, parallel=True)
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
            node_children = _collect_real_children(node)
            if _is_passthrough_case(node_children):
                passthrough_map[taxid] = str(node_children[0].name)
                passthrough += 1
                continue
            cap = caps.get(taxid, 0)
            budget = min(self.ctx.binary_budget, cap) if cap else self.ctx.binary_budget
            if budget <= 0:
                skipped += 1
                continue
            name = getattr(node, "scientific_name", taxid)

            pos_tasks = distribute_n_per_class_across_leaves(
                n_per_class=budget, children=[node], parent_taxid=taxid,
                parent_name=name, leaf_cache=leaf_cache,
                min_subseq_len=self.ctx.min_subseq_len,
            ).get(taxid, [])
            near, far = sample_reject_leaves(node, rng=random.Random(self.ctx.seed))
            neg_tasks = build_reject_tasks(
                near_leaves=near, far_leaves=far, n_reject=budget,
                near_far_ratio=self._reject_near_ratio(node),
                min_subseq_len=self.ctx.min_subseq_len,
            )
            if not pos_tasks or not neg_tasks:
                skipped += 1
                continue

            rng = random.Random(self.ctx.seed)
            pos_split = self.ctx._materialize_leaf_split(
                pos_tasks, 1, rng, min_genomes_for_genome_split=4)
            neg_split = self.ctx._materialize_leaf_split(
                neg_tasks, 0, rng, min_genomes_for_genome_split=4)
            parent_tasks = {s: pos_split[s] + neg_split[s] for s in _SPLITS}
            if not any(parent_tasks[s] for s in _SPLITS):
                skipped += 1
                continue

            path_parts = [p for p in node.path_name.split("/") if p]
            target_dir = os.path.join(self.ctx.output_dir, *path_parts)
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
                self.ctx.max_subseq_len, self.ctx.seed, self.ctx.output_format,
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
        if _is_passthrough_case(children_list):
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
            min_subclades_per_bucket=self.ctx.min_subclades_per_bucket,
            all_ranks=self.ctx.all_ranks,
        )
        _register_virtual_buckets(
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
            min_len=self.ctx.min_subseq_len,
            min_num_seqs=self.ctx.min_num_seqs,
            cutoff_percentage=self.ctx.cutoff_percentage,
            use_exact_capacity=self.ctx.use_exact_capacity,
            max_n_per_class=self.ctx.max_n_per_class,
            min_leaves_per_class=self.ctx.min_leaves_per_class,
            rare_taxa_strategy=self.ctx.rare_taxa_strategy,
            progress_callback=self._on_capacity_computed,
            capacity_override=self.ctx._all_capacities,
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
        next_children = _collect_real_children(child)

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
        if self.ctx._single_level:
            return
        for child in retained_children:
            child_rank = getattr(child, "rank", "")
            if is_recursion_terminator(child_rank):
                continue
            if self.ctx._depth_boundary is not None and is_below_boundary(
                child_rank, self.ctx._depth_boundary
            ):
                continue

            grand_children = _collect_real_children(child)
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
        start, end = self.ctx.reject_near_far_start, self.ctx.reject_near_far_end
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
            return self.ctx._dd_map
        depths: dict[int, int] = {}
        stack = [(root, 0)]
        while stack:
            n, parent_d = stack.pop()
            kids = [c for c in n.children if getattr(c, "rank", "") != "sequence"]
            d = parent_d + (0 if len(kids) == 1 else 1)   # passthrough adds 0
            depths[id(n)] = d
            for c in kids:
                stack.append((c, d))
        self.ctx._dd_root_id = id(root)
        self.ctx._dd_map = depths
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

        When ``self.ctx.reject_class`` is enabled, samples sequence leaves from
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
        if not self.ctx.reject_class:
            return retained_children

        near_leaves, far_leaves = sample_reject_leaves(
            current_node, rng=random.Random(self.ctx.seed)
        )
        n_reject = round(plan["n_per_class"] * self.ctx.reject_fraction)
        reject_tasks = build_reject_tasks(
            near_leaves=near_leaves,
            far_leaves=far_leaves,
            n_reject=n_reject,
            near_far_ratio=self._reject_near_ratio(current_node),
            min_subseq_len=self.ctx.min_subseq_len,
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
        target_dir = os.path.join(self.ctx.output_dir, *accumulated_path.split("/"))
        os.makedirs(target_dir, exist_ok=True)

        parent_tasks: dict[str, list[dict]] = {split: [] for split in _SPLITS}
        labels_metadata: dict[str, dict] = {}

        rng = random.Random(self.ctx.seed)

        for class_index, child in enumerate(retained_children):
            child_taxid = str(child.name)
            leaf_tasks = per_child_tasks.get(child_taxid, [])
            if not leaf_tasks:
                continue

            leaf_split = self.ctx._materialize_leaf_split(
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
            self.ctx.max_subseq_len,
            self.ctx.seed,
            self.ctx.output_format,
        )
