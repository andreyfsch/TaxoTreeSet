"""Stage-1 NCBI sync + selective-download manager for the orchestrator.

Extracted from ``generation_orchestrator.py``. ``_SyncManager`` wraps the
orchestrator (``ctx``) and owns the Stage-1 workflow: re-discovery, vault
reconciliation, the selective-download selection pass, capacity estimation from
the registry's ``total_sequence_length`` proxy, per-label target collection, and
the refinement loop. It reads config + the registry/downloader through ``ctx``
and writes back the shared ``ctx._selective_download_active`` flag. Intra-cluster
calls stay ``self._...``; anything on the orchestrator is ``self.ctx.``.
"""

import json
import logging
from typing import TYPE_CHECKING

from bigtree import Node

from taxotreeset.core.generation import (
    classify_children_by_rank,
    compute_balanced_extraction_plan,
)
from taxotreeset.core.generation.constants import is_recursion_terminator
from taxotreeset.core.orchestrator import DiscoveryOrchestrator
from taxotreeset.logging_utils import get_ui_logger
from taxotreeset.ranks import is_below_boundary

if TYPE_CHECKING:
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator

logger = logging.getLogger("TaxoTreeSet.Core.GenerationOrchestrator")
ui_logger = get_ui_logger()

_DOMAIN_GROUP_TO_TAXID: dict[str, str] = {
    "viruses": "10239",
    "bacteria": "2",
    "archaea": "2157",
    "eukaryotes": "2759",
}


class _SyncManager:
    """Stage-1 sync workflow bound to an orchestrator instance (``ctx``)."""

    def __init__(self, ctx: "GenerationOrchestrator") -> None:
        self.ctx = ctx

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
        domain_taxid = self.ctx._resolve_root_taxid(target_group)
        with open(self.ctx.config_path, encoding="utf-8") as handle:
            mapping_config = json.load(handle)
        discovery = DiscoveryOrchestrator(
            registry=self.ctx.registry,
            mapping_config=mapping_config,
            all_ranks=self.ctx.all_ranks,
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

        pending_volume = self.ctx.registry.get_pending_volume(domain_taxid)
        gib = pending_volume / 1024 ** 3
        threshold_gib = self.ctx.selective_download_threshold / 1024 ** 3
        if pending_volume >= self.ctx.selective_download_threshold:
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
        lineages = self.ctx.registry.registry.get("lineages", {})
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
        self.ctx._selective_download_active = True
        self.ctx.registry.reset_selection_flags(domain_taxid)

        estimation_tree = self.ctx._build_target_tree(domain_taxid)
        if estimation_tree is None or not estimation_tree.children:
            ui_logger.warning(
                "Could not build estimation tree for selective download; "
                "all pending accessions will be downloaded."
            )
            return

        estimated_capacities = self._estimate_capacities_from_registry(domain_taxid)
        downloaded_cap, pending_index = self._build_scope_accession_index(domain_taxid)

        domain_node = self.ctx._find_domain_node(estimation_tree, domain_taxid)
        if domain_node is None:
            return

        label_targets: dict[str, int] = {}
        self._collect_label_targets(
            node=domain_node,
            children_list=self.ctx._collect_real_children(domain_node),
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
        self.ctx.registry.mark_accessions_deferred(list(deferred))
        self.ctx.registry.save()

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
        lineages = self.ctx.registry.registry["lineages"]
        accessions = self.ctx.registry.registry["accessions"]
        taxons = self.ctx.registry.registry["taxons"]
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
        lineages = self.ctx.registry.registry["lineages"]
        accessions = self.ctx.registry.registry["accessions"]
        taxons = self.ctx.registry.registry["taxons"]
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
        lineages = self.ctx.registry.registry["lineages"]
        accessions = self.ctx.registry.registry["accessions"]
        taxons = self.ctx.registry.registry["taxons"]
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
        if self.ctx._is_passthrough_case(children_list):
            child = children_list[0]
            self._collect_label_targets(
                node=child,
                children_list=self.ctx._collect_real_children(child),
                estimated_capacities=estimated_capacities,
                targets=targets,
            )
            return

        effective_children, _ = classify_children_by_rank(
            node,
            children_list,
            min_subclades_per_bucket=self.ctx.min_subclades_per_bucket,
            all_ranks=self.ctx.all_ranks,
        )
        if not effective_children:
            return

        plan = compute_balanced_extraction_plan(
            parent_node=node,
            children=effective_children,
            leaf_cache={},
            min_len=self.ctx.min_subseq_len,
            min_num_seqs=self.ctx.min_num_seqs,
            cutoff_percentage=self.ctx.cutoff_percentage,
            use_exact_capacity=self.ctx.use_exact_capacity,
            max_n_per_class=self.ctx.max_n_per_class,
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

        if self.ctx._single_level:
            return

        for child in plan["retained_children"]:
            child_rank = getattr(child, "rank", "")
            if is_recursion_terminator(child_rank):
                continue
            if self.ctx._depth_boundary is not None and is_below_boundary(
                child_rank, self.ctx._depth_boundary
            ):
                continue
            grand_children = self.ctx._collect_real_children(child)
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
        capacity from ``ctx._all_capacities`` against its estimated
        target. For labels that fell short — meaning the size proxy
        over-estimated capacity due to repetitive genomic content —
        additional deferred accessions are undeferred (reference-assembly
        first, then by decreasing size) until the residual gap is covered
        or the deferred pool for that label is exhausted.

        Args:
            domain_taxid: Root TaxID of the scope being processed.
            tree_root: Taxonomic tree from the current download round.
                Only the node structure is used; sequence leaves are
                ignored because capacity comes from ``ctx._all_capacities``.

        Returns:
            True if at least one deferred accession was undeferred,
            indicating another download round is warranted. False when
            all labels meet their targets or no deferred accessions
            remain for shortfall labels.
        """
        estimated_capacities = self._estimate_capacities_from_registry(domain_taxid)
        label_targets: dict[str, int] = {}
        domain_node = self.ctx._find_domain_node(tree_root, domain_taxid)
        if domain_node is None:
            return False

        self._collect_label_targets(
            node=domain_node,
            children_list=self.ctx._collect_real_children(domain_node),
            estimated_capacities=estimated_capacities,
            targets=label_targets,
        )

        shortfall: dict[str, int] = {
            label: target - (self.ctx._all_capacities or {}).get(label, 0)
            for label, target in label_targets.items()
            if (self.ctx._all_capacities or {}).get(label, 0) < target
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

        registry_accessions = self.ctx.registry.registry["accessions"]
        for acc_id in newly_undeferred:
            if acc_id in registry_accessions:
                registry_accessions[acc_id]["download_deferred"] = False
        self.ctx.registry.save()

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
        lineages = self.ctx.registry.registry["lineages"]
        accessions = self.ctx.registry.registry["accessions"]
        taxons = self.ctx.registry.registry["taxons"]
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
        self.ctx.downloader.reconcile_with_vault()
