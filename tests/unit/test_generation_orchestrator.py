"""Tests for taxotreeset.core.generation_orchestrator — pure/isolated helpers."""

import json
import os
import random
from unittest.mock import MagicMock, patch

import pytest
from bigtree import Node

from taxotreeset.core.generation_orchestrator import (
    GenerationOrchestrator,
    _stratified_counts,
)


# ---------------------------------------------------------------------------
# _stratified_counts — within-genome subseq split (>=1 train; >=1 each at n>=3)
# ---------------------------------------------------------------------------


class TestStratifiedCounts:
    @pytest.mark.parametrize("n", list(range(3, 30)))
    def test_every_split_gets_at_least_one_at_n_ge_3(self, n):
        n_train, n_val, n_test = _stratified_counts(n)
        assert n_train >= 1
        assert n_val >= 1
        assert n_test >= 1
        assert n_train + n_val + n_test == n

    def test_train_never_empty_for_any_positive_n(self):
        # The regression: int(n*0.70) floored to 0 left a class untrainable.
        for n in range(1, 30):
            n_train, _, _ = _stratified_counts(n)
            assert n_train >= 1, f"n={n} left train empty"

    def test_zero_total_is_all_zero(self):
        assert _stratified_counts(0) == (0, 0, 0)

    def test_tiny_counts_fill_train_then_test(self):
        assert _stratified_counts(1) == (1, 0, 0)
        assert _stratified_counts(2) == (1, 0, 1)

    def test_three_splits_one_each(self):
        assert _stratified_counts(3) == (1, 1, 1)


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


def _make_orchestrator(tmp_path, registry_data=None):
    mock_registry = MagicMock()
    mock_registry.registry_path = str(tmp_path / "registry.json")
    mock_registry.registry = registry_data or {
        "taxons": {},
        "accessions": {},
        "lineages": {},
        "capacities": {},
    }
    return GenerationOrchestrator(
        registry=mock_registry,
        vault_path=str(tmp_path / "vault"),
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def orchestrator(tmp_path):
    return _make_orchestrator(tmp_path)


# ---------------------------------------------------------------------------
# __init__ — parameter validation
# ---------------------------------------------------------------------------


class TestGenerationOrchestratorInit:
    def test_raises_when_min_subseq_len_exceeds_max(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry_path = str(tmp_path / "registry.json")
        with pytest.raises(ValueError, match="min_subseq_len"):
            GenerationOrchestrator(
                registry=mock_registry,
                vault_path=str(tmp_path / "vault"),
                output_dir=str(tmp_path / "out"),
                min_subseq_len=500,
                max_subseq_len=200,
            )

    def test_stores_registry(self, orchestrator):
        assert orchestrator.registry is not None

    def test_creates_downloader(self, orchestrator):
        from taxotreeset.io.downloader import NCBIDownloader
        assert isinstance(orchestrator.downloader, NCBIDownloader)

    def test_creates_builder(self, orchestrator):
        from taxotreeset.dataset.builder import DatasetBuilder
        assert isinstance(orchestrator.builder, DatasetBuilder)


# ---------------------------------------------------------------------------
# _resolve_root_taxid
# ---------------------------------------------------------------------------


class TestResolveRootTaxid:
    def test_viruses_shortcut(self):
        assert GenerationOrchestrator._resolve_root_taxid("viruses") == "10239"

    def test_bacteria_shortcut(self):
        assert GenerationOrchestrator._resolve_root_taxid("bacteria") == "2"

    def test_archaea_shortcut(self):
        assert GenerationOrchestrator._resolve_root_taxid("archaea") == "2157"

    def test_eukaryotes_shortcut(self):
        assert GenerationOrchestrator._resolve_root_taxid("eukaryotes") == "2759"

    def test_numeric_taxid_passes_through(self):
        with patch(
            "taxotreeset.core.generation_orchestrator.resolve_to_taxid",
            return_value="11118",
        ):
            assert GenerationOrchestrator._resolve_root_taxid("11118") == "11118"

    def test_clade_name_delegates_to_resolve_to_taxid(self):
        with patch(
            "taxotreeset.core.generation_orchestrator.resolve_to_taxid",
            return_value="11234",
        ) as mock_resolve:
            result = GenerationOrchestrator._resolve_root_taxid("Coronaviridae")
        mock_resolve.assert_called_once_with("Coronaviridae")
        assert result == "11234"

    def test_all_resolves_to_none(self):
        assert GenerationOrchestrator._resolve_root_taxid("all") is None


# ---------------------------------------------------------------------------
# _domains_to_sync
# ---------------------------------------------------------------------------


class TestDomainsToSync:
    def test_returns_only_domains_present_in_lineages(self, tmp_path):
        reg_data = {
            "taxons": {},
            "accessions": {},
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "Virus"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ],
            },
            "capacities": {},
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        assert orch._domains_to_sync() == ["10239"]

    def test_falls_back_to_all_four_when_no_lineages(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        assert orch._domains_to_sync() == ["10239", "2", "2157", "2759"]


# ---------------------------------------------------------------------------
# _sync_plasmids — the --plasmids auto-sync (fetch + ingest + register)
# ---------------------------------------------------------------------------


class TestSyncPlasmids:
    def _plasmid_orch(self, tmp_path, no_fetch=False):
        mapping = tmp_path / "mapping.json"
        mapping.write_text('{"scopes": {}}', encoding="utf-8")
        orch = _make_orchestrator(tmp_path)
        orch.config_path = str(mapping)
        orch.plasmids = True
        orch.plasmid_release = None
        orch.plasmid_no_fetch = no_fetch
        return orch

    def test_fetches_ingests_and_registers_by_host(self, tmp_path):
        orch = self._plasmid_orch(tmp_path)
        reports = [{"accession": "NZ_P1.1"}]
        with (
            patch("taxotreeset.core._orchestration._sync.fetch_release") as m_fetch,
            patch("taxotreeset.core._orchestration._sync.iter_release_records",
                  return_value=iter([])),
            patch("taxotreeset.core._orchestration._sync.ingest_records_to_vault",
                  return_value=reports) as m_ingest,
            patch("taxotreeset.core._orchestration._sync.DiscoveryOrchestrator")
            as m_disc,
        ):
            orch._sync_plasmids()
        assert m_fetch.call_args.args[0].endswith("refseq_plasmid")  # default dir
        m_ingest.assert_called_once()
        call = m_disc.return_value.discover_from_reports.call_args
        assert call.args[0] == reports
        assert call.kwargs["root_id_str"] == "plasmids"
        assert call.kwargs["vault_lmdb_path"].endswith("sequences.lmdb")

    def test_no_fetch_skips_the_download(self, tmp_path):
        orch = self._plasmid_orch(tmp_path, no_fetch=True)
        with (
            patch("taxotreeset.core._orchestration._sync.fetch_release") as m_fetch,
            patch("taxotreeset.core._orchestration._sync.iter_release_records",
                  return_value=iter([])),
            patch("taxotreeset.core._orchestration._sync.ingest_records_to_vault",
                  return_value=[]),
            patch("taxotreeset.core._orchestration._sync.DiscoveryOrchestrator"),
        ):
            orch._sync_plasmids()
        m_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# _find_domain_node
# ---------------------------------------------------------------------------


class TestFindDomainNode:
    def test_returns_matching_child(self):
        root = Node("root")
        domain = Node("10239", parent=root)
        domain.rank = "superkingdom"
        result = GenerationOrchestrator._find_domain_node(root, "10239")
        assert result is domain

    def test_returns_none_when_not_found(self):
        root = Node("root")
        Node("11118", parent=root)
        result = GenerationOrchestrator._find_domain_node(root, "10239")
        assert result is None

    def test_returns_none_for_empty_root(self):
        root = Node("root")
        assert GenerationOrchestrator._find_domain_node(root, "10239") is None

    def test_returns_root_itself_for_none_domain(self):
        root = Node("root")
        Node("10239", parent=root)
        assert GenerationOrchestrator._find_domain_node(root, None) is root


# ---------------------------------------------------------------------------
# _collect_real_children
# ---------------------------------------------------------------------------


class TestCollectRealChildren:
    def test_returns_non_sequence_children(self):
        parent = Node("family")
        parent.rank = "family"
        child = Node("species", parent=parent)
        child.rank = "species"
        seq_leaf = Node("NC_001", parent=parent)
        seq_leaf.rank = "sequence"
        result = GenerationOrchestrator._collect_real_children(parent)
        assert child in result
        assert seq_leaf not in result

    def test_returns_empty_for_node_with_only_sequence_leaves(self):
        parent = Node("species")
        parent.rank = "species"
        seq = Node("NC_001", parent=parent)
        seq.rank = "sequence"
        result = GenerationOrchestrator._collect_real_children(parent)
        assert result == []

    def test_returns_empty_for_leaf_node(self):
        leaf = Node("NC_001")
        leaf.rank = "sequence"
        assert GenerationOrchestrator._collect_real_children(leaf) == []


# ---------------------------------------------------------------------------
# _is_passthrough_case
# ---------------------------------------------------------------------------


class TestIsPassthroughCase:
    def test_single_child_is_passthrough(self):
        child = Node("child")
        assert GenerationOrchestrator._is_passthrough_case([child]) is True

    def test_multiple_children_is_not_passthrough(self):
        children = [Node("a"), Node("b")]
        assert GenerationOrchestrator._is_passthrough_case(children) is False

    def test_empty_list_is_not_passthrough(self):
        assert GenerationOrchestrator._is_passthrough_case([]) is False


# ---------------------------------------------------------------------------
# _estimate_capacities_from_registry
# ---------------------------------------------------------------------------


class TestEstimateCapacitiesFromRegistry:
    def _registry_with_species(self, taxid, domain_taxid, seq_len, downloaded=True):
        return {
            "taxons": {taxid: ["acc1"]},
            "accessions": {
                "acc1": {
                    "total_sequence_length": seq_len,
                    "downloaded": downloaded,
                }
            },
            "lineages": {
                taxid: [
                    {"taxid": taxid, "rank": "species", "name": "SpeciesName"},
                    {"taxid": domain_taxid, "rank": "superkingdom", "name": "Viruses"},
                ]
            },
        }

    def test_estimates_species_capacity_from_seq_len(self, tmp_path):
        reg_data = self._registry_with_species("2697049", "10239", 30000)
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._estimate_capacities_from_registry("10239")
        # The leaf appears in its own stored lineage, but it is credited exactly
        # once (the set de-dupes the explicit leaf and the ancestor-loop leaf).
        assert result.get("2697049") == 30000

    def test_propagates_capacity_to_ancestor(self, tmp_path):
        reg_data = self._registry_with_species("2697049", "10239", 30000)
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._estimate_capacities_from_registry("10239")
        assert result.get("10239") == 30000

    def test_none_domain_includes_every_domain(self, tmp_path):
        reg_data = {
            "taxons": {"2697049": ["acc1"], "9999": ["acc2"]},
            "accessions": {
                "acc1": {"total_sequence_length": 30000, "downloaded": True},
                "acc2": {"total_sequence_length": 5000, "downloaded": True},
            },
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "Virus"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ],
                "9999": [
                    {"taxid": "9999", "rank": "species", "name": "Bacterium"},
                    {"taxid": "2", "rank": "superkingdom", "name": "Bacteria"},
                ],
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._estimate_capacities_from_registry(None)
        # None scope spans the whole registry: both domains get capacity.
        assert result.get("10239") == 30000
        assert result.get("2") == 5000

    def test_excludes_taxa_not_in_domain(self, tmp_path):
        reg_data = {
            "taxons": {"9999": ["acc2"]},
            "accessions": {"acc2": {"total_sequence_length": 5000, "downloaded": True}},
            "lineages": {
                "9999": [
                    {"taxid": "9999", "rank": "species", "name": "OutOfDomain"},
                    {"taxid": "2", "rank": "superkingdom", "name": "Bacteria"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._estimate_capacities_from_registry("10239")  # Viruses
        assert "9999" not in result

    def test_skips_taxa_with_zero_length(self, tmp_path):
        reg_data = self._registry_with_species("2697049", "10239", 0)
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._estimate_capacities_from_registry("10239")
        assert "2697049" not in result

    def test_aggregates_multiple_accessions_for_same_taxon(self, tmp_path):
        reg_data = {
            "taxons": {"2697049": ["acc1", "acc2"]},
            "accessions": {
                "acc1": {"total_sequence_length": 10000, "downloaded": True},
                "acc2": {"total_sequence_length": 20000, "downloaded": True},
            },
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._estimate_capacities_from_registry("10239")
        # 10000 + 20000 across the two accessions, credited once to the leaf.
        assert result.get("2697049") == 30000


# ---------------------------------------------------------------------------
# _collect_scope_pending_accessions
# ---------------------------------------------------------------------------


class TestCollectScopePendingAccessions:
    def test_returns_pending_accessions_in_scope(self, tmp_path):
        reg_data = {
            "taxons": {"2697049": ["acc1"]},
            "accessions": {"acc1": {"downloaded": False}},
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._collect_scope_pending_accessions("10239")
        assert "acc1" in result

    def test_excludes_already_downloaded(self, tmp_path):
        reg_data = {
            "taxons": {"2697049": ["acc1"]},
            "accessions": {"acc1": {"downloaded": True}},
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._collect_scope_pending_accessions("10239")
        assert "acc1" not in result

    def test_excludes_taxa_outside_scope(self, tmp_path):
        reg_data = {
            "taxons": {"9999": ["acc2"]},
            "accessions": {"acc2": {"downloaded": False}},
            "lineages": {
                "9999": [
                    {"taxid": "9999", "rank": "species", "name": "OutOfScope"},
                    {"taxid": "2", "rank": "superkingdom", "name": "Bacteria"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        result = orch._collect_scope_pending_accessions("10239")
        assert "acc2" not in result

    def test_empty_registry_returns_empty_set(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        result = orch._collect_scope_pending_accessions("10239")
        assert result == set()


# ---------------------------------------------------------------------------
# _build_scope_accession_index
# ---------------------------------------------------------------------------


class TestBuildScopeAccessionIndex:
    def test_pending_index_contains_accession_in_scope(self, tmp_path):
        reg_data = {
            "taxons": {"2697049": ["acc1"]},
            "accessions": {
                "acc1": {
                    "downloaded": False,
                    "is_reference": True,
                    "total_sequence_length": 30000,
                }
            },
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        downloaded_cap, pending_index = orch._build_scope_accession_index("10239")
        # Should appear under both species and ancestor labels
        assert "2697049" in pending_index or "10239" in pending_index

    def test_downloaded_accession_contributes_to_downloaded_cap(self, tmp_path):
        reg_data = {
            "taxons": {"2697049": ["acc1"]},
            "accessions": {
                "acc1": {
                    "downloaded": True,
                    "is_reference": True,
                    "total_sequence_length": 30000,
                }
            },
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ]
            },
        }
        orch = _make_orchestrator(tmp_path, reg_data)
        downloaded_cap, pending_index = orch._build_scope_accession_index("10239")
        # label_taxids = [taxid] + lineage entries, so species appears twice → 60000
        assert downloaded_cap.get("2697049", 0) == 60000

    def test_empty_registry_returns_empty_indices(self, tmp_path):
        orch = _make_orchestrator(tmp_path)
        downloaded_cap, pending_index = orch._build_scope_accession_index("10239")
        assert downloaded_cap == {}
        assert pending_index == {}


# ---------------------------------------------------------------------------
# _capture_tool_versions
# ---------------------------------------------------------------------------


class TestCaptureToolVersions:
    def test_returns_expected_keys_as_nonempty_strings(self):
        from taxotreeset.core.generation_orchestrator import _capture_tool_versions

        versions = _capture_tool_versions()
        assert set(versions) == {"datasets_cli", "taxoniq", "python", "platform"}
        assert all(isinstance(v, str) and v for v in versions.values())


# ---------------------------------------------------------------------------
# _maybe_add_reject_class
# ---------------------------------------------------------------------------


def _rej_leaf(name, parent):
    leaf = Node(str(name), parent=parent)
    leaf.rank = "sequence"
    leaf.header_id = str(name)
    leaf.fasta_path = "/fake/vault"
    return leaf


def _reject_node(name, parent, rank, sci):
    node = Node(str(name), parent=parent) if parent is not None else Node(str(name))
    node.rank = rank
    node.scientific_name = sci
    return node


def _reject_tree():
    """root → A → CA → [a1,a2]; root → B → [b1,b2]. Head under test is A."""
    root = _reject_node("1", None, "superkingdom", "Root")
    node_a = _reject_node("2", root, "kingdom", "A")
    node_b = _reject_node("3", root, "kingdom", "B")
    child_a = _reject_node("20", node_a, "family", "CA")
    _rej_leaf("a1", child_a)
    _rej_leaf("a2", child_a)
    _rej_leaf("b1", node_b)
    _rej_leaf("b2", node_b)
    return node_a, child_a


class TestMaybeAddRejectClass:
    def test_disabled_is_noop(self, orchestrator):
        orchestrator.reject_class = False
        node_a, child_a = _reject_tree()
        per_child_tasks = {"20": [{"header_id": "a1"}]}
        out = orchestrator._maybe_add_reject_class(
            current_node=node_a, retained_children=[child_a],
            per_child_tasks=per_child_tasks, plan={"n_per_class": 100},
            virtual_id_registry={},
        )
        assert out == [child_a]
        assert list(per_child_tasks.keys()) == ["20"]

    @patch(
        "taxotreeset.core.generation.distribution._read_sequence_cached",
        return_value="A" * 1000,
    )
    def test_enabled_appends_reject_from_external_leaves(self, _mock, orchestrator):
        orchestrator.reject_class = True
        orchestrator.reject_fraction = 1.0
        orchestrator.reject_near_far_start = 0.5
        orchestrator.reject_near_far_end = 0.5
        node_a, child_a = _reject_tree()
        per_child_tasks = {"20": []}
        registry: dict = {}

        out = orchestrator._maybe_add_reject_class(
            current_node=node_a, retained_children=[child_a],
            per_child_tasks=per_child_tasks, plan={"n_per_class": 100},
            virtual_id_registry=registry,
        )

        assert len(out) == 2
        reject_node = out[-1]
        assert reject_node.rank == "virtual_reject"
        reject_taxid = str(reject_node.name)
        assert reject_taxid in per_child_tasks and per_child_tasks[reject_taxid]
        headers = {task["header_id"] for task in per_child_tasks[reject_taxid]}
        assert headers <= {"b1", "b2"}            # external leaves only
        assert not (headers & {"a1", "a2"})       # never the head's own leaves
        assert registry[reject_taxid]["purpose"] == "reject"

    def test_near_ratio_interpolates_over_decidable_depth(self, orchestrator):
        # Branching (non-passthrough) tree — each node is a real decision with
        # >=2 taxonomic children. Decidable depths A=2, CA=3, GA=4; d_max=4, so
        # the near fraction runs start -> end linearly over decidable depth.
        root = _reject_node("1", None, "superkingdom", "Root")
        a = _reject_node("2", root, "kingdom", "A")
        _reject_node("3", root, "kingdom", "B")
        ca = _reject_node("20", a, "phylum", "CA")
        _reject_node("21", a, "phylum", "CB")
        ga = _reject_node("200", ca, "class", "GA")
        _reject_node("201", ca, "class", "GB")
        _rej_leaf("g1", ga)
        orchestrator.reject_near_far_start = 0.4
        orchestrator.reject_near_far_end = 0.9
        assert orchestrator._reject_near_ratio(a) == pytest.approx(0.4)     # d2
        assert orchestrator._reject_near_ratio(ca) == pytest.approx(0.65)   # d3
        assert orchestrator._reject_near_ratio(ga) == pytest.approx(0.9)    # d4

    def test_near_ratio_ignores_passthrough_depth(self, orchestrator):
        # A node under a long single-child (passthrough) chain has high RAW tree
        # depth but shallow DECIDABLE depth (passthroughs are not heads and prune
        # nothing) — its near fraction must be `start`, not near-heavy.
        root = _reject_node("1", None, "superkingdom", "Root")
        k1 = _reject_node("2", root, "clade", "k1")        # passthrough chain
        g1 = _reject_node("3", k1, "kingdom", "g1")
        g2 = _reject_node("4", g1, "phylum", "g2")
        pt_leaf = _reject_node("5", g2, "species", "ptleaf")
        _rej_leaf("s1", pt_leaf)
        k2 = _reject_node("6", root, "clade", "k2")        # branching side (deep)
        _reject_node("7", k2, "kingdom", "m2")
        m1 = _reject_node("8", k2, "kingdom", "m1")
        _reject_node("9", m1, "phylum", "x2")
        deep = _reject_node("10", m1, "phylum", "x1")
        _rej_leaf("d1", deep)
        orchestrator.reject_near_far_start = 0.5
        orchestrator.reject_near_far_end = 0.9
        assert pt_leaf.depth > deep.depth                  # raw depth: 5 > 4 ...
        assert orchestrator._reject_near_ratio(pt_leaf) == pytest.approx(0.5)  # ...but decidable d2
        assert orchestrator._reject_near_ratio(deep) == pytest.approx(0.9)     # decidable d4

    def test_near_ratio_flat_when_start_equals_end(self, orchestrator):
        orchestrator.reject_near_far_start = 0.6
        orchestrator.reject_near_far_end = 0.6
        node_a, child_a = _reject_tree()
        assert orchestrator._reject_near_ratio(node_a) == pytest.approx(0.6)
        assert orchestrator._reject_near_ratio(child_a) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# _materialize_leaf_split — < min_genomes window-slicing fallback (scarcity)
# ---------------------------------------------------------------------------


def _leaf_task(n: int, name: str = "acc") -> dict:
    return {"fasta_path": f"/vault/{name}.fa", "header_id": name, "n": n}


class TestMaterializeLeafSplitScarcity:
    def test_single_genome_n1_lands_in_train_not_test_only(self, orchestrator):
        # 1 genome (< 3 -> scarcity path) with n=1 used to go test-only, leaving
        # the class untrainable. It must now land in train.
        split = orchestrator._materialize_leaf_split(
            [_leaf_task(1)], class_index=0, rng=random.Random(0)
        )
        assert len(split["train"]) == 1
        assert split["val"] == []
        assert split["test"] == []

    def test_scarcity_n3_fills_all_three_splits(self, orchestrator):
        split = orchestrator._materialize_leaf_split(
            [_leaf_task(3)], class_index=0, rng=random.Random(0)
        )
        assert len(split["train"]) == 1
        assert len(split["val"]) == 1
        assert len(split["test"]) == 1

    def test_scarcity_never_leaves_train_empty(self, orchestrator):
        for n in range(1, 12):
            split = orchestrator._materialize_leaf_split(
                [_leaf_task(n)], class_index=0, rng=random.Random(0)
            )
            assert len(split["train"]) >= 1, f"n={n} left train empty"

    def test_enriched_tasks_carry_positional_fractions(self, orchestrator):
        split = orchestrator._materialize_leaf_split(
            [_leaf_task(5)], class_index=2, rng=random.Random(0)
        )
        assert split["train"][0]["start_pct"] == 0.0
        assert split["train"][0]["end_pct"] == 0.70
        assert split["test"][0]["end_pct"] == 1.0
        assert split["train"][0]["class_idx"] == 2


class TestMaterializeLeafSplitVolumeBalance:
    """Genome-level split balances WINDOW VOLUME, not just genome count."""

    # A few large genomes dominate the window volume (like a binary head's
    # negatives). Count-based splitting sent val's volume anywhere from 0% to 45%.
    _UNEQUAL = [5000, 3000, 2000, 500, 300, 200, 100, 80, 60, 40, 20, 10]

    def _volumes(self, orchestrator, seed):
        tasks = [_leaf_task(n, f"g{i}") for i, n in enumerate(self._UNEQUAL)]
        split = orchestrator._materialize_leaf_split(
            tasks, class_index=0, rng=random.Random(seed),
            min_genomes_for_genome_split=4)
        total = sum(self._UNEQUAL)
        return {s: sum(t["n"] for t in split[s]) / total for s in
                ("train", "val", "test")}

    def test_volume_split_is_stable_across_seeds(self, orchestrator):
        for seed in range(12):
            v = self._volumes(orchestrator, seed)
            assert 0.08 <= v["val"] <= 0.25, f"seed {seed}: val vol {v['val']:.2f}"
            assert 0.08 <= v["test"] <= 0.25, f"seed {seed}: test vol {v['test']:.2f}"
            assert 0.55 <= v["train"] <= 0.80, f"seed {seed}: train vol {v['train']:.2f}"

    def test_three_equal_genomes_fill_every_split(self, orchestrator):
        # Volume-greedy packing can starve a split; the non-empty guarantee holds.
        tasks = [_leaf_task(10, f"g{i}") for i in range(3)]
        split = orchestrator._materialize_leaf_split(
            tasks, class_index=0, rng=random.Random(0),
            min_genomes_for_genome_split=3)
        assert len(split["train"]) >= 1
        assert len(split["val"]) >= 1
        assert len(split["test"]) >= 1

    def test_whole_genomes_never_straddle_splits(self, orchestrator):
        # Leakage safety: each genome appears in exactly one split.
        tasks = [_leaf_task(n, f"g{i}") for i, n in enumerate(self._UNEQUAL)]
        split = orchestrator._materialize_leaf_split(
            tasks, class_index=0, rng=random.Random(3),
            min_genomes_for_genome_split=4)
        seen = [t["header_id"] for s in ("train", "val", "test") for t in split[s]]
        assert len(seen) == len(set(seen)) == len(self._UNEQUAL)


# ---------------------------------------------------------------------------
# _write_label_maps — duplicate class names stay bijective
# ---------------------------------------------------------------------------


class TestWriteLabelMapsCollision:
    def test_duplicate_class_names_disambiguated(self, tmp_path, orchestrator):
        head_dir = str(tmp_path / "head")
        artifacts = {
            "master_manifest": {
                "100": {
                    "directory_path": head_dir,
                    "scientific_name": "Parent",
                    "rank": "genus",
                    "labels": {
                        "1": {"class_idx": 0, "name": "Homonym", "rank": "species"},
                        "2": {"class_idx": 1, "name": "Homonym", "rank": "species"},
                    },
                }
            }
        }
        orchestrator._write_label_maps(artifacts)
        with open(os.path.join(head_dir, "label_map.json"), encoding="utf-8") as f:
            label_map = json.load(f)
        # Both classes survive the collision: no silent dict-key overwrite.
        assert len(label_map["id2label"]) == 2
        assert len(label_map["label2id"]) == 2
        assert set(label_map["id2label"].keys()) == {"0", "1"}
        # label2id is the inverse of id2label (bijective).
        assert {v: k for k, v in label_map["label2id"].items()} == {
            0: label_map["id2label"]["0"],
            1: label_map["id2label"]["1"],
        }


# ---------------------------------------------------------------------------
# _write_label_maps — keep-imbalance metadata (P5)
# ---------------------------------------------------------------------------


class TestWriteLabelMapsBalanceMode:
    def test_keep_mode_emits_counts_and_balanced_weights(self, tmp_path, orchestrator):
        head_dir = str(tmp_path / "head")
        artifacts = {
            "master_manifest": {
                "100": {
                    "directory_path": head_dir,
                    "scientific_name": "Parent",
                    "rank": "genus",
                    "balance_mode": "keep",
                    "labels": {
                        "1": {"class_idx": 0, "name": "A", "rank": "species",
                              "n_windows": 100},
                        "2": {"class_idx": 1, "name": "B", "rank": "species",
                              "n_windows": 300},
                    },
                }
            }
        }
        orchestrator._write_label_maps(artifacts)
        with open(os.path.join(head_dir, "label_map.json"), encoding="utf-8") as f:
            lm = json.load(f)
        assert lm["balance_mode"] == "keep"
        by_idx = {c["class_idx"]: c for c in lm["classes"]}
        assert by_idx[0]["n_windows"] == 100
        assert by_idx[1]["n_windows"] == 300
        # "balanced" weights: total=400, k=2 -> 400/(2*100)=2.0, 400/(2*300)=0.6667
        assert lm["class_weights"]["0"] == pytest.approx(2.0)
        assert lm["class_weights"]["1"] == pytest.approx(0.6667, abs=1e-3)

    def test_undersample_mode_has_no_weights(self, tmp_path, orchestrator):
        head_dir = str(tmp_path / "head")
        artifacts = {
            "master_manifest": {
                "100": {
                    "directory_path": head_dir, "scientific_name": "P",
                    "rank": "genus",
                    "labels": {
                        "1": {"class_idx": 0, "name": "A", "rank": "species"},
                        "2": {"class_idx": 1, "name": "B", "rank": "species"},
                    },
                }
            }
        }
        orchestrator._write_label_maps(artifacts)
        with open(os.path.join(head_dir, "label_map.json"), encoding="utf-8") as f:
            lm = json.load(f)
        assert lm["balance_mode"] == "undersample"  # default when absent
        assert "class_weights" not in lm


# ---------------------------------------------------------------------------
# Multi-root scope resolution + empty-root forest (P9 fatia 1)
# ---------------------------------------------------------------------------


class TestResolveScopeTaxids:
    def test_single_shortcut(self, orchestrator):
        assert orchestrator._resolve_scope_taxids("viruses") == frozenset({"10239"})

    def test_all_returns_none(self, orchestrator):
        assert orchestrator._resolve_scope_taxids("all") is None

    def test_multiple_shortcuts(self, orchestrator):
        assert orchestrator._resolve_scope_taxids("viruses,archaea") == frozenset(
            {"10239", "2157"}
        )

    def test_numeric_taxids(self, orchestrator):
        assert orchestrator._resolve_scope_taxids("10239,2157") == frozenset(
            {"10239", "2157"}
        )

    def test_whitespace_stripped_and_deduped(self, orchestrator):
        assert orchestrator._resolve_scope_taxids(" viruses , viruses ") == frozenset(
            {"10239"}
        )

    def test_all_combined_with_others_raises(self, orchestrator):
        with pytest.raises(ValueError, match="cannot be combined"):
            orchestrator._resolve_scope_taxids("all,viruses")

    def test_empty_raises(self, orchestrator):
        with pytest.raises(ValueError):
            orchestrator._resolve_scope_taxids("  ,  ")


class TestScopeAnchor:
    def test_none_scope_anchors_at_empty_root(self):
        assert GenerationOrchestrator._scope_anchor(None) is None

    def test_single_domain_anchors_at_its_node(self):
        assert GenerationOrchestrator._scope_anchor(frozenset({"10239"})) == "10239"

    def test_multi_domain_anchors_at_empty_root(self):
        assert GenerationOrchestrator._scope_anchor(frozenset({"10239", "2157"})) is None


class TestBuildTargetTreeForest:
    @staticmethod
    def _fake_tree(taxid):
        root = Node("root", rank="root")
        Node(str(taxid), parent=root, rank="superkingdom")
        return root

    def test_single_str_builds_one_tree(self, orchestrator):
        with patch(
            "taxotreeset.core.generation_orchestrator.generate_seqs_by_taxon_tree"
        ) as mock_gen:
            mock_gen.side_effect = lambda **kw: self._fake_tree(kw["domain_taxid"])
            tree = orchestrator._build_target_tree("10239")
        assert mock_gen.call_count == 1
        assert [c.name for c in tree.children] == ["10239"]

    def test_single_element_set_builds_one_tree(self, orchestrator):
        with patch(
            "taxotreeset.core.generation_orchestrator.generate_seqs_by_taxon_tree"
        ) as mock_gen:
            mock_gen.side_effect = lambda **kw: self._fake_tree(kw["domain_taxid"])
            orchestrator._build_target_tree(frozenset({"10239"}))
        assert mock_gen.call_count == 1

    def test_multi_root_merges_domains_under_empty_root(self, orchestrator):
        with patch(
            "taxotreeset.core.generation_orchestrator.generate_seqs_by_taxon_tree"
        ) as mock_gen:
            mock_gen.side_effect = lambda **kw: self._fake_tree(kw["domain_taxid"])
            tree = orchestrator._build_target_tree(frozenset({"10239", "2157"}))
        assert mock_gen.call_count == 2
        assert tree.rank == "root"
        # both domains become top-level children of the one empty root
        assert sorted(c.name for c in tree.children) == ["10239", "2157"]

    def test_multi_root_skips_failed_domain_builds(self, orchestrator):
        def _gen(**kw):
            return None if kw["domain_taxid"] == "2157" else self._fake_tree(kw["domain_taxid"])
        with patch(
            "taxotreeset.core.generation_orchestrator.generate_seqs_by_taxon_tree",
            side_effect=_gen,
        ):
            tree = orchestrator._build_target_tree(frozenset({"10239", "2157"}))
        assert [c.name for c in tree.children] == ["10239"]
