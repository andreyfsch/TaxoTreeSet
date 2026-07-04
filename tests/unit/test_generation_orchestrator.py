"""Tests for taxotreeset.core.generation_orchestrator — pure/isolated helpers."""

from unittest.mock import MagicMock, patch

import pytest
from bigtree import Node

from taxotreeset.core.generation_orchestrator import (
    GenerationOrchestrator,
    _stratified_cuts,
)


# ---------------------------------------------------------------------------
# _stratified_cuts — leaf-level split boundaries (>=1 leaf per split)
# ---------------------------------------------------------------------------


class TestStratifiedCuts:
    @pytest.mark.parametrize("leaf_count", list(range(3, 25)))
    def test_every_split_gets_at_least_one_leaf(self, leaf_count):
        train_cut, val_cut = _stratified_cuts(leaf_count)
        train = train_cut
        val = val_cut - train_cut
        test = leaf_count - val_cut
        assert train >= 1
        assert val >= 1
        assert test >= 1
        assert train + val + test == leaf_count

    def test_three_leaves_split_one_each(self):
        # The exact case that previously yielded test=0.
        assert _stratified_cuts(3) == (1, 2)

    def test_large_count_keeps_70_15_15_shape(self):
        train_cut, val_cut = _stratified_cuts(100)
        assert train_cut == 70
        assert val_cut - train_cut == 15
        assert 100 - val_cut == 15


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
        # Species appears in its own lineage list, so seq_len is added twice
        # (once via result[taxid] and once via the ancestor loop).
        assert result.get("2697049") == 60000

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
        # species_cap = 30000; counted twice (see test_estimates_species_capacity_from_seq_len)
        assert result.get("2697049") == 60000


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

    def test_near_ratio_interpolates_start_to_end_by_depth(self, orchestrator):
        # _reject_tree: A (kingdom, depth 2) → CA (family, depth 3) → seq leaves
        # (depth 4). d_min=2, tree max_depth=4, so the near fraction runs from
        # start (shallowest head) to end (deepest) linearly over depth.
        orchestrator.reject_near_far_start = 0.4
        orchestrator.reject_near_far_end = 0.9
        node_a, child_a = _reject_tree()
        assert orchestrator._reject_near_ratio(node_a) == pytest.approx(0.4)     # d2
        assert orchestrator._reject_near_ratio(child_a) == pytest.approx(0.65)   # d3

    def test_near_ratio_flat_when_start_equals_end(self, orchestrator):
        orchestrator.reject_near_far_start = 0.6
        orchestrator.reject_near_far_end = 0.6
        node_a, child_a = _reject_tree()
        assert orchestrator._reject_near_ratio(node_a) == pytest.approx(0.6)
        assert orchestrator._reject_near_ratio(child_a) == pytest.approx(0.6)
