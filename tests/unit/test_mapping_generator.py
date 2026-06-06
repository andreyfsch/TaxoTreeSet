"""Tests for taxotreeset.dataset.mapping_generator — DynamicMappingGenerator."""

import pytest
from bigtree import Node
from taxotreeset.dataset.mapping_generator import DynamicMappingGenerator


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_node(name, rank, taxid=None, parent=None):
    node = Node(str(name), parent=parent)
    node.rank = rank
    node.scientific_name = str(name)
    if taxid is not None:
        node.taxid = str(taxid)
    return node


def make_seq_leaf(header_id, parent=None):
    node = Node(str(header_id), parent=parent)
    node.rank = "sequence"
    return node


def _attach_leaves(node, n_leaves):
    for i in range(n_leaves):
        make_seq_leaf(f"leaf_{node.name}_{i}", parent=node)


# ---------------------------------------------------------------------------
# DynamicMappingGenerator.__init__
# ---------------------------------------------------------------------------


class TestDynamicMappingGeneratorInit:
    def test_default_abundance_threshold(self):
        gen = DynamicMappingGenerator()
        assert gen.abundance_threshold == 5

    def test_custom_abundance_threshold(self):
        gen = DynamicMappingGenerator(abundance_threshold=10)
        assert gen.abundance_threshold == 10


# ---------------------------------------------------------------------------
# compile_mapping — structure
# ---------------------------------------------------------------------------


class TestCompileMappingStructure:
    def test_returns_dict_with_scopes_key(self):
        root = make_node("root", "root", taxid="10239")
        gen = DynamicMappingGenerator(abundance_threshold=5)
        result = gen.compile_mapping(root, target_rank="family")
        assert "scopes" in result

    def test_scopes_keyed_by_root_taxid(self):
        root = make_node("root", "root", taxid="10239")
        gen = DynamicMappingGenerator(abundance_threshold=5)
        result = gen.compile_mapping(root, target_rank="family")
        assert "10239" in result["scopes"]

    def test_scopes_fallback_to_root_when_no_taxid(self):
        root = Node("root")
        root.rank = "root"
        gen = DynamicMappingGenerator(abundance_threshold=5)
        result = gen.compile_mapping(root, target_rank="family")
        assert "root" in result["scopes"]

    def test_redirections_and_virtual_id_labels_present(self):
        root = make_node("root", "root", taxid="10239")
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        assert "redirections" in scope
        assert "virtual_id_labels" in scope

    def test_virtual_id_labels_contains_fallback(self):
        root = make_node("root", "root", taxid="10239")
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family", virtual_fallback_id=999201)[
            "scopes"
        ]["10239"]
        assert "999201" in scope["virtual_id_labels"]


# ---------------------------------------------------------------------------
# compile_mapping — redirection rules
# ---------------------------------------------------------------------------


class TestCompileMappingRedirections:
    def _build_tree(self, n_leaves_a, n_leaves_b):
        root = make_node("root", "root", taxid="10239")
        family_a = make_node("Coronaviridae", "family", taxid="11118", parent=root)
        family_b = make_node("Adenoviridae", "family", taxid="12227", parent=root)
        _attach_leaves(family_a, n_leaves_a)
        _attach_leaves(family_b, n_leaves_b)
        return root

    def test_abundant_family_gets_self_redirect(self):
        root = self._build_tree(n_leaves_a=10, n_leaves_b=10)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        assert scope["redirections"]["11118"]["target_id"] == "11118"

    def test_rare_family_gets_fallback_redirect(self):
        root = self._build_tree(n_leaves_a=1, n_leaves_b=10)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family", virtual_fallback_id=999201)[
            "scopes"
        ]["10239"]
        assert scope["redirections"]["11118"]["target_id"] == "999201"

    def test_abundant_family_target_id_equals_source_id(self):
        root = self._build_tree(n_leaves_a=10, n_leaves_b=10)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        redir = scope["redirections"]["11118"]
        assert redir["target_id"] == "11118"

    def test_family_at_exact_threshold_passes_filter(self):
        root = self._build_tree(n_leaves_a=5, n_leaves_b=10)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        assert scope["redirections"]["11118"]["target_id"] == "11118"

    def test_node_without_taxid_is_skipped(self):
        root = make_node("root", "root", taxid="10239")
        family = make_node("UnknownFamily", "family", parent=root)  # no taxid
        _attach_leaves(family, 10)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        assert len(scope["redirections"]) == 0

    def test_empty_tree_produces_empty_redirections(self):
        root = make_node("root", "root", taxid="10239")
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        assert scope["redirections"] == {}

    def test_both_families_above_threshold_both_self_redirect(self):
        root = self._build_tree(n_leaves_a=8, n_leaves_b=12)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family")["scopes"]["10239"]
        assert scope["redirections"]["11118"]["target_id"] == "11118"
        assert scope["redirections"]["12227"]["target_id"] == "12227"

    def test_custom_fallback_id_used_in_redirect(self):
        root = self._build_tree(n_leaves_a=1, n_leaves_b=10)
        gen = DynamicMappingGenerator(abundance_threshold=5)
        scope = gen.compile_mapping(root, target_rank="family", virtual_fallback_id=888000)[
            "scopes"
        ]["10239"]
        assert scope["redirections"]["11118"]["target_id"] == "888000"
