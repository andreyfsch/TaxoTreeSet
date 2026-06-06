"""Tests for taxotreeset.dataset.analyzer — TaxonDiversityAnalyzer."""

from unittest.mock import patch

import pytest
from bigtree import Node
from taxotreeset.dataset.analyzer import TaxonDiversityAnalyzer

_MOCK_READ = "taxotreeset.dataset.analyzer._read_single_sequence"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_seq_leaf(header_id, fasta_path="/fake/vault", parent=None):
    node = Node(str(header_id), parent=parent)
    node.rank = "sequence"
    node.header_id = header_id
    node.fasta_path = fasta_path
    return node


def make_taxon_node(name, children=None):
    node = Node(str(name))
    node.rank = "species"
    node.scientific_name = str(name)
    if children:
        for child in children:
            child.parent = node
    return node


# ---------------------------------------------------------------------------
# TaxonDiversityAnalyzer.__init__
# ---------------------------------------------------------------------------


class TestTaxonDiversityAnalyzerInit:
    def test_default_max_subseq_len(self):
        analyzer = TaxonDiversityAnalyzer()
        assert analyzer.max_subseq_len == 2000

    def test_custom_max_subseq_len(self):
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=500)
        assert analyzer.max_subseq_len == 500


# ---------------------------------------------------------------------------
# get_unique_subseqs_count
# ---------------------------------------------------------------------------


class TestGetUniqueSubseqsCount:
    def test_returns_zero_for_node_with_no_sequence_leaves(self):
        node = make_taxon_node("root")
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=10)
        assert analyzer.get_unique_subseqs_count(node) == 0

    def test_returns_zero_when_sequence_shorter_than_window(self):
        leaf = make_seq_leaf("NC_001")
        parent = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=100)
        with patch(_MOCK_READ, return_value="ACGT" * 10):  # 40 bp < 100
            result = analyzer.get_unique_subseqs_count(parent)
        assert result == 0

    def test_returns_positive_count_for_long_sequence(self):
        leaf = make_seq_leaf("NC_001")
        parent = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=10)
        seq = "ACGT" * 100  # 400 bp
        with patch(_MOCK_READ, return_value=seq):
            result = analyzer.get_unique_subseqs_count(parent)
        assert result > 0

    def test_count_matches_expected_for_repetitive_sequence(self):
        # "AAAA" * n has only 1 unique 4-bp window (AAAA)
        leaf = make_seq_leaf("NC_001")
        parent = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=4)
        seq = "A" * 100
        with patch(_MOCK_READ, return_value=seq):
            result = analyzer.get_unique_subseqs_count(parent)
        assert result == 1

    def test_count_for_single_sliding_window_position(self):
        # seq length == window length → exactly 1 subseq
        leaf = make_seq_leaf("NC_001")
        parent = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=10)
        seq = "ACGTACGTAC"  # exactly 10 bp
        with patch(_MOCK_READ, return_value=seq):
            result = analyzer.get_unique_subseqs_count(parent)
        assert result == 1

    def test_count_is_not_negative(self):
        leaf = make_seq_leaf("NC_001")
        parent = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=50)
        with patch(_MOCK_READ, return_value="ACGT" * 200):
            result = analyzer.get_unique_subseqs_count(parent)
        assert result >= 0

    def test_returns_zero_when_read_returns_empty(self):
        leaf = make_seq_leaf("NC_001")
        parent = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=10)
        with patch(_MOCK_READ, return_value=""):
            result = analyzer.get_unique_subseqs_count(parent)
        assert result == 0

    def test_aggregates_across_multiple_leaves(self):
        leaf_a = make_seq_leaf("NC_A")
        leaf_b = make_seq_leaf("NC_B")
        parent = make_taxon_node("family", [leaf_a, leaf_b])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=4)

        def side_effect(fasta_path, header_id):
            return "ACGT" * 50 if header_id == "NC_A" else "TTTT" * 50

        with patch(_MOCK_READ, side_effect=side_effect):
            result = analyzer.get_unique_subseqs_count(parent)
        # ACGT*50 has several unique 4-mers; TTTT*50 has 1 (TTTT)
        assert result > 1


# ---------------------------------------------------------------------------
# calculate_bulk_capacities
# ---------------------------------------------------------------------------


class TestCalculateBulkCapacities:
    def test_returns_dict_with_one_key_per_node(self):
        leaf = make_seq_leaf("NC_001")
        node_a = make_taxon_node("species_a", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=4)
        with patch(_MOCK_READ, return_value="ACGT" * 50):
            result = analyzer.calculate_bulk_capacities([node_a])
        assert len(result) == 1

    def test_all_keys_are_path_strings(self):
        leaf = make_seq_leaf("NC_001")
        node = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=4)
        with patch(_MOCK_READ, return_value="ACGT" * 50):
            result = analyzer.calculate_bulk_capacities([node])
        for key in result:
            assert isinstance(key, str)
            assert "/" in key

    def test_empty_node_list_returns_empty_dict(self):
        analyzer = TaxonDiversityAnalyzer()
        assert analyzer.calculate_bulk_capacities([]) == {}

    def test_values_are_nonnegative_integers(self):
        leaf = make_seq_leaf("NC_001")
        node = make_taxon_node("species", [leaf])
        analyzer = TaxonDiversityAnalyzer(max_subseq_len=4)
        with patch(_MOCK_READ, return_value="ACGT" * 100):
            result = analyzer.calculate_bulk_capacities([node])
        for v in result.values():
            assert isinstance(v, int)
            assert v >= 0
