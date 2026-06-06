"""Shared fixtures for the TaxoTreeSet test suite."""

import json
import pytest
from bigtree import Node


def make_node(name, rank="no_rank", scientific_name=None, parent=None):
    """Create a bigtree Node with the attributes expected by the pipeline."""
    node = Node(str(name), parent=parent)
    node.rank = rank
    node.scientific_name = scientific_name or str(name)
    return node


def make_seq_leaf(header_id, seq_len, fasta_path="/tmp/test.fasta", parent=None):
    """Create a sequence-rank leaf node."""
    node = Node(str(header_id), parent=parent)
    node.rank = "sequence"
    node.header_id = header_id
    node.seq_len = seq_len
    node.fasta_path = fasta_path
    return node


@pytest.fixture
def minimal_mapping(tmp_path):
    """Write a minimal mapping.json and return its path."""
    mapping = {"domains": [], "redirects": {}}
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(mapping), encoding="utf-8")
    return str(p)


@pytest.fixture
def noise_config(tmp_path):
    """Write a minimal noise_patterns.json and return its path."""
    config = {
        "name_patterns": [
            {"regex": r"^unclassified\s+", "description": "unclassified containers"},
            {"regex": r"environmental samples", "description": "environmental samples"},
            {"regex": r"incertae sedis", "description": "placement uncertain"},
        ],
        "rank_blacklist": {"ranks": ["strain", "serotype", "subtype"]},
    }
    p = tmp_path / "noise_patterns.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return str(p)
