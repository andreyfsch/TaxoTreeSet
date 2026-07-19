"""Unit tests for the ``--single-level <taxid>`` scheduler helpers.

These cover the two pure pieces that let the cascade schedule a single head
anywhere in the tree (rather than only the root's): locating the target node and
rebuilding the accumulated TaxID path the full descent would have produced. The
end-to-end behaviour (only the target head emitted, negatives still sampled from
the whole tree) is covered in ``tests/integration/test_synthetic_pipeline.py``.
"""

from types import SimpleNamespace

from bigtree import Node

from taxotreeset.core._orchestration._scheduler import (
    _CascadeScheduler,
    _accumulated_path_to,
)


class TestAccumulatedPathTo:
    def test_rebuilds_domain_to_target_path(self):
        root = Node("root")
        domain = Node("10239", parent=root)
        family = Node("11118", parent=domain)
        species = Node("2697049", parent=family)
        # The cascade threads "10239" then appends each child on the way down;
        # the direct rebuild must reproduce it exactly so the head lands in the
        # same output directory a full run would use.
        assert _accumulated_path_to(domain, species) == "10239/11118/2697049"

    def test_domain_node_is_its_own_path(self):
        domain = Node("10239", parent=Node("root"))
        assert _accumulated_path_to(domain, domain) == "10239"

    def test_includes_passthrough_ancestors(self):
        # Passthrough (single-child) nodes are part of the cascade path, so the
        # rebuild must include them too — it walks the real ancestry.
        root = Node("root")
        domain = Node("10239", parent=root)
        passthrough = Node("12227", parent=domain)  # single-child clade
        leaf = Node("10509", parent=passthrough)
        assert _accumulated_path_to(domain, leaf) == "10239/12227/10509"


class TestFindSingleLevelTarget:
    def _scheduler(self, taxid):
        return _CascadeScheduler(SimpleNamespace(_single_level_taxid=taxid))

    def test_finds_a_descendant(self):
        domain = Node("10239", parent=Node("root"))
        family = Node("11118", parent=domain)
        assert self._scheduler("11118")._find_single_level_target(domain) is family

    def test_returns_the_domain_node_itself(self):
        domain = Node("10239", parent=Node("root"))
        assert self._scheduler("10239")._find_single_level_target(domain) is domain

    def test_missing_taxid_returns_none(self):
        domain = Node("10239", parent=Node("root"))
        Node("11118", parent=domain)
        assert self._scheduler("99999")._find_single_level_target(domain) is None
