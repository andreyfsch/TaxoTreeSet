"""Integration tests for the synthetic vault pipeline.

Covers the full chain from LMDB read-back → tree construction →
capacity computation → balanced extraction plan, using only local
synthetic data (no network access required).
"""

import zlib

import lmdb
import pytest

from taxotreeset.core.generation.balancing import compute_balanced_extraction_plan
from taxotreeset.core.generation.capacity import compute_all_capacities
from taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from taxotreeset.dataset.utils import _read_single_sequence

_MIN_LEN = 10


# ---------------------------------------------------------------------------
# 1. Vault read-back
# ---------------------------------------------------------------------------


class TestVaultReadBack:
    def test_all_headers_are_readable(self, synthetic_env):
        lmdb_path = synthetic_env["lmdb_path"]
        sequences = synthetic_env["sequences"]
        for header_id, expected_seq in sequences.items():
            read_seq = _read_single_sequence(lmdb_path, header_id)
            assert read_seq == expected_seq, f"{header_id} read-back mismatch"

    def test_unknown_header_returns_empty_string(self, synthetic_env):
        lmdb_path = synthetic_env["lmdb_path"]
        assert _read_single_sequence(lmdb_path, "NONEXISTENT_HEADER") == ""

    def test_raw_vault_key_is_zlib_compressed(self, synthetic_env):
        from taxotreeset.dataset.utils import _get_lmdb_env
        lmdb_dir = synthetic_env["lmdb_path"]
        env = _get_lmdb_env(lmdb_dir)
        with env.begin() as txn:
            raw = txn.get(b"NC_045512")
        assert raw is not None
        decompressed = zlib.decompress(raw).decode("utf-8")
        assert decompressed == synthetic_env["sequences"]["NC_045512"]


# ---------------------------------------------------------------------------
# 2. Tree construction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree_root(synthetic_env):
    return generate_seqs_by_taxon_tree(
        registry_path=synthetic_env["registry_path"],
        vault_path=synthetic_env["vault_dir"],
        domain_taxid=synthetic_env["domain_taxid"],
        mapping_path=synthetic_env["mapping_path"],
    )


class TestTreeConstruction:
    def test_root_node_exists(self, tree_root):
        assert tree_root is not None
        assert tree_root.name == "root"

    def test_domain_anchor_is_direct_child_of_root(self, tree_root):
        child_names = {c.name for c in tree_root.children}
        assert "10239" in child_names

    def test_two_families_under_domain(self, tree_root):
        domain_node = next(c for c in tree_root.children if c.name == "10239")
        family_names = {c.name for c in domain_node.children}
        assert "11118" in family_names
        assert "12227" in family_names

    def test_correct_species_under_coronaviridae(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        corona = next(c for c in domain.children if c.name == "11118")
        species_names = {c.name for c in corona.children}
        assert "2697049" in species_names
        assert "11234" in species_names

    def test_correct_species_under_adenoviridae(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        adeno = next(c for c in domain.children if c.name == "12227")
        species_names = {c.name for c in adeno.children}
        assert "10509" in species_names

    def test_sequence_leaves_are_attached(self, tree_root):
        all_leaves = list(tree_root.leaves)
        seq_leaves = [l for l in all_leaves if getattr(l, "rank", "") == "sequence"]
        assert len(seq_leaves) == 3

    def test_sequence_leaves_carry_fasta_path(self, tree_root, synthetic_env):
        all_leaves = list(tree_root.leaves)
        for leaf in all_leaves:
            if getattr(leaf, "rank", "") == "sequence":
                assert hasattr(leaf, "fasta_path")
                assert leaf.fasta_path == synthetic_env["lmdb_path"]

    def test_sequence_leaves_carry_header_id(self, tree_root):
        all_leaves = list(tree_root.leaves)
        header_ids = {
            leaf.header_id
            for leaf in all_leaves
            if getattr(leaf, "rank", "") == "sequence"
        }
        assert header_ids == {"NC_045512", "NC_001846", "NC_001407"}

    def test_taxonomic_nodes_carry_rank_attribute(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        assert domain.rank == "superkingdom"

        corona = next(c for c in domain.children if c.name == "11118")
        assert corona.rank == "family"

    def test_tree_is_idempotent_when_called_twice(self, synthetic_env):
        root1 = generate_seqs_by_taxon_tree(
            registry_path=synthetic_env["registry_path"],
            vault_path=synthetic_env["vault_dir"],
            domain_taxid=synthetic_env["domain_taxid"],
            mapping_path=synthetic_env["mapping_path"],
        )
        root2 = generate_seqs_by_taxon_tree(
            registry_path=synthetic_env["registry_path"],
            vault_path=synthetic_env["vault_dir"],
            domain_taxid=synthetic_env["domain_taxid"],
            mapping_path=synthetic_env["mapping_path"],
        )
        leaves1 = {l.name for l in root1.leaves}
        leaves2 = {l.name for l in root2.leaves}
        assert leaves1 == leaves2


# ---------------------------------------------------------------------------
# 3. Capacity computation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def all_capacities(tree_root):
    return compute_all_capacities(tree_root, min_len=_MIN_LEN)


class TestCapacityComputation:
    def test_returns_a_non_empty_dict(self, all_capacities):
        assert isinstance(all_capacities, dict)
        assert len(all_capacities) > 0

    def test_all_non_leaf_taxon_nodes_have_entries(self, all_capacities):
        expected_taxids = {"root", "10239", "11118", "12227", "2697049", "11234", "10509"}
        for taxid in expected_taxids:
            assert taxid in all_capacities, f"Missing capacity for taxid {taxid}"

    def test_all_capacities_are_positive(self, all_capacities):
        for taxid, cap in all_capacities.items():
            assert cap > 0, f"Zero or negative capacity for {taxid}"

    def test_parent_capacity_ge_child_capacity(self, all_capacities):
        assert all_capacities["11118"] >= all_capacities["2697049"]
        assert all_capacities["11118"] >= all_capacities["11234"]
        assert all_capacities["12227"] >= all_capacities["10509"]
        assert all_capacities["10239"] >= all_capacities["11118"]
        assert all_capacities["10239"] >= all_capacities["12227"]

    def test_species_capacity_matches_sequence_length_minus_window(
        self, all_capacities, synthetic_env
    ):
        seq = synthetic_env["sequences"]["NC_045512"]
        expected_upper_bound = len(seq) - _MIN_LEN + 1
        assert all_capacities["2697049"] <= expected_upper_bound

    def test_sequence_leaves_absent_from_capacities(self, all_capacities):
        header_ids = {"NC_045512", "NC_001846", "NC_001407"}
        for hid in header_ids:
            assert hid not in all_capacities

    def test_root_capacity_is_largest(self, all_capacities):
        root_cap = all_capacities["root"]
        for taxid, cap in all_capacities.items():
            assert root_cap >= cap, f"root capacity should be >= {taxid}: {root_cap} < {cap}"


# ---------------------------------------------------------------------------
# 4. Full pipeline: tree → capacity → balanced plan
# ---------------------------------------------------------------------------


class TestBalancingWithRealCapacities:
    def _get_domain_node_and_families(self, tree_root):
        domain = next(c for c in tree_root.children if c.name == "10239")
        families = list(domain.children)
        return domain, families

    def test_plan_has_valid_structure(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert "scenario" in plan
        assert "n_per_class" in plan
        assert "retained_children" in plan
        assert "low_capacity_children" in plan
        assert "capacities" in plan
        assert "rare_taxa_children" in plan

    def test_plan_scenario_is_a_known_value(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["scenario"] in ("level_all", "level_all_capped", "cutoff_applied")

    def test_n_per_class_is_positive(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["n_per_class"] > 0

    def test_retained_plus_low_capacity_equals_total(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        total = len(plan["retained_children"]) + len(plan["low_capacity_children"])
        assert total == len(families)

    def test_plan_with_low_min_num_seqs_triggers_level_all(self, tree_root, all_capacities):
        domain, families = self._get_domain_node_and_families(tree_root)
        min_cap = min(all_capacities.get(str(f.name), 0) for f in families)
        plan = compute_balanced_extraction_plan(
            parent_node=domain,
            children=families,
            leaf_cache={},
            capacity_override=all_capacities,
            min_num_seqs=1,
            max_n_per_class=min_cap + 1,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["scenario"] == "level_all"
        assert plan["n_per_class"] == min_cap

    def test_plan_without_capacity_override_uses_real_sequences(
        self, tree_root
    ):
        domain = next(c for c in tree_root.children if c.name == "10239")
        corona = next(c for c in domain.children if c.name == "11118")
        species = list(corona.children)

        plan = compute_balanced_extraction_plan(
            parent_node=corona,
            children=species,
            leaf_cache={},
            min_len=_MIN_LEN,
            use_exact_capacity=True,
            rare_taxa_strategy="keep",
            min_leaves_per_class=0,
        )
        assert plan["n_per_class"] > 0
        for taxid, cap in plan["capacities"].items():
            assert cap > 0, f"Zero capacity for {taxid} without override"
