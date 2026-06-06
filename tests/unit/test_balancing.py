"""Tests for taxotreeset.core.generation.balancing.compute_balanced_extraction_plan.

All tests use capacity_override and rare_taxa_strategy='keep' to avoid
requiring actual sequence data or LMDB access.
"""

import pytest
from bigtree import Node
from taxotreeset.core.generation.balancing import (
    _compute_n_per_class_from_retained,
    _compute_percentile_cutoff,
    _partition_by_leaf_count,
    compute_balanced_extraction_plan,
)


def _node(name, rank="species", scientific_name=None, parent=None):
    n = Node(str(name), parent=parent)
    n.rank = rank
    n.scientific_name = scientific_name or str(name)
    return n


def _seq_leaf(header_id, parent=None):
    n = Node(str(header_id), parent=parent)
    n.rank = "sequence"
    n.header_id = header_id
    return n


def _plan(
    parent,
    children,
    cap_override,
    min_num_seqs=1000,
    max_n_per_class=20_000,
    cutoff_percentage=98.0,
    min_leaves_per_class=0,
    rare_taxa_strategy="keep",
):
    return compute_balanced_extraction_plan(
        parent_node=parent,
        children=children,
        leaf_cache={},
        min_num_seqs=min_num_seqs,
        max_n_per_class=max_n_per_class,
        cutoff_percentage=cutoff_percentage,
        min_leaves_per_class=min_leaves_per_class,
        rare_taxa_strategy=rare_taxa_strategy,
        capacity_override=cap_override,
    )


# ---------------------------------------------------------------------------
# level_all scenario
# ---------------------------------------------------------------------------


class TestLevelAllScenario:
    def _build(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2, 3)]
        caps = {"1": 2000, "2": 3000, "3": 5000}
        return parent, children, caps

    def test_scenario_label_is_level_all(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, max_n_per_class=20_000)
        assert plan["scenario"] == "level_all"

    def test_n_per_class_equals_minimum_capacity(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps)
        assert plan["n_per_class"] == 2000

    def test_all_children_retained(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps)
        assert len(plan["retained_children"]) == 3
        assert plan["low_capacity_children"] == []

    def test_capacities_dict_matches_override(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps)
        assert plan["capacities"] == caps

    def test_rare_taxa_children_empty_with_keep_strategy(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps)
        assert plan["rare_taxa_children"] == []


# ---------------------------------------------------------------------------
# level_all_capped scenario
# ---------------------------------------------------------------------------


class TestLevelAllCappedScenario:
    def _build(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2, 3)]
        caps = {"1": 25_000, "2": 30_000, "3": 50_000}
        return parent, children, caps

    def test_scenario_label_is_level_all_capped(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, max_n_per_class=20_000)
        assert plan["scenario"] == "level_all_capped"

    def test_n_per_class_is_capped_at_max(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, max_n_per_class=20_000)
        assert plan["n_per_class"] == 20_000

    def test_all_children_still_retained(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps)
        assert len(plan["retained_children"]) == 3
        assert plan["low_capacity_children"] == []

    def test_capacities_preserved_in_capped_plan(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, max_n_per_class=20_000)
        assert plan["capacities"] == caps

    def test_cap_boundary_exact_minimum_not_capped(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2)]
        caps = {"1": 20_000, "2": 30_000}
        plan = _plan(parent, children, caps, max_n_per_class=20_000)
        assert plan["scenario"] == "level_all"
        assert plan["n_per_class"] == 20_000

    def test_one_above_cap_produces_capped_scenario(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2)]
        caps = {"1": 20_001, "2": 30_000}
        plan = _plan(parent, children, caps, max_n_per_class=20_000)
        assert plan["scenario"] == "level_all_capped"
        assert plan["n_per_class"] == 20_000


# ---------------------------------------------------------------------------
# cutoff_applied scenario
# ---------------------------------------------------------------------------


class TestCutoffAppliedScenario:
    def _build(self):
        parent = _node("root", rank="genus")
        names = list(range(10))
        children = [_node(str(i), parent=parent) for i in names]
        caps = {str(i): v for i, v in enumerate([50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000])}
        return parent, children, caps

    def test_scenario_label_is_cutoff_applied(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, cutoff_percentage=80.0)
        assert plan["scenario"] == "cutoff_applied"

    def test_low_capacity_children_are_populated(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, cutoff_percentage=80.0)
        assert len(plan["low_capacity_children"]) > 0

    def test_retained_plus_low_capacity_equals_total(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, cutoff_percentage=80.0)
        total = len(plan["retained_children"]) + len(plan["low_capacity_children"])
        assert total == len(children)

    def test_n_per_class_is_minimum_of_retained(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, cutoff_percentage=80.0)
        retained_caps = [caps[str(c.name)] for c in plan["retained_children"]]
        assert plan["n_per_class"] == min(retained_caps)

    def test_cutoff_at_98_with_small_list_retains_all(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in range(3)]
        caps = {"0": 10, "1": 500, "2": 5000}
        plan = _plan(parent, children, caps, min_num_seqs=1000, cutoff_percentage=98.0)
        assert plan["scenario"] == "cutoff_applied"
        assert len(plan["low_capacity_children"]) == 0

    def test_capacities_preserved_in_cutoff_plan(self):
        parent, children, caps = self._build()
        plan = _plan(parent, children, caps, min_num_seqs=1000, cutoff_percentage=80.0)
        assert plan["capacities"] == caps


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


class TestEmptyAndDegenerateInputs:
    def test_empty_children_list_returns_zero_plan(self):
        parent = _node("root")
        plan = _plan(parent, [], {})
        assert plan["n_per_class"] == 0
        assert plan["scenario"] == "level_all"
        assert plan["retained_children"] == []
        assert plan["low_capacity_children"] == []
        assert plan["rare_taxa_children"] == []
        assert plan["capacities"] == {}

    def test_single_child_triggers_level_all(self):
        parent = _node("root", rank="genus")
        child = _node("1", parent=parent)
        plan = _plan(parent, [child], {"1": 5000})
        assert plan["scenario"] in ("level_all", "level_all_capped")

    def test_all_zero_capacity_returns_cutoff_with_all_low(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in range(3)]
        caps = {"0": 0, "1": 0, "2": 0}
        plan = _plan(parent, children, caps, min_num_seqs=1)
        assert plan["scenario"] == "cutoff_applied"


# ---------------------------------------------------------------------------
# Rare-taxa partitioning (with fallback strategy + leaf counts)
# ---------------------------------------------------------------------------


class TestRareTaxaPartitioning:
    def test_fallback_strategy_diverts_children_below_leaf_floor(self):
        parent = _node("root", rank="genus")
        rich = [_node(str(i), parent=parent) for i in range(4)]
        rare = _node("99", parent=parent)

        for child in rich:
            for j in range(5):
                _seq_leaf(f"leaf_{child.name}_{j}", parent=child)

        caps = {str(c.name): 5000 for c in rich}
        caps["99"] = 5000

        plan = compute_balanced_extraction_plan(
            parent_node=parent,
            children=rich + [rare],
            leaf_cache={},
            min_leaves_per_class=3,
            rare_taxa_strategy="fallback",
            min_num_seqs=1000,
            capacity_override=caps,
        )
        assert rare in plan["rare_taxa_children"]

    def test_keep_strategy_ignores_leaf_count(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in range(3)]
        caps = {str(c.name): 5000 for c in children}

        plan = compute_balanced_extraction_plan(
            parent_node=parent,
            children=children,
            leaf_cache={},
            min_leaves_per_class=100,
            rare_taxa_strategy="keep",
            min_num_seqs=1000,
            capacity_override=caps,
        )
        assert plan["rare_taxa_children"] == []
        assert len(plan["retained_children"]) == 3

    def test_fallback_gate_keeps_all_when_fewer_than_two_eligible(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in range(2)]

        plan = compute_balanced_extraction_plan(
            parent_node=parent,
            children=children,
            leaf_cache={},
            min_leaves_per_class=10,
            rare_taxa_strategy="fallback",
            min_num_seqs=1000,
            capacity_override={"0": 5000, "1": 5000},
        )
        assert plan["rare_taxa_children"] == []
        assert len(plan["retained_children"]) == 2


# ---------------------------------------------------------------------------
# _compute_percentile_cutoff (internal helper)
# ---------------------------------------------------------------------------


class TestComputePercentileCutoff:
    def test_empty_list_returns_zero(self):
        assert _compute_percentile_cutoff([], 98.0) == 0

    def test_100_percent_retains_all_cutoff_is_minimum(self):
        caps = sorted([100, 200, 300, 400, 500])
        cutoff = _compute_percentile_cutoff(caps, 100.0)
        assert cutoff == caps[0]

    def test_typical_cutoff_with_known_result(self):
        caps = sorted([50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000])
        cutoff = _compute_percentile_cutoff(caps, 80.0)
        # 1.0 - 80.0/100.0 = 0.1999...9 (float precision), int(10 * 0.1999...) = 1
        assert cutoff == caps[1]


# ---------------------------------------------------------------------------
# _partition_by_leaf_count (internal helper)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# progress_callback with capacity_override
# ---------------------------------------------------------------------------


class TestProgressCallbackWithCapacityOverride:
    def test_callback_called_once_per_eligible_child(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2, 3)]
        caps = {"1": 2000, "2": 3000, "3": 5000}

        call_count = {"n": 0}

        def callback():
            call_count["n"] += 1

        compute_balanced_extraction_plan(
            parent_node=parent,
            children=children,
            leaf_cache={},
            min_num_seqs=1000,
            max_n_per_class=20_000,
            cutoff_percentage=98.0,
            capacity_override=caps,
            progress_callback=callback,
        )
        assert call_count["n"] == 3

    def test_no_callback_when_none(self):
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2)]
        caps = {"1": 2000, "2": 3000}
        plan = compute_balanced_extraction_plan(
            parent_node=parent,
            children=children,
            leaf_cache={},
            min_num_seqs=1000,
            max_n_per_class=20_000,
            cutoff_percentage=98.0,
            capacity_override=caps,
            progress_callback=None,
        )
        assert plan["n_per_class"] == 2000


# ---------------------------------------------------------------------------
# _compute_n_per_class_from_retained
# ---------------------------------------------------------------------------


class TestComputeNPerClassFromRetained:
    def test_returns_zero_when_no_retained_children(self):
        result = _compute_n_per_class_from_retained([], {}, max_n_per_class=1000)
        assert result == 0

    def test_returns_minimum_capacity_when_below_cap(self):
        parent = _node("root")
        children = [_node("a", parent=parent), _node("b", parent=parent)]
        caps = {"a": 500, "b": 1000}
        result = _compute_n_per_class_from_retained(children, caps, max_n_per_class=5000)
        assert result == 500

    def test_clamps_at_max_n_per_class(self):
        parent = _node("root")
        children = [_node("a", parent=parent), _node("b", parent=parent)]
        caps = {"a": 8000, "b": 9000}
        result = _compute_n_per_class_from_retained(children, caps, max_n_per_class=500)
        assert result == 500


# ---------------------------------------------------------------------------
# _partition_by_leaf_count (internal helper)
# ---------------------------------------------------------------------------


class TestPartitionByLeafCount:
    def test_keep_strategy_returns_all_eligible(self):
        parent = _node("root")
        children = [_node(str(i), parent=parent) for i in range(5)]
        eligible, rare = _partition_by_leaf_count(children, {}, 10, "keep")
        assert len(eligible) == 5
        assert rare == []

    def test_fallback_strategy_splits_by_leaf_count(self):
        parent = _node("root")
        rich_children = [_node(str(i), parent=parent) for i in range(4)]
        for child in rich_children:
            for j in range(5):
                _seq_leaf(f"leaf_{child.name}_{j}", parent=child)

        lean_child = _node("99", parent=parent)
        children = rich_children + [lean_child]
        eligible, rare = _partition_by_leaf_count(children, {}, min_leaves_per_class=3, rare_taxa_strategy="fallback")
        assert lean_child in rare
        assert lean_child not in eligible

    def test_fallback_gate_when_fewer_than_two_eligible(self):
        parent = _node("root")
        children = [_node(str(i), parent=parent) for i in range(2)]
        eligible, rare = _partition_by_leaf_count(children, {}, min_leaves_per_class=10, rare_taxa_strategy="fallback")
        assert len(eligible) == 2
        assert rare == []

    def test_leaf_cache_hit_path_returns_cached_count(self):
        # Pre-populate leaf_cache for child "42" — exercises line 215
        parent = _node("root")
        child = _node("42", parent=parent)
        # Fake cache entry: a list of 3 sequence leaves
        fake_leaves = [object(), object(), object()]
        leaf_cache = {"42": fake_leaves}
        eligible, rare = _partition_by_leaf_count(
            [child], leaf_cache, min_leaves_per_class=2, rare_taxa_strategy="fallback"
        )
        # 3 cached leaves >= 2 → child is eligible, not rare
        assert child in eligible
        assert rare == []


class TestComputeChildrenCapacitiesProgressCallback:
    def test_callback_called_when_capacity_override_is_none(self):
        # Exercises line 297: progress_callback() inside _compute_children_capacities.
        # capacity_override=None forces _compute_children_capacities to run.
        # Children with no sequence leaves produce capacity=0; callback still fires.
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2, 3)]
        call_count = {"n": 0}

        def callback():
            call_count["n"] += 1

        compute_balanced_extraction_plan(
            parent_node=parent,
            children=children,
            leaf_cache={},
            min_num_seqs=0,        # allow all-zero capacities to form a valid plan
            max_n_per_class=20_000,
            cutoff_percentage=98.0,
            capacity_override=None,
            progress_callback=callback,
        )
        assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_min_capacity_equal_to_min_num_seqs_produces_level_all(self):
        # ID 32: >= vs >; exactly equal should yield level_all, not cutoff
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2, 3)]
        caps = {"1": 1000, "2": 2000, "3": 3000}
        plan = _plan(parent, children, caps, min_num_seqs=1000, max_n_per_class=50_000)
        assert plan["scenario"] == "level_all"
        assert plan["n_per_class"] == 1000

    def test_child_with_exactly_min_leaves_is_eligible(self):
        # ID 48: >= vs >; exactly at threshold must remain eligible
        parent = _node("root", rank="genus")
        rich = [_node(str(i), parent=parent) for i in range(4)]
        for child in rich:
            for j in range(5):
                _seq_leaf(f"sl_{child.name}_{j}", parent=child)
        exact = _node("99", parent=parent)
        for j in range(3):
            _seq_leaf(f"sl_exact_{j}", parent=exact)

        eligible, rare = _partition_by_leaf_count(
            rich + [exact], {}, min_leaves_per_class=3, rare_taxa_strategy="fallback"
        )
        assert exact in eligible
        assert exact not in rare

    def test_fallback_gate_inactive_with_exactly_two_eligible(self):
        # IDs 49 (<= 2) and 50 (< 3): gate fires only when eligible < 2,
        # so exactly 2 eligible must NOT trigger it.
        parent = _node("root", rank="genus")
        rich_1 = _node("1", parent=parent)
        rich_2 = _node("2", parent=parent)
        lean = _node("3", parent=parent)
        for j in range(5):
            _seq_leaf(f"s1_{j}", parent=rich_1)
            _seq_leaf(f"s2_{j}", parent=rich_2)

        eligible, rare = _partition_by_leaf_count(
            [rich_1, rich_2, lean], {}, min_leaves_per_class=3, rare_taxa_strategy="fallback"
        )
        assert lean in rare
        assert rich_1 in eligible
        assert rich_2 in eligible


# ---------------------------------------------------------------------------
# Cache and leaf-count correctness
# ---------------------------------------------------------------------------


class TestCacheAndLeafCountCorrectness:
    def test_capacity_override_missing_key_defaults_to_zero(self):
        # ID 23: default 0 vs 1 when child is absent from capacity_override
        parent = _node("root", rank="genus")
        children = [_node("1", parent=parent), _node("2", parent=parent)]
        caps = {"1": 5000}  # child "2" is intentionally absent
        plan = _plan(parent, children, caps)
        assert plan["capacities"]["2"] == 0

    def test_leaf_cache_hit_used_when_eligible_gate_does_not_fire(self):
        # ID 37: cached=None mutant skips cache; need 3 children so gate doesn't activate
        parent = _node("root", rank="genus")
        cached_child = _node("cached", parent=parent)  # no bigtree seq-leaves, but in cache
        rich_1 = _node("r1", parent=parent)
        rich_2 = _node("r2", parent=parent)
        for j in range(5):
            _seq_leaf(f"s1_{j}", parent=rich_1)
            _seq_leaf(f"s2_{j}", parent=rich_2)

        leaf_cache = {"cached": [object()] * 5}  # 5 cached leaves
        eligible, rare = _partition_by_leaf_count(
            [cached_child, rich_1, rich_2], leaf_cache,
            min_leaves_per_class=3, rare_taxa_strategy="fallback",
        )
        # cache says cached_child has 5 leaves >= 3 → eligible
        # without cache (mutant), it would have 0 leaves < 3 → rare (gate: 2 eligible)
        assert cached_child in eligible

    def test_capacity_keys_use_child_name_as_string(self):
        # ID 138: child_name = None makes all children share one key in capacities dict
        # With capacity_override=None, _compute_children_capacities is called.
        # If child_name=None, all children overwrite the same key → len(capacities)==1.
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in (1, 2, 3)]
        plan = compute_balanced_extraction_plan(
            parent_node=parent,
            children=children,
            leaf_cache={},
            min_num_seqs=0,
            max_n_per_class=20_000,
            cutoff_percentage=98.0,
            capacity_override=None,
        )
        assert set(plan["capacities"].keys()) == {"1", "2", "3"}

    def test_sequence_leaf_count_not_doubled(self):
        # ID 39: sum(2 for leaf…) doubles count; test where doubling changes gate behaviour
        parent = _node("root", rank="genus")
        children = [_node(str(i), parent=parent) for i in range(3)]
        for c in children[:2]:
            for j in range(3):
                _seq_leaf(f"sl_{c.name}_{j}", parent=c)
        # children[0]/[1]: 3 seq-leaves each; children[2]: 0 seq-leaves
        # min_leaves_per_class=5: correct count (3) < 5 → all rare → gate fires → all eligible
        # with ID 39 (×2): count 6 >= 5 → eligible=[0,1], rare=[2] → gate: 2 >= 2 → no gate
        eligible, rare = _partition_by_leaf_count(
            children, {}, min_leaves_per_class=5, rare_taxa_strategy="fallback"
        )
        assert len(eligible) == 3  # gate fired; all returned as eligible
        assert rare == []
