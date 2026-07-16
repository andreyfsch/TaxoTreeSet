"""Tests for taxotreeset.dataset.tree_builder internal helpers.

All helpers tested here are pure transformations or deterministic tree
mutations — no network access or LMDB I/O is required.
"""

import json

import pytest
from bigtree import Node
from taxotreeset.dataset.tree_builder import (
    _accumulate_sequence_leaves,
    _anchor_lineage_to_domain,
    _annotate_node_metadata,
    _apply_noise_filter_to_lineage,
    _apply_scope_redirections,
    _attach_sequence_leaves,
    _build_lineage_path,
    _drop_child_indexes,
    _flush_pending_leaves,
    _enumerate_accession_tasks,
    _find_child_by_name,
    _lineage_ids_from_registry,
    _load_json,
    _load_optional_json,
    _maybe_append_target_taxid,
    _process_accession,
    _register_child,
    _resolve_scope_config,
    generate_seqs_by_taxon_tree,
)
from taxotreeset.io.noise_filter import NoiseFilter


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_root():
    root = Node("root")
    root.rank = "root"
    return root


def permissive_noise_filter(tmp_path):
    """NoiseFilter with no patterns — passes everything."""
    config = {"name_patterns": [], "rank_blacklist": {"ranks": []}}
    p = tmp_path / "noise_patterns.json"
    p.write_text(json.dumps(config))
    return NoiseFilter(str(p))


# ---------------------------------------------------------------------------
# _load_json / _load_optional_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_loads_valid_json(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"key": "value"}')
        result = _load_json(str(p))
        assert result == {"key": "value"}

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_json(str(tmp_path / "nonexistent.json"))


class TestLoadOptionalJson:
    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        result = _load_optional_json(str(tmp_path / "missing.json"))
        assert result == {}

    def test_returns_parsed_content_when_file_exists(self, tmp_path):
        p = tmp_path / "mapping.json"
        p.write_text('{"scopes": {}}')
        result = _load_optional_json(str(p))
        assert result == {"scopes": {}}


# ---------------------------------------------------------------------------
# _lineage_ids_from_registry
# ---------------------------------------------------------------------------


class TestLineageIdsFromRegistry:
    def test_returns_root_to_leaf_order(self):
        lineages = {
            "2697049": [
                {"taxid": "2697049", "rank": "species", "name": "SARS-CoV-2"},
                {"taxid": "11118", "rank": "family", "name": "Coronaviridae"},
                {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
            ]
        }
        result = _lineage_ids_from_registry("2697049", lineages)
        assert result == ["10239", "11118", "2697049"]

    def test_returns_empty_for_unknown_taxid(self):
        result = _lineage_ids_from_registry("99999", {})
        assert result == []

    def test_single_entry_lineage(self):
        lineages = {"1": [{"taxid": "1", "rank": "root", "name": "root"}]}
        result = _lineage_ids_from_registry("1", lineages)
        assert result == ["1"]

    def test_integer_taxid_is_coerced_to_string_key(self):
        lineages = {"10239": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]}
        result = _lineage_ids_from_registry(10239, lineages)
        assert result == ["10239"]


# ---------------------------------------------------------------------------
# _apply_noise_filter_to_lineage
# ---------------------------------------------------------------------------


class TestApplyNoiseFilterToLineage:
    def test_passes_clean_lineage_unchanged(self, tmp_path):
        nf = permissive_noise_filter(tmp_path)
        lineage = ["10239", "11118", "2697049"]
        taxon_info = {
            "10239": ("Viruses", "superkingdom"),
            "11118": ("Coronaviridae", "family"),
            "2697049": ("SARS-CoV-2", "species"),
        }
        result = _apply_noise_filter_to_lineage(lineage, nf, taxon_info)
        assert result == lineage

    def test_removes_rank_blacklisted_node(self, tmp_path):
        config = {"name_patterns": [], "rank_blacklist": {"ranks": ["strain"]}}
        p = tmp_path / "noise.json"
        p.write_text(json.dumps(config))
        nf = NoiseFilter(str(p))

        lineage = ["10239", "11118", "2697049", "9999999"]
        taxon_info = {
            "10239": ("Viruses", "superkingdom"),
            "11118": ("Coronaviridae", "family"),
            "2697049": ("SARS-CoV-2", "species"),
            "9999999": ("SomeStrain", "strain"),
        }
        result = _apply_noise_filter_to_lineage(lineage, nf, taxon_info)
        assert "9999999" not in result
        assert result == ["10239", "11118", "2697049"]

    def test_keeps_taxid_absent_from_index(self, tmp_path):
        nf = permissive_noise_filter(tmp_path)
        lineage = ["10239", "UNKNOWN_99"]
        taxon_info = {"10239": ("Viruses", "superkingdom")}
        result = _apply_noise_filter_to_lineage(lineage, nf, taxon_info)
        assert "UNKNOWN_99" in result

    def test_empty_lineage_returns_empty(self, tmp_path):
        nf = permissive_noise_filter(tmp_path)
        assert _apply_noise_filter_to_lineage([], nf, {}) == []


# ---------------------------------------------------------------------------
# _anchor_lineage_to_domain
# ---------------------------------------------------------------------------


class TestAnchorLineageToDomain:
    def test_trims_lineage_to_start_at_domain(self):
        lineage = ["1", "2", "10239", "11118", "2697049"]
        result = _anchor_lineage_to_domain(lineage, "10239")
        assert result == ["10239", "11118", "2697049"]

    def test_prepends_domain_when_absent_from_lineage(self):
        lineage = ["11118", "2697049"]
        result = _anchor_lineage_to_domain(lineage, "10239")
        assert result == ["10239", "11118", "2697049"]

    def test_no_domain_returns_lineage_unchanged(self):
        lineage = ["10239", "11118"]
        assert _anchor_lineage_to_domain(lineage, None) == lineage

    def test_domain_at_start_of_lineage_unchanged(self):
        lineage = ["10239", "11118"]
        assert _anchor_lineage_to_domain(lineage, "10239") == lineage

    def test_domain_as_only_element(self):
        lineage = ["10239"]
        assert _anchor_lineage_to_domain(lineage, "10239") == ["10239"]


# ---------------------------------------------------------------------------
# _resolve_scope_config
# ---------------------------------------------------------------------------


class TestResolveScopeConfig:
    def test_returns_known_domain_config(self):
        mapping = {
            "scopes": {
                "10239": {
                    "default_id": "999000",
                    "redirections": {"12227": {"target_id": "999001"}},
                    "virtual_id_labels": {"999000": "Unclassified Viruses"},
                }
            }
        }
        result = _resolve_scope_config(mapping, "10239")
        assert result["default_id"] == "999000"
        assert "12227" in result["redirections"]
        assert "999000" in result["virtual_labels"]

    def test_returns_empty_config_for_unknown_domain(self):
        result = _resolve_scope_config({}, "10239")
        assert result["default_id"] is None
        assert result["redirections"] == {}
        assert result["virtual_labels"] == {}

    def test_none_domain_returns_empty_config(self):
        result = _resolve_scope_config({}, None)
        assert result["default_id"] is None


# ---------------------------------------------------------------------------
# _apply_scope_redirections
# ---------------------------------------------------------------------------


class TestApplyScopeRedirections:
    def _scope(self, redirections=None, default_id=None):
        return {
            "redirections": redirections or {},
            "default_id": default_id,
            "virtual_labels": {},
        }

    def test_no_domain_returns_lineage_unchanged(self):
        lineage = ["11118", "2697049"]
        scope = self._scope()
        result = _apply_scope_redirections(lineage, None, scope)
        assert result == lineage

    def test_single_element_lineage_returned_unchanged(self):
        lineage = ["10239"]
        scope = self._scope()
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == lineage

    def test_self_redirect_preserves_lineage(self):
        lineage = ["10239", "11118", "2697049"]
        scope = self._scope(redirections={"11118": {"target_id": "11118"}})
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == lineage

    def test_explicit_redirect_inserts_virtual_group(self):
        lineage = ["10239", "11118", "2697049"]
        scope = self._scope(redirections={"11118": {"target_id": "999001"}})
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == ["10239", "999001", "11118", "2697049"]

    def test_default_fallback_when_no_explicit_rule(self):
        lineage = ["10239", "11118", "2697049"]
        scope = self._scope(default_id="999000")
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == ["10239", "999000", "2697049"]

    def test_no_rule_and_no_default_leaves_lineage_unchanged(self):
        lineage = ["10239", "11118", "2697049"]
        scope = self._scope()  # default_id=None
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == lineage

    def test_lineage_not_anchored_at_domain_returned_unchanged(self):
        lineage = ["11118", "2697049"]  # doesn't start with domain
        scope = self._scope(default_id="999000")
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == lineage


class TestApplyScopeRedirectionsAllRanks:
    """All-ranks scan: the redirectable group may sit below clade/realm."""

    def _scope(self, redirections=None, default_id=None, all_ranks=True):
        return {
            "redirections": redirections or {},
            "default_id": default_id,
            "virtual_labels": {},
            "all_ranks": all_ranks,
        }

    def test_kingdom_below_clade_preserves_backbone(self):
        # Regression: Viruses -> clade Riboviria -> kingdom Orthornavirae -> ...
        # Canonically this collapsed the whole subtree to [domain, 999000, leaf].
        lineage = ["10239", "2559587", "2732396", "2732408", "76804", "2508233"]
        scope = self._scope(
            redirections={"2732396": {"target_id": "2732396"}},
            default_id="999000",
        )
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == lineage  # full backbone kept, not collapsed

    def test_virtual_insert_below_clade_keeps_intermediate_ranks(self):
        lineage = ["10239", "2559587", "2732090", "10501"]
        scope = self._scope(
            redirections={"2732090": {"target_id": "999001"}},
            default_id="999000",
        )
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == ["10239", "2559587", "999001", "2732090", "10501"]

    def test_match_at_index_one_matches_canonical(self):
        lineage = ["10239", "2732396", "2732408", "2508233"]
        scope = self._scope(redirections={"2732396": {"target_id": "2732396"}})
        assert _apply_scope_redirections(lineage, "10239", scope) == lineage

    def test_unclassified_still_flattens_to_default(self):
        lineage = ["10239", "9999999", "8888888"]
        scope = self._scope(default_id="999000")
        result = _apply_scope_redirections(lineage, "10239", scope)
        assert result == ["10239", "999000", "8888888"]

    def test_canonical_mode_does_not_scan_deeper(self):
        # Same lineage, all_ranks=False: the deeper key is ignored and the
        # taxon falls to the default fallback (documents the gating).
        lineage = ["10239", "2559587", "2732396", "2508233"]
        redirections = {"2732396": {"target_id": "2732396"}}
        canon = self._scope(
            redirections=redirections, default_id="999000", all_ranks=False)
        assert _apply_scope_redirections(lineage, "10239", canon) == [
            "10239", "999000", "2508233"]
        ar = self._scope(redirections=redirections, default_id="999000")
        assert _apply_scope_redirections(lineage, "10239", ar) == lineage


# ---------------------------------------------------------------------------
# _enumerate_accession_tasks
# ---------------------------------------------------------------------------


class TestEnumerateAccessionTasks:
    def _registry(self, taxons, lineages):
        return {"taxons": taxons, "lineages": lineages, "accessions": {}}

    def test_returns_all_tasks_when_no_domain_filter(self):
        reg = self._registry(
            taxons={"A": ["acc1"], "B": ["acc2"]},
            lineages={},
        )
        tasks = _enumerate_accession_tasks(reg, domain_taxid=None)
        assert set(tasks) == {("A", "acc1"), ("B", "acc2")}

    def test_filters_by_domain_taxid(self):
        reg = self._registry(
            taxons={"2697049": ["acc1"], "9999": ["acc2"]},
            lineages={
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ],
                "9999": [],  # no lineage → excluded
            },
        )
        tasks = _enumerate_accession_tasks(reg, domain_taxid="10239")
        assert ("2697049", "acc1") in tasks
        assert all(t[0] != "9999" for t in tasks)

    def test_taxon_with_multiple_accessions_produces_multiple_tasks(self):
        reg = self._registry(
            taxons={"2697049": ["acc1", "acc2", "acc3"]},
            lineages={
                "2697049": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]
            },
        )
        tasks = _enumerate_accession_tasks(reg, domain_taxid="10239")
        assert len(tasks) == 3

    def test_taxon_with_empty_lineage_excluded_when_domain_set(self):
        reg = self._registry(
            taxons={"9999": ["acc1"]},
            lineages={"9999": []},
        )
        tasks = _enumerate_accession_tasks(reg, domain_taxid="10239")
        assert tasks == []


# ---------------------------------------------------------------------------
# _find_child_by_name
# ---------------------------------------------------------------------------


class TestFindChildByName:
    def test_returns_matching_child(self):
        root = make_root()
        child = Node("10239", parent=root)
        assert _find_child_by_name(root, "10239") is child

    def test_returns_none_when_no_match(self):
        root = make_root()
        Node("11118", parent=root)
        assert _find_child_by_name(root, "99999") is None

    def test_returns_none_when_node_has_no_children(self):
        root = make_root()
        assert _find_child_by_name(root, "10239") is None

    def test_returns_first_match_among_multiple_children(self):
        root = make_root()
        child_a = Node("A", parent=root)
        Node("B", parent=root)
        assert _find_child_by_name(root, "A") is child_a


# ---------------------------------------------------------------------------
# _annotate_node_metadata
# ---------------------------------------------------------------------------


class TestAnnotateNodeMetadata:
    def test_annotates_from_taxon_lookup(self):
        node = Node("11118")
        taxon_info = {"11118": ("Coronaviridae", "family")}
        _annotate_node_metadata(node, "11118", {}, taxon_info)
        assert node.rank == "family"
        assert node.scientific_name == "Coronaviridae"

    def test_annotates_from_virtual_labels_when_present(self):
        node = Node("999000")
        virtual_labels = {"999000": "Unclassified Viruses"}
        _annotate_node_metadata(node, "999000", virtual_labels, {})
        assert node.rank == "realm_group"
        assert node.scientific_name == "Unclassified Viruses"

    def test_defaults_when_taxid_absent_from_both_sources(self):
        node = Node("UNKNOWN")
        _annotate_node_metadata(node, "UNKNOWN", {}, {})
        assert node.rank == "unknown"
        assert node.scientific_name == "UNKNOWN"

    def test_rank_is_lowercased_and_stripped(self):
        node = Node("11118")
        taxon_info = {"11118": ("Coronaviridae", "  FAMILY  ")}
        _annotate_node_metadata(node, "11118", {}, taxon_info)
        assert node.rank == "family"

    def test_virtual_label_takes_priority_over_taxon_lookup(self):
        node = Node("11118")
        virtual_labels = {"11118": "Virtual Group A"}
        taxon_info = {"11118": ("Coronaviridae", "family")}
        _annotate_node_metadata(node, "11118", virtual_labels, taxon_info)
        assert node.scientific_name == "Virtual Group A"
        assert node.rank == "realm_group"


# ---------------------------------------------------------------------------
# _attach_sequence_leaves
# ---------------------------------------------------------------------------


class TestAttachSequenceLeaves:
    def test_attaches_leaves_for_each_header(self):
        parent = Node("2697049")
        parent.rank = "species"
        accession_info = {
            "organism": "SARS-CoV-2",
            "headers": [{"id": "NC_045512", "name": "Complete genome"}],
        }
        _attach_sequence_leaves(parent, accession_info, "/vault")
        seq_leaves = [c for c in parent.children if c.name == "NC_045512"]
        assert len(seq_leaves) == 1
        leaf = seq_leaves[0]
        assert leaf.rank == "sequence"
        assert leaf.header_id == "NC_045512"
        assert leaf.fasta_path.endswith("sequences.lmdb")

    def test_idempotent_on_repeated_call(self):
        parent = Node("2697049")
        parent.rank = "species"
        accession_info = {
            "organism": "SARS-CoV-2",
            "headers": [{"id": "NC_045512", "name": "Complete genome"}],
        }
        _attach_sequence_leaves(parent, accession_info, "/vault")
        _attach_sequence_leaves(parent, accession_info, "/vault")
        seq_leaves = [c for c in parent.children if c.name == "NC_045512"]
        assert len(seq_leaves) == 1

    def test_skips_header_entry_without_id(self):
        parent = Node("2697049")
        parent.rank = "species"
        accession_info = {
            "organism": "SARS-CoV-2",
            "headers": [{"name": "No ID here"}],  # missing "id" key
        }
        _attach_sequence_leaves(parent, accession_info, "/vault")
        assert len(parent.children) == 0

    def test_skips_non_dict_header_entries(self):
        parent = Node("2697049")
        parent.rank = "species"
        accession_info = {
            "organism": "SARS-CoV-2",
            "headers": ["not_a_dict"],
        }
        _attach_sequence_leaves(parent, accession_info, "/vault")
        assert len(parent.children) == 0

    def test_organism_becomes_scientific_name_on_leaf(self):
        parent = Node("2697049")
        parent.rank = "species"
        accession_info = {
            "organism": "SARS-CoV-2",
            "headers": [{"id": "NC_045512", "name": "genome"}],
        }
        _attach_sequence_leaves(parent, accession_info, "/vault")
        leaf = parent.children[0]
        assert leaf.scientific_name == "SARS-CoV-2"

    def test_empty_organism_sets_empty_scientific_name(self):
        parent = Node("2697049")
        parent.rank = "species"
        accession_info = {
            "headers": [{"id": "NC_045512", "name": "genome"}],
        }
        _attach_sequence_leaves(parent, accession_info, "/vault")
        assert parent.children[0].scientific_name == ""


# ---------------------------------------------------------------------------
# _build_lineage_path
# ---------------------------------------------------------------------------


class TestBuildLineagePath:
    def test_creates_intermediate_nodes(self):
        root = make_root()
        lineage = ["10239", "11118", "2697049"]
        taxon_info = {
            "10239": ("Viruses", "superkingdom"),
            "11118": ("Coronaviridae", "family"),
            "2697049": ("SARS-CoV-2", "species"),
        }
        leaf = _build_lineage_path(root, lineage, {}, taxon_info)
        assert leaf.name == "2697049"

    def test_path_nodes_carry_correct_rank(self):
        root = make_root()
        lineage = ["10239", "11118"]
        taxon_info = {
            "10239": ("Viruses", "superkingdom"),
            "11118": ("Coronaviridae", "family"),
        }
        _build_lineage_path(root, lineage, {}, taxon_info)
        domain = _find_child_by_name(root, "10239")
        assert domain.rank == "superkingdom"

    def test_reuses_existing_nodes(self):
        root = make_root()
        lineage = ["10239", "11118"]
        taxon_info = {
            "10239": ("Viruses", "superkingdom"),
            "11118": ("Coronaviridae", "family"),
        }
        first = _build_lineage_path(root, lineage, {}, taxon_info)
        second = _build_lineage_path(root, lineage, {}, taxon_info)
        assert first is second

    def test_empty_lineage_returns_root(self):
        root = make_root()
        result = _build_lineage_path(root, [], {}, {})
        assert result is root

    def test_single_element_lineage_creates_one_child(self):
        root = make_root()
        leaf = _build_lineage_path(root, ["10239"], {}, {"10239": ("Viruses", "superkingdom")})
        assert leaf.name == "10239"
        assert len(root.children) == 1


# ---------------------------------------------------------------------------
# _maybe_append_target_taxid
# ---------------------------------------------------------------------------


class TestMaybeAppendTargetTaxid:
    def _permissive_filter(self):
        return NoiseFilter(config_path="/nonexistent_path")

    def test_appends_target_when_info_present_and_not_noise(self):
        filtered_lineage = []
        taxon_info = {"2697049": ("SARS-CoV-2", "species")}
        nf = self._permissive_filter()
        _maybe_append_target_taxid(filtered_lineage, 2697049, nf, taxon_info)
        assert "2697049" in filtered_lineage

    def test_appends_target_when_info_absent(self):
        filtered_lineage = []
        _maybe_append_target_taxid(filtered_lineage, 9999999, self._permissive_filter(), {})
        assert "9999999" in filtered_lineage

    def test_does_not_append_when_target_is_noise(self, tmp_path):
        import json as _json
        config = {"name_patterns": [{"regex": r"^noisy$", "description": "noise"}]}
        p = tmp_path / "noise.json"
        p.write_text(_json.dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))
        taxon_info = {"2697049": ("noisy", "species")}
        filtered_lineage = []
        _maybe_append_target_taxid(filtered_lineage, 2697049, nf, taxon_info)
        assert filtered_lineage == []


# ---------------------------------------------------------------------------
# _process_accession — early-return paths
# ---------------------------------------------------------------------------


class TestProcessAccessionEarlyReturn:
    def _make_root(self):
        return make_root()

    def _permissive_filter(self):
        return NoiseFilter(config_path="/nonexistent_path")

    def test_skips_when_no_lineage_stored(self):
        root = self._make_root()
        _process_accession(
            root=root,
            taxid_str="2697049",
            accession_id="GCF_001",
            accession_info={"taxid": "2697049"},
            domain_taxid="10239",
            scope_config={},
            noise_filter=self._permissive_filter(),
            vault_path="/fake/vault",
            lineages={},
        )
        assert len(list(root.descendants)) == 0

    def test_skips_when_entire_lineage_filtered_out(self, tmp_path):
        import json as _json
        config = {"name_patterns": [{"regex": r".*", "description": "filter all"}]}
        p = tmp_path / "noise.json"
        p.write_text(_json.dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))

        root = self._make_root()
        lineages = {
            "2697049": [
                {"taxid": "2697049", "rank": "species", "name": "match_all"},
                {"taxid": "10239", "rank": "superkingdom", "name": "match_all"},
            ]
        }
        _process_accession(
            root=root,
            taxid_str="2697049",
            accession_id="GCF_001",
            accession_info={"taxid": "2697049"},
            domain_taxid="10239",
            scope_config={},
            noise_filter=nf,
            vault_path="/fake/vault",
            lineages=lineages,
        )
        assert len(list(root.descendants)) == 0

    def test_calls_maybe_append_when_target_filtered_but_lineage_non_empty(self, tmp_path):
        # Line 359: target taxid filtered out but ancestor lineage survives
        import json as _json
        # Only the species name "FilterMe" is filtered; "Viruses" passes
        config = {"name_patterns": [{"regex": r"^FilterMe$", "description": "filter species"}]}
        p = tmp_path / "noise.json"
        p.write_text(_json.dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))

        root = self._make_root()
        lineages = {
            "2697049": [
                {"taxid": "2697049", "rank": "species", "name": "FilterMe"},
                {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
            ]
        }
        # After noise-filter: "2697049" is removed, "10239" survives
        # → target_str "2697049" not in filtered_lineage ["10239"] → line 359 hit
        _process_accession(
            root=root,
            taxid_str="2697049",
            accession_id="GCF_001",
            accession_info={"taxid": "2697049"},
            domain_taxid="10239",
            scope_config={"default_id": None, "redirections": {}, "virtual_labels": {}},
            noise_filter=nf,
            vault_path="/fake/vault",
            lineages=lineages,
        )


# ---------------------------------------------------------------------------
# _apply_noise_filter_to_lineage — debug log path
# ---------------------------------------------------------------------------


class TestApplyNoiseFilterToLineageDebug:
    def test_debug_log_triggered_on_filtered_taxid(self, tmp_path, caplog):
        import json as _json
        import logging
        config = {"name_patterns": [{"regex": r"^filtered$", "description": "test"}]}
        p = tmp_path / "noise.json"
        p.write_text(_json.dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))

        taxon_info = {
            "1111": ("filtered", "species"),
            "10239": ("Viruses", "superkingdom"),
        }
        with caplog.at_level(logging.DEBUG, logger="TaxoTreeSet"):
            result = _apply_noise_filter_to_lineage(["1111", "10239"], nf, taxon_info)

        assert "1111" not in result
        assert "10239" in result


# ---------------------------------------------------------------------------
# generate_seqs_by_taxon_tree — dangling accession skip (line 132)
# ---------------------------------------------------------------------------


class TestGenerateSeqsByTaxonTreeDanglingAccession:
    def test_dangling_accession_skipped_gracefully(self, tmp_path):
        """Accession in taxons but absent from accessions → continue (line 132)."""
        registry = {
            "taxons": {"10239": ["GCF_DANGLING"]},
            "accessions": {},  # GCF_DANGLING not here
            # Lineage for taxon 10239 must include 10239 so _enumerate_accession_tasks
            # includes the task (otherwise it's filtered out before the loop body).
            "lineages": {
                "10239": [{"taxid": "10239", "rank": "superkingdom", "name": "Viruses"}]
            },
            "capacities": {},
        }
        registry_path = str(tmp_path / "registry.json")
        with open(registry_path, "w", encoding="utf-8") as fh:
            json.dump(registry, fh)

        noise_path = str(tmp_path / "noise.json")
        with open(noise_path, "w", encoding="utf-8") as fh:
            json.dump({"name_patterns": [], "rank_blacklist": {"ranks": []}}, fh)

        root = generate_seqs_by_taxon_tree(
            registry_path=registry_path,
            vault_path=str(tmp_path / "vault"),
            domain_taxid="10239",
            noise_patterns_path=noise_path,
        )
        # Dangling accession was skipped, tree has no taxa nodes
        assert root is not None
        assert all(
            getattr(n, "rank", "") != "sequence" for n in root.descendants
        )


# ---------------------------------------------------------------------------
# per-node child index (O(1) lookup) — behaviour preservation + cleanup
# ---------------------------------------------------------------------------


class TestChildIndex:
    def test_register_child_makes_lookup_find_it(self):
        # A child created and registered is found in O(1) by the index.
        root = make_root()
        child = Node("X", parent=root)
        _register_child(root, child)
        assert _find_child_by_name(root, "X") is child

    def test_index_seeds_from_preexisting_children(self):
        # A node whose children predate the index is still resolved (lazy seed).
        root = make_root()
        pre = Node("PRE", parent=root)          # created without registering
        assert _find_child_by_name(root, "PRE") is pre

    def test_drop_child_indexes_is_idempotent_and_total(self):
        root = make_root()
        a = Node("a", parent=root)
        _register_child(root, a)
        _find_child_by_name(a, "none")          # forces an index on `a` too
        _drop_child_indexes(root)
        _drop_child_indexes(root)               # second call must not raise
        assert all(
            "_child_index" not in n.__dict__ for n in (root, *root.descendants)
        )

    def _build_tree(self, tmp_path):
        # One species, two accessions sharing header H2 (dedup), registered
        # under a single taxon — exercises the index-backed leaf dedup path.
        registry = {
            "taxons": {"2697049": ["ACC1", "ACC2"]},
            "accessions": {
                "ACC1": {"taxid": "2697049", "organism": "SARS-CoV-2",
                         "headers": [{"id": "H1"}, {"id": "H2"}]},
                "ACC2": {"taxid": "2697049", "organism": "SARS-CoV-2",
                         "headers": [{"id": "H2"}, {"id": "H3"}]},
            },
            "lineages": {
                "2697049": [
                    {"taxid": "2697049", "rank": "species", "name": "SARS-CoV-2"},
                    {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
                ]
            },
            "capacities": {},
        }
        registry_path = str(tmp_path / "registry.json")
        with open(registry_path, "w", encoding="utf-8") as fh:
            json.dump(registry, fh)
        noise_path = str(tmp_path / "noise.json")
        with open(noise_path, "w", encoding="utf-8") as fh:
            json.dump({"name_patterns": [], "rank_blacklist": {"ranks": []}}, fh)
        # Empty scopes so no redirection/default-fallback node is inserted —
        # keeps the test hermetic (independent of configs/mapping.json).
        mapping_path = str(tmp_path / "mapping.json")
        with open(mapping_path, "w", encoding="utf-8") as fh:
            json.dump({"scopes": {}}, fh)
        return generate_seqs_by_taxon_tree(
            registry_path=registry_path,
            vault_path=str(tmp_path / "vault"),
            domain_taxid="10239",
            mapping_path=mapping_path,
            noise_patterns_path=noise_path,
        )

    def test_shared_header_dedups_across_accessions(self, tmp_path):
        root = self._build_tree(tmp_path)
        species = _find_child_by_name(_find_child_by_name(root, "10239"), "2697049")
        leaves = sorted(
            c.name for c in species.children if getattr(c, "rank", "") == "sequence"
        )
        assert leaves == ["H1", "H2", "H3"]   # H2 not duplicated

    def test_construction_index_dropped_from_returned_tree(self, tmp_path):
        root = self._build_tree(tmp_path)
        assert all(
            "_child_index" not in n.__dict__ for n in (root, *root.descendants)
        )


# ---------------------------------------------------------------------------
# bulk leaf attachment — deferred accumulate + single flush per node
# ---------------------------------------------------------------------------


class TestBulkLeafAttachment:
    def test_accumulate_defers_then_flush_attaches_unique_leaves(self):
        node = Node("sp")
        node.rank = "species"
        pending = {}
        _accumulate_sequence_leaves(
            node, {"organism": "O1", "headers": [{"id": "H1"}, {"id": "H2"}]},
            "/v", pending,
        )
        _accumulate_sequence_leaves(
            node, {"organism": "O2", "headers": [{"id": "H2"}, {"id": "H3"}]},
            "/v", pending,
        )
        assert len(node.children) == 0          # nothing attached until flush
        _flush_pending_leaves(pending)
        assert sorted(c.name for c in node.children) == ["H1", "H2", "H3"]
        assert all(c.rank == "sequence" for c in node.children)
        assert node.children[0].fasta_path.endswith("sequences.lmdb")

    def test_first_accession_wins_on_duplicate_header(self):
        node = Node("sp")
        pending = {}
        _accumulate_sequence_leaves(
            node, {"organism": "FIRST", "headers": [{"id": "H"}]}, "/v", pending)
        _accumulate_sequence_leaves(
            node, {"organism": "SECOND", "headers": [{"id": "H"}]}, "/v", pending)
        _flush_pending_leaves(pending)
        assert node.children[0].scientific_name == "FIRST"

    def test_flush_skips_header_colliding_with_taxon_child(self):
        node = Node("sp")
        taxon_child = Node("12345", parent=node)
        taxon_child.rank = "strain"
        pending = {}
        _accumulate_sequence_leaves(
            node, {"organism": "O", "headers": [{"id": "12345"}, {"id": "H1"}]},
            "/v", pending,
        )
        _flush_pending_leaves(pending)
        collisions = [c for c in node.children if c.name == "12345"]
        assert len(collisions) == 1
        assert collisions[0].rank == "strain"   # the taxon child, not a leaf
        assert any(c.name == "H1" and c.rank == "sequence" for c in node.children)

    def test_flush_empty_pending_is_noop(self):
        _flush_pending_leaves({})                # must not raise

    def test_leaves_across_two_nodes_each_bulk_attached(self):
        a = Node("A")
        b = Node("B")
        pending = {}
        _accumulate_sequence_leaves(a, {"headers": [{"id": "Ha"}]}, "/v", pending)
        _accumulate_sequence_leaves(b, {"headers": [{"id": "Hb"}]}, "/v", pending)
        _flush_pending_leaves(pending)
        assert [c.name for c in a.children] == ["Ha"]
        assert [c.name for c in b.children] == ["Hb"]
