"""Tests for taxotreeset.io.noise_filter.NoiseFilter."""

import json
import pytest
from taxotreeset.io.noise_filter import NoiseFilter


@pytest.fixture
def noise_config_path(tmp_path):
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


@pytest.fixture
def nf(noise_config_path):
    return NoiseFilter(config_path=noise_config_path)


# ---------------------------------------------------------------------------
# Permissive behaviour when no configuration file exists
# ---------------------------------------------------------------------------


class TestPermissiveWithMissingConfig:
    def test_does_not_raise_when_config_absent(self, tmp_path):
        nf = NoiseFilter(config_path=str(tmp_path / "nonexistent.json"))
        assert nf is not None

    def test_all_nodes_pass_when_no_config(self, tmp_path):
        nf = NoiseFilter(config_path=str(tmp_path / "nonexistent.json"))
        assert not nf.is_noise("unclassified Caudoviricetes")
        assert not nf.is_noise("environmental samples", rank="no_rank")
        assert not nf.is_noise("E. coli", rank="strain")


# ---------------------------------------------------------------------------
# Name-pattern matching
# ---------------------------------------------------------------------------


class TestNamePatternMatching:
    def test_unclassified_prefix_matches(self, nf):
        assert nf.is_noise("unclassified Caudoviricetes")

    def test_environmental_samples_matches_anywhere(self, nf):
        assert nf.is_noise("environmental samples in deep sea")

    def test_incertae_sedis_matches_as_substring(self, nf):
        assert nf.is_noise("Bacillus incertae sedis group")

    def test_normal_scientific_names_pass(self, nf):
        assert not nf.is_noise("Escherichia coli")
        assert not nf.is_noise("Bacillus subtilis")
        assert not nf.is_noise("SARS-CoV-2")

    def test_matching_is_case_insensitive(self, nf):
        assert nf.is_noise("UNCLASSIFIED Viruses")
        assert nf.is_noise("Environmental Samples")
        assert nf.is_noise("Incertae Sedis")

    def test_empty_name_does_not_match_any_pattern(self, nf):
        assert not nf.is_noise("")


# ---------------------------------------------------------------------------
# Rank blacklist
# ---------------------------------------------------------------------------


class TestRankBlacklist:
    def test_strain_rank_is_blocked(self, nf):
        assert nf.is_noise("E. coli K-12", rank="strain")

    def test_serotype_rank_is_blocked(self, nf):
        assert nf.is_noise("Salmonella serovar", rank="serotype")

    def test_subtype_rank_is_blocked(self, nf):
        assert nf.is_noise("H1N1", rank="subtype")

    def test_species_rank_passes(self, nf):
        assert not nf.is_noise("Bacillus subtilis", rank="species")

    def test_genus_rank_passes(self, nf):
        assert not nf.is_noise("Bacillus", rank="genus")

    def test_rank_blacklist_check_is_case_insensitive(self, nf):
        assert nf.is_noise("E. coli strain", rank="STRAIN")

    def test_rank_check_does_not_examine_name_when_blacklisted(self, nf):
        assert nf.is_noise("Bacillus subtilis", rank="strain")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStats:
    def test_evaluated_counter_increments_per_call(self, nf):
        for _ in range(5):
            nf.is_noise("Escherichia coli", rank="species")
        stats = nf.stats()
        assert stats["evaluated"] == 5

    def test_name_hits_counter_increments_on_name_match(self, nf):
        nf.is_noise("unclassified X", rank="species")
        nf.is_noise("Bacillus subtilis", rank="species")
        stats = nf.stats()
        assert stats["name_hits"] == 1

    def test_rank_hits_counter_increments_on_rank_match(self, nf):
        nf.is_noise("anything", rank="strain")
        stats = nf.stats()
        assert stats["rank_hits"] == 1

    def test_rank_hit_does_not_increment_name_counter(self, nf):
        nf.is_noise("unclassified X", rank="strain")
        stats = nf.stats()
        assert stats["rank_hits"] == 1
        assert stats["name_hits"] == 0

    def test_stats_accumulate_across_calls(self, nf):
        nf.is_noise("Bacillus", rank="species")
        nf.is_noise("unclassified X", rank="species")
        nf.is_noise("E. coli", rank="strain")
        stats = nf.stats()
        assert stats["evaluated"] == 3
        assert stats["name_hits"] == 1
        assert stats["rank_hits"] == 1

    def test_reset_stats_clears_all_counters(self, nf):
        nf.is_noise("unclassified X", rank="species")
        nf.is_noise("E. coli", rank="strain")
        nf.reset_stats()
        stats = nf.stats()
        assert stats["evaluated"] == 0
        assert stats["name_hits"] == 0
        assert stats["rank_hits"] == 0


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_returns_none_for_passing_nodes(self, nf):
        assert nf.explain("Bacillus subtilis", rank="species") is None

    def test_returns_string_for_name_match(self, nf):
        explanation = nf.explain("unclassified X")
        assert explanation is not None
        assert isinstance(explanation, str)

    def test_returns_string_for_rank_blacklist_hit(self, nf):
        explanation = nf.explain("E. coli", rank="strain")
        assert explanation is not None
        assert "strain" in explanation.lower()

    def test_does_not_increment_stats(self, nf):
        nf.explain("unclassified X")
        nf.explain("E. coli", rank="strain")
        stats = nf.stats()
        assert stats["evaluated"] == 0

    def test_returns_none_for_empty_scientific_name(self, nf):
        assert nf.explain("", rank="species") is None


# ---------------------------------------------------------------------------
# _load_patterns — invalid regex handling
# ---------------------------------------------------------------------------


class TestLoadPatternsInvalidRegex:
    def test_invalid_regex_does_not_raise(self, tmp_path):
        config = {
            "name_patterns": [
                {"regex": "[invalid(", "description": "bad pattern"},
                {"regex": r"^valid$", "description": "good pattern"},
            ]
        }
        p = tmp_path / "noise.json"
        p.write_text(__import__("json").dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))
        assert nf is not None

    def test_valid_pattern_after_invalid_still_matches(self, tmp_path):
        config = {
            "name_patterns": [
                {"regex": "[invalid(", "description": "bad"},
                {"regex": r"^valid_name$", "description": "good"},
            ]
        }
        p = tmp_path / "noise.json"
        p.write_text(__import__("json").dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))
        assert nf.is_noise("valid_name")

    def test_empty_regex_entry_is_skipped(self, tmp_path):
        config = {
            "name_patterns": [
                {"regex": "", "description": "empty regex"},
                {"regex": r"^match_this$", "description": "real pattern"},
            ]
        }
        p = tmp_path / "noise.json"
        p.write_text(__import__("json").dumps(config), encoding="utf-8")
        nf = NoiseFilter(config_path=str(p))
        assert nf.is_noise("match_this")
        assert not nf.is_noise("other")
