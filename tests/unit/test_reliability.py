"""Tests for the per-head reliability annotator (P12-B)."""

from taxotreeset.benchmark.reliability import annotate_reliability

_AP_LOW = {"belongs_genomes": 6, "a_priori_flag": "low", "split_mode": "genome-level"}
_AP_OK = {"belongs_genomes": 20, "a_priori_flag": "ok", "split_mode": "genome-level"}


class TestApriori:
    def test_low_flag_without_training(self):
        r = annotate_reliability(_AP_LOW)
        assert r["verdict"] == "noisy-metrics"
        assert r["verdict_source"] == "a_priori"
        assert r["belongs_genomes"] == 6  # a-priori preserved

    def test_ok_flag_without_training(self):
        assert annotate_reliability(_AP_OK)["verdict"] == "reliable"

    def test_no_apriori_defaults_reliable(self):
        assert annotate_reliability(None)["verdict"] == "reliable"


class TestTrainingDetermines:
    def test_not_learned_is_unreliable(self):
        r = annotate_reliability(
            _AP_OK, {"learned": False, "test_f1": 0.5, "val_f1s": [0.5, 0.5]})
        assert r["verdict"] == "unreliable"
        assert r["verdict_source"] == "training"

    def test_stable_learned_is_reliable(self):
        r = annotate_reliability(
            _AP_OK, {"learned": True, "test_f1": 0.90, "val_f1s": [0.88, 0.89, 0.90]})
        assert r["verdict"] == "reliable"
        assert r["posterior"]["val_test_gap"] == 0.0

    def test_oscillating_val_is_noisy(self):
        # Lavidaviridae-like: val f1 swings while test ends high
        r = annotate_reliability(
            _AP_LOW,
            {"learned": True, "test_f1": 0.93, "val_f1s": [0.83, 0.58, 0.70, 0.67, 0.59]})
        assert r["verdict"] == "noisy-metrics"
        assert r["posterior"]["val_f1_std"] > 0.08

    def test_large_val_test_gap_is_noisy(self):
        r = annotate_reliability(
            _AP_OK, {"learned": True, "test_f1": 0.90, "val_f1s": [0.66, 0.66, 0.66]})
        assert r["verdict"] == "noisy-metrics"
        assert r["posterior"]["val_test_gap"] == 0.24

    def test_training_overrides_low_apriori(self):
        # a-priori "low" (few genomes) but it learned stably -> reliable
        r = annotate_reliability(
            _AP_LOW, {"learned": True, "test_f1": 0.93, "val_f1s": [0.92, 0.93, 0.93]})
        assert r["verdict"] == "reliable"
        assert r["verdict_source"] == "training"
