"""Tests for taxotreeset.core.generation.virtual_id — deterministic bucket ID generation."""

from taxotreeset.core.generation.virtual_id import make_virtual_id


class TestMakeVirtualId:
    def test_output_starts_with_nine(self):
        assert make_virtual_id("10239", "low_capacity").startswith("9")

    def test_output_is_exactly_nine_characters(self):
        assert len(make_virtual_id("10239", "low_capacity")) == 9

    def test_output_is_all_digits(self):
        assert make_virtual_id("10239", "low_capacity").isdigit()

    def test_deterministic_for_same_inputs(self):
        vid_a = make_virtual_id("10239", "low_capacity")
        vid_b = make_virtual_id("10239", "low_capacity")
        assert vid_a == vid_b

    def test_known_fixed_value_from_docstring(self):
        assert make_virtual_id("10239", "low_capacity") == "956419858"

    def test_different_parents_produce_different_ids(self):
        vid_a = make_virtual_id("10239", "low_capacity")
        vid_b = make_virtual_id("10240", "low_capacity")
        assert vid_a != vid_b

    def test_different_purposes_produce_different_ids(self):
        vid_a = make_virtual_id("10239", "low_capacity")
        vid_b = make_virtual_id("10239", "misc")
        assert vid_a != vid_b

    def test_virtual_parent_taxid_is_accepted(self):
        vid = make_virtual_id("956419858", "misc")
        assert vid.startswith("9")
        assert len(vid) == 9

    def test_all_purposes_produce_valid_ids(self):
        purposes = ["low_capacity", "misc", "rare_taxa", "rank_family", "rank_genus"]
        ids = [make_virtual_id("12345", p) for p in purposes]
        for vid in ids:
            assert vid.startswith("9")
            assert len(vid) == 9
        assert len(set(ids)) == len(ids)
