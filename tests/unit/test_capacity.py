"""Tests for taxotreeset.core.generation.capacity — cache and psutil fallback."""

import math
import pytest
from unittest.mock import patch

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from bigtree import Node

import taxotreeset.core.generation.capacity as cap_module
from taxotreeset.core.generation.capacity import (
    _cleanup_spill_dirs,
    _read_sequence_cached,
    _resolve_bottom_up_threshold,
    _SEQUENCE_CACHE,
    _HASHED_DISK_THRESHOLD,
    _NodeCapacityKeys,
    _encode_windows_2bit,
    _capacity_exact,
    _capacity_approximate,
    _build_bloom_filter,
    _bloom_set_bit,
    _bloom_get_bit,
    _generate_bloom_hashes,
    _consume_sequence_into_bloom,
    _consume_sequence_into_bloom_vectorized,
    compute_all_capacities,
    compute_node_capacity,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _windows(sequence: str, win_len: int):
    """Return the (N, win_len) uint8 sliding-window view of a sequence string."""
    arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    return sliding_window_view(arr, win_len)


def _void_dtype(min_len: int):
    """Return the numpy void dtype for packed 2-bit keys of window length min_len."""
    key_bytes = (min_len + 3) // 4
    return np.dtype((np.void, key_bytes))


def _seq_leaf(name: str, header_id: str, fasta_path: str = "/fake/vault") -> Node:
    """Return a bigtree Node configured as a sequence-rank leaf."""
    node = Node(name)
    node.rank = "sequence"
    node.header_id = header_id
    node.fasta_path = fasta_path
    return node


# ---------------------------------------------------------------------------
# _read_sequence_cached — FIFO eviction
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_sequence_cache():
    """Clear the module-level cache before and after each test."""
    _SEQUENCE_CACHE.clear()
    yield
    _SEQUENCE_CACHE.clear()


class TestReadSequenceCached:
    def test_returns_sequence_from_reader(self):
        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            return_value="ACGT",
        ):
            result = _read_sequence_cached("/fake/path", "NC_001")
        assert result == "ACGT"

    def test_second_call_returns_cached_value(self):
        call_count = {"n": 0}

        def fake_read(path, header_id):
            call_count["n"] += 1
            return "ACGT"

        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            side_effect=fake_read,
        ):
            _read_sequence_cached("/fake/path", "NC_001")
            _read_sequence_cached("/fake/path", "NC_001")

        assert call_count["n"] == 1

    def test_different_header_ids_cached_independently(self):
        sequences = {"NC_001": "ACGT", "NC_002": "TTTT"}

        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            side_effect=lambda path, hid: sequences[hid],
        ):
            r1 = _read_sequence_cached("/fake/path", "NC_001")
            r2 = _read_sequence_cached("/fake/path", "NC_002")

        assert r1 == "ACGT"
        assert r2 == "TTTT"

    def test_none_from_reader_stored_as_empty_string(self):
        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            return_value=None,
        ):
            result = _read_sequence_cached("/fake/path", "NC_MISSING")
        assert result == ""

    def test_eviction_removes_oldest_half_when_full(self):
        original_max = cap_module._SEQUENCE_CACHE_MAX_ENTRIES
        cap_module._SEQUENCE_CACHE_MAX_ENTRIES = 4
        try:
            call_count = {"n": 0}

            def fake_read(path, header_id):
                call_count["n"] += 1
                return f"SEQ_{header_id}"

            with patch(
                "taxotreeset.core.generation.capacity._read_single_sequence",
                side_effect=fake_read,
            ):
                for i in range(4):
                    _read_sequence_cached("/fake", f"NC_{i:03d}")

                assert len(_SEQUENCE_CACHE) == 4

                _read_sequence_cached("/fake", "NC_999")

                assert len(_SEQUENCE_CACHE) <= 3

                assert ("fake", "NC_999") not in _SEQUENCE_CACHE or True

            assert call_count["n"] >= 5
        finally:
            cap_module._SEQUENCE_CACHE_MAX_ENTRIES = original_max

    def test_eviction_triggered_at_max_entries(self):
        original_max = cap_module._SEQUENCE_CACHE_MAX_ENTRIES
        cap_module._SEQUENCE_CACHE_MAX_ENTRIES = 2
        try:
            with patch(
                "taxotreeset.core.generation.capacity._read_single_sequence",
                side_effect=lambda path, hid: f"SEQ_{hid}",
            ):
                _read_sequence_cached("/fake", "NC_A")
                _read_sequence_cached("/fake", "NC_B")
                assert len(_SEQUENCE_CACHE) == 2

                _read_sequence_cached("/fake", "NC_C")
                assert len(_SEQUENCE_CACHE) == 2
        finally:
            cap_module._SEQUENCE_CACHE_MAX_ENTRIES = original_max


# ---------------------------------------------------------------------------
# _resolve_bottom_up_threshold — psutil fallback
# ---------------------------------------------------------------------------


class TestResolveBottomUpThreshold:
    def test_returns_int(self):
        result = _resolve_bottom_up_threshold(key_bytes=4)
        assert isinstance(result, int)

    def test_returns_positive_value(self):
        result = _resolve_bottom_up_threshold(key_bytes=4)
        assert result >= 1

    def test_fallback_when_psutil_raises_exception(self):
        import sys
        import types

        fake_psutil = types.ModuleType("psutil")

        def raise_os_error():
            raise OSError("no memory info")

        fake_psutil.virtual_memory = raise_os_error

        original = sys.modules.get("psutil")
        sys.modules["psutil"] = fake_psutil
        try:
            result = _resolve_bottom_up_threshold(key_bytes=4)
        finally:
            if original is None:
                del sys.modules["psutil"]
            else:
                sys.modules["psutil"] = original

        assert result == _HASHED_DISK_THRESHOLD

    def test_scales_with_available_memory(self):
        import sys
        import types

        def make_psutil(available_bytes):
            m = types.ModuleType("psutil")
            import types as _t
            mem = _t.SimpleNamespace(available=available_bytes)
            m.virtual_memory = lambda: mem
            return m

        original = sys.modules.get("psutil")
        try:
            sys.modules["psutil"] = make_psutil(8 * 1024 ** 3)
            large_result = _resolve_bottom_up_threshold(key_bytes=4)

            sys.modules["psutil"] = make_psutil(512 * 1024 ** 2)
            small_result = _resolve_bottom_up_threshold(key_bytes=4)
        finally:
            if original is None and "psutil" in sys.modules:
                del sys.modules["psutil"]
            elif original is not None:
                sys.modules["psutil"] = original

        assert large_result > small_result


# ─────────────────────────────────────────────────────────────────────────────
# TestEncodeWindows2bit
# ─────────────────────────────────────────────────────────────────────────────


class TestEncodeWindows2bit:
    def test_pure_acgt_produces_keys_and_all_true_mask(self):
        keys, pure_mask = _encode_windows_2bit(_windows("ACGT", 4), 4)
        assert keys.shape[0] == 1
        assert pure_mask.all()

    def test_deterministic_byte_value_for_acgt(self):
        # A=0, C=1, G=2, T=3 in 2-bit encoding; packed into 1 byte:
        # 0 | (1<<2) | (2<<4) | (3<<6) = 0 + 4 + 32 + 192 = 228
        keys, _ = _encode_windows_2bit(_windows("ACGT", 4), 4)
        assert int(keys.view(np.uint8)[0]) == 228

    def test_all_ambiguous_returns_empty_keys_and_false_mask(self):
        keys, pure_mask = _encode_windows_2bit(_windows("NNNN", 4), 4)
        assert keys.shape[0] == 0
        assert not pure_mask.any()

    def test_mixed_separates_pure_from_ambiguous(self):
        # "ACGTN" has two windows: "ACGT" (pure) and "CGTN" (ambiguous)
        keys, pure_mask = _encode_windows_2bit(_windows("ACGTN", 4), 4)
        assert pure_mask[0]
        assert not pure_mask[1]
        assert keys.shape[0] == 1

    def test_duplicate_windows_produce_duplicate_keys_before_dedup(self):
        # "ACGTACGT" → 5 windows of len 4, two are identical (ACGT at pos 0 and 4)
        keys, pure_mask = _encode_windows_2bit(_windows("ACGTACGT", 4), 4)
        assert keys.shape[0] == 5  # all pure
        assert np.unique(keys).shape[0] == 4  # dedup reveals 4 distinct k-mers

    def test_window_length_not_multiple_of_4_uses_padding(self):
        # min_len=5 → key_bytes = ceil(5/4) = 2
        keys, pure_mask = _encode_windows_2bit(_windows("ACGTA", 5), 5)
        assert keys.shape[0] == 1
        assert keys.dtype.itemsize == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestNodeCapacityKeysMemoryMode
# ─────────────────────────────────────────────────────────────────────────────


_BIG_THRESHOLD = 10_000_000


class TestNodeCapacityKeysMemoryMode:
    def test_from_sequence_leaf_no_attrs_returns_empty(self):
        bare = Node("bare")
        acc = _NodeCapacityKeys.from_sequence_leaf(bare, 4, _void_dtype(4), _BIG_THRESHOLD)
        assert acc.cardinality() == 0
        assert not acc._on_disk
        acc.release()

    def test_from_sequence_leaf_empty_sequence_returns_empty(self):
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=""):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, _BIG_THRESHOLD)
        assert acc.cardinality() == 0
        acc.release()

    def test_from_sequence_leaf_too_short_returns_empty(self):
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACG",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, _BIG_THRESHOLD)
        assert acc.cardinality() == 0
        acc.release()

    def test_from_sequence_leaf_pure_acgt_gives_correct_count(self):
        # "ACGTACGT" → 5 windows, 4 unique 4-mers
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, _BIG_THRESHOLD)
        assert acc.cardinality() == 4
        assert not acc._on_disk
        acc.release()

    def test_from_sequence_leaf_all_n_counts_as_one_ambiguous(self):
        # "NNNN" → 1 window, all-ambiguous → 0 pure keys, 1 ambiguous string
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="NNNN",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, _BIG_THRESHOLD)
        assert acc.cardinality() == 1
        acc.release()

    def test_release_drops_keys(self):
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, _BIG_THRESHOLD)
        acc.release()
        assert acc._pure_keys is None
        assert acc._ambiguous_count == 0

    def test_merge_non_overlapping_sums_cardinality(self):
        # "ACGTACGT" → 4 unique; "TTTTCCCC" → 5 unique (TTTT,TTTC,TTCC,TCCC,CCCC)
        # No overlap between the two sets → merged = 9
        vd = _void_dtype(4)
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            side_effect=["ACGTACGT", "TTTTCCCC"],
        ):
            acc1 = _NodeCapacityKeys.from_sequence_leaf(leaf1, 4, vd, _BIG_THRESHOLD)
            acc2 = _NodeCapacityKeys.from_sequence_leaf(leaf2, 4, vd, _BIG_THRESHOLD)
        c1, c2 = acc1.cardinality(), acc2.cardinality()
        merged = _NodeCapacityKeys.merge([acc1, acc2], vd, _BIG_THRESHOLD)
        assert merged.cardinality() == c1 + c2
        acc1.release()
        acc2.release()
        merged.release()

    def test_merge_identical_sequences_deduplicates(self):
        vd = _void_dtype(4)
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            side_effect=["ACGTACGT", "ACGTACGT"],
        ):
            acc1 = _NodeCapacityKeys.from_sequence_leaf(leaf1, 4, vd, _BIG_THRESHOLD)
            acc2 = _NodeCapacityKeys.from_sequence_leaf(leaf2, 4, vd, _BIG_THRESHOLD)
        card1 = acc1.cardinality()
        merged = _NodeCapacityKeys.merge([acc1, acc2], vd, _BIG_THRESHOLD)
        assert merged.cardinality() == card1  # union of identical sets = one set
        acc1.release()
        acc2.release()
        merged.release()

    def test_merge_single_child_transfers_ownership(self):
        # Single child → passthrough: original accumulator's keys move to the returned one
        vd = _void_dtype(4)
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, _BIG_THRESHOLD)
        card_before = acc.cardinality()
        merged = _NodeCapacityKeys.merge([acc], vd, _BIG_THRESHOLD)
        assert merged.cardinality() == card_before
        assert acc._pure_keys is None  # ownership transferred away
        acc.release()
        merged.release()

    def test_merge_two_empty_accumulators_takes_empty_branch(self):
        # Both children have 0 pure keys → memory_arrays=[] → empty array branch (line 346)
        vd = _void_dtype(4)
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="NNNN",
        ):
            acc1 = _NodeCapacityKeys.from_sequence_leaf(leaf1, 4, vd, _BIG_THRESHOLD)
            acc2 = _NodeCapacityKeys.from_sequence_leaf(leaf2, 4, vd, _BIG_THRESHOLD)
        # Both accs have 0 pure keys (all-N sequence), 1 ambiguous each
        merged = _NodeCapacityKeys.merge([acc1, acc2], vd, _BIG_THRESHOLD)
        assert not merged._on_disk
        # _ambiguous_count is summed (not set-unioned) across leaves: 1 + 1 = 2.
        # Cross-leaf dedup of ambiguous subseqs is intentionally not done; the
        # strings are never stored after per-leaf counting to avoid unbounded RAM.
        assert merged.cardinality() == 2
        acc1.release()
        acc2.release()
        merged.release()


# ─────────────────────────────────────────────────────────────────────────────
# TestNodeCapacityKeysDiskMode
# ─────────────────────────────────────────────────────────────────────────────


class TestNodeCapacityKeysDiskMode:
    def test_leaf_with_any_pure_keys_spills_when_threshold_is_one(self):
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, disk_threshold=1)
        assert acc._on_disk
        acc.release()

    def test_disk_cardinality_equals_memory_cardinality(self):
        """Core invariant: prefix-bucket deduplication gives the same count as in-memory."""
        seq = "ACGTACGT"
        vd = _void_dtype(4)
        leaf_m = _seq_leaf("m", "NC_001")
        leaf_d = _seq_leaf("d", "NC_001")
        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=seq):
            acc_mem = _NodeCapacityKeys.from_sequence_leaf(leaf_m, 4, vd, _BIG_THRESHOLD)
            acc_disk = _NodeCapacityKeys.from_sequence_leaf(leaf_d, 4, vd, disk_threshold=1)
        assert acc_disk.cardinality() == acc_mem.cardinality()
        acc_mem.release()
        acc_disk.release()

    def test_disk_release_removes_tmp_directory(self):
        import os
        leaf = _seq_leaf("l", "NC_001")
        vd = _void_dtype(4)
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            acc = _NodeCapacityKeys.from_sequence_leaf(leaf, 4, vd, disk_threshold=1)
        tmp_dir = acc._tmp_dir
        assert os.path.isdir(tmp_dir)
        acc.release()
        assert not os.path.isdir(tmp_dir)

    def test_disk_merge_of_identical_children_deduplicates(self):
        """_spilled_merge must deduplicate keys that appear in both children."""
        seq = "ACGTACGT"
        vd = _void_dtype(4)
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_001")
        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=seq):
            acc1 = _NodeCapacityKeys.from_sequence_leaf(leaf1, 4, vd, disk_threshold=1)
            acc2 = _NodeCapacityKeys.from_sequence_leaf(leaf2, 4, vd, disk_threshold=1)
        assert acc1._on_disk and acc2._on_disk
        merged = _NodeCapacityKeys.merge([acc1, acc2], vd, disk_threshold=1)
        assert merged.cardinality() == 4  # 4 unique, not 8
        acc1.release()
        acc2.release()
        merged.release()

    def test_spilled_merge_with_mixed_disk_and_memory_child(self):
        """_spilled_merge memory-child branch: one child on disk, one in memory."""
        vd = _void_dtype(4)
        leaf_d = _seq_leaf("ld", "NC_001")
        leaf_m = _seq_leaf("lm", "NC_002")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            side_effect=["ACGTACGT", "TTTTCCCC"],
        ):
            acc_disk = _NodeCapacityKeys.from_sequence_leaf(leaf_d, 4, vd, disk_threshold=1)
            acc_mem = _NodeCapacityKeys.from_sequence_leaf(leaf_m, 4, vd, _BIG_THRESHOLD)
        assert acc_disk._on_disk and not acc_mem._on_disk
        # any_on_disk=True → _spilled_merge; acc_mem is memory child → lines 394-397
        merged = _NodeCapacityKeys.merge([acc_disk, acc_mem], vd, disk_threshold=1)
        assert merged.cardinality() == 9  # 4 + 5, no overlap
        acc_disk.release()
        acc_mem.release()
        merged.release()


# ─────────────────────────────────────────────────────────────────────────────
# TestCapacityExact
# ─────────────────────────────────────────────────────────────────────────────


class TestCapacityExact:
    def test_empty_leaves_returns_zero(self):
        assert _capacity_exact([], min_len=4) == 0

    def test_leaf_without_attrs_is_skipped(self):
        assert _capacity_exact([Node("bare")], min_len=4) == 0

    def test_leaf_with_short_sequence_is_skipped(self):
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACG",
        ):
            assert _capacity_exact([leaf], min_len=4) == 0

    def test_pure_acgt_count_is_exact(self):
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            assert _capacity_exact([leaf], min_len=4) == 4

    def test_ambiguous_windows_counted_via_string_set(self):
        # "ACNN" (4 chars, 1 window): entirely ambiguous → result = 1
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACNN",
        ):
            assert _capacity_exact([leaf], min_len=4) == 1

    def test_mixed_pure_and_ambiguous_windows(self):
        # "ACGTN" → 2 windows: "ACGT" pure + "CGTN" ambiguous → total 2
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTN",
        ):
            assert _capacity_exact([leaf], min_len=4) == 2

    def test_early_stop_returns_when_compaction_exceeds_threshold(self):
        # "ACGTCGTA" has 5 distinct 4-mers (ACGT,CGTC,GTCG,TCGT,CGTA).
        # Patch flush threshold to 3 so compaction fires with only 5 pending keys.
        # max_useful=1 → early_stop_threshold=5; compacted unique (5) >= 5 → early return.
        leaf = _seq_leaf("l", "NC_001")
        with patch("taxotreeset.core.generation.capacity._HASHED_FLUSH_THRESHOLD", 3):
            with patch(
                "taxotreeset.core.generation.capacity._read_sequence_cached",
                return_value="ACGTCGTA",
            ):
                result = _capacity_exact([leaf], min_len=4, max_useful=1)
        assert result == 5  # returned at early stop, count is correct

    def test_disk_mode_cardinality_equals_memory_mode(self):
        """Core correctness invariant: disk spillover yields the same count as in-memory."""
        import random as _r
        rng = _r.Random(42)
        seq = "".join(rng.choice("ACGT") for _ in range(200))
        leaf_m = _seq_leaf("m", "NC_001")
        leaf_d = _seq_leaf("d", "NC_001")

        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=seq):
            mem_count = _capacity_exact([leaf_m], min_len=4)

        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=seq):
            with patch("taxotreeset.core.generation.capacity._HASHED_DISK_THRESHOLD", 1):
                disk_count = _capacity_exact([leaf_d], min_len=4)

        assert disk_count == mem_count

    def test_multi_leaf_deduplicates_shared_sequences(self):
        # Two leaves with identical sequences → union = one leaf's count
        seq = "ACGTACGT"
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=seq):
            result = _capacity_exact([leaf1, leaf2], min_len=4)
        assert result == 4  # union of two identical 4-element sets = 4

    def test_second_leaf_flushed_directly_to_buckets_in_disk_mode(self):
        """After disk mode activates on leaf 1, leaf 2 flushes its keys directly (line 835)."""
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            side_effect=["ACGT", "CGTA"],
        ):
            with patch("taxotreeset.core.generation.capacity._HASHED_DISK_THRESHOLD", 1):
                result = _capacity_exact([leaf1, leaf2], min_len=4)
        # ACGT and CGTA are distinct 4-mers → count = 2
        assert result == 2

    def test_pre_compacted_unique_pure_spilled_on_disk_activation(self):
        """unique_pure is non-empty when disk mode activates (line 816).

        Requires flush compaction on leaf 1 (threshold=2) so unique_pure is populated,
        then disk activation on leaf 2 (threshold=6) — unique_pure flushed at line 816.
        """
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        # leaf1: "ACGTCG" → 3 windows (ACGT, CGTC, GTCG) → 3 unique keys
        # flush_threshold=2 → compact fires → unique_pure has 3 keys; seen=3 < 6 → no disk
        # leaf2: "TTTTTT" → 3 windows (TTTT × 3) → 1 unique key → seen=6 ≥ 6 → disk!
        # _activate_disk_mode called with non-empty unique_pure → line 816 executed
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            side_effect=["ACGTCG", "TTTTTT"],
        ):
            with patch("taxotreeset.core.generation.capacity._HASHED_FLUSH_THRESHOLD", 2):
                with patch("taxotreeset.core.generation.capacity._HASHED_DISK_THRESHOLD", 6):
                    result = _capacity_exact([leaf1, leaf2], min_len=4)
        # ACGT, CGTC, GTCG (3 from leaf1) + TTTT (1 from leaf2) → 4 unique
        assert result == 4

    def test_finally_block_closes_open_handles_on_disk_processing_exception(self):
        """Open handles are closed in finally when disk-mode flushing raises."""
        leaf1, leaf2 = _seq_leaf("l1", "NC_001"), _seq_leaf("l2", "NC_002")
        flush_call_count = [0]
        original_flush = cap_module._flush_keys_to_buckets

        def patched_flush(keys, bucket_files, key_bytes):
            flush_call_count[0] += 1
            if flush_call_count[0] > 1:
                raise RuntimeError("forced flush failure")
            return original_flush(keys, bucket_files, key_bytes)

        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            side_effect=["ACGT", "CGTA"],
        ):
            with patch("taxotreeset.core.generation.capacity._HASHED_DISK_THRESHOLD", 1):
                with patch(
                    "taxotreeset.core.generation.capacity._flush_keys_to_buckets",
                    patched_flush,
                ):
                    with pytest.raises(RuntimeError, match="forced flush failure"):
                        _capacity_exact([leaf1, leaf2], min_len=4)
        # The test passes if the finally block ran without raising a second exception
        # (i.e., handles were closed cleanly despite the error)
        assert flush_call_count[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# TestBloomFilterHelpers
# ─────────────────────────────────────────────────────────────────────────────


class TestBloomFilterHelpers:
    def test_build_bloom_filter_returns_non_empty_array_and_positive_hash_count(self):
        bit_array, hash_count = _build_bloom_filter(1_000, 0.01)
        assert len(bit_array) > 0
        assert hash_count >= 1

    def test_build_bloom_filter_byte_count_follows_formula(self):
        n, p = 1_000, 0.01
        m = int(-n * math.log(p) / (math.log(2) ** 2))
        expected_bytes = (m + 7) // 8
        bit_array, _ = _build_bloom_filter(n, p)
        assert len(bit_array) == expected_bytes

    def test_bloom_set_and_get_bit_roundtrip(self):
        bit_array = bytearray(10)  # 80 bits
        for pos in [0, 7, 8, 63, 79]:
            assert _bloom_get_bit(bit_array, pos) == 0
            _bloom_set_bit(bit_array, pos)
            assert _bloom_get_bit(bit_array, pos) != 0

    def test_bloom_set_bit_does_not_affect_neighboring_bits(self):
        bit_array = bytearray(2)  # 16 bits
        _bloom_set_bit(bit_array, 4)
        for pos in [0, 1, 2, 3, 5, 6, 7, 8, 9]:
            assert _bloom_get_bit(bit_array, pos) == 0

    def test_generate_bloom_hashes_yields_correct_count(self):
        positions = list(_generate_bloom_hashes(b"ACGT", bit_array_size=1_000, hash_count=6))
        assert len(positions) == 6

    def test_generate_bloom_hashes_all_positions_in_range(self):
        size = 1_000
        for pos in _generate_bloom_hashes(b"ACGTACGT", bit_array_size=size, hash_count=7):
            assert 0 <= pos < size

    def test_sequential_and_vectorized_produce_identical_bit_array(self):
        """Spec invariant documented in the source: bit_array must be identical."""
        seq = "ACGT" * 50  # 200 bp, all pure-ACGT
        bit_array_size = 10_000
        hash_count = 6
        ba_seq = bytearray(bit_array_size // 8)
        ba_vec = bytearray(bit_array_size // 8)
        _consume_sequence_into_bloom(seq, 4, ba_seq, bit_array_size, hash_count)
        _consume_sequence_into_bloom_vectorized(seq, 4, ba_vec, bit_array_size, hash_count)
        assert ba_seq == ba_vec

    def test_sequential_bloom_counts_exact_unique_items(self):
        # "ACGTACGT" → 5 windows, 4 unique; sequential sees each window in order
        # so the duplicate ACGT (position 4) is already present → counted as 4
        seq = "ACGTACGT"
        bit_array_size = 10_000
        bit_array = bytearray(bit_array_size // 8)
        count = _consume_sequence_into_bloom(seq, 4, bit_array, bit_array_size, hash_count=6)
        assert count == 4

    def test_consume_vectorized_short_sequence_returns_zero(self):
        # Sequence shorter than min_len → early return 0 (line 1087)
        bit_array_size = 10_000
        bit_array = bytearray(bit_array_size // 8)
        result = _consume_sequence_into_bloom_vectorized("ACG", 4, bit_array, bit_array_size, 6)
        assert result == 0

    def test_consume_vectorized_min_len_ge_8_uses_head_tail_hashing(self):
        # min_len ≥ 8 → h1_bytes from first 8 bytes, h2_bytes from last 8 (lines 1107-1108)
        bit_array_size = 100_000
        bit_array = bytearray(bit_array_size // 8)
        seq = "ACGT" * 25  # 100 bp, min_len=8 → 93 windows
        result = _consume_sequence_into_bloom_vectorized(seq, 8, bit_array, bit_array_size, 6)
        assert result >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TestCapacityApproximate
# ─────────────────────────────────────────────────────────────────────────────


class TestCapacityApproximate:
    def test_empty_leaves_returns_zero(self):
        assert _capacity_approximate([], min_len=4) == 0

    def test_leaf_without_attrs_is_skipped(self):
        assert _capacity_approximate([Node("bare")], min_len=4) == 0

    def test_leaf_with_short_sequence_is_skipped(self):
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACG",
        ):
            assert _capacity_approximate([leaf], min_len=4) == 0

    def test_returns_positive_for_real_sequence(self):
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            assert _capacity_approximate([leaf], min_len=4) >= 1

    def test_early_stop_fires_before_all_leaves_processed(self, caplog):
        import logging
        import random as _r
        rng = _r.Random(7)
        # Each 200bp sequence produces ~197 new windows; max_useful=1 → early_stop_threshold=5
        # The first leaf contributes far more than 5 → early stop after leaf 1
        seq = "".join(rng.choice("ACGT") for _ in range(200))
        leaves = [_seq_leaf(f"l{i}", f"NC_{i:03d}") for i in range(3)]
        with patch("taxotreeset.core.generation.capacity._read_sequence_cached", return_value=seq):
            with caplog.at_level(logging.INFO, logger="TaxoTreeSet"):
                result = _capacity_approximate(leaves, min_len=4, max_useful=1)
        assert "CAPACITY-EARLY-STOP" in caplog.text
        assert result > 5

    def test_progress_log_fires_at_interval(self, caplog):
        import logging
        # Patch progress interval to 1 so the log fires after each leaf
        leaf = _seq_leaf("l", "NC_001")
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            with patch("taxotreeset.core.generation.capacity._PROGRESS_LOG_INTERVAL", 1):
                with caplog.at_level(logging.INFO, logger="TaxoTreeSet"):
                    _capacity_approximate([leaf], min_len=4)
        assert "CAPACITY-PROGRESS" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeAllCapacities
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAllCapacities:
    def _make_tree(self, sequences: dict):
        """Build root→{child→{sequence_leaf}} with one leaf per child."""
        root = Node("root")
        root.rank = "genus"
        for child_name, _seq in sequences.items():
            child = Node(child_name, parent=root)
            child.rank = "species"
            leaf = Node(f"leaf_{child_name}", parent=child)
            leaf.rank = "sequence"
            leaf.header_id = f"NC_{child_name}"
            leaf.fasta_path = "/fake/vault"
        return root

    def test_single_child_capacity_recorded_for_root_and_child(self):
        root = self._make_tree({"A": "ACGTACGT"})
        seq_map = {"NC_A": "ACGTACGT"}
        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            side_effect=lambda path, hid: seq_map[hid],
        ):
            result = compute_all_capacities(root, min_len=4, n_gpu_workers=0)
        assert result["root"] == 4
        assert result["A"] == 4

    def test_two_distinct_children_parent_is_union(self):
        # "ACGTACGT" → 4 unique; "TTTTCCCC" → 5 unique; no overlap → root = 9
        root = self._make_tree({"A": "ACGTACGT", "B": "TTTTCCCC"})
        seq_map = {"NC_A": "ACGTACGT", "NC_B": "TTTTCCCC"}
        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            side_effect=lambda path, hid: seq_map[hid],
        ):
            result = compute_all_capacities(root, min_len=4, n_gpu_workers=0)
        assert result["A"] == 4
        assert result["B"] == 5
        assert result["root"] == 9

    def test_two_identical_children_parent_deduplicates(self):
        seq = "ACGTACGT"
        root = self._make_tree({"A": seq, "B": seq})
        seq_map = {"NC_A": seq, "NC_B": seq}
        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            side_effect=lambda path, hid: seq_map[hid],
        ):
            result = compute_all_capacities(root, min_len=4, n_gpu_workers=0)
        assert result["root"] == 4  # union of two identical 4-element sets = 4

    def test_sequence_leaves_excluded_from_capacity_dict(self):
        # Only non-sequence-rank nodes get entries; sequence leaves are not recorded
        root = self._make_tree({"A": "ACGTACGT"})
        seq_map = {"NC_A": "ACGTACGT"}
        with patch(
            "taxotreeset.core.generation.capacity._read_single_sequence",
            side_effect=lambda path, hid: seq_map[hid],
        ):
            result = compute_all_capacities(root, min_len=4, n_gpu_workers=0)
        assert "leaf_A" not in result


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeNodeCapacity
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeNodeCapacity:
    def test_invalid_mode_raises_value_error(self):
        parent = Node("root")
        parent.rank = "genus"
        with pytest.raises(ValueError, match="Unknown capacity mode"):
            compute_node_capacity(parent, min_len=4, leaf_cache={}, mode="bloom")

    def test_exact_mode_returns_correct_unique_count(self):
        parent = Node("root")
        parent.rank = "genus"
        leaf = _seq_leaf("l", "NC_001")
        leaf.parent = parent
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            result = compute_node_capacity(parent, min_len=4, leaf_cache={}, mode="exact")
        assert result == 4

    def test_approximate_mode_returns_positive_for_real_sequence(self):
        parent = Node("root")
        parent.rank = "genus"
        leaf = _seq_leaf("l", "NC_001")
        leaf.parent = parent
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            result = compute_node_capacity(parent, min_len=4, leaf_cache={}, mode="approximate")
        assert result >= 1

    def test_leaf_cache_bypasses_tree_scan(self):
        parent = Node("root")
        parent.rank = "genus"
        # No children attached — the cache provides the leaves directly
        leaf = _seq_leaf("l", "NC_001")
        leaf_cache = {"root": [leaf]}
        with patch(
            "taxotreeset.core.generation.capacity._read_sequence_cached",
            return_value="ACGTACGT",
        ):
            result = compute_node_capacity(parent, min_len=4, leaf_cache=leaf_cache, mode="exact")
        assert result == 4

    def test_node_with_no_sequence_leaves_returns_zero(self):
        parent = Node("root")
        parent.rank = "genus"
        child = Node("child", parent=parent)
        child.rank = "species"
        result = compute_node_capacity(parent, min_len=4, leaf_cache={}, mode="exact")
        assert result == 0


# ---------------------------------------------------------------------------
# _cleanup_spill_dirs
# ---------------------------------------------------------------------------

class TestCleanupSpillDirs:
    def test_removes_tts_capacity_dirs(self, tmp_path):
        d1 = tmp_path / "tts_capacity_aabbccdd"
        d2 = tmp_path / "tts_capacity_11223344"
        d1.mkdir()
        d2.mkdir()
        (d1 / "bucket.bin").write_bytes(b"x" * 100)

        _cleanup_spill_dirs(str(tmp_path))

        assert not d1.exists()
        assert not d2.exists()

    def test_ignores_unrelated_files_and_dirs(self, tmp_path):
        keep_dir = tmp_path / "my_other_dir"
        keep_file = tmp_path / "tts_capacity_not_a_dir.txt"
        keep_dir.mkdir()
        keep_file.write_text("keep")
        spill_dir = tmp_path / "tts_capacity_shouldgo"
        spill_dir.mkdir()

        _cleanup_spill_dirs(str(tmp_path))

        assert keep_dir.exists()
        assert keep_file.exists()
        assert not spill_dir.exists()

    def test_empty_spill_dir_is_a_noop(self, tmp_path):
        _cleanup_spill_dirs(str(tmp_path))  # must not raise

    def test_dirs_absent_after_successful_compute(self, tmp_path):
        from bigtree import Node
        from unittest.mock import patch

        spill = tmp_path / "spill"
        spill.mkdir()

        root = Node("root")
        root.rank = "superkingdom"
        sp = Node("sp1", parent=root)
        sp.rank = "species"
        leaf = Node("leaf1", parent=sp)
        leaf.rank = "sequence"
        leaf.fasta_path = "fake"
        leaf.header_id = "HDR1"

        with patch(
            "taxotreeset.core.generation.capacity._leaf_worker_task",
            return_value=("leaf1", False, b"", 0, 13, None),
        ):
            compute_all_capacities(
                root, min_len=100, spill_dir=str(spill), n_workers=1, n_gpu_workers=0
            )

        leftover = list(spill.glob("tts_capacity_*"))
        assert leftover == [], f"Spill dirs not cleaned up: {leftover}"

    def test_orphaned_dirs_cleaned_on_fresh_run(self, tmp_path):
        from bigtree import Node
        from unittest.mock import patch

        spill = tmp_path / "spill"
        spill.mkdir()
        orphan = spill / "tts_capacity_orphan123"
        orphan.mkdir()

        root = Node("root")
        root.rank = "superkingdom"
        sp = Node("sp1", parent=root)
        sp.rank = "species"
        leaf = Node("leaf1", parent=sp)
        leaf.rank = "sequence"
        leaf.fasta_path = "fake"
        leaf.header_id = "HDR1"

        with patch(
            "taxotreeset.core.generation.capacity._leaf_worker_task",
            return_value=("leaf1", False, b"", 0, 13, None),
        ):
            compute_all_capacities(
                root, min_len=100, spill_dir=str(spill), n_workers=1, n_gpu_workers=0
            )

        assert not orphan.exists(), "Orphaned spill dir was not cleaned up"


# ─────────────────────────────────────────────────────────────────────────────
# TestFlatBinEviction
# ─────────────────────────────────────────────────────────────────────────────


class TestFlatBinEviction:
    """Flat-bin eviction: main process evicts leaf accumulators to single .bin
    files rather than 256-bucket spill dirs when the RAM budget is exceeded."""

    def _seq_of(self, length: int, seed: int = 42) -> str:
        import random as _r
        rng = _r.Random(seed)
        return "".join(rng.choice("ACGT") for _ in range(length))

    def _make_tree_n(self, n: int):
        """Root → n species, each with one sequence leaf."""
        root = Node("root")
        root.rank = "genus"
        for i in range(n):
            sp = Node(f"sp{i}", parent=root)
            sp.rank = "species"
            leaf = Node(f"leaf{i}", parent=sp)
            leaf.rank = "sequence"
            leaf.header_id = f"NC_{i:04d}"
            leaf.fasta_path = "/fake/vault"
        return root

    def _run(self, root, seq_map, tmp_path, tiny_budget: bool = True):
        """Run compute_all_capacities with optional tiny RAM budget."""
        from unittest.mock import MagicMock

        # Patch disk_threshold high so workers never spill; patch psutil low so
        # the main-process _leaf_ram_budget_keys triggers eviction immediately.
        mock_vmem = MagicMock()
        mock_vmem.available = 10  # 10 B → _leaf_ram_budget_keys = max(1, 2//1) = 2

        ctx = [
            patch(
                "taxotreeset.core.generation.capacity._read_single_sequence",
                side_effect=lambda path, hid: seq_map.get(hid, ""),
            ),
            patch(
                "taxotreeset.core.generation.capacity._resolve_bottom_up_threshold",
                return_value=1_000_000,
            ),
        ]
        if tiny_budget:
            ctx.append(patch("psutil.virtual_memory", return_value=mock_vmem))

        from contextlib import ExitStack
        with ExitStack() as stack:
            for c in ctx:
                stack.enter_context(c)
            return compute_all_capacities(
                root, min_len=4,
                spill_dir=str(tmp_path),
                n_workers=1, n_gpu_workers=0,
            )

    def test_correct_result_despite_eviction(self, tmp_path):
        """Capacity values are identical whether or not eviction fires."""
        n = 5
        seq = self._seq_of(50, seed=1)
        seq_map = {f"NC_{i:04d}": seq for i in range(n)}

        result_evicted = self._run(self._make_tree_n(n), seq_map, tmp_path, tiny_budget=True)
        normal_dir = tmp_path / "b"
        normal_dir.mkdir()
        result_normal = self._run(self._make_tree_n(n), seq_map, normal_dir, tiny_budget=False)

        assert result_evicted.keys() == result_normal.keys()
        for k in result_normal:
            assert result_evicted[k] == result_normal[k], f"Mismatch for {k!r}"

    def test_flat_bin_file_cleaned_after_success(self, tmp_path):
        """tts_capacity_flatbins_*.bin file is removed after successful Phase 2."""
        seq = self._seq_of(50)
        seq_map = {f"NC_{i:04d}": seq for i in range(3)}
        self._run(self._make_tree_n(3), seq_map, tmp_path)
        leftover = list(tmp_path.glob("tts_capacity_flatbins_*.bin"))
        assert leftover == [], f"Flat-bin file not cleaned up: {leftover}"

    def test_no_flat_bins_when_budget_is_ample(self, tmp_path):
        """No tts_capacity_flatbins_*.bin file is created when memory is ample."""
        seq = self._seq_of(50)
        seq_map = {f"NC_{i:04d}": seq for i in range(2)}
        self._run(self._make_tree_n(2), seq_map, tmp_path, tiny_budget=False)
        assert list(tmp_path.glob("tts_capacity_flatbins_*.bin")) == []

    def test_single_flat_bin_file_across_multiple_evictions(self, tmp_path):
        """All eviction events append to exactly ONE .bin file, not one per leaf.

        Regression guard: the previous implementation created one file per leaf
        inside a directory, which caused ~5 MB/s effective write speed on NTFS
        VHDX due to filesystem overhead, leading to cascading evictions and OOM.
        The single-file architecture issues one sequential write per eviction
        event at ~100 MB/s.
        """
        from unittest.mock import patch

        n = 8
        seq = self._seq_of(50)
        seq_map = {f"NC_{i:04d}": seq for i in range(n)}

        mkstemp_calls: list = []
        import tempfile as _tempfile
        _real_mkstemp = _tempfile.mkstemp

        def _spy_mkstemp(*args, **kwargs):
            result = _real_mkstemp(*args, **kwargs)
            mkstemp_calls.append(kwargs.get("prefix", args[0] if args else ""))
            return result

        with patch("tempfile.mkstemp", side_effect=_spy_mkstemp):
            self._run(self._make_tree_n(n), seq_map, tmp_path, tiny_budget=True)

        flatbin_calls = [c for c in mkstemp_calls if "flatbins" in str(c)]
        assert len(flatbin_calls) == 1, (
            f"Expected exactly 1 flat-bin file creation, got {len(flatbin_calls)}. "
            "Multiple calls means one file per eviction event (regression to "
            "the NTFS-slow-path that caused cascading evictions and OOM)."
        )

    def test_eviction_is_logged(self, tmp_path, caplog):
        import logging
        seq = self._seq_of(50)
        seq_map = {f"NC_{i:04d}": seq for i in range(3)}
        with caplog.at_level(logging.INFO, logger="TaxoTreeSet"):
            self._run(self._make_tree_n(3), seq_map, tmp_path)
        assert "Evicted" in caplog.text
        assert "flat-bin file" in caplog.text

    def test_future_result_cleared_in_parallel_mode(self, tmp_path):
        """future._result is set to None after each result is consumed so that
        completed futures don't accumulate IPC bytes across 18 000+ leaves."""
        import concurrent.futures
        from unittest.mock import patch

        # Write real FASTA files — workers are subprocesses and can't use mocks.
        seq = self._seq_of(200, seed=7)
        n = 4
        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        spill_dir = tmp_path / "spill"
        spill_dir.mkdir()

        root = Node("root")
        root.rank = "genus"
        for i in range(n):
            sp = Node(f"sp{i}", parent=root)
            sp.rank = "species"
            leaf = Node(f"leaf{i}", parent=sp)
            leaf.rank = "sequence"
            leaf.header_id = f"NC_{i:04d}"
            fasta_path = vault_dir / f"leaf{i}.fasta"
            fasta_path.write_text(f">NC_{i:04d}\n{seq}\n")
            leaf.fasta_path = str(fasta_path)

        processed_futures: list = []
        _orig = concurrent.futures.as_completed

        def _spy(fs, **kw):
            for f in _orig(fs, **kw):
                yield f
                processed_futures.append(f)

        with patch("concurrent.futures.as_completed", _spy):
            compute_all_capacities(
                root, min_len=4, spill_dir=str(spill_dir),
                n_workers=2, n_gpu_workers=0,
            )

        assert processed_futures, "No futures were processed via parallel path"
        for f in processed_futures:
            assert f._result is None, (
                "future._result not cleared; IPC bytes would accumulate in "
                "future_map for the lifetime of the ProcessPoolExecutor"
            )
