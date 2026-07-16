"""Tests for taxotreeset.dataset.utils — LMDB cache, read helpers."""

import os
import zlib
from unittest.mock import MagicMock, patch

import lmdb
import pytest

from taxotreeset.dataset import utils as utils_module
from taxotreeset.dataset.utils import (
    _get_fasta_sequence_length,
    _get_lmdb_env,
    _pool_worker_initializer,
    _read_single_sequence,
)


# ---------------------------------------------------------------------------
# helpers & fixtures
# ---------------------------------------------------------------------------


def _make_lmdb(base_path, data: dict[str, str]) -> str:
    lmdb_dir = base_path / "test.lmdb"
    lmdb_dir.mkdir(parents=True)
    env = lmdb.open(str(lmdb_dir), map_size=1024 * 1024, max_dbs=0)
    with env.begin(write=True) as txn:
        for key, value in data.items():
            txn.put(key.encode("utf-8"), zlib.compress(value.encode("utf-8")))
    env.close()
    return str(lmdb_dir)


@pytest.fixture(autouse=True)
def reset_lmdb_cache():
    """Isolate each test from leftover LMDB state."""
    for env in list(utils_module._LMDB_ENV_CACHE.values()):
        try:
            env.close()
        except Exception:
            pass
    utils_module._LMDB_ENV_CACHE = {}
    utils_module._LMDB_CACHE_PID = None
    yield
    for env in list(utils_module._LMDB_ENV_CACHE.values()):
        try:
            env.close()
        except Exception:
            pass
    utils_module._LMDB_ENV_CACHE = {}
    utils_module._LMDB_CACHE_PID = None


# ---------------------------------------------------------------------------
# _pool_worker_initializer
# ---------------------------------------------------------------------------


class TestPoolWorkerInitializer:
    def test_clears_cache(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {"k": "ACGT"})
        _get_lmdb_env(lmdb_path)
        assert len(utils_module._LMDB_ENV_CACHE) > 0
        _pool_worker_initializer()
        assert utils_module._LMDB_ENV_CACHE == {}

    def test_resets_pid_to_none(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {"k": "ACGT"})
        _get_lmdb_env(lmdb_path)
        _pool_worker_initializer()
        assert utils_module._LMDB_CACHE_PID is None


# ---------------------------------------------------------------------------
# _get_lmdb_env
# ---------------------------------------------------------------------------


class TestGetLmdbEnv:
    def test_returns_open_environment(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {"key": "value"})
        env = _get_lmdb_env(lmdb_path)
        assert env is not None

    def test_same_path_returns_cached_env(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {"key": "value"})
        env1 = _get_lmdb_env(lmdb_path)
        env2 = _get_lmdb_env(lmdb_path)
        assert env1 is env2

    def test_raises_file_not_found_for_nonexistent_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _get_lmdb_env(str(tmp_path / "nonexistent.lmdb"))

    def test_pid_change_clears_cache_and_reopens(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {"key": "value"})
        _get_lmdb_env(lmdb_path)
        # Simulate a stale PID from a forked parent
        utils_module._LMDB_CACHE_PID = os.getpid() + 99999
        # Should clear the stale cache and open a fresh env
        env = _get_lmdb_env(lmdb_path)
        assert utils_module._LMDB_CACHE_PID == os.getpid()
        assert env is not None

    def test_different_paths_cached_independently(self, tmp_path):
        path_a = _make_lmdb(tmp_path / "a", {"k": "seq_a"})
        path_b = _make_lmdb(tmp_path / "b", {"k": "seq_b"})
        env_a = _get_lmdb_env(path_a)
        env_b = _get_lmdb_env(path_b)
        assert env_a is not env_b


# ---------------------------------------------------------------------------
# _read_single_sequence
# ---------------------------------------------------------------------------


class TestReadSingleSequence:
    def test_reads_stored_sequence(self, tmp_path):
        seq = "ACGTACGT" * 50
        lmdb_path = _make_lmdb(tmp_path, {"NC_001": seq})
        assert _read_single_sequence(lmdb_path, "NC_001") == seq

    def test_soft_masked_lowercase_is_uppercased(self, tmp_path):
        # NCBI eukaryotic genomes are soft-masked (lowercase = repeat regions).
        # The read boundary normalizes to canonical ACGT for every consumer.
        stored = "acgtACGTnnnNacGT"
        lmdb_path = _make_lmdb(tmp_path, {"NC_soft": stored})
        assert _read_single_sequence(lmdb_path, "NC_soft") == "ACGTACGTNNNNACGT"

    def test_returns_empty_for_missing_key(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {})
        assert _read_single_sequence(lmdb_path, "MISSING_KEY") == ""

    def test_returns_empty_for_nonexistent_path(self):
        assert _read_single_sequence("/no/such/path.lmdb", "NC_001") == ""

    def test_multiple_sequences_in_same_vault(self, tmp_path):
        seqs = {
            "SEQ_A": "AAAA" * 100,
            "SEQ_B": "CCCC" * 100,
            "SEQ_C": "GGGG" * 100,
        }
        lmdb_path = _make_lmdb(tmp_path, seqs)
        for header, expected in seqs.items():
            assert _read_single_sequence(lmdb_path, header) == expected

    def test_returns_empty_when_lmdb_txn_raises(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {"NC_001": "ACGT" * 10})
        mock_env = MagicMock()
        mock_env.begin.side_effect = lmdb.Error("forced lmdb error")
        with patch("taxotreeset.dataset.utils._get_lmdb_env", return_value=mock_env):
            result = _read_single_sequence(lmdb_path, "NC_001")
        assert result == ""

    def test_returns_empty_when_stored_data_is_not_valid_zlib(self, tmp_path):
        # Write raw (non-compressed) bytes directly to skip zlib.compress
        lmdb_dir = tmp_path / "corrupt.lmdb"
        lmdb_dir.mkdir()
        env = lmdb.open(str(lmdb_dir), map_size=1024 * 1024)
        with env.begin(write=True) as txn:
            txn.put(b"NC_BAD", b"not zlib compressed data!!!!")
        env.close()
        utils_module._LMDB_ENV_CACHE = {}
        utils_module._LMDB_CACHE_PID = None

        result = _read_single_sequence(str(lmdb_dir), "NC_BAD")
        assert result == ""


# ---------------------------------------------------------------------------
# _get_fasta_sequence_length
# ---------------------------------------------------------------------------


class TestGetFastaSequenceLength:
    def test_returns_exact_length(self, tmp_path):
        seq = "ACGT" * 50  # 200 bp
        lmdb_path = _make_lmdb(tmp_path, {"NC_001": seq})
        assert _get_fasta_sequence_length(lmdb_path, "NC_001") == 200

    def test_returns_zero_for_missing_key(self, tmp_path):
        lmdb_path = _make_lmdb(tmp_path, {})
        assert _get_fasta_sequence_length(lmdb_path, "MISSING") == 0

    def test_returns_zero_for_nonexistent_path(self):
        assert _get_fasta_sequence_length("/no/such/path.lmdb", "NC_001") == 0
