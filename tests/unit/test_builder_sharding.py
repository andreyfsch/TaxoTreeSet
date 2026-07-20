"""Tests for task-level parquet sharding in the dataset builder.

Sharding splits a head's per-split tasks across workers (part files) and merges
the parts back into the single ``<split>.parquet`` downstream expects. Correctness
rests on: the per-task row count being independent of which shard runs it, the
merge being a lossless row-group concatenation, and resume/idempotence at the
split, shard, and merge level. Driven through a real temp LMDB vault
(``_vault_fixture``) so the production reader path is exercised.

Sequences are long enough (``len >= 2*n*max_len``) to trigger the non-overlapping
sampler, which returns exactly ``n`` rows per task — so total row counts are
deterministic even though the subseq sampling RNG is unseeded.
"""

import os

import pyarrow.parquet as pq
import pytest

from taxotreeset.dataset import builder as B
from taxotreeset.dataset.builder import (
    DatasetBuilder,
    _merge_worker,
    _partition_tasks,
    _plan_shards,
    _shard_worker,
    extract_parent_node_worker,
)
from tests.unit._vault_fixture import make_test_vault

_MAX_LEN = 200
_N = 5              # samples/task -> exactly _N rows/task (non-overlapping)
_SEQ_LEN = 4000     # >= 2*_N*_MAX_LEN = 2000, so non-overlapping sampler is used


def _make_vault(tmp_path, n_seqs):
    seqs = {f"seq{i}": ("ACGT" * (_SEQ_LEN // 4)) for i in range(n_seqs)}
    return make_test_vault(tmp_path, seqs)


def _task(vault, header_id, class_idx):
    return {
        "fasta_path": vault, "header_id": header_id, "n": _N,
        "class_idx": class_idx, "start_pct": 0.0, "end_pct": 1.0,
    }


def _job(taxid, target_dir, parent_tasks):
    return (taxid, target_dir, parent_tasks, _MAX_LEN, 42, "parquet")


def _rows(path):
    return pq.read_table(path).num_rows if os.path.exists(path) else 0


def _class_counts(path):
    if not os.path.exists(path):
        return {}
    col = pq.read_table(path).column("class_idx").to_pylist()
    return {c: col.count(c) for c in set(col)}


def _parts(target_dir, split):
    return [f for f in os.listdir(target_dir) if f.startswith(f"{split}.part")]


# ---------------------------------------------------------------------------
# _partition_tasks
# ---------------------------------------------------------------------------

class TestPartitionTasks:
    def test_below_target_is_one_shard(self):
        tasks = [{"n": 10}, {"n": 10}]
        assert len(_partition_tasks(tasks, shard_rows_target=1000)) == 1

    def test_single_task_never_splits(self):
        # one huge genome cannot be split by this pass (intra-genome is deferred)
        assert len(_partition_tasks([{"n": 10_000}], shard_rows_target=100)) == 1

    def test_splits_and_balances(self):
        tasks = [{"n": 10} for _ in range(12)]  # 120 rows, target 30 -> 4 shards
        shards = _partition_tasks(tasks, shard_rows_target=30)
        assert len(shards) == 4
        loads = [sum(t["n"] for t in s) for s in shards]
        assert max(loads) - min(loads) <= 10   # balanced within one task


# ---------------------------------------------------------------------------
# _plan_shards resume
# ---------------------------------------------------------------------------

class TestPlanShards:
    def test_skips_already_built_split(self, tmp_path):
        target = tmp_path / "head"
        target.mkdir()
        (target / "train.parquet").write_bytes(b"x")   # pretend train is built
        parent_tasks = {"train": [{"n": 5}], "val": [{"n": 5}], "test": []}
        shard_jobs, merge_jobs = _plan_shards(
            [_job("t", str(target), parent_tasks)], shard_rows_target=1
        )
        built_splits = {mj[1] for mj in merge_jobs}
        assert "train" not in built_splits     # skipped (final exists)
        assert "val" in built_splits
        assert all(sj[1] != "train" for sj in shard_jobs)

    def test_part_names_are_content_hashed(self, tmp_path):
        target = tmp_path / "head"
        target.mkdir()
        tasks = {"train": [{"n": 5, "fasta_path": "/f", "header_id": "H",
                            "start_pct": 0.0, "end_pct": 1.0, "class_idx": 0}]}
        shard_jobs, _ = _plan_shards(
            [_job("t", str(target), tasks)], shard_rows_target=1000
        )
        name = os.path.basename(shard_jobs[0][2])   # part_path
        # <split>.part{idx:05d}.{hash}.{fmt}
        assert name.startswith("train.part00000.")
        assert name.endswith(".parquet")
        assert len(name.split(".")) == 4

    def test_changed_tasks_get_new_part_and_clean_stale(self, tmp_path):
        # D: a resumed run with a changed schedule must not reuse a stale part,
        # and the superseded part is removed rather than orphaned on disk.
        target = tmp_path / "head"
        target.mkdir()
        t1 = {"n": 5, "fasta_path": "/f", "header_id": "H1",
              "start_pct": 0.0, "end_pct": 1.0, "class_idx": 0}
        sj1, _ = _plan_shards([_job("t", str(target), {"train": [t1]})], 1000)
        open(sj1[0][2], "wb").close()                # part left by a prior run
        t2 = dict(t1, header_id="H2")                # the schedule changed
        sj2, _ = _plan_shards([_job("t", str(target), {"train": [t2]})], 1000)
        assert sj2[0][2] != sj1[0][2]                # new content hash -> new path
        assert not os.path.exists(sj1[0][2])         # stale part cleaned up


# ---------------------------------------------------------------------------
# end-to-end: sharded == serial, merge, resume
# ---------------------------------------------------------------------------

def _build_sharded_direct(jobs, shard_rows_target):
    """Run the sharded pipeline in-process (no pool) — same logic, fast."""
    shard_jobs, merge_jobs = _plan_shards(jobs, shard_rows_target)
    for sj in shard_jobs:
        _shard_worker(sj)
    for mj in merge_jobs:
        _merge_worker(mj)


class TestShardedBuild:
    def _parent_tasks(self, vault):
        # train: 8 tasks (2 classes), val: 2, test: 2 -> non-trivial split sizes
        return {
            "train": [_task(vault, f"seq{i}", class_idx=i % 2) for i in range(8)],
            "val": [_task(vault, f"seq{i}", class_idx=0) for i in (8, 9)],
            "test": [_task(vault, f"seq{i}", class_idx=1) for i in (10, 11)],
        }

    def test_sharded_matches_serial(self, tmp_path):
        vault = _make_vault(tmp_path, 12)
        parent_tasks = self._parent_tasks(vault)

        serial_dir = tmp_path / "serial"
        serial_dir.mkdir()
        extract_parent_node_worker(_job("t", str(serial_dir), parent_tasks))

        shard_dir = tmp_path / "shard"
        shard_dir.mkdir()
        _build_sharded_direct(
            [_job("t", str(shard_dir), parent_tasks)], shard_rows_target=15
        )

        for split in ("train", "val", "test"):
            s_path = os.path.join(str(serial_dir), f"{split}.parquet")
            h_path = os.path.join(str(shard_dir), f"{split}.parquet")
            assert _rows(h_path) == _rows(s_path), split
            assert _class_counts(h_path) == _class_counts(s_path), split
        # train had 8 tasks * 5 = 40 rows, sharded > 1 then merged clean
        assert _rows(os.path.join(str(shard_dir), "train.parquet")) == 40
        assert not _parts(str(shard_dir), "train")   # parts deleted after merge

    def test_test_novel_holdout_split_is_written_both_paths(self, tmp_path):
        # The optional 4th split (cluster-aware novel-lineage holdout) must be
        # written by both the serial and the sharded extraction paths.
        vault = _make_vault(tmp_path, 14)
        parent_tasks = self._parent_tasks(vault)
        parent_tasks["test_novel"] = [
            _task(vault, f"seq{i}", class_idx=1) for i in (12, 13)
        ]
        serial_dir = tmp_path / "serial_novel"
        serial_dir.mkdir()
        extract_parent_node_worker(_job("t", str(serial_dir), parent_tasks))
        shard_dir = tmp_path / "shard_novel"
        shard_dir.mkdir()
        _build_sharded_direct(
            [_job("t", str(shard_dir), parent_tasks)], shard_rows_target=15)
        for d in (serial_dir, shard_dir):
            assert _rows(os.path.join(str(d), "test_novel.parquet")) == 2 * _N

    def test_no_test_novel_file_without_a_holdout(self, tmp_path):
        # Heads without a holdout must not emit an (empty) test_novel parquet.
        vault = _make_vault(tmp_path, 12)
        target = tmp_path / "plain"
        target.mkdir()
        extract_parent_node_worker(_job("t", str(target), self._parent_tasks(vault)))
        assert not os.path.exists(os.path.join(str(target), "test_novel.parquet"))

    def test_single_task_head_one_shard(self, tmp_path):
        vault = _make_vault(tmp_path, 1)
        target = tmp_path / "one"
        target.mkdir()
        _build_sharded_direct(
            [_job("t", str(target), {"train": [_task(vault, "seq0", 1)],
                                     "val": [], "test": []})],
            shard_rows_target=15,
        )
        assert _rows(os.path.join(str(target), "train.parquet")) == _N
        assert not _parts(str(target), "train")

    def test_resume_is_idempotent(self, tmp_path):
        vault = _make_vault(tmp_path, 12)
        parent_tasks = self._parent_tasks(vault)
        target = tmp_path / "resume"
        target.mkdir()
        jobs = [_job("t", str(target), parent_tasks)]

        _build_sharded_direct(jobs, shard_rows_target=15)
        first = {s: _rows(os.path.join(str(target), f"{s}.parquet"))
                 for s in ("train", "val", "test")}
        # a fully-built head plans no work
        shard_jobs, merge_jobs = _plan_shards(jobs, shard_rows_target=15)
        assert shard_jobs == [] and merge_jobs == []
        # re-running changes nothing
        _build_sharded_direct(jobs, shard_rows_target=15)
        second = {s: _rows(os.path.join(str(target), f"{s}.parquet"))
                  for s in ("train", "val", "test")}
        assert first == second

    @pytest.mark.parametrize("monkeypatched_target", [15])
    def test_build_node_dataset_pool(self, tmp_path, monkeypatch, monkeypatched_target):
        # exercises the real spawn pool path end to end
        monkeypatch.setattr(B, "_SHARD_ROWS_TARGET", monkeypatched_target)
        vault = _make_vault(tmp_path, 12)
        target = tmp_path / "pool"
        target.mkdir()
        builder = DatasetBuilder(
            output_dir=str(tmp_path), max_subseq_len=_MAX_LEN, seed=42,
            output_format="parquet",
        )
        builder.build_node_dataset(
            [_job("t", str(target), self._parent_tasks(vault))], parallel=True
        )
        assert _rows(os.path.join(str(target), "train.parquet")) == 40
        assert _rows(os.path.join(str(target), "val.parquet")) == 10
        assert not _parts(str(target), "train")
