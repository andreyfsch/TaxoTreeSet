"""Integration test for DatasetBuilder.build_node_dataset in parallel mode.

Uses the synthetic vault fixture (no network required). Constructs a real job
tuple referencing synthetic LMDB sequences and verifies that the spawn-based
worker pool produces Parquet output files.
"""

import os

import pyarrow.parquet as pq

from taxotreeset.dataset.builder import DatasetBuilder


_MAX_SUBSEQ_LEN = 200
_SEED = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_tasks(lmdb_path: str, header_id: str, class_idx: int) -> list[dict]:
    return [
        {
            "fasta_path": lmdb_path,
            "header_id": header_id,
            "start_pct": 0.0,
            "end_pct": 1.0,
            "n": 3,
            "class_idx": class_idx,
        }
    ]


def _make_job(
    label: str,
    output_dir: str,
    lmdb_path: str,
    header_id: str,
) -> tuple:
    """Build a job tuple with one sequence leaf per split."""
    tasks = _make_tasks(lmdb_path, header_id, class_idx=0)
    parent_tasks = {"train": tasks, "val": tasks, "test": tasks}
    return (label, output_dir, parent_tasks, _MAX_SUBSEQ_LEN, _SEED, "parquet")


# ---------------------------------------------------------------------------
# parallel-mode integration
# ---------------------------------------------------------------------------


class TestParallelBuilderIntegration:
    def test_parallel_mode_produces_parquet_files(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "corona_output")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="11118",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root_output"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        results = builder.build_node_dataset([job], parallel=True)

        assert results == [True]

        for split in ("train", "val", "test"):
            parquet_path = os.path.join(output_dir, f"{split}.parquet")
            assert os.path.exists(parquet_path), f"Missing {split}.parquet"

    def test_parallel_output_has_expected_columns(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "col_check")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="2697049",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        builder.build_node_dataset([job], parallel=True)

        table = pq.read_table(os.path.join(output_dir, "train.parquet"))
        assert "seq" in table.schema.names
        assert "class_idx" in table.schema.names

    def test_parallel_output_contains_rows(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "rows_check")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="2697049",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        builder.build_node_dataset([job], parallel=True)

        table = pq.read_table(os.path.join(output_dir, "train.parquet"))
        assert table.num_rows > 0

    def test_parallel_output_seq_lengths_within_bounds(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "len_check")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="2697049",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        builder.build_node_dataset([job], parallel=True)

        table = pq.read_table(os.path.join(output_dir, "train.parquet"))
        seqs = table.column("seq").to_pylist()
        for seq in seqs:
            assert len(seq) <= _MAX_SUBSEQ_LEN

    def test_parallel_class_idx_is_int(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "idx_check")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="2697049",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        builder.build_node_dataset([job], parallel=True)

        import pyarrow as pa
        table = pq.read_table(os.path.join(output_dir, "train.parquet"))
        assert table.schema.field("class_idx").type == pa.int32()

    def test_parallel_multiple_jobs_all_succeed(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        jobs = []
        for i, (header_id, taxid) in enumerate(
            [("NC_045512", "2697049"), ("NC_001846", "11234"), ("NC_001407", "10509")]
        ):
            output_dir = str(tmp_path / f"job_{i}")
            os.makedirs(output_dir, exist_ok=True)
            jobs.append(_make_job(taxid, output_dir, lmdb_path, header_id))

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        results = builder.build_node_dataset(jobs, parallel=True)

        assert all(r is True for r in results)
        for i in range(3):
            output_dir = str(tmp_path / f"job_{i}")
            assert os.path.exists(os.path.join(output_dir, "train.parquet"))


# ---------------------------------------------------------------------------
# serial mode (parallel=False) — exercises _write_split_parquet directly
# ---------------------------------------------------------------------------


class TestSerialBuilderIntegration:
    def test_serial_mode_produces_parquet_files(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "serial_output")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="11118",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )

        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root_output"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        results = builder.build_node_dataset([job], parallel=False)

        assert results == [True]
        for split in ("train", "val", "test"):
            parquet_path = os.path.join(output_dir, f"{split}.parquet")
            assert os.path.exists(parquet_path), f"Missing {split}.parquet"

    def test_serial_output_has_expected_columns(self, synthetic_env, tmp_path):
        lmdb_path = synthetic_env["lmdb_path"]
        output_dir = str(tmp_path / "serial_cols")
        os.makedirs(output_dir, exist_ok=True)

        job = _make_job(
            label="2697049",
            output_dir=output_dir,
            lmdb_path=lmdb_path,
            header_id="NC_045512",
        )
        builder = DatasetBuilder(
            output_dir=str(tmp_path / "root"),
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=_SEED,
            output_format="parquet",
        )
        builder.build_node_dataset([job], parallel=False)

        table = pq.read_table(os.path.join(output_dir, "train.parquet"))
        assert "seq" in table.schema.names
        assert "class_idx" in table.schema.names
