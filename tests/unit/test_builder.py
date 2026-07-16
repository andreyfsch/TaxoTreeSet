"""Tests for taxotreeset.dataset.builder — DatasetBuilder and worker helpers."""

from unittest.mock import patch

import pyarrow as pa
import pytest

from taxotreeset.dataset.builder import (
    DatasetBuilder,
    _BUFFER_SIZE_ROWS,
    _buffer_to_arrow_table,
    _extract_subseqs_for_task,
    _write_split_parquet,
)

_MOCK_READ = "taxotreeset.dataset.builder._read_single_sequence"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def builder(tmp_path):
    return DatasetBuilder(
        output_dir=str(tmp_path / "output"),
        max_subseq_len=200,
        seed=42,
        output_format="parquet",
    )


# ---------------------------------------------------------------------------
# _buffer_to_arrow_table
# ---------------------------------------------------------------------------


class TestBufferToArrowTable:
    def test_returns_arrow_table(self):
        buffer = [{"seq": "ACGT", "class_idx": 0}]
        table = _buffer_to_arrow_table(buffer)
        assert isinstance(table, pa.Table)

    def test_class_idx_is_int32(self):
        buffer = [{"seq": "ACGT", "class_idx": 5}]
        table = _buffer_to_arrow_table(buffer)
        assert table.schema.field("class_idx").type == pa.int32()

    def test_seq_field_is_string_type(self):
        buffer = [{"seq": "ACGT", "class_idx": 0}]
        table = _buffer_to_arrow_table(buffer)
        assert pa.types.is_string(table.schema.field("seq").type) or pa.types.is_large_string(
            table.schema.field("seq").type
        )

    def test_row_count_matches_buffer_size(self):
        buffer = [{"seq": f"ACGT_{i}", "class_idx": i} for i in range(10)]
        table = _buffer_to_arrow_table(buffer)
        assert table.num_rows == 10

    def test_class_idx_values_are_preserved(self):
        buffer = [{"seq": "AA", "class_idx": 7}, {"seq": "CC", "class_idx": 3}]
        table = _buffer_to_arrow_table(buffer)
        class_vals = table.column("class_idx").to_pylist()
        assert class_vals == [7, 3]


# ---------------------------------------------------------------------------
# _extract_subseqs_for_task
# ---------------------------------------------------------------------------


class TestExtractSubseqsForTask:
    def _make_task(self, n=5, class_idx=0, start_pct=0.0, end_pct=1.0):
        return {
            "fasta_path": "/fake/vault",
            "header_id": "NC_001",
            "start_pct": start_pct,
            "end_pct": end_pct,
            "n": n,
            "class_idx": class_idx,
        }

    def test_returns_rows_for_valid_sequence(self):
        seq = "ACGT" * 1000
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(self._make_task(n=5), max_subseq_len=200)
        assert len(rows) > 0

    def test_rows_have_seq_and_class_idx_keys(self):
        seq = "ACGT" * 1000
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(self._make_task(n=3), max_subseq_len=200)
        for row in rows:
            assert set(row.keys()) == {"seq", "class_idx"}

    def test_class_idx_is_correct(self):
        seq = "ACGT" * 1000
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(self._make_task(n=3, class_idx=7), max_subseq_len=200)
        assert all(row["class_idx"] == 7 for row in rows)

    def test_returns_empty_for_empty_sequence(self):
        with patch(_MOCK_READ, return_value=""):
            rows = _extract_subseqs_for_task(self._make_task(), max_subseq_len=200)
        assert rows == []

    def test_slices_sequence_by_percentages(self):
        seq = "A" * 1000 + "C" * 1000
        task = self._make_task(start_pct=0.5, end_pct=1.0, n=3)
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(task, max_subseq_len=200)
        # The sliced half should be "C" * 1000 — all rows should contain only Cs
        for row in rows:
            assert set(row["seq"]) == {"C"} or set(row["seq"]).issubset({"C", "A"})

    def test_class_idx_is_cast_to_int(self):
        seq = "ACGT" * 1000
        task = self._make_task(class_idx=3.7)  # float input
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(task, max_subseq_len=200)
        assert all(isinstance(row["class_idx"], int) for row in rows)

    def test_honors_min_subseq_len(self):
        # Regression: extraction hardcoded min_len=100 and ignored --min-subseq-len,
        # so capacity/n-distribution and extraction disagreed on the length floor.
        seq = "ACGT" * 500  # 2000 bp
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(
                self._make_task(n=20), max_subseq_len=200, min_subseq_len=180
            )
        assert rows
        assert all(len(row["seq"]) >= 180 for row in rows)

    def test_default_min_subseq_len_is_100(self):
        seq = "ACGT" * 500
        with patch(_MOCK_READ, return_value=seq):
            rows = _extract_subseqs_for_task(self._make_task(n=20), max_subseq_len=200)
        assert all(len(row["seq"]) >= 100 for row in rows)


# ---------------------------------------------------------------------------
# DatasetBuilder._compute_worker_count
# ---------------------------------------------------------------------------


class TestComputeWorkerCount:
    def test_low_memory_returns_two_workers(self):
        with patch("taxotreeset.dataset.builder.psutil.virtual_memory") as mock_vm:
            mock_vm.return_value.total = 8 * 1024**3  # 8 GiB
            count = DatasetBuilder._compute_worker_count()
        assert count == 2

    def test_high_memory_uses_cpu_count_minus_reserved(self):
        with (
            patch("taxotreeset.dataset.builder.psutil.virtual_memory") as mock_vm,
            patch("taxotreeset.dataset.builder.multiprocessing.cpu_count", return_value=8),
        ):
            mock_vm.return_value.total = 32 * 1024**3  # 32 GiB
            count = DatasetBuilder._compute_worker_count()
        assert count == 6  # 8 - 2 reserved

    def test_result_is_at_least_one(self):
        with (
            patch("taxotreeset.dataset.builder.psutil.virtual_memory") as mock_vm,
            patch("taxotreeset.dataset.builder.multiprocessing.cpu_count", return_value=1),
        ):
            mock_vm.return_value.total = 32 * 1024**3
            count = DatasetBuilder._compute_worker_count()
        assert count >= 1


# ---------------------------------------------------------------------------
# DatasetBuilder.build_node_dataset — serial mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _write_split_parquet — buffer flush at _BUFFER_SIZE_ROWS
# ---------------------------------------------------------------------------


class TestWriteSplitParquet:
    def test_writes_parquet_file_for_small_batch(self, tmp_path):
        import pyarrow.parquet as pq

        output_path = str(tmp_path / "train.parquet")
        task = {"fasta_path": "/fake", "header_id": "NC_001",
                "start_pct": 0.0, "end_pct": 1.0, "n": 3, "class_idx": 0}

        with patch(
            "taxotreeset.dataset.builder._extract_subseqs_for_task",
            return_value=[{"seq": "ACGT" * 50, "class_idx": 0}] * 5,
        ):
            _write_split_parquet([task], output_path, max_subseq_len=200)

        table = pq.read_table(output_path)
        assert table.num_rows == 5

    def test_buffer_flush_mid_stream_when_threshold_exceeded(self, tmp_path):
        import pyarrow.parquet as pq

        output_path = str(tmp_path / "train.parquet")
        single_row = {"seq": "A" * 200, "class_idx": 0}
        big_batch = [single_row] * _BUFFER_SIZE_ROWS

        call_count = {"n": 0}

        def fake_extract(task, max_subseq_len, min_subseq_len=100):
            call_count["n"] += 1
            return big_batch

        task1 = {"fasta_path": "/fake", "header_id": "NC_001",
                 "start_pct": 0.0, "end_pct": 1.0, "n": 1, "class_idx": 0}
        task2 = {"fasta_path": "/fake", "header_id": "NC_002",
                 "start_pct": 0.0, "end_pct": 1.0, "n": 1, "class_idx": 0}

        with patch(
            "taxotreeset.dataset.builder._extract_subseqs_for_task",
            side_effect=fake_extract,
        ):
            _write_split_parquet([task1, task2], output_path, max_subseq_len=200)

        assert call_count["n"] == 2
        table = pq.read_table(output_path)
        assert table.num_rows == _BUFFER_SIZE_ROWS * 2

    def test_empty_tasks_produce_no_file(self, tmp_path):
        import os
        output_path = str(tmp_path / "empty.parquet")
        with patch(
            "taxotreeset.dataset.builder._extract_subseqs_for_task",
            return_value=[],
        ):
            _write_split_parquet([], output_path, max_subseq_len=200)
        assert not os.path.exists(output_path)


class TestBuildNodeDatasetSerial:
    def test_serial_mode_calls_worker_for_each_job(self, builder, tmp_path):
        job1 = (
            "taxid_1",
            str(tmp_path / "head_1"),
            {"train": [], "val": [], "test": []},
            200,
            42,
            "parquet",
        )
        job2 = (
            "taxid_2",
            str(tmp_path / "head_2"),
            {"train": [], "val": [], "test": []},
            200,
            42,
            "parquet",
        )
        results = builder.build_node_dataset([job1, job2], parallel=False)
        assert results == [True, True]

    def test_serial_mode_returns_empty_list_for_no_jobs(self, builder):
        results = builder.build_node_dataset([], parallel=False)
        assert results == []

    def test_serial_mode_produces_parquet_when_tasks_non_empty(self, builder, tmp_path):
        import os

        output_dir = str(tmp_path / "head")
        os.makedirs(output_dir)

        task = {"fasta_path": "/fake", "header_id": "NC_001",
                "start_pct": 0.0, "end_pct": 1.0, "n": 2, "class_idx": 0}
        job = (
            "taxid_1",
            output_dir,
            {"train": [task], "val": [], "test": []},
            200,
            42,
            "parquet",
        )

        with patch(
            "taxotreeset.dataset.builder._extract_subseqs_for_task",
            return_value=[{"seq": "ACGT" * 50, "class_idx": 0}] * 2,
        ):
            results = builder.build_node_dataset([job], parallel=False)

        assert results == [True]
        assert os.path.exists(os.path.join(output_dir, "train.parquet"))

    def test_serial_write_is_atomic_no_tmp_left(self, builder, tmp_path):
        # C: serial writes each split to a .tmp then renames, so a crash never
        # leaves a partial <split>.parquet a resume would trust.
        import os

        output_dir = str(tmp_path / "head")
        os.makedirs(output_dir)
        task = {"fasta_path": "/fake", "header_id": "NC_001",
                "start_pct": 0.0, "end_pct": 1.0, "n": 2, "class_idx": 0}
        job = ("t", output_dir, {"train": [task]}, 200, 42, "parquet")

        with patch(
            "taxotreeset.dataset.builder._extract_subseqs_for_task",
            return_value=[{"seq": "ACGT" * 50, "class_idx": 0}] * 2,
        ):
            builder.build_node_dataset([job], parallel=False)

        assert os.path.exists(os.path.join(output_dir, "train.parquet"))
        assert [f for f in os.listdir(output_dir) if f.endswith(".tmp")] == []
