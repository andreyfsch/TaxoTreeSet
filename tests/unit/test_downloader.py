"""Tests for taxotreeset.io.downloader — pure/testable methods of NCBIDownloader."""

import subprocess
from unittest.mock import MagicMock, patch

import lmdb
import pytest
import zlib

from taxotreeset.io.downloader import NCBIDownloader


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def downloader(tmp_path):
    """NCBIDownloader wired to a mock registry and a real tmp vault dir."""
    mock_registry = MagicMock()
    mock_registry.registry = {"accessions": {}}
    return NCBIDownloader(
        registry=mock_registry,
        vault_path=str(tmp_path / "vault"),
    )


# ---------------------------------------------------------------------------
# _collect_pending_accessions
# ---------------------------------------------------------------------------


class TestCollectPendingAccessions:
    def test_returns_not_downloaded_not_deferred(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False, "download_deferred": False},
            }
        }
        assert downloader._collect_pending_accessions() == ["ACC_A"]

    def test_excludes_already_downloaded(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": True, "download_deferred": False},
            }
        }
        assert downloader._collect_pending_accessions() == []

    def test_excludes_deferred_accessions(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False, "download_deferred": True},
            }
        }
        assert downloader._collect_pending_accessions() == []

    def test_mixed_accessions_returns_only_eligible(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False, "download_deferred": False},
                "ACC_B": {"downloaded": True, "download_deferred": False},
                "ACC_C": {"downloaded": False, "download_deferred": True},
                "ACC_D": {"downloaded": True, "download_deferred": True},
            }
        }
        result = downloader._collect_pending_accessions()
        assert result == ["ACC_A"]

    def test_empty_registry_returns_empty_list(self, downloader):
        downloader.registry.registry = {"accessions": {}}
        assert downloader._collect_pending_accessions() == []


# ---------------------------------------------------------------------------
# _split_into_chunks
# ---------------------------------------------------------------------------


class TestSplitIntoChunks:
    def test_exact_multiple_splits_evenly(self, downloader):
        downloader.chunk_size = 3
        result = downloader._split_into_chunks(["A", "B", "C", "D", "E", "F"])
        assert result == [["A", "B", "C"], ["D", "E", "F"]]

    def test_trailing_chunk_is_smaller(self, downloader):
        downloader.chunk_size = 3
        result = downloader._split_into_chunks(["A", "B", "C", "D", "E"])
        assert result == [["A", "B", "C"], ["D", "E"]]

    def test_empty_list_returns_empty_list(self, downloader):
        result = downloader._split_into_chunks([])
        assert result == []

    def test_fewer_than_chunk_size_returns_single_chunk(self, downloader):
        downloader.chunk_size = 10
        result = downloader._split_into_chunks(["A", "B"])
        assert result == [["A", "B"]]

    def test_chunk_size_of_one_returns_singleton_chunks(self, downloader):
        downloader.chunk_size = 1
        result = downloader._split_into_chunks(["X", "Y", "Z"])
        assert result == [["X"], ["Y"], ["Z"]]


# ---------------------------------------------------------------------------
# _update_registry_for_batch
# ---------------------------------------------------------------------------


class TestUpdateRegistryForBatch:
    def test_marks_downloaded_and_sets_local_path(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False, "local_path": None},
            }
        }
        batch_results = {"ACC_A": [{"id": "NC_001", "name": "Seq one"}]}
        downloader._update_registry_for_batch(["ACC_A"], batch_results, ["ACC_A"])
        entry = downloader.registry.registry["accessions"]["ACC_A"]
        assert entry["downloaded"] is True
        assert entry["local_path"] == downloader.lmdb_path

    def test_sets_headers_from_batch_results(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False},
            }
        }
        headers = [{"id": "NC_001", "name": "Genome"}, {"id": "NC_002", "name": "Plasmid"}]
        downloader._update_registry_for_batch(["ACC_A"], {"ACC_A": headers}, ["ACC_A"])
        entry = downloader.registry.registry["accessions"]["ACC_A"]
        assert entry["headers"] == headers

    def test_transient_miss_not_in_attempted_left_untouched(self, downloader):
        # ACC_B was requested but the CLI never returned it (not attempted):
        # a transient/whole-batch miss must not count against it.
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False, "local_path": None},
                "ACC_B": {"downloaded": False, "local_path": None},
            }
        }
        batch_results = {"ACC_A": [{"id": "NC_001", "name": "Seq"}]}
        downloader._update_registry_for_batch(
            ["ACC_A", "ACC_B"], batch_results, ["ACC_A"]
        )
        entry_b = downloader.registry.registry["accessions"]["ACC_B"]
        assert entry_b["downloaded"] is False
        assert "download_attempts" not in entry_b  # not attempted -> untouched

    def test_empty_batch_results_leaves_all_untouched(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": False},
            }
        }
        downloader._update_registry_for_batch(["ACC_A"], {}, [])
        assert downloader.registry.registry["accessions"]["ACC_A"]["downloaded"] is False

    def test_attempted_but_failed_bumps_retry_counter(self, downloader):
        downloader.registry.registry = {
            "accessions": {"ACC_A": {"downloaded": False}}
        }
        downloader._update_registry_for_batch(["ACC_A"], {}, ["ACC_A"])
        entry = downloader.registry.registry["accessions"]["ACC_A"]
        assert entry["downloaded"] is False
        assert entry["download_attempts"] == 1

    def test_success_clears_retry_counter(self, downloader):
        downloader.registry.registry = {
            "accessions": {"ACC_A": {"downloaded": False, "download_attempts": 2}}
        }
        downloader._update_registry_for_batch(
            ["ACC_A"], {"ACC_A": [{"id": "NC_001", "name": "S"}]}, ["ACC_A"]
        )
        entry = downloader.registry.registry["accessions"]["ACC_A"]
        assert entry["downloaded"] is True
        assert "download_attempts" not in entry

    def test_exhausted_attempts_excluded_from_pending(self, downloader):
        # After _MAX_DOWNLOAD_ATTEMPTS ingest failures the accession is given up
        # on: _collect_pending_accessions no longer offers it for re-download.
        cap = NCBIDownloader._MAX_DOWNLOAD_ATTEMPTS
        downloader.registry.registry = {
            "accessions": {"ACC_A": {"downloaded": False}}
        }
        for _ in range(cap):
            downloader._update_registry_for_batch(["ACC_A"], {}, ["ACC_A"])
        entry = downloader.registry.registry["accessions"]["ACC_A"]
        assert entry["download_attempts"] == cap
        assert downloader._collect_pending_accessions() == []


# ---------------------------------------------------------------------------
# _find_fasta_in_directory
# ---------------------------------------------------------------------------


class TestFindFastaInDirectory:
    def test_finds_fna_file(self, tmp_path):
        (tmp_path / "genome.fna").write_text(">seq\nACGT\n")
        result = NCBIDownloader._find_fasta_in_directory(str(tmp_path))
        assert result == str(tmp_path / "genome.fna")

    def test_finds_fasta_extension(self, tmp_path):
        (tmp_path / "genome.fasta").write_text(">seq\nACGT\n")
        result = NCBIDownloader._find_fasta_in_directory(str(tmp_path))
        assert result == str(tmp_path / "genome.fasta")

    def test_finds_fa_extension(self, tmp_path):
        (tmp_path / "genome.fa").write_text(">seq\nACGT\n")
        result = NCBIDownloader._find_fasta_in_directory(str(tmp_path))
        assert result == str(tmp_path / "genome.fa")

    def test_returns_none_when_no_fasta_present(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a fasta")
        result = NCBIDownloader._find_fasta_in_directory(str(tmp_path))
        assert result is None

    def test_returns_none_for_empty_directory(self, tmp_path):
        result = NCBIDownloader._find_fasta_in_directory(str(tmp_path))
        assert result is None


# ---------------------------------------------------------------------------
# _parse_fasta_file
# ---------------------------------------------------------------------------


class TestParseFastaFile:
    def test_single_record(self, tmp_path):
        fasta = tmp_path / "genome.fna"
        fasta.write_text(">NC_001 Some description\nACGT\nACGT\n")
        seqs, headers = NCBIDownloader._parse_fasta_file(str(fasta))
        assert seqs == {"NC_001": "ACGTACGT"}
        assert headers == [{"id": "NC_001", "name": "Some description"}]

    def test_multiple_records(self, tmp_path):
        fasta = tmp_path / "genome.fna"
        fasta.write_text(">SEQ_A Record A\nAAAA\n>SEQ_B Record B\nCCCC\n")
        seqs, headers = NCBIDownloader._parse_fasta_file(str(fasta))
        assert seqs == {"SEQ_A": "AAAA", "SEQ_B": "CCCC"}
        assert len(headers) == 2
        assert headers[0] == {"id": "SEQ_A", "name": "Record A"}
        assert headers[1] == {"id": "SEQ_B", "name": "Record B"}

    def test_header_without_description_uses_id_as_name(self, tmp_path):
        fasta = tmp_path / "genome.fna"
        fasta.write_text(">NC_001\nACGT\n")
        seqs, headers = NCBIDownloader._parse_fasta_file(str(fasta))
        assert headers == [{"id": "NC_001", "name": "NC_001"}]

    def test_sequence_lines_concatenated(self, tmp_path):
        fasta = tmp_path / "genome.fna"
        fasta.write_text(">SEQ_A desc\nACGT\nACGT\nACGT\n")
        seqs, _ = NCBIDownloader._parse_fasta_file(str(fasta))
        assert seqs == {"SEQ_A": "ACGTACGTACGT"}

    def test_blank_lines_in_sequence_are_skipped(self, tmp_path):
        fasta = tmp_path / "genome.fna"
        fasta.write_text(">SEQ_A desc\nACGT\n\nACGT\n")
        seqs, _ = NCBIDownloader._parse_fasta_file(str(fasta))
        assert seqs == {"SEQ_A": "ACGTACGT"}

    def test_returns_empty_for_empty_file(self, tmp_path):
        fasta = tmp_path / "empty.fna"
        fasta.write_text("")
        seqs, headers = NCBIDownloader._parse_fasta_file(str(fasta))
        assert seqs == {}
        assert headers == []

    def test_duplicate_record_id_keeps_first_and_stays_consistent(self, tmp_path):
        # A repeated id within one FASTA is malformed; keep the first occurrence
        # so `sequences` and `headers_metadata` stay one-to-one (LMDB key = id).
        fasta = tmp_path / "dup.fna"
        fasta.write_text(">NC_001 first\nAAAA\n>NC_001 second\nCCCC\n>NC_002 c\nGGGG\n")
        seqs, headers = NCBIDownloader._parse_fasta_file(str(fasta))
        assert seqs == {"NC_001": "AAAA", "NC_002": "GGGG"}  # first NC_001 kept
        assert [h["id"] for h in headers] == ["NC_001", "NC_002"]  # one per id


# ---------------------------------------------------------------------------
# reconcile_with_vault
# ---------------------------------------------------------------------------


class TestReconcileWithVault:
    def _make_lmdb(self, tmp_path, keys):
        """Create a minimal LMDB with the given keys (empty values)."""
        env = lmdb.open(
            str(tmp_path / "sequences.lmdb"),
            map_size=1024 * 1024,
            max_dbs=0,
        )
        with env.begin(write=True) as txn:
            for key in keys:
                txn.put(key.encode("utf-8"), zlib.compress(b"ACGT"))
        env.close()

    def test_returns_zero_when_vault_missing(self, downloader):
        downloader.registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": True, "headers": [{"id": "NC_001", "name": "x"}]},
            }
        }
        result = downloader.reconcile_with_vault()
        assert result == 0

    def test_returns_zero_when_all_headers_present(self, tmp_path):
        self._make_lmdb(tmp_path, ["NC_001"])
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "ACC_A": {
                    "downloaded": True,
                    "headers": [{"id": "NC_001", "name": "x"}],
                }
            }
        }
        dl = NCBIDownloader(
            registry=mock_registry,
            vault_path=str(tmp_path),
        )
        result = dl.reconcile_with_vault()
        assert result == 0
        assert mock_registry.registry["accessions"]["ACC_A"]["downloaded"] is True

    def test_resets_accession_with_missing_header(self, tmp_path):
        self._make_lmdb(tmp_path, [])  # vault exists but is empty
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "ACC_A": {
                    "downloaded": True,
                    "headers": [{"id": "NC_MISSING", "name": "x"}],
                    "local_path": "/old/path",
                }
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))
        result = dl.reconcile_with_vault()
        assert result == 1
        entry = mock_registry.registry["accessions"]["ACC_A"]
        assert entry["downloaded"] is False
        assert entry["local_path"] is None
        assert "headers" not in entry

    def test_saves_registry_when_accession_reset(self, tmp_path):
        self._make_lmdb(tmp_path, [])
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "ACC_A": {
                    "downloaded": True,
                    "headers": [{"id": "NC_MISSING", "name": "x"}],
                }
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))
        dl.reconcile_with_vault()
        mock_registry.save.assert_called_once()

    def test_skips_not_downloaded_accessions(self, tmp_path):
        self._make_lmdb(tmp_path, [])
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "ACC_A": {
                    "downloaded": False,
                    "headers": [{"id": "NC_001", "name": "x"}],
                }
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))
        result = dl.reconcile_with_vault()
        assert result == 0

    def test_skips_accessions_without_headers(self, tmp_path):
        self._make_lmdb(tmp_path, [])
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "ACC_A": {"downloaded": True, "headers": []},
                "ACC_B": {"downloaded": True},  # no headers key
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))
        result = dl.reconcile_with_vault()
        assert result == 0


# ---------------------------------------------------------------------------
# _invoke_ncbi_datasets_cli
# ---------------------------------------------------------------------------


class TestInvokeNcbiDatasetsCli:
    def test_returns_true_on_success(self, tmp_path):
        archive_path = str(tmp_path / "batch.zip")
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))

        (tmp_path / "batch.zip").write_bytes(b"PK\x03\x04fakecontent")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            result = dl._invoke_ncbi_datasets_cli(["GCF_001"], archive_path)

        assert result is True

    def test_returns_false_on_subprocess_error(self, tmp_path):
        archive_path = str(tmp_path / "batch.zip")
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "datasets", stderr="error"),
        ):
            result = dl._invoke_ncbi_datasets_cli(["GCF_001"], archive_path)

        assert result is False

    def test_returns_false_when_archive_not_created(self, tmp_path):
        archive_path = str(tmp_path / "nonexistent.zip")
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))

        with patch("subprocess.run"):
            result = dl._invoke_ncbi_datasets_cli(["GCF_001"], archive_path)

        assert result is False

    def test_returns_false_when_archive_is_empty(self, tmp_path):
        archive_path = str(tmp_path / "empty.zip")
        (tmp_path / "empty.zip").write_bytes(b"")
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))

        with patch("subprocess.run"):
            result = dl._invoke_ncbi_datasets_cli(["GCF_001"], archive_path)

        assert result is False

    def test_command_includes_all_accessions(self, tmp_path):
        archive_path = str(tmp_path / "batch.zip")
        (tmp_path / "batch.zip").write_bytes(b"PK\x03\x04fakecontent")
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return MagicMock()

        with patch("subprocess.run", side_effect=fake_run):
            dl._invoke_ncbi_datasets_cli(["GCF_001", "GCF_002"], archive_path)

        assert "GCF_001" in captured["cmd"]
        assert "GCF_002" in captured["cmd"]


# ---------------------------------------------------------------------------
# _reset_state_if_lmdb_missing
# ---------------------------------------------------------------------------


class TestResetStateIfLmdbMissing:
    def test_resets_downloaded_flags_when_lmdb_is_missing(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "GCF_A": {"downloaded": True, "local_path": "/old", "headers": ["h1"]},
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))
        dl._reset_state_if_lmdb_missing()
        assert mock_registry.registry["accessions"]["GCF_A"]["downloaded"] is False
        assert mock_registry.save.called

    def test_no_reset_when_none_are_downloaded(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "GCF_A": {"downloaded": False},
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))
        dl._reset_state_if_lmdb_missing()
        assert not mock_registry.save.called

    def test_no_reset_when_lmdb_exists_and_valid_size(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        lmdb_dir = vault / "sequences.lmdb"
        lmdb_dir.mkdir()
        data_file = lmdb_dir / "data.mdb"
        data_file.write_bytes(b"X" * 10000)

        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {"GCF_A": {"downloaded": True}},
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(vault))
        dl._reset_state_if_lmdb_missing()
        assert not mock_registry.save.called


# ---------------------------------------------------------------------------
# download_all_pending — main entry point
# ---------------------------------------------------------------------------


class TestDownloadAllPending:
    def test_returns_immediately_when_no_pending(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {"GCF_A": {"downloaded": True, "download_deferred": False}},
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))

        with patch.object(dl, "_reset_state_if_lmdb_missing"):
            dl.download_all_pending()

        mock_registry.save.assert_not_called()

    def test_calls_process_chunks_when_pending_exist(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "GCF_A": {"downloaded": False, "download_deferred": False},
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))

        with (
            patch.object(dl, "_reset_state_if_lmdb_missing"),
            patch.object(dl, "_process_chunks") as mock_chunks,
        ):
            dl.download_all_pending()

        mock_chunks.assert_called_once()

    def test_process_chunks_dispatches_to_download_batch(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {
            "accessions": {
                "GCF_A": {"downloaded": True, "download_deferred": False},
            }
        }
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))
        dl._env = MagicMock()

        with patch.object(dl, "download_batch", return_value=({}, [])) as mock_batch:
            dl._process_chunks(
                chunks=[["GCF_A"]],
                bar_total=1,
                bar_initial=0,
            )

        mock_batch.assert_called_once_with(["GCF_A"])


# ---------------------------------------------------------------------------
# download_batch — core batch download logic
# ---------------------------------------------------------------------------


class TestDownloadBatch:
    def _make_downloader(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))
        dl._env = MagicMock()
        return dl

    def test_returns_empty_when_cli_fails(self, tmp_path):
        dl = self._make_downloader(tmp_path)
        with patch.object(dl, "_invoke_ncbi_datasets_cli", return_value=False):
            result, attempted = dl.download_batch(["GCF_001"])
        assert result == {}
        assert attempted == []  # CLI failed -> nothing attempted (no retry count)

    def test_returns_empty_when_extract_fails(self, tmp_path):
        dl = self._make_downloader(tmp_path)
        with (
            patch.object(dl, "_invoke_ncbi_datasets_cli", return_value=True),
            patch.object(dl, "_extract_assembly_archive", return_value=None),
        ):
            result, attempted = dl.download_batch(["GCF_001"])
        assert result == {}
        assert attempted == []

    def test_returns_headers_when_ingest_succeeds(self, tmp_path):
        dl = self._make_downloader(tmp_path)
        fake_headers = [{"id": "NC_001", "name": "NC_001.1 genome"}]
        with (
            patch.object(dl, "_invoke_ncbi_datasets_cli", return_value=True),
            patch.object(dl, "_extract_assembly_archive", return_value="/fake/extracted"),
            patch.object(dl, "_ingest_accession_fasta", return_value=fake_headers),
        ):
            result, attempted = dl.download_batch(["GCF_001"])
        assert "GCF_001" in result
        assert result["GCF_001"] == fake_headers
        assert attempted == ["GCF_001"]

    def test_omits_accession_on_ingest_failure(self, tmp_path):
        # None from ingestion = genuine failure -> omitted so it stays pending.
        dl = self._make_downloader(tmp_path)
        with (
            patch.object(dl, "_invoke_ncbi_datasets_cli", return_value=True),
            patch.object(dl, "_extract_assembly_archive", return_value="/fake/extracted"),
            patch.object(dl, "_ingest_accession_fasta", return_value=None),
        ):
            result, attempted = dl.download_batch(["GCF_001"])
        assert "GCF_001" not in result
        assert attempted == ["GCF_001"]  # attempted (CLI ok) but failed to ingest

    def test_includes_accession_when_all_sequences_filtered(self, tmp_path):
        # Empty list = processed but every sequence filtered -> recorded (with
        # empty headers) so the accession is marked downloaded, not retried.
        dl = self._make_downloader(tmp_path)
        with (
            patch.object(dl, "_invoke_ncbi_datasets_cli", return_value=True),
            patch.object(dl, "_extract_assembly_archive", return_value="/fake/extracted"),
            patch.object(dl, "_ingest_accession_fasta", return_value=[]),
        ):
            result, _ = dl.download_batch(["GCF_001"])
        assert result["GCF_001"] == []

    def test_ingest_exception_is_isolated(self, tmp_path):
        # A single accession raising during ingest must not abort the batch;
        # the others still ingest and the bad one is left out (stays pending).
        dl = self._make_downloader(tmp_path)

        def ingest(acc, _root):
            if acc == "GCF_BAD":
                raise ValueError("corrupt FASTA")
            return [{"id": "NC_001", "name": "ok"}]

        with (
            patch.object(dl, "_invoke_ncbi_datasets_cli", return_value=True),
            patch.object(dl, "_extract_assembly_archive", return_value="/fake"),
            patch.object(dl, "_ingest_accession_fasta", side_effect=ingest),
        ):
            result, attempted = dl.download_batch(["GCF_BAD", "GCF_GOOD"])
        assert "GCF_GOOD" in result
        assert "GCF_BAD" not in result
        assert attempted == ["GCF_BAD", "GCF_GOOD"]


# ---------------------------------------------------------------------------
# _extract_assembly_archive
# ---------------------------------------------------------------------------


class TestExtractAssemblyArchive:
    def _make_downloader(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        return NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))

    def test_returns_dataset_root_from_valid_archive(self, tmp_path):
        import zipfile as zf

        archive_path = str(tmp_path / "batch.zip")
        ncbi_data = tmp_path / "ncbi_dataset" / "data"
        ncbi_data.mkdir(parents=True)

        with zf.ZipFile(archive_path, "w") as arc:
            arc.writestr("ncbi_dataset/data/.keep", "")

        dl = self._make_downloader(tmp_path)
        result = dl._extract_assembly_archive(archive_path, str(tmp_path / "extract"))
        assert result is not None
        assert "ncbi_dataset/data" in result or "ncbi_dataset" in result

    def test_returns_none_when_expected_layout_absent(self, tmp_path):
        import zipfile as zf

        archive_path = str(tmp_path / "weird.zip")
        with zf.ZipFile(archive_path, "w") as arc:
            arc.writestr("unexpected_layout.txt", "nothing here")

        dl = self._make_downloader(tmp_path)
        result = dl._extract_assembly_archive(archive_path, str(tmp_path / "extract"))
        assert result is None


# ---------------------------------------------------------------------------
# _ingest_accession_fasta
# ---------------------------------------------------------------------------


class TestIngestAccessionFasta:
    def _make_downloader(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))
        dl._env = MagicMock()
        return dl

    def test_returns_none_when_accession_dir_absent(self, tmp_path):
        dl = self._make_downloader(tmp_path)
        result = dl._ingest_accession_fasta("GCF_001", str(tmp_path / "data"))
        assert result is None

    def test_returns_none_when_no_fasta_found(self, tmp_path):
        dataset_root = tmp_path / "data"
        acc_dir = dataset_root / "GCF_001"
        acc_dir.mkdir(parents=True)
        (acc_dir / "not_a_fasta.txt").write_text("data", encoding="utf-8")

        dl = self._make_downloader(tmp_path)
        result = dl._ingest_accession_fasta("GCF_001", str(dataset_root))
        assert result is None

    def test_returns_headers_when_fasta_parsed_and_persisted(self, tmp_path):
        # Covers lines 482-487: parse_fasta returns sequences → persist called → headers returned
        dataset_root = tmp_path / "data"
        acc_dir = dataset_root / "GCF_001"
        acc_dir.mkdir(parents=True)
        fasta = acc_dir / "genome.fna"
        fasta.write_text(">NC_001 SARS-CoV-2\nACGTACGT\n", encoding="utf-8")

        dl = self._make_downloader(tmp_path)
        # Patch _persist_sequences_to_lmdb so we don't need a real LMDB env
        with patch.object(dl, "_persist_sequences_to_lmdb"):
            result = dl._ingest_accession_fasta("GCF_001", str(dataset_root))

        assert len(result) == 1
        assert result[0]["id"] == "NC_001"

    def test_returns_none_when_fasta_has_no_sequences(self, tmp_path):
        # FASTA file found but _parse_fasta_file returns no sequences -> treated
        # as a failed/empty download (None) so it stays pending and is retried.
        dataset_root = tmp_path / "data"
        acc_dir = dataset_root / "GCF_001"
        acc_dir.mkdir(parents=True)
        fasta = acc_dir / "genome.fna"
        # File exists but has no sequence lines (only comments/blank lines)
        fasta.write_text("# not a real fasta\n\n", encoding="utf-8")

        dl = self._make_downloader(tmp_path)
        result = dl._ingest_accession_fasta("GCF_001", str(dataset_root))
        assert result is None


# ---------------------------------------------------------------------------
# _persist_sequences_to_lmdb — raises when env is None
# ---------------------------------------------------------------------------


class TestPersistSequencesToLmdb:
    def test_raises_when_env_is_not_open(self, tmp_path):
        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path / "vault"))
        with pytest.raises(RuntimeError, match="LMDB environment is not open"):
            dl._persist_sequences_to_lmdb({"NC_001": "ACGT"})

    def test_writes_compressed_sequences_to_lmdb(self, tmp_path):
        # Covers lines 572-575: real LMDB env → txn.put compressed data
        lmdb_dir = tmp_path / "sequences.lmdb"
        lmdb_dir.mkdir()
        env = lmdb.open(str(lmdb_dir), map_size=1024 * 1024)

        mock_registry = MagicMock()
        mock_registry.registry = {"accessions": {}}
        dl = NCBIDownloader(registry=mock_registry, vault_path=str(tmp_path))
        dl._env = env

        dl._persist_sequences_to_lmdb({"NC_TEST": "ACGTACGT"})

        with env.begin() as txn:
            raw = txn.get(b"NC_TEST")
        env.close()

        assert raw is not None
        assert zlib.decompress(raw) == b"ACGTACGT"


# ---------------------------------------------------------------------------
# molecule-type (plasmid) filter
# ---------------------------------------------------------------------------


class TestIsExcludedMolecule:
    def test_true_for_plasmid_defline(self):
        assert NCBIDownloader._is_excluded_molecule(
            "Escherichia coli strain K12 plasmid pXYZ, complete sequence"
        )

    def test_true_for_megaplasmid(self):
        assert NCBIDownloader._is_excluded_molecule(
            "Sinorhizobium meliloti megaplasmid pSymA, complete sequence"
        )

    def test_case_insensitive(self):
        assert NCBIDownloader._is_excluded_molecule("X PLASMID p1")

    def test_false_for_chromosome_and_genome(self):
        assert not NCBIDownloader._is_excluded_molecule(
            "Escherichia coli str. K-12 substr. MG1655, complete genome"
        )
        assert not NCBIDownloader._is_excluded_molecule(
            "Homo sapiens chromosome 1, GRCh38"
        )


class TestDropExcludedMolecules:
    def test_drops_plasmid_from_both_maps(self, downloader):
        sequences = {"CHR": "ACGT", "PLA": "TTTT"}
        headers = [
            {"id": "CHR", "name": "E. coli, complete genome"},
            {"id": "PLA", "name": "E. coli plasmid pX, complete sequence"},
        ]
        seqs, hdrs = downloader._drop_excluded_molecules("GCF_X", sequences, headers)
        assert seqs == {"CHR": "ACGT"}
        assert hdrs == [{"id": "CHR", "name": "E. coli, complete genome"}]

    def test_keeps_all_when_no_plasmid(self, downloader):
        sequences = {"CHR": "ACGT"}
        headers = [{"id": "CHR", "name": "E. coli, complete genome"}]
        seqs, hdrs = downloader._drop_excluded_molecules("GCF_X", sequences, headers)
        assert seqs == sequences
        assert hdrs == headers


def _write_accession_fasta(root, accession, content):
    """Create ``root/<accession>/genome.fna`` and return ``str(root)``."""
    acc_dir = root / accession
    acc_dir.mkdir(parents=True)
    (acc_dir / "genome.fna").write_text(content)
    return str(root)


class TestIngestAccessionFastaFilter:
    _CHR = ">NC_1 E. coli, complete genome\nACGTACGT\n"
    _PLA = ">NC_2 E. coli plasmid pX, complete sequence\nTTTTCCCC\n"

    def test_missing_directory_returns_none(self, downloader):
        assert downloader._ingest_accession_fasta("GCF_X", "/nonexistent") is None

    def test_no_filter_keeps_all(self, tmp_path, downloader):
        root = _write_accession_fasta(tmp_path / "data", "GCF_A", self._CHR + self._PLA)
        with patch.object(downloader, "_persist_sequences_to_lmdb") as persist:
            headers = downloader._ingest_accession_fasta("GCF_A", root)
        assert {h["id"] for h in headers} == {"NC_1", "NC_2"}
        assert set(persist.call_args.args[0]) == {"NC_1", "NC_2"}

    def test_filter_drops_plasmid(self, tmp_path, downloader):
        downloader.exclude_plasmids = True
        root = _write_accession_fasta(tmp_path / "data", "GCF_B", self._CHR + self._PLA)
        with patch.object(downloader, "_persist_sequences_to_lmdb") as persist:
            headers = downloader._ingest_accession_fasta("GCF_B", root)
        assert [h["id"] for h in headers] == ["NC_1"]
        assert set(persist.call_args.args[0]) == {"NC_1"}

    def test_all_plasmid_returns_empty_list_not_none(self, tmp_path, downloader):
        downloader.exclude_plasmids = True
        root = _write_accession_fasta(tmp_path / "data", "GCF_C", self._PLA)
        with patch.object(downloader, "_persist_sequences_to_lmdb") as persist:
            headers = downloader._ingest_accession_fasta("GCF_C", root)
        # Empty list (not None) -> processed, marked downloaded, not retried.
        assert headers == []
        assert persist.call_args.args[0] == {}
