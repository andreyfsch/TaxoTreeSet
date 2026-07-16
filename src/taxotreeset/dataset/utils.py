"""LMDB I/O utilities for the dataset generation pipeline.

This module provides low-level helpers for reading compressed genome
sequences from an LMDB vault. These helpers are designed to be used by
the parallel worker processes spawned during the generation phase: each
worker reads from the same LMDB vault produced by the discovery phase
and needs lightweight, fork-safe access to the sequences it indexes.

Key design considerations:

1. **Fork-safe LMDB caching**. The lmdb.Environment object is not safe
   to share across forked processes because it embeds an mmap region
   and file descriptors that were established in the parent. To avoid
   subtle corruption when workers are forked (rather than spawned),
   this module detects PID changes and discards the cached environment
   to force a clean reopen in the child process.

2. **Read-only access from workers**. The vault is written exclusively
   by the downloader during the discovery phase. Workers open the
   environment with readonly=True and lock=False, allowing concurrent
   reads from many processes without contention.

3. **zlib compression at the value layer**. Sequences are stored
   compressed to reduce vault size on disk. Decompression happens
   transparently on read; the cost is a few percent of CPU per read,
   which is negligible compared to the bigger work the workers do.

Typical usage::

    from taxotreeset.dataset.utils import _read_single_sequence

    sequence = _read_single_sequence(
        lmdb_path="data/vault/sequences.lmdb",
        header_id="NC_001416.1",
    )
    if sequence:
        process(sequence)

Module-private helpers are prefixed with a single underscore because
they are not part of a stable public API; callers within the project
import them directly but external code should not rely on their names.
"""

import logging
import os
import threading
import zlib

import lmdb

logger = logging.getLogger("TaxoTreeSet.Dataset.Utils")

_LMDB_ENV_CACHE: dict[str, lmdb.Environment] = {}
_LMDB_CACHE_PID: int | None = None
_LMDB_CACHE_LOCK = threading.Lock()


def _pool_worker_initializer() -> None:
    """Reset the LMDB cache when a worker process starts.

    This function is intended to be passed as the ``initializer``
    argument to ``multiprocessing.Pool``. It runs exactly once per
    worker, immediately after the worker process is created.

    By clearing the cache inherited from the parent, we guarantee that
    each worker opens its own LMDB handle on first access rather than
    reusing the parent's mmap region. This is essential for forked
    workers; spawned workers start with empty module state anyway and
    are unaffected.
    """
    from taxotreeset.dataset import utils

    utils._LMDB_ENV_CACHE = {}
    utils._LMDB_CACHE_PID = None


def _get_lmdb_env(lmdb_path: str) -> lmdb.Environment:
    """Return a process-local LMDB environment for the given vault path.

    Manages a per-process cache of LMDB environments keyed by path.
    When the cache was populated by a different process (detected via
    PID comparison), it is discarded and the environment is reopened
    fresh in the current process. This prevents corruption from
    sharing mmap regions across forked workers.

    The environment is opened in read-only mode without locking, which
    allows concurrent reads from many worker processes against the same
    vault without contention.

    Args:
        lmdb_path: Filesystem path to the LMDB environment directory.

    Returns:
        An open LMDB environment ready for read transactions.

    Raises:
        FileNotFoundError: If the LMDB directory does not exist at the
            given path.
    """
    global _LMDB_ENV_CACHE, _LMDB_CACHE_PID
    current_pid = os.getpid()

    with _LMDB_CACHE_LOCK:
        if _LMDB_CACHE_PID != current_pid:
            if _LMDB_CACHE_PID is not None:
                logger.debug(
                    f"[LMDB-FORK] PID changed {_LMDB_CACHE_PID} -> "
                    f"{current_pid}; discarding inherited cache."
                )
            _LMDB_ENV_CACHE = {}
            _LMDB_CACHE_PID = current_pid

        env = _LMDB_ENV_CACHE.get(lmdb_path)
        if env is None:
            if not os.path.exists(lmdb_path):
                logger.error(f"LMDB vault path does not exist: {lmdb_path}")
                raise FileNotFoundError(lmdb_path)
            env = lmdb.open(
                lmdb_path,
                readonly=True,
                lock=False,
                max_dbs=0,
                readahead=False,
            )
            logger.debug(f"[LMDB-OPEN] PID={current_pid} path={lmdb_path}")
            _LMDB_ENV_CACHE[lmdb_path] = env

    return env


def _read_single_sequence(lmdb_path: str, header_id: str) -> str:
    """Read and decompress a single sequence from the LMDB vault.

    Looks up the compressed value associated with ``header_id`` in the
    LMDB environment at ``lmdb_path``, decompresses it with zlib, and
    decodes the result as UTF-8 text.

    The function returns an empty string in all error cases (missing
    vault, missing key, I/O failure, decompression failure) rather
    than raising. This permits callers to safely process possibly
    incomplete data without wrapping every call in a try/except.

    Args:
        lmdb_path: Filesystem path to the LMDB vault directory.
        header_id: Sequence header identifier used as the LMDB key.

    Returns:
        The decompressed sequence as a string, or an empty string on
        any failure.
    """
    try:
        env = _get_lmdb_env(lmdb_path)
    except FileNotFoundError:
        return ""

    try:
        with env.begin(write=False) as txn:
            compressed_seq = txn.get(header_id.encode("utf-8"))
    except lmdb.Error as exc:
        logger.error(f"LMDB I/O error fetching '{header_id}' from {lmdb_path}: {exc}")
        return ""

    if not compressed_seq:
        logger.warning(f"Header '{header_id}' not found in {lmdb_path}")
        return ""

    try:
        # Uppercase at the single read boundary so every consumer — capacity,
        # subseq extraction, separability, the tokenizer — sees canonical ACGT.
        # NCBI ships eukaryotic genomes soft-masked (lowercase acgt for repeats);
        # left as-is, those bases read as ambiguous (capacity) and as unexpected
        # tokens (training). The vault keeps the original bytes; only reads normalize.
        return zlib.decompress(compressed_seq).decode("utf-8").upper()
    except (zlib.error, UnicodeDecodeError) as exc:
        logger.error(f"Failed to decompress '{header_id}': {exc}")
        return ""


def _get_fasta_sequence_length(lmdb_path: str, header_id: str) -> int:
    """Return the length in base pairs of a sequence stored in the vault.

    Equivalent to ``len(_read_single_sequence(lmdb_path, header_id))``
    but expresses the intent more clearly at call sites where only the
    length is needed for capacity computations.

    Note that this still incurs the full decompression cost because
    zlib does not expose the uncompressed length without inflating the
    stream. For workloads that read the length many times for the same
    sequence, callers should cache the result.

    Args:
        lmdb_path: Filesystem path to the LMDB vault directory.
        header_id: Sequence header identifier used as the LMDB key.

    Returns:
        Length of the decompressed sequence in characters, or 0 if the
        sequence could not be read.
    """
    sequence = _read_single_sequence(lmdb_path, header_id)
    return len(sequence)
