"""``_NodeCapacityKeys``: the packed-key set for one taxonomic node.

Holds a node's unique sliding-window keys during capacity computation: pure-ACGT
windows packed at 2 bits/base plus an exact set of IUPAC-ambiguous window
strings. The set is memory-backed (a deduplicated numpy array) until it grows
large enough to spill to prefix-bucketed files on disk, after which merges and
the final unique count run bucket-by-bucket with bounded peak memory.

Extracted from ``capacity.py`` (P7 Part C) and re-exported there. The sequence
readers (``_read_sequence_cached`` / ``_read_single_sequence``) are imported
lazily inside ``from_sequence_leaf`` so this module does not import ``capacity``
at load time (breaking the cycle) while tests that patch
``capacity._read_sequence_cached`` still take effect.
"""

import contextlib

from taxotreeset.core.generation._capacity._diskdedup import (
    _bucket_writer_paths,
    _count_unique_bucketed_on_disk,
    _flush_keys_to_buckets,
)
from taxotreeset.core.generation._capacity._encoding import _encode_windows_2bit


class _NodeCapacityKeys:
    """Accumulator of a node's unique capacity keys.

    A node's capacity is the number of unique fixed-length subsequences
    found across all sequence leaves beneath it. This class accumulates
    those subsequences as deduplicated keys so that a parent node can be
    resolved by merging its children's accumulators, instead of rescanning
    every descendant leaf from scratch.

    Pure-ACGT subsequences are stored as packed 2-bit keys; subsequences
    containing IUPAC ambiguity codes are kept as exact strings. The two
    groups are disjoint by construction, so their counts add up without
    double counting.

    The packed keys are held in one of two interchangeable representations:

    * In memory, as a deduplicated array. This is the fast common path for
      clades whose keys fit in RAM.
    * On disk, partitioned into prefix-bucket files by each key's first
      byte, once a clade's keys would risk exhausting RAM. Keys in
      different buckets can never be equal, so each bucket is deduplicated
      independently and the counts summed, bounding peak memory regardless
      of clade size.

    The ambiguous subsequences, always few, stay in memory in both modes.
    """

    def __init__(
        self,
        pure_keys,
        ambiguous_count: int,
        key_bytes: int,
        bucket_paths=None,
        tmp_dir=None,
    ):
        """Store the deduplicated key groups in memory or on disk.

        Args:
            pure_keys: Deduplicated array of packed 2-bit keys (memory
                mode), or None when the keys live on disk.
            ambiguous_count: Count of unique ambiguous subsequences in
                this accumulator. Stored as an integer — not the strings
                themselves — so it does not accumulate unbounded memory
                when propagated up through tens of thousands of leaves.
            key_bytes: Width of one packed key in bytes.
            bucket_paths: The 256 prefix-bucket file paths (disk mode), or
                None when the keys live in memory.
            tmp_dir: Temporary directory holding the bucket files (disk
                mode), removed on release, or None in memory mode.
        """
        self._pure_keys = pure_keys
        self._ambiguous_count = ambiguous_count
        self._key_bytes = key_bytes
        self._bucket_paths = bucket_paths
        self._tmp_dir = tmp_dir

    @property
    def _on_disk(self) -> bool:
        """Return True when the packed keys are stored on disk."""
        return self._bucket_paths is not None

    @classmethod
    def from_sequence_leaf(
        cls,
        leaf,
        min_len: int,
        void_dtype,
        disk_threshold: int,
        use_cache: bool = True,
        spill_dir: str | None = None,
    ):
        """Build an accumulator from a single sequence leaf.

        Reads the leaf's sequence, enumerates its sliding windows, and
        deduplicates them into the two key groups. A leaf whose unique
        keys exceed the disk threshold spills to bucket files immediately,
        which matters for very large single genomes.

        Args:
            leaf: A sequence-rank leaf node with ``fasta_path`` and
                ``header_id`` attributes.
            min_len: Sliding window size in base pairs.
            void_dtype: The numpy void dtype sized to one packed key.
            disk_threshold: Maximum in-memory keys before spilling the
                leaf's keys to bucket files.
            use_cache: When False, bypasses the module-level sequence
                cache. Pass False for the bottom-up pass, where each leaf
                is read exactly once and caching only inflates RSS.

        Returns:
            A populated accumulator, empty when the leaf has no readable
            sequence or one shorter than ``min_len``.
        """
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        # lazy so this module doesn't import capacity at load time, while a test
        # patch of capacity._read_sequence_cached still applies (see module docstring)
        from taxotreeset.core.generation.capacity import (
            _read_sequence_cached,
            _read_single_sequence,
        )

        key_bytes = void_dtype.itemsize
        empty = np.empty((0,), dtype=void_dtype)
        fasta_path = getattr(leaf, "fasta_path", "")
        header_id = getattr(leaf, "header_id", "")
        if not fasta_path or not header_id:
            return cls(empty, 0, key_bytes)
        if use_cache:
            sequence = _read_sequence_cached(fasta_path, header_id)
        else:
            sequence = _read_single_sequence(fasta_path, header_id) or ""
        if not sequence or len(sequence) < min_len:
            return cls(empty, 0, key_bytes)
        seq_arr = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        n_windows = max(0, seq_arr.shape[0] - min_len + 1)
        if n_windows == 0:
            return cls(empty, 0, key_bytes)
        # _encode_windows_2bit allocates a codes array of shape
        # (n_windows, min_len) uint8. For large scaffolds this can reach
        # several GB and OOM before any disk-spillover threshold fires.
        # Cap individual batches to ~512 MiB of codes.
        _encode_batch = max(100_000, (512 << 20) // min_len)
        if n_windows > _encode_batch:
            # Delete the Python string before entering the chunked path.
            # seq_arr's backing bytes object (from sequence.encode) is kept
            # alive by seq_arr.base; only the string itself is freed here,
            # saving sequence_length bytes for the duration of the call.
            del sequence
            return cls._from_chunked_sequence(
                seq_arr, min_len, _encode_batch, void_dtype, key_bytes,
                spill_dir=spill_dir,
            )
        windows = sliding_window_view(seq_arr, min_len)
        keys, pure_mask = _encode_windows_2bit(windows, min_len)
        pure_keys = np.unique(keys) if keys.shape[0] else empty
        ambiguous: set = set()
        if not pure_mask.all():
            ambiguous = {
                sequence[i : i + min_len]
                for i in np.flatnonzero(~pure_mask).tolist()
            }
        if pure_keys.shape[0] >= disk_threshold:
            return cls._spilled_from_arrays(
                [pure_keys], len(ambiguous), key_bytes, spill_dir=spill_dir,
            )
        return cls(pure_keys, len(ambiguous), key_bytes)

    @classmethod
    def _from_chunked_sequence(
        cls,
        seq_arr,
        min_len: int,
        chunk_size: int,
        void_dtype,
        key_bytes: int,
        spill_dir: str | None = None,
    ):
        """Encode a large sequence in batches to avoid peak-memory OOM.

        ``_encode_windows_2bit`` creates a ``codes`` array of shape
        ``(n_windows, min_len)`` uint8. For scaffolds larger than ~5 MB
        (at min_len=100) this intermediate allocation reaches several GB and
        triggers an OOM before the disk-spillover threshold can react. This
        method processes the sequence in ``chunk_size``-window batches,
        flushing each batch's keys into prefix-bucketed disk files and
        deduplicating bucket-by-bucket at the end.

        Ambiguous windows (containing N or IUPAC codes) are intentionally
        not tracked here. Building a per-sequence set of ambiguous strings
        across all chunks allocates O(n_ambiguous × min_len) bytes for a
        single scaffold, which can reach several GiB for large chromosomes
        with many N-runs. Because ambiguous subsequences are filtered out
        during dataset generation, omitting them from the capacity count
        does not affect the usable training-sample estimate.

        Args:
            seq_arr: The full sequence encoded as a uint8 numpy array.
                The caller must ensure ``seq_arr``'s backing buffer remains
                alive for the duration of this call.
            min_len: Sliding window size in bases.
            chunk_size: Number of windows to encode per batch.
            void_dtype: Numpy void dtype sized to one packed key.
            key_bytes: Width of one packed key in bytes.

        Returns:
            A disk-mode accumulator owning a fresh bucket directory.
        """
        import os
        import tempfile

        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view

        tmp_dir = tempfile.mkdtemp(prefix="tts_capacity_", dir=spill_dir)
        bucket_paths = _bucket_writer_paths(tmp_dir)
        n_windows = seq_arr.shape[0] - min_len + 1

        with contextlib.ExitStack() as stack:
            bucket_files = [stack.enter_context(open(p, "ab")) for p in bucket_paths]
            for chunk_start in range(0, n_windows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_windows)
                # Include the trailing bases needed to form complete windows.
                sub = seq_arr[chunk_start: chunk_end + min_len - 1]
                windows_chunk = sliding_window_view(sub, min_len)
                keys, pure_mask = _encode_windows_2bit(windows_chunk, min_len)
                if keys.shape[0]:
                    _flush_keys_to_buckets(keys, bucket_files, key_bytes)
                # Ambiguous windows are intentionally skipped — see docstring.

        for path in bucket_paths:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            raw = np.fromfile(path, dtype=np.uint8)
            unique_bucket = np.unique(raw.view(void_dtype))
            unique_bucket.tofile(path)

        return cls(None, 0, key_bytes, bucket_paths, tmp_dir)

    @classmethod
    def _spilled_from_arrays(
        cls, pure_arrays: list, ambiguous_count: int, key_bytes: int,
        spill_dir: str | None = None,
    ):
        """Create a disk-mode accumulator from in-memory key arrays.

        Args:
            pure_arrays: Arrays of packed keys to write to buckets.
            ambiguous_count: Count of unique ambiguous subsequences.
            key_bytes: Width of one packed key in bytes.

        Returns:
            A disk-mode accumulator owning a fresh bucket directory.
        """
        import tempfile

        tmp_dir = tempfile.mkdtemp(prefix="tts_capacity_", dir=spill_dir)
        bucket_paths = _bucket_writer_paths(tmp_dir)
        with contextlib.ExitStack() as stack:
            bucket_files = [stack.enter_context(open(p, "wb")) for p in bucket_paths]
            for keys in pure_arrays:
                if keys is not None and keys.shape[0]:
                    _flush_keys_to_buckets(keys, bucket_files, key_bytes)
        return cls(None, ambiguous_count, key_bytes, bucket_paths, tmp_dir)

    @classmethod
    def merge(cls, parts: list, void_dtype, disk_threshold: int):
        """Merge several accumulators into one, deduplicating across them.

        The union is exact and captures subsequences shared between
        siblings (conserved regions), so a parent's count is the size of
        the union, never the naive sum. When the combined pure keys stay
        small they are unioned in memory with ``np.unique``; when they
        would exhaust RAM the merge spills to bucket files, appending each
        child's keys (already-bucketed children are concatenated bucket by
        bucket, since the partition is identical).

        Args:
            parts: Accumulators to merge (typically a node's children).
            void_dtype: The numpy void dtype sized to one packed key.
            disk_threshold: Maximum combined in-memory keys before the
                merge spills to bucket files.

        Returns:
            A single accumulator holding the deduplicated union.
        """
        import numpy as np

        key_bytes = void_dtype.itemsize
        ambiguous_count = sum(p._ambiguous_count for p in parts)

        # A single child (e.g. a passthrough node) needs no reprocessing:
        # its set already is the union, so adopt it directly rather than
        # re-deduplicating millions of keys. Ownership of the keys moves
        # to the new accumulator, leaving the child empty so its later
        # release does not drop the adopted storage.
        if len(parts) == 1:
            return parts[0]._transfer_ownership(ambiguous_count)

        any_on_disk = any(part._on_disk for part in parts)
        memory_arrays = [
            part._pure_keys
            for part in parts
            if not part._on_disk and part._pure_keys.shape[0]
        ]
        memory_total = sum(arr.shape[0] for arr in memory_arrays)

        if not any_on_disk and memory_total < disk_threshold:
            if memory_arrays:
                pure_keys = np.unique(np.concatenate(memory_arrays))
            else:
                pure_keys = np.empty((0,), dtype=void_dtype)
            return cls(pure_keys, ambiguous_count, key_bytes)

        return cls._spilled_merge(parts, ambiguous_count, key_bytes)

    @classmethod
    def _spilled_merge(
        cls, parts: list, ambiguous_count: int, key_bytes: int,
        spill_dir: str | None = None,
    ):
        """Merge accumulators into a fresh disk-mode accumulator.

        In-memory children are flushed into the new buckets; disk-mode
        children have each of their bucket files appended to the matching
        new bucket, which is exact because both use the same first-byte
        partition.

        Args:
            parts: Accumulators to merge.
            ambiguous_count: Sum of per-part ambiguous counts.
            key_bytes: Width of one packed key in bytes.

        Returns:
            A disk-mode accumulator owning a fresh bucket directory.
        """
        import os
        import shutil
        import tempfile

        import numpy as np

        void_dtype = np.dtype((np.void, key_bytes))
        tmp_dir = tempfile.mkdtemp(prefix="tts_capacity_", dir=spill_dir)
        bucket_paths = _bucket_writer_paths(tmp_dir)

        # Write every child's keys into the new buckets one child at a time
        # (streaming, never holding more than one child in memory), then
        # deduplicate one bucket at a time. A node thus propagates an
        # already-unique set so dedup work neither repeats nor accumulates
        # as sets rise through the tree, while peak memory stays bounded by
        # a single bucket (~1/256 of the keys) rather than the whole set.
        with contextlib.ExitStack() as stack:
            bucket_files = [stack.enter_context(open(p, "ab")) for p in bucket_paths]
            for part in parts:
                if part._on_disk:
                    for index, child_bucket in enumerate(part._bucket_paths):
                        if os.path.exists(child_bucket) and os.path.getsize(
                            child_bucket
                        ):
                            with open(child_bucket, "rb") as source:
                                shutil.copyfileobj(source, bucket_files[index])
                elif (
                    part._pure_keys is not None and part._pure_keys.shape[0]
                ):
                    _flush_keys_to_buckets(
                        part._pure_keys, bucket_files, key_bytes
                    )

        for path in bucket_paths:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            raw = np.fromfile(path, dtype=np.uint8)
            unique_bucket = np.unique(raw.view(void_dtype))
            unique_bucket.tofile(path)
        return cls(None, ambiguous_count, key_bytes, bucket_paths, tmp_dir)

    def _inplace_extend(self, child: "_NodeCapacityKeys", key_bytes: int) -> None:
        """Append a child's keys directly into this accumulator's bucket files.

        Requires self to be in disk mode. The child is not modified or
        released by this call. This avoids the O(self) copy that
        ``_spilled_merge`` would incur when updating a growing accumulator
        one child at a time: only the child's data is written, not a copy
        of the existing accumulator.

        Args:
            child: Accumulator whose keys are appended to self's buckets.
            key_bytes: Width of one packed key in bytes.
        """
        import os
        import shutil

        assert self._on_disk, "_inplace_extend requires a disk-mode accumulator"

        self._ambiguous_count += child._ambiguous_count
        with contextlib.ExitStack() as stack:
            bucket_files = [
                stack.enter_context(open(p, "ab")) for p in self._bucket_paths
            ]
            if child._on_disk:
                for index, child_bucket in enumerate(child._bucket_paths):
                    if os.path.exists(child_bucket) and os.path.getsize(child_bucket):
                        with open(child_bucket, "rb") as source:
                            shutil.copyfileobj(source, bucket_files[index])
            elif child._pure_keys is not None and child._pure_keys.shape[0]:
                _flush_keys_to_buckets(child._pure_keys, bucket_files, key_bytes)

    def _transfer_ownership(self, ambiguous_count: int):
        """Move this accumulator's key storage into a new accumulator.

        Used when a parent has a single child: the child's set already is
        the union, so the parent adopts its in-memory array or its bucket
        files directly. This object is left empty afterwards so that the
        traversal's later call to its release does not drop the storage now
        owned by the returned accumulator.

        Args:
            ambiguous_count: Ambiguous subsequence count for the new
                accumulator (already summed by the caller).

        Returns:
            A new accumulator owning this one's pure-key storage.
        """
        adopted = _NodeCapacityKeys(
            self._pure_keys,
            ambiguous_count,
            self._key_bytes,
            self._bucket_paths,
            self._tmp_dir,
        )
        self._pure_keys = None
        self._bucket_paths = None
        self._tmp_dir = None
        self._ambiguous_count = 0
        return adopted

    def cardinality(self) -> int:
        """Return the number of unique subsequences accumulated.

        Returns:
            The count of unique pure keys plus unique ambiguous
            subsequences.
        """
        if self._on_disk:
            pure_count = _count_unique_bucketed_on_disk(
                self._bucket_paths, self._key_bytes
            )
        else:
            pure_count = int(self._pure_keys.shape[0])
        return pure_count + self._ambiguous_count

    def release(self) -> None:
        """Free the accumulated keys once the node has been resolved.

        Drops the in-memory array, or closes and removes the bucket files
        and their temporary directory in disk mode.
        """
        import os

        self._pure_keys = None
        self._ambiguous_count = 0
        if self._bucket_paths is not None:
            for path in self._bucket_paths:
                if os.path.exists(path):
                    os.remove(path)
            if self._tmp_dir is not None and os.path.isdir(self._tmp_dir):
                os.rmdir(self._tmp_dir)
            self._bucket_paths = None
            self._tmp_dir = None
