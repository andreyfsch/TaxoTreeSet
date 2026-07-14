"""Prefix-bucketed on-disk deduplication of packed capacity keys.

When a node's pure-ACGT key count would risk exhausting RAM, exact deduplication
switches from in-memory ``np.unique`` to a disk strategy: keys are partitioned
into 256 files by their first packed byte (the first four 2-bit bases), each
bucket is uniqued independently, and the per-bucket counts are summed. Keys in
different buckets can never be equal, so the sum is exact while peak memory is
bounded to a single bucket.

These helpers are pure key/file machinery (numpy + filesystem); they hold no
sequence I/O and no memory-vs-disk policy — the caller (``_NodeCapacityKeys`` /
``_capacity_exact``) decides when to spill. Extracted from ``capacity.py`` (P7
Part C) and re-exported there so existing imports keep working.
"""

from taxotreeset.core.generation._capacity._encoding import _HASHED_PREFIX_BUCKETS


def _bucket_writer_paths(tmp_dir: str) -> list:
    """Return the 256 prefix-bucket file paths under a temp directory.

    Args:
        tmp_dir: Directory to hold the per-bucket key files.

    Returns:
        List of 256 file paths, indexed by the first packed byte of a key.
    """
    import os

    return [
        os.path.join(tmp_dir, f"bucket_{i:03d}.bin")
        for i in range(_HASHED_PREFIX_BUCKETS)
    ]


def _count_unique_bucketed_on_disk(bucket_paths: list, key_bytes: int) -> int:
    """Count unique keys across prefix-bucket files, one bucket at a time.

    Each file holds raw packed keys (fixed ``key_bytes`` per key) whose first
    byte equals the bucket index. Keys in different buckets cannot be equal,
    so de-duplicating each bucket independently and summing the per-bucket
    unique counts yields the exact global unique count, while never holding
    more than one bucket in memory at once.

    Args:
        bucket_paths: The 256 per-bucket file paths.
        key_bytes: Width of each packed key in bytes.

    Returns:
        Total number of unique keys across all buckets.
    """
    import os

    import numpy as np

    void_dtype = np.dtype((np.void, key_bytes))
    total_unique = 0
    for path in bucket_paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        raw = np.fromfile(path, dtype=np.uint8)
        keys = raw.reshape(-1, key_bytes).view(void_dtype).reshape(-1)
        total_unique += int(np.unique(keys).shape[0])
        del raw, keys
    return total_unique


def _flush_keys_to_buckets(keys, bucket_files, key_bytes: int) -> None:
    """Append a batch of packed keys to their prefix-bucket files.

    Keys are partitioned by their first byte (the first four 2-bit bases),
    which is the prefix that defines the bucket. Writing in first-byte order
    groups the file appends so each bucket file is touched at most once per
    flush.

    Args:
        keys: (N,) void-typed array of packed keys to distribute.
        bucket_files: List of 256 open binary file handles, indexed by byte.
        key_bytes: Width of each packed key in bytes.
    """
    import numpy as np

    as_bytes = keys.view(np.uint8).reshape(-1, key_bytes)
    first_byte = as_bytes[:, 0]
    order = np.argsort(first_byte, kind="stable")
    sorted_bytes = as_bytes[order]
    sorted_first = first_byte[order]
    # Boundaries between distinct first-byte values in the sorted array.
    split_points = np.flatnonzero(np.diff(sorted_first)) + 1
    starts = np.concatenate([[0], split_points])
    ends = np.concatenate([split_points, [len(sorted_first)]])
    for start, end in zip(starts, ends):
        bucket = int(sorted_first[start])
        bucket_files[bucket].write(sorted_bytes[start:end].tobytes())


def _compact_pure_keys(unique_pure, pending):
    """Merge the pending key chunks into ``unique_pure`` and deduplicate.

    Returns the deduplicated array and a reset ``(pending, pending_count)``.
    """
    if not pending:
        return unique_pure, [], 0
    import numpy as np

    return np.unique(np.concatenate([unique_pure, *pending])), [], 0


def _cleanup_key_buckets(tmp_dir, bucket_files, bucket_paths) -> None:
    """Close any open bucket files and delete the temporary bucket directory."""
    import os

    if tmp_dir is None:
        return
    for handle in bucket_files or []:
        if not handle.closed:
            handle.close()
    for path in bucket_paths or []:
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(tmp_dir)
