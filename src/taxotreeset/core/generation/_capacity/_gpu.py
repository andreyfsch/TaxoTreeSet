"""GPU-accelerated capacity kernels (require CuPy + CUDA).

Pure compute kernels extracted from ``capacity.py``: CUDA-device detection, a
GPU sliding-window view, GPU row-dedup, and the chunked encode-and-dedup of one
sequence entirely on the device. CuPy / numpy / psutil are imported lazily inside
the functions so the module loads without a GPU. The multiprocessing leaf workers
— which read sequences, assign devices, and fall back to CPU on OOM — stay in
``capacity.py`` and call ``_gpu_encode_unique``.
"""

from typing import TYPE_CHECKING

from taxotreeset.core.generation._capacity._encoding import (
    _BASES_PER_BYTE,
    _HASHED_PREFIX_BUCKETS,
    _get_acgt_lut,
)

if TYPE_CHECKING:
    import numpy as np


def _detect_cuda_device_count() -> int:
    """Return the number of available CUDA devices, or 0 if CuPy is absent."""
    try:
        import cupy as cp
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _gpu_sliding_window_view(seq_gpu, window_len: int):
    """Sliding-window view over a 1-D CuPy uint8 array.

    Uses ``sliding_window_view`` when available (CuPy ≥ 12) and falls
    back to ``as_strided`` for older releases.
    """
    import cupy as cp

    try:
        return cp.lib.stride_tricks.sliding_window_view(seq_gpu, window_len)
    except AttributeError:
        n = seq_gpu.shape[0]
        n_windows = n - window_len + 1
        strides = (seq_gpu.strides[0], seq_gpu.strides[0])
        return cp.lib.stride_tricks.as_strided(
            seq_gpu, shape=(n_windows, window_len), strides=strides,
        )


def _gpu_unique_rows(packed_gpu):
    """Return deduplicated rows of a 2-D uint8 CuPy array.

    Uses ``cp.lexsort`` (GPU radix sort column-by-column) rather than
    ``cp.unique(axis=0)``.  On CuPy ≥ 14 the latter silently calls
    ``numpy.unique`` on the host for void-typed keys, turning a 50 ms GPU
    operation into a multi-minute CPU sort for large arrays.
    ``cp.lexsort`` dispatches to thrust and stays on the device.
    """
    import cupy as cp

    n = packed_gpu.shape[0]
    if n <= 1:
        return packed_gpu
    # cp.lexsort requires a 2-D CuPy ndarray, not a Python sequence.
    # Transposing gives shape (key_bytes, n); reversing columns puts
    # column-0 (most significant byte) on the last row, which lexsort
    # treats as the primary sort key.
    keys_2d = cp.ascontiguousarray(packed_gpu[:, ::-1].T)
    idx = cp.lexsort(keys_2d)
    del keys_2d
    sorted_arr = packed_gpu[idx]
    diff = cp.any(sorted_arr[1:] != sorted_arr[:-1], axis=1)
    mask = cp.concatenate([cp.array([True]), diff])
    return sorted_arr[mask]


def _gpu_encode_unique(
    seq_arr: "np.ndarray",
    min_len: int,
    device_id: int,
    key_bytes: int,
) -> "np.ndarray":
    """Encode and globally deduplicate a sequence's k-mers entirely on GPU.

    Processes the sequence in VRAM-sized chunks, deduplicates each chunk
    on the GPU, then performs a final cross-chunk dedup.  Only the unique
    packed keys are transferred back to the host — a small fraction of all
    windows for repetitive sequences.

    Ambiguous windows (non-ACGT bases) are intentionally not tracked; the
    returned count is therefore exact for pure-ACGT sequences and an
    under-estimate otherwise.  This matches ``_from_chunked_sequence``.

    Args:
        seq_arr: Host-side uint8 array of ASCII-encoded sequence bases.
        min_len: Sliding-window size in bases.
        device_id: CUDA device index.
        key_bytes: Packed key width = ceil(min_len / 4).

    Returns:
        (M,) void-dtype numpy array of unique 2-bit-packed keys.
    """
    import numpy as np
    import cupy as cp

    cp.cuda.Device(device_id).use()
    lut_gpu = cp.array(_get_acgt_lut())

    n = seq_arr.shape[0]
    n_windows = n - min_len + 1

    # Budget 30 % of free VRAM per chunk.  Peak per window: codes array
    # (min_len bytes) + filtered pure (≈ 0.9 × min_len) + packed (key_bytes).
    free_vram, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_window = int(min_len * 2.0 + key_bytes)
    chunk_size = max(500_000, int(free_vram * 0.30 / bytes_per_window))

    # Upload the full sequence to VRAM once.  For sequences larger than
    # available VRAM this raises cupy.cuda.memory.OutOfMemoryError, which
    # propagates out of _gpu_encode_unique and is caught by
    # _leaf_worker_task_auto, routing the leaf to the CPU disk-spill path.
    # This is the correct filter: only sequences that genuinely fit in VRAM
    # are processed on GPU; everything else goes to CPU automatically.
    seq_gpu = cp.asarray(seq_arr)

    # Accumulate per-chunk unique arrays on CPU (not GPU) to avoid exhausting
    # VRAM on GPUs with small memory budgets (e.g. GTX 1650 with 4 GB).
    # Guard: if accumulated data exceeds 30 % of *currently* available RAM,
    # raise MemoryError to trigger CPU fallback before the host OOMs.
    # psutil.available adapts automatically: ~2 GB locally, ~75 GB on
    # HoreKa Green, so large chromosomes are handled in-memory on big nodes
    # without unnecessary spill.
    import psutil as _psutil
    _ACCUM_LIMIT = int(_psutil.virtual_memory().available * 0.30)
    all_unique_cpu: list = []
    _accumulated_bytes = 0

    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_windows)

        sub = seq_gpu[chunk_start: chunk_end + min_len - 1]
        windows = _gpu_sliding_window_view(sub, min_len)

        codes = lut_gpu[windows]                              # (N, min_len) uint8
        del windows
        pure_mask = cp.all(codes != cp.uint8(255), axis=1)   # (N,) bool
        pure = codes[pure_mask]
        n_pure = int(pure.shape[0])
        del codes, pure_mask

        if n_pure == 0:
            del pure
            cp.get_default_memory_pool().free_all_blocks()
            continue

        pad = (-min_len) % _BASES_PER_BYTE
        if pad:
            pure = cp.concatenate(
                [pure, cp.zeros((n_pure, pad), dtype=cp.uint8)], axis=1,
            )
        groups = pure.reshape(n_pure, key_bytes, _BASES_PER_BYTE)
        packed = (
            groups[:, :, 0]
            | (groups[:, :, 1] << cp.uint8(2))
            | (groups[:, :, 2] << cp.uint8(4))
            | (groups[:, :, 3] << cp.uint8(6))
        ).astype(cp.uint8)                                    # (n_pure, key_bytes)
        del pure, groups

        # Dedup this chunk on GPU, transfer only unique keys to CPU, then
        # release the slab once per chunk.
        unique_chunk_gpu = _gpu_unique_rows(packed)
        unique_cpu = cp.asnumpy(unique_chunk_gpu)
        del packed, unique_chunk_gpu
        cp.get_default_memory_pool().free_all_blocks()  # one sync per chunk

        _accumulated_bytes += unique_cpu.nbytes
        if _accumulated_bytes > _ACCUM_LIMIT:
            del seq_gpu, lut_gpu, all_unique_cpu, unique_cpu
            cp.get_default_memory_pool().free_all_blocks()
            raise MemoryError(
                f"GPU encode accumulation reached "
                f"{_accumulated_bytes / 1024**3:.1f} GB — "
                "routing to CPU spill path"
            )
        all_unique_cpu.append(unique_cpu)

    del seq_gpu, lut_gpu
    cp.get_default_memory_pool().free_all_blocks()

    void_dtype = np.dtype((np.void, key_bytes))

    if not all_unique_cpu:
        return np.empty((0,), dtype=void_dtype)

    # Final cross-chunk dedup.  If the concatenated unique-per-chunk data fits
    # in VRAM (3× for sort scratch), run it on GPU.  Otherwise fall back to a
    # CPU bucket sort, which is the same O(N/256) per-bucket np.unique the CPU
    # leaf worker uses.
    all_cpu = np.concatenate(all_unique_cpu, axis=0)          # (total, key_bytes) uint8
    del all_unique_cpu

    try:
        free_vram, _ = cp.cuda.runtime.memGetInfo()
        if all_cpu.nbytes * 3 < free_vram:
            merged_gpu = cp.asarray(all_cpu)
            del all_cpu
            unique_gpu = _gpu_unique_rows(merged_gpu)
            del merged_gpu
            cp.get_default_memory_pool().free_all_blocks()
            result_cpu = cp.asnumpy(unique_gpu)
            del unique_gpu
            return np.ascontiguousarray(result_cpu).view(void_dtype).reshape(-1)
    except Exception:
        pass

    # CPU bucket-based dedup: mirror of _flush_keys_to_buckets / _from_chunked_sequence.
    n_buckets = _HASHED_PREFIX_BUCKETS
    cpu_buckets: list[list] = [[] for _ in range(n_buckets)]
    first_byte = all_cpu[:, 0]
    order = np.argsort(first_byte, kind="stable")
    sb = all_cpu[order]
    sf = first_byte[order]
    del all_cpu, order, first_byte

    split_pts = np.flatnonzero(np.diff(sf)) + 1
    starts = np.concatenate([[0], split_pts])
    ends = np.concatenate([split_pts, [len(sf)]])
    for s, e in zip(starts, ends):
        b = int(sf[s])
        cpu_buckets[b].append(sb[s:e])
    del sb, sf

    result_parts = []
    for bucket_arrs in cpu_buckets:
        if not bucket_arrs:
            continue
        merged = np.concatenate(bucket_arrs, axis=0)
        keys_v = np.ascontiguousarray(merged).view(void_dtype).reshape(-1)
        result_parts.append(np.unique(keys_v))

    if not result_parts:
        return np.empty((0,), dtype=void_dtype)
    return np.concatenate(result_parts)
