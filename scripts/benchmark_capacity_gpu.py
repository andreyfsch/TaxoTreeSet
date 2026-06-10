#!/usr/bin/env python3
"""Benchmark CPU vs GPU for the capacity leaf processing hotpath.

Profiles every sub-phase of the chunked leaf pipeline:
  LMDB read simulation → window encoding → sort/unique → bucket flush

Runs the CPU (numpy) path on any machine. Adds a CuPy/GPU path when
--gpu is requested. Finishes with an Amdahl projection showing the
realistic end-to-end speedup as a function of the I/O fraction.

Usage:
    # CPU only — works everywhere:
    python scripts/benchmark_capacity_gpu.py

    # Larger synthetic sequence (Mbp):
    python scripts/benchmark_capacity_gpu.py --size 500

    # CPU + one GPU (needs CuPy and a CUDA device):
    python scripts/benchmark_capacity_gpu.py --gpu

    # CPU + four GPUs (HoreKa Green: 4x A100-40):
    python scripts/benchmark_capacity_gpu.py --gpu --n-gpus 4

    # Full A100-40 test at realistic scale:
    python scripts/benchmark_capacity_gpu.py --size 1000 --min-len 100 --gpu --n-gpus 4
"""

import argparse
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASES_PER_BYTE = 4
_HASHED_PREFIX_BUCKETS = 256


def _get_acgt_lut() -> np.ndarray:
    lut = np.full(256, 255, dtype=np.uint8)
    for code, base in enumerate(b"ACGT"):
        lut[base] = code
    return lut


def _synthetic_sequence(n_bases: int, repetitive_fraction: float = 0.3) -> np.ndarray:
    """Generate a realistic synthetic genome as a uint8 (ASCII) array.

    A ``repetitive_fraction`` share of the sequence consists of a
    telomere-like 6-mer repeat (TTAGGG) tiled to fill the window.
    The rest is uniformly random ACGT, giving a sequence with roughly
    ``repetitive_fraction × 100 %`` of windows mapping to duplicate keys
    — matching what we see in large eukaryotic chromosomes.
    """
    rng = np.random.default_rng(42)
    seq = rng.choice(np.frombuffer(b"ACGT", dtype=np.uint8), size=n_bases)

    # Splice in the telomeric repeat block.
    rep_len = int(n_bases * repetitive_fraction)
    telomere = np.frombuffer(b"TTAGGG" * ((rep_len // 6) + 1), dtype=np.uint8)
    seq[:rep_len] = telomere[:rep_len]

    return seq  # uint8 ASCII-encoded


# ---------------------------------------------------------------------------
# CPU (numpy) benchmark
# ---------------------------------------------------------------------------

def _encode_windows_2bit_cpu(windows: np.ndarray, min_len: int, lut: np.ndarray):
    """Pure-numpy version mirroring capacity.py:_encode_windows_2bit."""
    codes = lut[windows]
    pure_mask = np.all(codes != np.uint8(255), axis=1)
    pure = codes[pure_mask]
    n_pure = pure.shape[0]
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
    if n_pure == 0:
        return np.empty((0, key_bytes), dtype=np.uint8), pure_mask

    pad = (-min_len) % _BASES_PER_BYTE
    if pad:
        pure = np.concatenate([pure, np.zeros((n_pure, pad), dtype=np.uint8)], axis=1)
    groups = pure.reshape(n_pure, key_bytes, _BASES_PER_BYTE)
    packed = (
        groups[:, :, 0]
        | (groups[:, :, 1] << np.uint8(2))
        | (groups[:, :, 2] << np.uint8(4))
        | (groups[:, :, 3] << np.uint8(6))
    ).astype(np.uint8)
    return packed, pure_mask


def _flush_keys_to_buckets_cpu(packed: np.ndarray) -> dict[int, np.ndarray]:
    """Sort keys by first byte and group into buckets (in-memory dict)."""
    first_byte = packed[:, 0]
    order = np.argsort(first_byte, kind="stable")
    sorted_bytes = packed[order]
    sorted_first = first_byte[order]
    split_points = np.flatnonzero(np.diff(sorted_first)) + 1
    starts = np.concatenate([[0], split_points])
    ends = np.concatenate([split_points, [len(sorted_first)]])
    return {
        int(sorted_first[s]): sorted_bytes[s:e]
        for s, e in zip(starts, ends)
    }


def run_cpu_benchmark(seq: np.ndarray, min_len: int, chunk_size: int) -> dict:
    """Run the full chunked leaf pipeline on CPU and return per-phase timings."""
    from numpy.lib.stride_tricks import sliding_window_view

    lut = _get_acgt_lut()
    n_windows = seq.shape[0] - min_len + 1
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE

    # Phase 1: chunked encode + bucket flush
    t0 = time.perf_counter()
    buckets: dict[int, list[np.ndarray]] = {}
    total_windows = 0
    total_pure = 0

    for chunk_start in range(0, n_windows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_windows)
        sub = seq[chunk_start: chunk_end + min_len - 1]
        windows_chunk = sliding_window_view(sub, min_len)
        packed, _ = _encode_windows_2bit_cpu(windows_chunk, min_len, lut)
        total_windows += windows_chunk.shape[0]
        total_pure += packed.shape[0]
        if packed.shape[0]:
            chunk_buckets = _flush_keys_to_buckets_cpu(packed)
            for b_idx, b_data in chunk_buckets.items():
                if b_idx not in buckets:
                    buckets[b_idx] = []
                buckets[b_idx].append(b_data)
    t_encode = time.perf_counter() - t0

    # Phase 2: per-bucket deduplication (= final unique pass)
    t0 = time.perf_counter()
    void_dtype = np.dtype((np.void, key_bytes))
    total_unique = 0
    for b_idx in range(_HASHED_PREFIX_BUCKETS):
        parts = buckets.get(b_idx)
        if not parts:
            continue
        merged = np.concatenate(parts, axis=0)
        keys_void = np.ascontiguousarray(merged).view(void_dtype).reshape(-1)
        total_unique += int(np.unique(keys_void).shape[0])
    t_unique = time.perf_counter() - t0

    return {
        "n_windows": total_windows,
        "n_pure": total_pure,
        "total_unique": total_unique,
        "t_encode_s": t_encode,
        "t_unique_s": t_unique,
        "t_total_s": t_encode + t_unique,
        "chunk_size": chunk_size,
        "n_chunks": (n_windows + chunk_size - 1) // chunk_size,
        "key_bytes": key_bytes,
    }


# ---------------------------------------------------------------------------
# GPU (CuPy) benchmark
# ---------------------------------------------------------------------------

def _encode_windows_2bit_gpu(seq_gpu, min_len: int, lut_gpu):
    """CuPy equivalent of _encode_windows_2bit; returns (packed, pure_mask)."""
    import cupy as cp
    from cupy.lib.stride_tricks import sliding_window_view as cp_swv

    windows = cp_swv(seq_gpu, min_len)          # (N, min_len) uint8 — view, no copy
    codes = lut_gpu[windows]                     # (N, min_len) uint8 — LUT gather
    pure_mask = cp.all(codes != cp.uint8(255), axis=1)
    pure = codes[pure_mask]
    n_pure = int(pure.shape[0])
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE

    if n_pure == 0:
        return cp.empty((0, key_bytes), dtype=cp.uint8), pure_mask

    pad = (-min_len) % _BASES_PER_BYTE
    if pad:
        pure = cp.concatenate([pure, cp.zeros((n_pure, pad), dtype=cp.uint8)], axis=1)
    groups = pure.reshape(n_pure, key_bytes, _BASES_PER_BYTE)
    packed = (
        groups[:, :, 0]
        | (groups[:, :, 1] << cp.uint8(2))
        | (groups[:, :, 2] << cp.uint8(4))
        | (groups[:, :, 3] << cp.uint8(6))
    ).astype(cp.uint8)
    return packed, pure_mask


def run_gpu_benchmark(
    seq: np.ndarray,
    min_len: int,
    gpu_chunk_size: int,
    device_id: int = 0,
) -> dict:
    """Run the chunked leaf pipeline on a single GPU and return per-phase timings.

    The GPU chunk size is chosen to fill VRAM (up to 40 GB for A100-40),
    dramatically reducing the number of Python-loop iterations vs CPU.
    """
    import cupy as cp

    with cp.cuda.Device(device_id):
        cp.cuda.Stream.null.synchronize()

        lut_cpu = _get_acgt_lut()

        # --- Transfer sequence to GPU once ---
        t0 = time.perf_counter()
        seq_gpu = cp.array(seq)          # H2D: full sequence
        lut_gpu = cp.array(lut_cpu)
        cp.cuda.Stream.null.synchronize()
        t_h2d = time.perf_counter() - t0

        n_windows = seq.shape[0] - min_len + 1
        key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE

        # --- Phase 1: chunked encode on GPU ---
        t0 = time.perf_counter()
        gpu_buckets: dict[int, list] = {}
        total_windows = 0
        total_pure = 0

        for chunk_start in range(0, n_windows, gpu_chunk_size):
            chunk_end = min(chunk_start + gpu_chunk_size, n_windows)
            sub_gpu = seq_gpu[chunk_start: chunk_end + min_len - 1]
            packed_gpu, _ = _encode_windows_2bit_gpu(sub_gpu, min_len, lut_gpu)
            total_windows += chunk_end - chunk_start
            total_pure += int(packed_gpu.shape[0])

            if packed_gpu.shape[0]:
                first_byte_gpu = packed_gpu[:, 0]
                # Sort on GPU by first byte
                order = cp.argsort(first_byte_gpu, kind="stable")
                sorted_gpu = packed_gpu[order]
                sorted_first_gpu = first_byte_gpu[order]
                split_pts = cp.flatnonzero(cp.diff(sorted_first_gpu)) + 1
                starts_gpu = cp.concatenate([cp.array([0]), split_pts])
                ends_gpu = cp.concatenate([split_pts, cp.array([sorted_first_gpu.shape[0]])])

                starts_cpu = cp.asnumpy(starts_gpu)
                ends_cpu = cp.asnumpy(ends_gpu)
                first_labels = cp.asnumpy(sorted_first_gpu[starts_gpu])

                for s, e, b_idx in zip(starts_cpu, ends_cpu, first_labels):
                    chunk_arr = cp.asnumpy(sorted_gpu[s:e])    # D2H: bucket slice
                    b_idx = int(b_idx)
                    if b_idx not in gpu_buckets:
                        gpu_buckets[b_idx] = []
                    gpu_buckets[b_idx].append(chunk_arr)

        cp.cuda.Stream.null.synchronize()
        t_encode = time.perf_counter() - t0

        # --- Phase 2: bucket dedup on CPU (same as CPU path) ---
        t0 = time.perf_counter()
        void_dtype = np.dtype((np.void, key_bytes))
        total_unique = 0
        for b_idx in range(_HASHED_PREFIX_BUCKETS):
            parts = gpu_buckets.get(b_idx)
            if not parts:
                continue
            merged = np.concatenate(parts, axis=0)
            keys_void = np.ascontiguousarray(merged).view(void_dtype).reshape(-1)
            total_unique += int(np.unique(keys_void).shape[0])
        t_unique = time.perf_counter() - t0

        return {
            "device": device_id,
            "n_windows": total_windows,
            "n_pure": total_pure,
            "total_unique": total_unique,
            "t_h2d_s": t_h2d,
            "t_encode_s": t_encode,
            "t_unique_s": t_unique,
            "t_total_s": t_h2d + t_encode + t_unique,
            "gpu_chunk_size": gpu_chunk_size,
            "n_chunks": (n_windows + gpu_chunk_size - 1) // gpu_chunk_size,
            "key_bytes": key_bytes,
        }


# ---------------------------------------------------------------------------
# GPU VRAM capacity detection
# ---------------------------------------------------------------------------

def _detect_gpu_chunk_size(device_id: int, min_len: int, safety: float = 0.4) -> int:
    """Return a GPU chunk size that uses ~40 % of VRAM for the codes array.

    The intermediate ``codes`` array has shape ``(chunk_size, min_len)`` uint8.
    We cap usage at ``safety`` of free VRAM so there is room for the packed
    keys, sort buffers, and CuPy overhead.
    """
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            free, _total = cp.cuda.runtime.memGetInfo()
        budget = int(free * safety)
        chunk_size = budget // min_len
        chunk_size = max(1_000_000, chunk_size)
        # Round down to nearest million for readability.
        chunk_size = (chunk_size // 1_000_000) * 1_000_000
        return chunk_size
    except Exception:
        return 50_000_000  # 50M windows as safe fallback


# ---------------------------------------------------------------------------
# Amdahl projection
# ---------------------------------------------------------------------------

def _amdahl_projection(
    t_io: float,
    t_compute_cpu: float,
    t_compute_gpu: float,
    n_cpu_workers: int,
    n_gpus: int,
):
    """Print an Amdahl-law table for parallel leaf processing.

    Parameters
    ----------
    t_io : float
        I/O time (LMDB read) for a single leaf, in seconds.
    t_compute_cpu : float
        Encoding + unique compute time for one leaf on one CPU core.
    t_compute_gpu : float
        Encoding + unique compute time for one leaf on one GPU.
    n_cpu_workers : int
        Number of parallel CPU worker processes.
    n_gpus : int
        Number of GPUs available.
    """
    t_cpu_parallel = t_io + t_compute_cpu   # per-leaf, already parallelised across workers
    t_gpu_total = t_io + t_compute_gpu       # per-leaf, one GPU

    print()
    print("=" * 64)
    print("  AMDAHL PROJECTION (per-leaf wall time)")
    print("=" * 64)
    print(f"  I/O time (LMDB read, not parallelisable)  : {t_io:.2f} s")
    print(f"  Compute: CPU (1 core)                     : {t_compute_cpu:.2f} s")
    if t_compute_gpu is not None:
        print(f"  Compute: GPU (1 x A100-40)                : {t_compute_gpu:.2f} s")
        compute_speedup = t_compute_cpu / max(t_compute_gpu, 1e-6)
        leaf_speedup = t_cpu_parallel / max(t_gpu_total, 1e-6)
        io_fraction = t_io / max(t_cpu_parallel, 1e-6)
        print()
        print(f"  Compute speedup (GPU vs 1 CPU core)       : {compute_speedup:.1f}x")
        print(f"  Per-leaf speedup (GPU vs 1 CPU core)      : {leaf_speedup:.1f}x")
        print(f"  I/O fraction of CPU leaf time             : {io_fraction*100:.1f}%")
        print(f"  Amdahl ceiling (I/O-limited)              : {1/io_fraction:.1f}x")

        print()
        print("  MULTI-WORKER THROUGHPUT ESTIMATE")
        print(f"  {n_cpu_workers} CPU workers, each at {t_cpu_parallel:.1f}s/leaf:")
        print(f"    Throughput = {n_cpu_workers / t_cpu_parallel:.2f} leaves/s")
        print(f"  {n_gpus} GPU(s), each at {t_gpu_total:.1f}s/leaf:")
        print(f"    Throughput = {n_gpus / t_gpu_total:.2f} leaves/s")
        throughput_ratio = (n_gpus / max(t_gpu_total, 1e-6)) / (n_cpu_workers / max(t_cpu_parallel, 1e-6))
        print(f"  GPU-pool vs CPU-pool throughput           : {throughput_ratio:.1f}x")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs GPU for TaxoTreeSet capacity leaf processing"
    )
    parser.add_argument(
        "--size", type=int, default=200,
        help="Synthetic sequence length in Mbp (default: 200)",
    )
    parser.add_argument(
        "--min-len", type=int, default=100,
        help="Sliding window size (default: 100)",
    )
    parser.add_argument(
        "--repetitive", type=float, default=0.30,
        help="Fraction of sequence that is telomeric repeat (default: 0.30)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Also benchmark the CuPy/GPU path",
    )
    parser.add_argument(
        "--n-gpus", type=int, default=1,
        help="Number of GPUs for throughput projection (default: 1)",
    )
    parser.add_argument(
        "--cpu-workers", type=int, default=75,
        help="CPU workers for Amdahl projection (default: 75 = Xeon 8368 - 1)",
    )
    parser.add_argument(
        "--io-speed-mbs", type=float, default=500.0,
        help="Assumed LMDB read speed in MB/s for I/O estimate (default: 500)",
    )
    args = parser.parse_args()

    min_len = args.min_len
    n_bases = args.size * 1_000_000
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE

    # CPU chunk size mirrors capacity.py
    cpu_chunk_size = max(100_000, (512 << 20) // min_len)  # 5.37M for min_len=100

    print()
    print("=" * 64)
    print("  TaxoTreeSet capacity leaf benchmark")
    print("=" * 64)
    print(f"  Sequence size      : {args.size} Mbp ({n_bases:,} bases)")
    print(f"  Window (min_len)   : {min_len} bp")
    print(f"  Key bytes          : {key_bytes}")
    print(f"  Repetitive frac    : {args.repetitive*100:.0f}%")
    print(f"  CPU chunk size     : {cpu_chunk_size:,} windows")
    print(f"  CPU workers        : {args.cpu_workers}")
    print()

    print("  Generating synthetic sequence…", end=" ", flush=True)
    t0 = time.perf_counter()
    seq = _synthetic_sequence(n_bases, repetitive_fraction=args.repetitive)
    t_gen = time.perf_counter() - t0
    print(f"done ({t_gen:.1f}s)")

    # --- I/O estimate ---
    io_time_s = n_bases / (args.io_speed_mbs * 1e6)

    # ---------------------------------------------------------------
    # CPU benchmark
    # ---------------------------------------------------------------
    print()
    print("  [CPU] Running chunked leaf pipeline (numpy)…", flush=True)
    cpu = run_cpu_benchmark(seq, min_len, cpu_chunk_size)

    pure_pct = 100.0 * cpu["n_pure"] / max(cpu["n_windows"], 1)
    dup_pct = 100.0 * (1 - cpu["total_unique"] / max(cpu["n_pure"], 1))
    print()
    print("  --- CPU results ---")
    print(f"  Windows            : {cpu['n_windows']:>14,}")
    print(f"  Pure ACGT          : {cpu['n_pure']:>14,}  ({pure_pct:.1f}%)")
    print(f"  Unique keys        : {cpu['total_unique']:>14,}  ({dup_pct:.1f}% duplicates)")
    print(f"  Chunks processed   : {cpu['n_chunks']:>14,}")
    print(f"  Time — encode+flush: {cpu['t_encode_s']:>10.2f} s")
    print(f"  Time — bucket dedup: {cpu['t_unique_s']:>10.2f} s")
    print(f"  Time — total compute:{cpu['t_total_s']:>10.2f} s")
    print(f"  (I/O estimate @ {args.io_speed_mbs:.0f} MB/s: {io_time_s:.2f} s)")
    print(f"  Estimated leaf time (I/O + compute): {io_time_s + cpu['t_total_s']:.2f} s")

    # ---------------------------------------------------------------
    # GPU benchmark
    # ---------------------------------------------------------------
    gpu_result: dict | None = None
    if args.gpu:
        try:
            import cupy as cp
        except ImportError:
            print()
            print("  [GPU] CuPy not installed — skipping GPU benchmark.")
            print("        Install via: pip install cupy-cuda12x  (or matching CUDA version)")
        else:
            gpu_chunk_size = _detect_gpu_chunk_size(0, min_len)
            vram_usage_gb = (gpu_chunk_size * min_len) / 1e9
            print()
            print(f"  [GPU] CuPy found. GPU chunk size: {gpu_chunk_size:,} windows")
            print(f"        (codes array peak: {vram_usage_gb:.1f} GB VRAM)")
            print(f"        Running on device 0…", flush=True)

            gpu_result = run_gpu_benchmark(seq, min_len, gpu_chunk_size, device_id=0)

            print()
            print("  --- GPU results (device 0) ---")
            print(f"  Windows            : {gpu_result['n_windows']:>14,}")
            print(f"  Pure ACGT          : {gpu_result['n_pure']:>14,}")
            print(f"  Unique keys        : {gpu_result['total_unique']:>14,}")
            print(f"  Chunks (GPU)       : {gpu_result['n_chunks']:>14,}  "
                  f"(vs CPU: {cpu['n_chunks']:,})")
            print(f"  Time — H2D transfer: {gpu_result['t_h2d_s']:>10.3f} s")
            print(f"  Time — encode+flush: {gpu_result['t_encode_s']:>10.2f} s")
            print(f"  Time — bucket dedup: {gpu_result['t_unique_s']:>10.2f} s")
            print(f"  Time — total compute:{gpu_result['t_total_s']:>10.2f} s")
            print(f"  (I/O estimate: {io_time_s:.2f} s)")
            print(f"  Estimated leaf time (I/O + compute): {io_time_s + gpu_result['t_total_s']:.2f} s")

            speedup_compute = cpu["t_total_s"] / max(gpu_result["t_total_s"], 1e-9)
            speedup_leaf = (io_time_s + cpu["t_total_s"]) / max(
                io_time_s + gpu_result["t_total_s"], 1e-9
            )
            print()
            print(f"  Compute speedup (GPU vs 1 CPU core): {speedup_compute:.1f}x")
            print(f"  Leaf speedup (incl. I/O estimate)  : {speedup_leaf:.1f}x")

    # ---------------------------------------------------------------
    # Amdahl projection
    # ---------------------------------------------------------------
    gpu_compute_time = gpu_result["t_total_s"] if gpu_result else None
    if gpu_compute_time is not None:
        _amdahl_projection(
            t_io=io_time_s,
            t_compute_cpu=cpu["t_total_s"],
            t_compute_gpu=gpu_compute_time,
            n_cpu_workers=args.cpu_workers,
            n_gpus=args.n_gpus,
        )

    if args.gpu and gpu_result:
        print("NOTE: The GPU path moves bucket-dedup back to CPU.")
        print("      A fully GPU-resident dedup (cupy.unique on sorted chunks)")
        print("      would eliminate the D2H bucket transfers and is the next")
        print("      optimisation step if the above numbers justify it.")


if __name__ == "__main__":
    main()
