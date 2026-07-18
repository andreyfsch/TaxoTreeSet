"""Bottom-up capacity computation: parallel leaf phase + sequential merge.

``_BottomUpCapacityComputer`` computes every node's capacity in one pass over the
tree: a **parallel leaf phase** (a spawn ``ProcessPoolExecutor`` of CPU/GPU workers
turns each sequence leaf into a packed-key accumulator) followed by a **sequential
merge phase** (children's key sets fold bottom-up so shared subsequences are counted
once at every ancestor). This is the engine behind ``compute_all_capacities``.

Extracted from ``capacity.py`` (P7 Part C). The pool workers, the pool initializer,
and the ``_WORKER_GPU_DEVICE_ID`` module global all live HERE together — spawn
pickles the workers by qualified name and the initializer sets the device id that
the workers read, so they must share one module. ``_resolve_bottom_up_threshold``
stays in ``capacity.py`` (a patch anchor); it is imported lazily where used so this
module does not import ``capacity`` at load time (cycle-free) while
``patch("capacity._resolve_bottom_up_threshold")`` still applies.
"""

import logging

from tqdm import tqdm

from taxotreeset.core.generation._capacity._encoding import _BASES_PER_BYTE
from taxotreeset.core.generation._capacity._gpu import (
    _detect_cuda_device_count,
    _gpu_encode_unique,
)
from taxotreeset.core.generation._capacity._keys import _NodeCapacityKeys
from taxotreeset.core.generation._capacity._spill import (
    _cleanup_spill_dirs,
    _delete_leaf_checkpoint,
    _load_leaf_checkpoint,
    _save_leaf_checkpoint,
)

logger = logging.getLogger("TaxoTreeSet.Core.Generation.Capacity")

# Minimum sequence length (bases) for the GPU path to be profitable. Below this
# the per-leaf H2D transfer + CUDA launch overhead exceeds the bandwidth gain.
_GPU_MIN_BASES: int = 500_000

# Process-local CUDA device index, set once by _leaf_pool_initializer when the
# worker process starts. -1 means this worker has no GPU assignment.
_WORKER_GPU_DEVICE_ID: int = -1


def _leaf_worker_task(
    fasta_path: str,
    header_id: str,
    leaf_name,
    min_len: int,
    key_bytes: int,
    disk_threshold: int,
    spill_dir: str | None,
) -> tuple:
    """Process one sequence leaf and return a serialisable result tuple.

    Top-level (non-nested) so the multiprocessing machinery can pickle it
    for both fork and spawn start methods.  Returns:
        (leaf_name, on_disk, data, ambiguous_count, key_bytes, tmp_dir)
    where ``data`` is a list of bucket paths when ``on_disk`` is True, or
    the raw bytes of the packed-key array when False.
    """
    import numpy as np

    void_dtype = np.dtype((np.void, key_bytes))

    class _LeafProxy:
        rank = "sequence"

    proxy = _LeafProxy()
    proxy.fasta_path = fasta_path
    proxy.header_id = header_id
    proxy.name = leaf_name

    acc = _NodeCapacityKeys.from_sequence_leaf(
        proxy, min_len, void_dtype, disk_threshold,
        use_cache=False, spill_dir=spill_dir,
    )

    if acc._on_disk:
        bucket_paths = acc._bucket_paths
        tmp_dir = acc._tmp_dir
        amb = acc._ambiguous_count
        # Null storage refs so the accumulator's release() won't delete the
        # spill files when the worker process exits.
        acc._bucket_paths = None
        acc._tmp_dir = None
        return (leaf_name, True, bucket_paths, amb, key_bytes, tmp_dir)

    raw = (
        acc._pure_keys.tobytes()
        if acc._pure_keys is not None and acc._pure_keys.shape[0]
        else b""
    )
    return (leaf_name, False, raw, acc._ambiguous_count, key_bytes, None)


def _reconstruct_leaf_keys(result: tuple, void_dtype) -> "_NodeCapacityKeys":
    """Rebuild a _NodeCapacityKeys from a _leaf_worker_task result."""
    import numpy as np

    _, on_disk, data, amb, kb, tmp_dir = result
    if on_disk:
        return _NodeCapacityKeys(None, amb, kb, data, tmp_dir)
    arr = (
        np.frombuffer(data, dtype=void_dtype).copy()
        if data
        else np.empty((0,), dtype=void_dtype)
    )
    return _NodeCapacityKeys(arr, amb, kb)


class _BottomUpCapacityComputer:
    """Two-phase bottom-up capacity computation for a taxonomic tree.

    Implements :func:`compute_all_capacities`. Phase 1 processes sequence
    leaves in parallel into packed-key accumulators, evicting the oldest
    in-memory accumulators to a single flat-bin file when the RAM budget is
    exceeded. Phase 2 folds the accumulators bottom-up by set-union, recording
    each internal node's capacity. State shared across the phases (the leaf
    accumulators, the flat-bin index, the running counters and the progress
    bar) lives on the instance.
    """

    def __init__(
        self,
        min_len: int,
        spill_dir: str | None,
        n_workers: int | None,
        n_gpu_workers: int | None,
    ) -> None:
        import logging
        import os

        import numpy as np
        import psutil

        # lazy: _resolve_bottom_up_threshold stays in capacity.py as a patch anchor;
        # importing it at call time avoids a capacity<->_bottomup import cycle while
        # keeping patch("capacity._resolve_bottom_up_threshold") effective.
        from taxotreeset.core.generation.capacity import _resolve_bottom_up_threshold

        if n_workers is None:
            n_workers = max(1, (os.cpu_count() or 1) - 1)
        n_workers = max(1, n_workers)
        if n_gpu_workers is None:
            n_gpu_workers = _detect_cuda_device_count()
        n_gpu_workers = max(0, n_gpu_workers)

        self.min_len = min_len
        self.spill_dir = spill_dir
        self.n_workers = n_workers
        self.n_gpu_workers = n_gpu_workers
        self.key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
        self.void_dtype = np.dtype((np.void, self.key_bytes))
        self.disk_threshold = _resolve_bottom_up_threshold(self.key_bytes)
        self.logger = logging.getLogger("TaxoTreeSet.Core.Generation.Capacity")

        self.capacities: dict[str, int] = {}
        self.accumulators: dict[str, _NodeCapacityKeys] = {}
        # name → (bin_path, byte_offset, n_keys, amb_count)
        self.flat_bins: dict[str, tuple[str, int, int, int]] = {}
        self.flat_bin_file: str | None = None
        self.in_memory_key_count = 0
        # Reserve 25 % of currently available RAM for leaf accumulation.
        self.ram_budget_keys = max(
            1, int(psutil.virtual_memory().available * 0.25) // self.key_bytes
        )
        self.leaves_done = 0
        self.total_leaves = 0
        self.pbar = None

    # ── orchestration ────────────────────────────────────────────────────

    def run(self, tree_root) -> dict:
        """Compute every node's capacity and return the name → capacity map."""
        all_leaves = [
            n for n in tree_root.leaves if getattr(n, "rank", "") == "sequence"
        ]
        self.total_leaves = len(all_leaves)
        self._log_start()

        self._leaf_phase(all_leaves)

        root_set = self._merge_subtree(tree_root)
        root_set.release()
        self.logger.info(
            "[bottom-up] Done: %d nodes resolved.", len(self.capacities)
        )

        self._cleanup()
        return self.capacities

    def _log_start(self) -> None:
        import psutil

        gpu_info = (
            f", gpu_workers={self.n_gpu_workers}"
            f" (devices 0-{self.n_gpu_workers - 1})"
            if self.n_gpu_workers > 0
            else " (CPU-only)"
        )
        self.logger.info(
            "[bottom-up] Starting: %d sequence leaves, disk_threshold=%s keys "
            "(%.2f GiB), sys_avail=%.2f GiB, cpu_workers=%d%s",
            self.total_leaves,
            f"{self.disk_threshold:,}",
            self.disk_threshold * self.key_bytes / 2**30,
            psutil.virtual_memory().available / 2**30,
            self.n_workers,
            gpu_info,
        )

    def _empty_acc(self) -> "_NodeCapacityKeys":
        import numpy as np

        return _NodeCapacityKeys(
            np.empty((0,), dtype=self.void_dtype), 0, self.key_bytes
        )

    # ── Phase 1: parallel leaf processing ────────────────────────────────

    def _leaf_phase(self, all_leaves: list) -> None:
        """Resume any checkpoint, process the remaining leaves, checkpoint."""
        self._resume_from_checkpoint()
        self.leaves_done = len(self.accumulators)

        # Progress bar visible on any TTY or file stream (nohup included).
        # Dynamic miniters keeps the bar from flooding the log on fast datasets
        # while still updating at least every 60 seconds on slow ones.
        self.pbar = tqdm(
            total=self.total_leaves,
            initial=self.leaves_done,
            desc="Unique k-mer analysis",
            unit="leaf",
            dynamic_ncols=True,
            miniters=1,
            smoothing=0.05,
        )

        valid_leaves = self._select_valid_leaves(all_leaves)
        self._process_valid_leaves(valid_leaves)
        self.pbar.close()
        self._save_leaf_checkpoint_maybe(valid_leaves)

    def _resume_from_checkpoint(self) -> None:
        """Load a valid leaf checkpoint, or clear stale spill dirs on a fresh run."""
        if not self.spill_dir:
            return
        restored = _load_leaf_checkpoint(
            self.spill_dir, self.min_len, self.void_dtype
        )
        if restored:
            self.accumulators.update(restored)
            self.logger.info(
                "[bottom-up] Resuming from checkpoint: %d/%d leaves already computed.",
                len(restored), self.total_leaves,
            )
        else:
            # Fresh run with no valid checkpoint — evict any tts_capacity_*
            # directories left by previous failed runs before starting Phase 1.
            _cleanup_spill_dirs(self.spill_dir)

    def _select_valid_leaves(self, all_leaves: list) -> list:
        """Mark unusable leaves empty; return the leaves still to process."""
        for leaf in all_leaves:
            if not getattr(leaf, "fasta_path", "") or not getattr(leaf, "header_id", ""):
                self.accumulators[str(leaf.name)] = self._empty_acc()

        # Exclude leaves whose accumulators were restored from the checkpoint.
        return [
            leaf for leaf in all_leaves
            if getattr(leaf, "fasta_path", "") and getattr(leaf, "header_id", "")
            and str(leaf.name) not in self.accumulators
        ]

    def _process_valid_leaves(self, valid_leaves: list) -> None:
        """Run the leaf worker pool (or a sequential fallback) and record results."""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        total_pool = self.n_workers + self.n_gpu_workers
        task_fn = (
            _leaf_worker_task_auto if self.n_gpu_workers > 0 else _leaf_worker_task
        )

        if total_pool > 1 and valid_leaves:
            # When GPU workers are present we must use the spawn start method so
            # each worker process initialises its own CUDA context from scratch.
            # Forking a process that has already touched CUDA state is undefined
            # behaviour in the CUDA driver.  The spawn overhead (~300 ms per
            # worker) is negligible compared to leaf processing times.
            pool_kwargs: dict = {"max_workers": total_pool}
            if self.n_gpu_workers > 0:
                spawn_ctx = multiprocessing.get_context("spawn")
                pool_kwargs["mp_context"] = spawn_ctx
                # Value and Lock must be created in the same spawn context so
                # they can be safely passed as initializer args to spawn workers.
                gpu_counter = spawn_ctx.Value("i", 0)
                gpu_lock = spawn_ctx.Lock()
                pool_kwargs["initializer"] = _leaf_pool_initializer
                pool_kwargs["initargs"] = (gpu_counter, gpu_lock, self.n_gpu_workers)

            with ProcessPoolExecutor(**pool_kwargs) as executor:
                future_map = {
                    executor.submit(
                        task_fn,
                        leaf.fasta_path, leaf.header_id, leaf.name,
                        self.min_len, self.key_bytes, self.disk_threshold,
                        self.spill_dir,
                    ): leaf.name
                    for leaf in valid_leaves
                }
                for future in as_completed(future_map):
                    try:
                        result = future.result()
                        # Release the IPC bytes immediately so completed futures
                        # don't accumulate ~500 KB each across 18 000+ leaves.
                        # CPython holds the result in Future._result until the
                        # Future is GC'd; as_completed keeps all futures alive
                        # internally, so we must clear it ourselves.
                        future._result = None
                        self._record(result)
                    except Exception as exc:
                        leaf_name = future_map[future]
                        self.logger.error(
                            "[bottom-up] leaf %s failed: %s — using empty accumulator",
                            leaf_name, exc,
                        )
                        self.accumulators[str(leaf_name)] = self._empty_acc()
                        self.leaves_done += 1
                        self.pbar.update(1)
        else:
            for leaf in valid_leaves:
                try:
                    result = _leaf_worker_task(
                        leaf.fasta_path, leaf.header_id, leaf.name,
                        self.min_len, self.key_bytes, self.disk_threshold,
                        self.spill_dir,
                    )
                    self._record(result)
                except Exception as exc:
                    self.logger.error(
                        "[bottom-up] leaf %s failed: %s — using empty accumulator",
                        leaf.name, exc,
                    )
                    self.accumulators[str(leaf.name)] = self._empty_acc()
                    self.leaves_done += 1
                    self.pbar.update(1)

    def _record(self, result: tuple) -> None:
        """Store a leaf result, evict if over the RAM budget, update progress."""
        import psutil

        leaf_name = result[0]
        acc = _reconstruct_leaf_keys(result, self.void_dtype)
        self.accumulators[str(leaf_name)] = acc
        # Snapshot cardinality and on-disk flag BEFORE potential eviction, which
        # nulls acc._pure_keys to allow GC — accessing it afterwards would fail.
        n_keys = acc.cardinality()
        acc_on_disk = acc._on_disk
        if not acc_on_disk and acc._pure_keys is not None:
            self.in_memory_key_count += acc._pure_keys.shape[0]
        if self.in_memory_key_count > self.ram_budget_keys:
            self._evict_to_flat_bins()
        self.leaves_done += 1
        self.pbar.update(1)
        if self.leaves_done % 50 == 0 or self.leaves_done == self.total_leaves:
            avail = psutil.virtual_memory().available / 2**30
            self.logger.info(
                "[bottom-up] leaves %d/%d  sys_avail=%.2f GiB  leaf=%s  keys=%s%s",
                self.leaves_done, self.total_leaves, avail,
                leaf_name,
                f"{n_keys:,}",
                "  [on-disk]" if acc_on_disk else "",
            )

    def _evict_to_flat_bins(self) -> None:
        """Evict ~50 % of in-memory leaf accumulators to a single flat-bin file.

        All evicted leaves are appended sequentially to ONE file per run,
        recording each leaf's byte offset in flat_bins.  This issues a
        single file-create + one large sequential write per eviction event
        instead of N small file creates, which is critical on NTFS/VHDX
        where per-file overhead limits effective throughput to ~5 MB/s
        with thousands of small files vs ~100 MB/s for sequential writes.
        """
        import os
        import tempfile

        if self.flat_bin_file is None:
            fd, path = tempfile.mkstemp(
                prefix="tts_capacity_flatbins_", dir=self.spill_dir, suffix=".bin"
            )
            os.close(fd)
            self.flat_bin_file = path
        bin_path = self.flat_bin_file
        target = max(1, self.in_memory_key_count // 2)
        evicted_keys = 0
        to_evict = []
        for name, acc in self.accumulators.items():
            if acc._on_disk or acc._pure_keys is None or acc._pure_keys.shape[0] == 0:
                continue
            to_evict.append((name, acc))
            evicted_keys += acc._pure_keys.shape[0]
            if evicted_keys >= target:
                break
        # One sequential write per eviction event — all leaves appended to the
        # same file.  f.tell() gives the byte offset before each array is written.
        with open(bin_path, "ab") as f:
            for name, acc in to_evict:
                offset = f.tell()
                n_keys = acc._pure_keys.shape[0]
                acc._pure_keys.tofile(f)
                self.flat_bins[name] = (bin_path, offset, n_keys, acc._ambiguous_count)
                del self.accumulators[name]
                acc._pure_keys = None
                acc._ambiguous_count = 0
        self.in_memory_key_count -= evicted_keys
        self.logger.info(
            "[bottom-up] Evicted %d leaves to flat-bin file (%.2f GiB); "
            "in-memory keys remaining: %d",
            len(to_evict), evicted_keys * self.key_bytes / 2**30,
            self.in_memory_key_count,
        )

    def _save_leaf_checkpoint_maybe(self, valid_leaves: list) -> None:
        """Save a leaf checkpoint, unless flat-bin eviction made it incomplete.

        Skipped when flat-bin eviction occurred: evicted leaves are absent from
        accumulators, so the checkpoint would be incomplete.  A crash during
        Phase 2 in that case requires re-running Phase 1 from scratch.
        """
        if self.spill_dir and valid_leaves and not self.flat_bins:
            _save_leaf_checkpoint(
                self.accumulators, self.spill_dir, self.min_len, self.void_dtype
            )
            self.logger.info(
                "[bottom-up] Leaf checkpoint saved (%d leaves) to %s",
                len(self.accumulators), self.spill_dir,
            )
        elif self.flat_bins:
            self.logger.info(
                "[bottom-up] Checkpoint skipped: %d leaves are in flat-bin storage "
                "(Phase 2 crash requires re-running Phase 1).",
                len(self.flat_bins),
            )

    # ── Phase 2: sequential bottom-up merge ──────────────────────────────

    def _merge_subtree(self, node) -> "_NodeCapacityKeys":
        """Resolve a node's key set by bottom-up union of its children.

        Leaf nodes load their accumulator from storage; internal nodes fold
        their children one at a time (bounding peak memory to one child's keys
        alongside the running accumulator), record their capacity, and return
        the merged accumulator.
        """
        import psutil

        if getattr(node, "rank", "") == "sequence":
            return self._resolve_leaf(str(node.name))

        if not node.children:
            return self._empty_acc()

        # Progressive accumulation: process one child at a time so that
        # at most one child's key array is live simultaneously alongside
        # the running accumulator. This bounds peak memory to
        # O(disk_threshold) regardless of how many children a node has,
        # instead of O(n_children × disk_threshold) with a batch merge.
        running = self._merge_subtree(node.children[0])
        for child_node in node.children[1:]:
            child_set = self._merge_subtree(child_node)
            running = self._merge_pair(running, child_set)

        cap = running.cardinality()
        self.capacities[str(node.name)] = cap
        avail = psutil.virtual_memory().available / 2**30
        self.logger.info(
            "[bottom-up] node %s (%s)  cap=%s  sys_avail=%.2f GiB%s",
            node.name, getattr(node, "rank", ""), f"{cap:,}", avail,
            "  [on-disk]" if running._on_disk else "",
        )
        return running

    def _resolve_leaf(self, leaf_name: str) -> "_NodeCapacityKeys":
        """Pop and return a leaf accumulator from memory or flat-bin storage."""
        import os

        import numpy as np

        acc = self.accumulators.pop(leaf_name, None)
        if acc is not None:
            return acc
        if leaf_name in self.flat_bins:
            bin_path, offset, n_keys, amb = self.flat_bins.pop(leaf_name)
            if os.path.exists(bin_path):
                with open(bin_path, "rb") as f:
                    f.seek(offset)
                    raw = np.fromfile(f, dtype=self.void_dtype, count=n_keys)
            else:
                raw = np.empty((0,), dtype=self.void_dtype)
            # The file is shared; deletion happens in bulk after Phase 2.
            return _NodeCapacityKeys(raw, amb, self.key_bytes)
        return self._empty_acc()

    def _merge_pair(
        self,
        running: "_NodeCapacityKeys",
        child_set: "_NodeCapacityKeys",
    ) -> "_NodeCapacityKeys":
        """Fold ``child_set`` into ``running`` by set-union, spilling as needed.

        Returns the merged accumulator; both inputs are released appropriately.
        """
        import numpy as np

        if running._on_disk:
            running._inplace_extend(child_set, self.key_bytes)
            child_set.release()
            return running
        if child_set._on_disk:
            spilled = _NodeCapacityKeys._spilled_from_arrays(
                [running._pure_keys],
                running._ambiguous_count,
                self.key_bytes,
                spill_dir=self.spill_dir,
            )
            running.release()
            spilled._inplace_extend(child_set, self.key_bytes)
            child_set.release()
            return spilled
        r_count = (
            running._pure_keys.shape[0]
            if running._pure_keys is not None else 0
        )
        c_count = (
            child_set._pure_keys.shape[0]
            if child_set._pure_keys is not None else 0
        )
        if r_count + c_count < self.disk_threshold:
            arrays = [
                a for a in [running._pure_keys, child_set._pure_keys]
                if a is not None and a.shape[0]
            ]
            pure_keys = (
                np.unique(np.concatenate(arrays))
                if arrays
                else np.empty((0,), dtype=self.void_dtype)
            )
            new_running = _NodeCapacityKeys(
                pure_keys,
                running._ambiguous_count + child_set._ambiguous_count,
                self.key_bytes,
            )
            running.release()
            child_set.release()
            return new_running
        spilled = _NodeCapacityKeys._spilled_merge(
            [running, child_set],
            running._ambiguous_count + child_set._ambiguous_count,
            self.key_bytes,
            spill_dir=self.spill_dir,
        )
        running.release()
        child_set.release()
        return spilled

    # ── cleanup ──────────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        """Remove the flat-bin file, leaf checkpoint and stale spill dirs."""
        import os

        # Phase 2 completed successfully — flat-bin file, checkpoint, and spill
        # dirs are no longer needed.  Clean up all so the spill_dir stays lean.
        if self.flat_bin_file is not None:
            try:
                os.remove(self.flat_bin_file)
            except OSError:
                pass
            self.flat_bin_file = None
        if self.spill_dir:
            _delete_leaf_checkpoint(self.spill_dir)
            _cleanup_spill_dirs(self.spill_dir)




def _leaf_pool_initializer(counter, lock, n_gpus: int) -> None:
    """Assign each worker a GPU device ID or mark it as CPU-only.

    Called once when a worker process starts.  The first ``n_gpus``
    workers to call this initializer receive CUDA device indices
    0 … n_gpus-1; subsequent workers are marked CPU-only
    (``_WORKER_GPU_DEVICE_ID = -1``).

    A shared counter protected by a lock guarantees exactly one worker
    per device regardless of process-start timing.
    """
    global _WORKER_GPU_DEVICE_ID
    with lock:
        idx = counter.value
        counter.value += 1
    device_id = idx if idx < n_gpus else -1
    _WORKER_GPU_DEVICE_ID = device_id
    if device_id >= 0:
        try:
            import cupy as cp
            cp.cuda.Device(device_id).use()
        except Exception:
            _WORKER_GPU_DEVICE_ID = -1


def _leaf_worker_task_auto(
    fasta_path: str,
    header_id: str,
    leaf_name,
    min_len: int,
    key_bytes: int,
    disk_threshold: int,
    spill_dir: str | None,
) -> tuple:
    """Dispatch a leaf to the GPU path or the CPU path.

    GPU path: used when this worker was assigned a CUDA device
    (``_WORKER_GPU_DEVICE_ID >= 0``) and the sequence is at least
    ``_GPU_MIN_BASES`` long.

    CPU path: used for short sequences, CPU-only workers, or when the
    GPU path raises any exception (OOM, driver error, …).  The returned
    tuple is identical in format to ``_leaf_worker_task``.
    """
    import numpy as np
    from taxotreeset.dataset.utils import _read_single_sequence

    device_id = _WORKER_GPU_DEVICE_ID

    if device_id >= 0:
        seq = _read_single_sequence(fasta_path, header_id) or ""
        if len(seq) >= _GPU_MIN_BASES:
            try:
                seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
                keys_void = _gpu_encode_unique(seq_arr, min_len, device_id, key_bytes)
                # For very large unique key sets spill to disk so the IPC pipe
                # doesn't carry gigabytes of raw bytes.  The threshold mirrors
                # the CPU path's disk_threshold parameter.
                if keys_void.shape[0] > disk_threshold:
                    acc = _NodeCapacityKeys._spilled_from_arrays(
                        [keys_void], 0, key_bytes, spill_dir=spill_dir,
                    )
                    bpaths = acc._bucket_paths
                    tdir = acc._tmp_dir
                    acc._bucket_paths = None
                    acc._tmp_dir = None
                    return (leaf_name, True, bpaths, 0, key_bytes, tdir)
                return (leaf_name, False, keys_void.tobytes(), 0, key_bytes, None)
            except Exception as exc:
                logger.warning(
                    "[bottom-up-gpu] leaf %s GPU encode failed (%s) — retrying on CPU",
                    leaf_name, exc,
                )

    return _leaf_worker_task(
        fasta_path, header_id, leaf_name, min_len, key_bytes, disk_threshold, spill_dir,
    )


