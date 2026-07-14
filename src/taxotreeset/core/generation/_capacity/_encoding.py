"""2-bit packing of ACGT sliding windows (capacity encoding primitives).

Pure helpers extracted from ``capacity.py``: the ACGT→2-bit lookup table and
the window packer used by the exact-capacity path (and the GPU path). Windows
holding any non-ACGT symbol (IUPAC ambiguity codes or N) are not packed — the
caller tracks those in an exact string set, keeping the two domains disjoint so
their unique counts sum without double counting. numpy is imported lazily inside
the functions, matching the rest of the generation subpackage.
"""

# Exact-hashed encoding: each ACGT base packs into 2 bits, 4 bases per byte.
# The packed key length in bytes is derived from min_len at call time as
# ceil(min_len / _BASES_PER_BYTE), so the method holds for any window size.
_BASES_PER_BYTE: int = 4

# Packed keys are bucketed for disk/CPU dedup by their first packed byte (the
# first four 2-bit bases) — 256 possible values. Shared by the exact disk-spill
# path (capacity.py) and the GPU CPU-fallback (_gpu.py); lives here, with the
# packing, to keep both importers free of a circular dependency.
_HASHED_PREFIX_BUCKETS: int = 256

_ACGT_LUT_CACHE = None


def _get_acgt_lut():
    """Return the cached 256-entry ACGT-to-2-bit lookup table, building it once.

    Non-ACGT bytes map to the sentinel 255, which marks a window as
    ambiguous (containing IUPAC ambiguity codes or N) so it is routed to
    the exact string-set path rather than the 2-bit-packed path. The table
    is built lazily on first use because numpy is imported lazily within
    this module.

    Returns:
        numpy uint8 array of length 256; index by ASCII byte value.
    """
    global _ACGT_LUT_CACHE
    if _ACGT_LUT_CACHE is None:
        import numpy as np

        lut = np.full(256, 255, dtype=np.uint8)
        for code, base in enumerate(b"ACGT"):
            lut[base] = code
        _ACGT_LUT_CACHE = lut
    return _ACGT_LUT_CACHE


def _encode_windows_2bit(windows, min_len: int):
    """Pack pure-ACGT sliding windows into fixed-length 2-bit byte keys.

    Each base occupies 2 bits and four bases pack into one byte, so a
    window of ``min_len`` bases packs into ceil(min_len / 4) bytes. The
    key length is derived from ``min_len`` here, so the encoding is valid
    for any window size, not just the default of 100.

    Windows containing any non-ACGT symbol are not encoded; the returned
    boolean mask marks which windows were pure ACGT. Ambiguous windows are
    handled separately by the caller via an exact string set, keeping the
    two domains disjoint so their unique counts sum without double counting.

    Args:
        windows: (N, min_len) uint8 array of ASCII base values, typically
            a ``sliding_window_view`` over one sequence.
        min_len: Window size in bases; determines the packed key length.

    Returns:
        Two-tuple ``(packed_keys, pure_mask)``:
            - packed_keys: (M,) array of void-typed keys of width
              ceil(min_len / 4) bytes, one per pure-ACGT window (M <= N).
            - pure_mask: (N,) boolean array, True where the window was
              pure ACGT.
    """
    import numpy as np

    codes = _get_acgt_lut()[windows]
    pure_mask = np.all(codes != np.uint8(255), axis=1)
    pure = codes[pure_mask]
    n_pure = pure.shape[0]
    key_bytes = (min_len + _BASES_PER_BYTE - 1) // _BASES_PER_BYTE
    if n_pure == 0:
        empty = np.empty((0,), dtype=np.dtype((np.void, key_bytes)))
        return empty, pure_mask

    # Pad the base axis up to a multiple of 4 with zeros so it reshapes
    # cleanly into groups of four 2-bit codes. The padding is identical for
    # every window, so it cannot introduce a spurious collision.
    pad = (-min_len) % _BASES_PER_BYTE
    if pad:
        pure = np.concatenate(
            [pure, np.zeros((n_pure, pad), dtype=np.uint8)], axis=1
        )
    groups = pure.reshape(n_pure, key_bytes, _BASES_PER_BYTE)
    packed = (
        groups[:, :, 0]
        | (groups[:, :, 1] << np.uint8(2))
        | (groups[:, :, 2] << np.uint8(4))
        | (groups[:, :, 3] << np.uint8(6))
    ).astype(np.uint8)
    keys = np.ascontiguousarray(packed).view(
        np.dtype((np.void, key_bytes))
    ).reshape(n_pure)
    return keys, pure_mask
