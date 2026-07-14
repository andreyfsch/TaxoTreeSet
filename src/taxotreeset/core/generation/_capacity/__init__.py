"""Private implementation package for ``capacity.py``.

Holds the capacity subsystem's building blocks — split out of the former
``capacity.py`` god-module: 2-bit encoding (``_encoding``), Bloom estimation
(``_bloom``), GPU kernels (``_gpu``), the packed-key accumulator (``_keys``),
prefix-bucket disk dedup (``_diskdedup``), leaf checkpoint/spill (``_spill``),
and the parallel bottom-up computer + pool workers (``_bottomup``). The public
face stays in ``taxotreeset.core.generation.capacity``, which re-exports what
callers and the tests reference (and remains the single patch-anchor namespace
for the I/O cache and thresholds).
"""
