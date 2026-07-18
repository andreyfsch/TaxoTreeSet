"""Private implementation package for ``generation_orchestrator.py``.

Holds the generation orchestrator's building blocks — split out of the former
``generation_orchestrator.py`` god-class: leaf-level train/val/test splitting
(``_splits``), manifest / label-map / run-metadata writers (``_manifest``), the
Stage-1 sync + selective-download manager (``_sync``), and the recursive cascade
scheduler (``_scheduler``). The public face stays in
``taxotreeset.core.generation_orchestrator.GenerationOrchestrator``, which
composes these parts and remains the import path and patch/test anchor.
"""
