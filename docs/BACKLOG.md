# Backlog

Improvement backlog distilled from a code-review analysis of the repository
(2026-06-18), ordered by methodological impact rather than effort. Scope is
TaxoTreeSet (dataset generation); inference/evaluation items that belong to the
downstream PhyloCascadeGLM project are listed separately at the end so they are
not lost.

Priority: 🔴 critical · 🟠 high · 🟡 medium · 🟢 low. Entries keep their original
numbers; the badge on each is its *current* priority — P1 was downgraded after a
2026-06-18 diagnostic (see its entry).

---

## 🟢 P1 — Phylogenetic redundancy across splits (largely N/A for RefSeq)

**Status (2026-06-18 diagnostic): not a material problem for the current
pipeline.** Measured per class of every trained head, the unique-k-mer content
tracks the raw sequence volume (`capacity / Σ lengths` ≈ 0.74–0.99) and the clone
factor (`genomes / (capacity / median genome length)`) ≈ 1.0 — i.e. the genomes
are genuinely diverse, not near-clones. SARS-CoV-2 (taxid 2697049) has a single
genome; species classes hold 1–3 genomes, kingdom classes hold thousands but
they are distinct *species*. Exact subsequence overlap between train and test is
also ~0%. **Root cause:** TaxoTreeSet ingests **RefSeq reference assemblies**
(~1 curated genome per species), so the epidemiological oversampling this item
worried about (GenBank-style strain collections) is preempted at the source.

**Conclusion.** The pilot test-F1 (Viruses 0.729, Orthornavirae 0.648,
Bamfordvirae 0.897) are **not** inflated by near-clone memorization. The
diagnostic is reproducible from the registry + manifest (per-class `capacity`
vs. summed genome length); a genome-level Mash/MinHash check would confirm it but
is unnecessary given the RefSeq curation.

**Original concern (now a guard-rail for a future GenBank expansion).** The
whole-genome split prevents window leakage *within* a genome but not near-
identical strains spanning train/test, and exact-dedup `capacity` would count
such near-clones as diversity — inflating both the download budget and
`n_per_class`. This only bites with non-RefSeq (GenBank) data. If that expansion
happens, cluster genomes by similarity (Mash/MinHash, or a CD-HIT identity
cutoff) and assign *whole clusters* to one split, and reconsider whether
`capacity` should reflect genuine diversity rather than raw volume. Couple with
the molecule-type/scope work (P3, `data_scope_questions.md` item 1).

**Effort (only if/when GenBank).** High — new dependency, all-vs-all compute,
touches the split logic and the capacity metric.

Files: `dataset/sequence_utils.py` (split), `core/generation/capacity.py`,
`core/generation/balancing.py`.

---

## 🟢 P2 — Reproducible NCBI snapshot (capture done; re-fetch pinning pending)

**Status (2026-06-18): implemented (capture/export).** Each generate run now
writes a `provenance` block in `run_metadata_<group>.json` (NCBI `datasets` CLI
version, taxoniq version, Python/platform, registry `last_update`) plus a full
`accession_snapshot_<group>.json` — the sorted versioned accessions (immutable
`GCF_….N`) and a SHA-256 digest. The digest is a citable snapshot ID; the
accession list is the manifest to re-fetch. (`NCBIRegistry.accession_snapshot`,
`_capture_tool_versions`.)

**Problem (recap).** `discover` captures "the current state of RefSeq", a moving
target, so without this a run is not citable/reproducible.

**Remaining (follow-up).** Active **pinning**: a path that re-fetches exactly the
accessions in a saved snapshot (instead of re-querying NCBI for the current set)
so a benchmark reproduces byte-for-byte. The capture above already enables manual
reproduction.

Files: `io/registry.py`, `core/generation_orchestrator.py`.

---

## 🟢 P3 — Molecule-type filter (plasmids) — implemented (opt-in, heuristic)

**Status (2026-06-18): implemented.** `generate --exclude-plasmids` drops
plasmid sequences at ingestion, matched heuristically from the FASTA defline, so
they never enter the vault nor become training leaves. Off by default (no effect
on viruses). Plumbed CLI -> `GenerationOrchestrator` -> `NCBIDownloader`
(`_is_excluded_molecule` / `_drop_excluded_molecules`). An all-plasmid accession
is still marked processed (not retried) via a None-vs-empty-list ingestion
contract in `_ingest_accession_fasta`.

**Problem (recap).** Plasmids are horizontally transferred across distant taxa
and carry little reliable host phylogenetic signal; mixing them with chromosomal
sequences adds noise. Larger effect for Bacteria than viruses.

**Remaining / upgrades.**
- Authoritative signal: `datasets download --include seq-report` +
  `assigned_molecule_location_type` instead of defline text (more robust;
  moderate). Also lets the keyword set extend to organelles.
- `total_sequence_length` (per accession) still includes plasmids (it comes from
  the discover-time summary), so selective-download volume estimates slightly
  overcount after filtering — safe direction; optional recompute.
- Ingestion-time only: pre-existing downloads are not retro-filtered (re-download
  to change). Fine for the not-yet-fetched Bacteria scope.

Files: `io/downloader.py`, `core/generation_orchestrator.py`, `cli/generate.py`.

---

## 🟠 P4 — OOD routing: reject class (done, intra-virus) + validation + non-virus gate

**Update (2026-06-21).** This was the decisive finding of the pilot inference work.
Closed-set heads **confidently mis-accept** out-of-subtree inputs (a SARS-CoV-2
sequence forced down the wrong Bamfordvirae subtree scored 0.94–1.00 at every
level), so uncertainty alone cannot route a foreign input to a fallback. The fix —
an explicit **reject class** of out-of-subtree negatives per head — is now
implemented (`generate --reject-class`; near siblings + far clades) and validated
on one head (rejects near 97% / far 99%, 7% false-reject, ~0 accuracy cost). The
downstream **dominant-reject termination** lives in PhyloCascadeGLM's `_traverser.py`.

**Remaining.**
- **Validate end-to-end** on the reject-trained pilot (held-out-taxa protocol:
  exclude whole taxa from `discover`, confirm the cascade rejects them).
- **Phase 2 — non-virus domain gate:** the intra-virus reject covers "not in this
  clade", not "not a virus at all". The root/shallow heads need **non-virus
  negatives** (Bacteria/Archaea/Eukaryota RefSeq) — a new cross-domain sampling
  capability in generation (discover/download outside Viruses).

**Effort.** Reject class: done. Non-virus gate: moderate (new data acquisition).

Files: `core/generation/reject_bucket.py`, `core/generation_orchestrator.py`.

---

## 🟡 P5 — Preserve signal: train-time rebalancing vs definitive undersampling

**Problem.** `n_per_class = min(sibling capacities)` discards data from rich
clades. The percentile cutoff (`--cutoff-percentage`) and `--max-n-per-class`
mitigate the extremes, but balancing is still lossy by construction.

**Approach.** Offer keeping richer datasets and rebalancing at fine-tuning time
(class weights, or oversampling with the existing reverse-complement
augmentation in `sequence_utils.py`) instead of definitive subsampling on disk.
Trade-off: this breaks the "balanced, model-agnostic on disk" property, so it
should be opt-in. Relevant for the HoreKa scenario (more tokens available).

**Effort.** Moderate. Files: `core/generation/balancing.py`,
`dataset/sequence_utils.py`.

---

## 🟢 P6 — Audit length / compositional confound in virtual buckets

**Problem.** Virtual buckets aggregate taxonomically heterogeneous taxa, which
may be separable for non-phylogenetic reasons. (The "sequence length as a
shortcut" framing is weak — the model sees bounded windows, not whole-genome
length; the real risk is *compositional* artifacts in the virtual classes.)

**Approach.** Cheap diagnostic: audit the per-class length and composition
distributions within a head, especially for virtual classes, and check they are
comparable to the canonical classes.

**Effort.** Low (analysis only).

---

## 🟡 P7 — Decompose `capacity.py` / `generation_orchestrator.py`

**Problem.** `capacity.py` and the generation orchestrator are large relative to
the "modular pipeline" framing, hindering granular review and testing.

**Approach.** Split into submodules mirroring the glossary concepts (exact vs
approximate/Bloom capacity, GPU encoding, disk spill, checkpointing).

**Progress.**
- **Part A — done (2026-06-22).** Conservative extraction of the pure, patch-free
  helper groups into submodules, re-exported from `capacity.py` so every existing
  import/patch keeps working with zero test edits: `_encoding.py` (2-bit packing
  LUT + window packer), `_bloom.py` (filter sizing + sliding-window insertion +
  bit ops), `_gpu.py` (CUDA detection + GPU encode/dedup kernels). `capacity.py`
  2422 → 1893 lines. Earlier work also extracted `_BottomUpCapacityComputer` from
  `compute_all_capacities` and split `run_preflight` into helpers.
- **Part B — done (2026-06-22).** Test groundwork for the *later* full split: a
  real-LMDB vault fixture (`tests/unit/_vault_fixture.py`) + behavioral tests
  (`test_capacity_behavioral.py`) that drive `_capacity_exact`,
  `_capacity_approximate`, `compute_node_capacity` through a genuine temp vault
  with **no monkeypatching**, so they survive the I/O core moving out of
  `capacity.py`.
- **Part C — in progress (2026-07-13).** Extracted the key-machinery layer, keeping
  the I/O cache (`_read_sequence_cached` / `_SEQUENCE_CACHE`) and the `_HASHED_*`
  thresholds IN `capacity.py` as the patch anchors, so **no test needed editing**
  (912 pass, ruff clean). New submodules, all re-exported from `capacity.py`:
  - `_diskdedup.py` — the pure prefix-bucket machinery (`_bucket_writer_paths`,
    `_count_unique_bucketed_on_disk`, `_flush_keys_to_buckets`, `_compact_pure_keys`,
    `_cleanup_key_buckets`). `_open_key_buckets` stays in `capacity.py` so both its
    and `_capacity_exact`'s flush calls resolve through the one patchable namespace.
  - `_spill.py` — checkpoint/spill (`_save`/`_load`/`_delete_leaf_checkpoint`,
    `_cleanup_spill_dirs`, `_LEAF_CHECKPOINT_FNAME`); `_load` lazily imports
    `_NodeCapacityKeys` to avoid a cycle.
  - `_keys.py` — `_NodeCapacityKeys` (the 445-line accumulator); its sequence reads
    lazily `from …capacity import _read_sequence_cached`, which breaks the cycle AND
    keeps `patch("capacity._read_sequence_cached")` working (verified by the 7 patch
    tests).
  `capacity.py` **1893 → 1179** so far.
  - **Still remaining.** Move `_BottomUpCapacityComputer` (~470 lines) + the pool
    worker tasks (`_leaf_worker_task[_auto]`, `_leaf_pool_initializer`,
    `_reconstruct_leaf_keys`, `_WORKER_GPU_DEVICE_ID`) into `_bottomup.py`. CAREFUL:
    multiprocessing pickles workers by qualified name and `_leaf_pool_initializer`
    sets a module-global GPU device id read by the tasks — initializer + tasks must
    land in the SAME module. Not covered by a real-pool test, so higher risk.

**Effort.** Remaining `_bottomup.py` move moderate but multiprocessing-sensitive;
do it as its own verified step.

---

## Cross-repo — PhyloCascadeGLM

Inference/evaluation items live in the PhyloCascadeGLM repo's own `docs/BACKLOG.md`.
As of 2026-06-21 the decision policy is implemented there (depth-normalized "mean"
ranking + dominant-reject termination); what remains is end-to-end validation on
the reject-trained pilot, theta/temperature calibration, the two weak heads
(kingdom; collapsed `694009`), and a hierarchy-aware eval metric. The non-virus
domain gate (P4 Phase 2) is the TaxoTreeSet-side follow-up.
