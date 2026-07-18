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
  - `_bottomup.py` (2026-07-14) — `_BottomUpCapacityComputer` + the pool workers
    (`_leaf_worker_task[_auto]`, `_leaf_pool_initializer`, `_reconstruct_leaf_keys`)
    + the `_WORKER_GPU_DEVICE_ID` global, all in ONE module (spawn pickles workers
    by qualified name and the initializer sets the device id the workers read).
    `_resolve_bottom_up_threshold` stays in `capacity.py` (patch anchor) and is
    imported lazily. First closed the test-net gap with a parallel-vs-serial
    value test (`compute_all_capacities` n_workers=2 == n_workers=1 == ground
    truth over a real vault) — the spawn workers can't be mocked, so this guards
    the pickle/global. Only 2 tests needed a patch-path repoint
    (`capacity._leaf_worker_task` → `_bottomup._leaf_worker_task`).

  **`capacity.py` decomposition DONE:** **2422 → 541** (−78%); cohesive submodules
  `_bottomup` (668), `_keys` (476), `_bloom` (249), `_gpu` (245), `_spill` (200),
  `_diskdedup` (121), `_encoding` (102). Suite 921, ruff clean.

- **Orchestrator half — done (2026-07-18).** Decomposed `generation_orchestrator.py`
  **2553 → 819** into a private subpackage `core/_orchestration/` (mirrors
  `_capacity/`), in 4 staged, suite-green, byte-identical commits: `_splits.py` (leaf
  train/val/test split helpers — also dropped the dead `_prepare_stratified_split`
  trio), `_manifest.py` (label-map / run-metadata / artifact writers via a `ctx`
  handle), `_sync.py` (`_SyncManager` — Stage-1 sync / selective-download /
  refinement), `_scheduler.py` (`_CascadeScheduler` + module-func tree helpers — the
  recursive cascade). Same playbook as capacity: the orchestrator stays the public
  face + patch/test anchor and keeps thin delegators for every private method the
  tests call on the instance; extracted code reads config via `ctx=self`. The
  behavioral net was the pre-existing `tests/integration/test_synthetic_pipeline.py`
  (end-to-end `run_pipeline` → parquet + `label_map` contracts). Suite 957, ruff clean.

**P7 DONE** — both `capacity.py` and `generation_orchestrator.py` decomposed; no
`src/` module now exceeds ~920 lines.

## 🟢 P8 — Extraction parallelism (HoreKa)

**Problem.** Parquet extraction (Stage 3/4) pooled one worker per HEAD (`cpu-2`
workers), tasks within a head serial — so a head with many source genomes was a
single-worker straggler that idled cores at the end of each batch.

**Done (2026-07-13) — task-level sharding.** `build_node_dataset` now fans each
head's per-split tasks into work-balanced shards (`_partition_tasks` /
`_plan_shards`), pools the shard-jobs (`_shard_worker` → `<split>.part*.parquet`),
then a merge pass (`_merge_worker`) row-group-concatenates the parts into
`<split>.parquet`. Crash-safe (`.tmp` + atomic rename) and resumable at the split,
shard, and merge level. Orchestrator + downstream readers unchanged (single file
per split preserved). Correct because the subseq sampling is order-independent →
total rows per split/class invariant under sharding. New `_SHARD_ROWS_TARGET`
(50k). Tests: `tests/unit/test_builder_sharding.py` (8, incl. a real spawn-pool
run) over the P7-Part-B `_vault_fixture`.

**Deferred follow-ups (independent, additive; today's change is their prerequisite).**
- **Intra-genome chunking** for eukaryotes: split one huge genome's `n` across
  fraction ranges so a single giant genome isn't an unsplittable straggler. The
  real memory/I/O win needs a **storage-layout change** — genomes stored as
  independently-decompressible blocks (chunked LMDB records or BGZF) + a ranged
  reader; zlib whole-genome blobs have no random access. Reuse the streaming-read
  pattern of `_from_chunked_sequence` (now in `core/generation/_keys.py`). NOT the
  LMDB engine itself — just the value layout + reader.
- **`--shard i/N`** multi-node partition of the head set (SLURM job array) — heads
  are independent; the biggest raw win on HoreKa.
- **Vectorize `extract_subseqs`** — the per-subseq Python `rng.randint`+slice loop
  is the per-core CPU cost.

---

## 🟠 P9 — Multi-root scope + plasmid datasets (host-taxonomy, tool-free)

**Goal.** Parametrize several scopes as roots (e.g. `--root Viruses,Plasmids`) and,
specifically, let TaxoTreeSet build datasets for **plasmid recognition in isolation,
without downloading Bacteria as a whole**.

**Why it isn't covered.** `--root` resolves to one TaxID and walks that taxonomic
subtree. "Plasmids" is **not a taxon** — there is no NCBI TaxID that roots all
plasmids; each plasmid record is taxonomically assigned to its **host** organism. So
a plasmid scope needs a different acquisition path and a non-taxonomic (or
host-taxonomic) labeling. There is also no 100%-viral or plasmid-only benchmark in
CAMI — CAMI datasets are mixed metagenomes by environment, with viruses and plasmids
bundled as a minority ("plasmids and viruses" / "circular elements"), which is why a
purpose-built viral/plasmid generator is needed rather than reusing a CAMI track.

**Architecture (decided).** The binary belongs/not-belongs methodology already gives
this for free: `--root all` + `--binary-only` is exactly "empty virtual root → each
top-level child is a binary head (positive = its subtree, not-belongs = out-of-subtree
near/far windows)". So:
- **Empty root** with top-level children `{Viruses, Plasmids}` — two otherwise-isolated
  subtrees joined only by the empty root.
- **"Virus vs not-virus"** = the Viruses node's binary head (its not-belongs already
  includes plasmid windows, being outside the viral subtree); **"plasmid vs
  not-plasmid"** = the Plasmids node's binary head. No special virus-vs-plasmid
  discriminator is needed.
- Cross-domain reject negatives at shallow heads only **if leakage is observed** later
  (deferred; ties to P4's domain gate). The field's canonical plasmid negative is
  *chromosome*, not virus — if real metagenomic plasmid detection becomes a target,
  chromosome negatives enter here (P4).

**Plasmid-branch labeling (decided): host taxonomy — tool-free (v1).** Every RefSeq
plasmid record already carries its host organism's TaxID, and the existing lineage
machinery (taxoniq → lineage, with the NCBI fallback) turns that into a tree, built by
the **same cascade code**. Only the hosts that *have* plasmids are materialized (a
sparse subset), so "without Bacteria as a whole" holds — no full-kingdom crawl. Task
framing: this is **host prediction** for plasmids (place a plasmid in its host's tree),
not intrinsic plasmid typing.

Rejected/deferred alternatives (the field's plasmid schemes all need a reference/tool):
- Recognition (plasmid vs chromosome) is binary k-mer/ML (PlasClass, PlasFlow, Platon,
  Deeplasmid, …) — matches our method but is the *root binary head*, not a branch label.
- Intrinsic typing — replicon/Inc (PlasmidFinder), MOB (MOB-suite, MOBFinder [LM-based]),
  or **PTU (COPLA)** — is the real "plasmid taxonomy" but each is defined by its
  reference DB. **Future upgrade (option 2):** ingest a **precomputed** PTU/replicon
  table (PLSDB/COPLA publish assignments) as a static data input (like `mapping.json`),
  giving intrinsic typing with **no runtime tool dependency**. Reference-free de-novo
  k-mer clustering (mge-cluster style) is a last resort — self-contained but produces
  uninterpretable cluster labels + new ML.

**New architectural pieces (the actual work).**
1. **Multi-root plumbing** — `--root` accepts a list; the empty-root forest schedules
   each top-level child. Small (generalizes the existing `all` empty-root path).
2. **Plasmid acquisition** — pull from the RefSeq `plasmid` division (a curated plasmid
   collection) directly, reusing the P3 `--exclude-plasmids` molecule detector to
   *select* plasmids at ingestion. No full-Bacteria download.
3. **Accession-set-driven discovery entry point** — plasmids have no root taxon to walk
   top-down, so discovery starts from the plasmid **accession set** (bottom-up: resolve
   each record's host lineage, assemble the host subtree). Reuses lineage resolution +
   the cascade; the new part is the entry point, not the tree/head logic.

**Effort.** Moderate. Multi-root is small; the plasmid path (RefSeq plasmid acquisition
+ accession-driven discovery) is the bulk; host-lineage labeling reuses existing code.

Files: `cli/generate.py` (`--root` list), `core/orchestrator.py` (accession-set-driven
discovery entry), `taxonomy.py` (resolve multiple roots), `io/downloader.py` (RefSeq
plasmid acquisition; reuse `_is_excluded_molecule`), `core/generation_orchestrator.py`
(empty-root forest / multi-scope scheduling). Related: P3 (plasmid detection primitive),
P4 (cross-domain negatives), the `--binary-only` machinery.

---

## Cross-repo — PhyloCascadeGLM

Inference/evaluation items live in the PhyloCascadeGLM repo's own `docs/BACKLOG.md`.
As of 2026-06-21 the decision policy is implemented there (depth-normalized "mean"
ranking + dominant-reject termination); what remains is end-to-end validation on
the reject-trained pilot, theta/temperature calibration, the two weak heads
(kingdom; collapsed `694009`), and a hierarchy-aware eval metric. The non-virus
domain gate (P4 Phase 2) is the TaxoTreeSet-side follow-up.
