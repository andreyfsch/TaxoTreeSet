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

## 🟡 P5 — Preserve signal: train-time rebalancing vs definitive undersampling — DONE

**Problem.** `n_per_class = min(sibling capacities)` discards data from rich
clades. The percentile cutoff (`--cutoff-percentage`) and `--max-n-per-class`
mitigate the extremes, but balancing is still lossy by construction.

**Status (2026-07-18): implemented (opt-in `--keep-imbalance`).** With the flag,
each class keeps up to its OWN capacity (capped by `--max-n-per-class`) instead of
being undersampled to the sibling minimum: the scheduler builds a per-child target
`min(capacity, max_n_per_class)` (falling back to the balanced `n_per_class` for
classes with no recorded capacity, e.g. virtual buckets) and passes it to
`distribute_n_per_class_across_leaves` via a new backward-compatible `per_child_n`
override. The on-disk dataset is then imbalanced, so each head's `label_map.json`
records `balance_mode` (`"keep"`/`"undersample"`), per-class `n_windows`, and
suggested `class_weights` (sklearn "balanced": `total / (n_classes * n_c)`) so a
trainer can offset the imbalance with class weights or oversampling (reusing the
reverse-complement augmentation in `sequence_utils.py`). Off by default — the
dataset stays balanced and model-agnostic. Tests: distribution `per_child_n` +
label-map metadata; the default (undersample) path is unchanged (integration green).

**Effort.** Moderate — done. Files: `cli/generate.py`,
`core/generation_orchestrator.py`, `core/_orchestration/_scheduler.py`,
`core/_orchestration/_manifest.py`, `core/generation/distribution.py`.
**Deferred:** actual train-time oversampling helper (this ships the *metadata*; the
trainer applies the weights). The `class_weights` are a convenience — a trainer can
also read per-class row counts straight from the parquet.

---

## 🟢 P6 — Audit length / compositional confound in virtual buckets — DONE

**Problem.** Virtual buckets aggregate taxonomically heterogeneous taxa, which
may be separable for non-phylogenetic reasons. (The "sequence length as a
shortcut" framing is weak — the model sees bounded windows, not whole-genome
length; the real risk is *compositional* artifacts in the virtual classes.)

**Status (2026-07-18): implemented.** New `dataset/composition.py` + a
`taxotreeset composition <dataset_dir>` subcommand (numpy-only, no sklearn).
`audit_head` groups a split's rows by class and reports per-class length +
nucleotide composition (mean/std GC, A/C/G/T fractions), then compares each
**virtual** class (`rank` starts with `virtual_`) against the head's canonical
classes: a GC **z-score** when there are >= 2 canonical classes, else a raw GC
**gap** for binary / single-canonical heads. Virtual classes past the threshold
(`|z| > 2` or `|gap| > 0.05`) are flagged as possible non-phylogenetic
separators. `survey_dataset` walks all heads, writes a compact
`composition_audit` summary into each `label_map.json` (atomic, like the
separability diagnostic) and returns aggregate rows (`--csv` to export). Length
is reported as a sanity check on the length-confound fix (windows drawn uniformly
in [min_len, max_len] regardless of class → per-class lengths should match).
Tests: `tests/unit/test_composition.py` (13). **Interpretation is still manual** —
the tool surfaces flagged heads; deciding whether a flagged virtual class is a
genuine confound is a human call.

**Effort.** Low (analysis only) — done.

Files: `dataset/composition.py`, `cli/composition.py`, `__main__.py`.

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
1. **Multi-root plumbing — DONE (2026-07-18).** `--root` accepts a comma-separated
   list. `_resolve_scope_taxids` resolves it to a frozenset of TaxIDs (or None for
   `all`, which can't be combined); `_scope_anchor` keeps a one-domain scope anchored
   at its node (unchanged) while `all`/multi anchor at the empty root. `_build_target_tree`
   builds several domains by calling the **unchanged** per-domain tree builder once each
   (own scope config / anchoring / redirections) and grafting their anchors under one
   empty `root` — the scheduler reuses its empty-root (`None`) path untouched, and
   `tree_builder.py` is not modified. Sync discovers each requested domain. Tests:
   14 in `test_generation_orchestrator.py` (resolution, anchor, forest merge); the
   single-root path is exercised end-to-end by the synthetic-pipeline integration test.
   NOTE this needs REAL taxa on both sides — `--root Viruses,Plasmids` still awaits the
   plasmid acquisition (pieces 2-3 below), since "Plasmids" has no TaxID yet.
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

## 🟢 P10 — Cluster-aware split (non-i.i.d. genomes across train/val/test) — DONE

**Problem (diagnosed on the pilot, 2026-07-19).** Head `1335638` (a deep binary
belongs/not-belongs head) trained to **test f1 0.98** but its **eval f1 peaked at
0.75 then degraded to 0.49** (eval_loss rising) → early stop. Both val and test are
large + balanced (~6000 rows), so it is not a small-sample artifact. A k-mer LR
baseline reproduces it (VAL 0.66 vs TEST 0.83, **model-agnostic**), and the tell is
**belongs recall 0.47 on val vs 0.98 on test**: the val "belongs" genomes are a
distinct **sub-lineage** the model never trained on. **Root cause:** the random
genome-level split (`_materialize_leaf_split`) treats genomes as i.i.d., but a
clade's genomes are phylogenetically clustered — a whole sub-lineage can land in
val, so val (unseen lineage) tanks while test (train-like lineages) looks great.
This is P1 in reverse: P1 feared clustering *inflating* test via near-clone leakage;
here it *deflated* val. **General lesson: test ≫ val on a deep head is a non-i.i.d.
split, not model quality — the val↔test gap is the signal, and a single per-head F1
is untrustworthy.**

**Design (decided with the user): hybrid + MinHash, conditional + per-class.** The
clustering self-verifies the need: cluster a class's genomes (MinHash, tool-free)
and only act on >= 2 well-separated clusters (else keep the random split) — applied
to positives and negatives independently (negatives can also use the free near/far
reject tag). Two philosophies combined: stratify clusters across train/val/test for
STABLE, representative metrics, plus a disjoint holdout for HONEST novel-lineage
generalization.

**Phase 1 — DONE (2026-07-19): conditional cluster-STRATIFIED split (opt-in).** New
`core/_orchestration/_cluster.py`: `cluster_genomes` sketches each genome with a
bottom-k MinHash (stdlib `zlib.crc32` over k-mers — no external tool), single-linkage
clusters by the KMV Jaccard estimate, and returns clusters only when actionable
(>= 2 clusters, the two largest each >= `min_cluster_genomes`). `_materialize_leaf_split`
gains an opt-in `cluster_aware` path (flag `--cluster-aware-split`): it spreads each
cluster across train/val/test (small clusters → train), and **falls back to the
random split if there is no structure OR the cluster split would empty a split** (so
the >= 1-per-split guarantee always holds). Off by default → byte-identical.
Self-verifying, so homogeneous heads pay nothing. Params (k=21, sketch=200,
threshold=0.30, min_cluster_genomes=2, max_genomes=300 for the O(n^2) cap) are module
constants — **need tuning on real data** (validate on `1335638`). Tests:
`tests/unit/test_cluster.py` (15). Threading: `cli/generate.py` →
`GenerationOrchestrator.cluster_aware_split` → the `_materialize_leaf_split` delegator.

**Phase 1b — DONE (2026-07-19): block-stratified window slicing (2nd mechanism).**
Validating Phase 1 on `1335638` exposed a *distinct* mechanism: that head has ONE
belongs genome, so it uses the window-slicing path (not the multi-genome clustering
Phase 1 fixes). Its contiguous cut (train 0-70% / val 70-85% / test 85-100%) put
compositionally-distinct genome thirds in different splits (GC 0.356 / 0.407 / 0.316)
→ val diverges from train on the same genome. Under the SAME `--cluster-aware-split`
flag, the window-slicing branch now cuts the genome into `L // max_subseq_len` blocks
(each >= max_subseq_len, so windows keep full length — no length confound) and assigns
them by an interleaving pattern (~5:1:1) that spreads val/test blocks AMONG train
blocks; windows stay confined to their block (leakage-safe). Falls back to the
contiguous cut when unreadable or too short (< 6 blocks). Off by default →
byte-identical. Validated on NC_021333.1: per-split GC 0.356/0.407/0.316 → 0.359/0.345/
0.360 (val/test now match train). So P10 addresses **two mechanisms**: multi-genome
sub-lineage (Phase 1) and single-genome positional (Phase 1b).

**Phase 1 validated on a multi-genome head (2732529, 146 belongs genomes, the biggest
val/test gap among trained heads): its genomes are DIVERSE, not clustered** (largest
MinHash cluster is 4 even at Jaccard 0.10) — as expected for RefSeq (~1 genome/species,
P1's rationale). So the multi-genome mechanism is largely a **no-op on RefSeq**; it would
pay off on GenBank-style strain collections. The gate was tightened accordingly (an
actionable cluster must cover >= 10% of the genomes, >= 2 such) so diverse heads return
None cleanly instead of clustering + falling back. Net: **Phase 1b (single-genome
positional) is the mechanism that actually helps this dataset; Phase 1 is a correct but
mostly-dormant guard for future GenBank data.** The 2732529 gap is therefore NOT
sub-lineage — likely the near/far negatives (unaddressed) or normal variance.

**Single-head regeneration — DONE (2026-07-19): `--single-level <taxid>`.**
Regenerating one existing head (to swap a fixed split into the training queue)
needs the head's negatives sampled from *outside* its subtree — for binary
(not-belongs) AND multi-class (reject bucket) alike. `sample_reject_leaves` reads
`node.root`, so the pool is defined by the **tree** (`--root`), not by which head
is scheduled: scoping `--root <taxid>` to the target empties the pool and breaks
the head. Fix: `--single-level` now takes an optional TaxID — keep `--root
<ancestor>` (e.g. `viruses`, with `--no-sync`) so the whole tree is built, and the
flag schedules only that one head. Binary path filters the descendant node list;
multi path locates the node and schedules its decision point directly (rebuilding
the accumulated path from the ancestry). Same tree + same reject pool + same
per-head seed → a byte-identical drop-in except for whatever `--cluster-aware-split`
changes. Files: `_scheduler.py`, `generation_orchestrator.py`, `cli/generate.py`,
`_manifest.py`. Tested: binary drop-in parity vs a full run, out-of-subtree
negatives retained, multi interior-node targeting.

**Regeneration verified on the two single-genome pilot heads (2026-07-19).** Of
the 14 binary pilot heads, only `1335638` and `2739681` take the block-stratified
window path (1 belongs genome each; the other 12 are diverse multi-genome → Phase 1
returns None). Regenerated both via `--root viruses --single-level <taxid>
--cluster-aware-split --no-sync`:
- `1335638`: belongs-GC spread **0.095 → 0.035** (0.316/0.411/0.316 → 0.362/0.327/
  0.346). Real composition shift, materially fixed → **replace + retrain**.
- `2739681`: spread already **0.035** (0.570/0.594/0.605); regen 0.033. The path
  fired but the genome is compositionally uniform, so there was no shift to fix —
  the windows moved (interior blocks, leakage-safe) but the distribution is
  unchanged. Replacing is optional (robustness insurance, not a metric fix).
So "fires the mechanism" != "meaningfully changes the split": `1335638` was the
only genuinely broken head.

**MinHash params as flags — DONE (2026-07-20).** The clustering knobs were
hardcoded constants, but the split rarely fires on RefSeq (diverse genomes), so
tuning matters for denser data. New frozen `ClusterParams` value object (defaults
= the old constants) threaded CLI -> orchestrator -> `_materialize_leaf_split` ->
`_cluster_stratified_split` -> `cluster_genomes`. Three decision knobs exposed:
`--cluster-jaccard-threshold`, `--cluster-min-genomes`, `--cluster-min-frac`
(k / sketch_size / max_genomes stay as ClusterParams defaults). Run metadata now
records `cluster_aware_split` + `cluster_params`.

**Default flipped ON — DONE (2026-07-20).** The cluster-aware split is now the
**default** (`--no-cluster-aware-split` opts out), because the plain split has a
known latent confound (the single-genome contiguous cut — 1335638) and the
cluster-aware path is self-verifying (falls back where genomes aren't clustered,
so it is never *wrong*, only occasionally unnecessary). To keep the common
single-genome case cheap, the block-stratified path no longer re-reads the genome
for its length: `distribute_n_per_class_across_leaves` attaches `length` to each
task, recovered for free from the share weight (`weight = len - min_subseq_len + 1`),
and `_block_stratified_windows` reads `task["length"]` (falling back to a read only
for tasks without it, e.g. reject-bucket negatives). Only the multi-genome MinHash
clustering still reads sequences (bounded, `max_genomes` cap). Note: default-on is
NOT byte-identical to prior runs. The orchestrator/CLI default is True; the pure
`_materialize_leaf_split` helper keeps `cluster_aware=False` so a bare call is
unchanged.

**Phase 2 (test_novel holdout) — BUILT then REMOVED (2026-07-20).** A
`--cluster-novel-holdout` flag was implemented (carve the smallest of >= 3 MinHash
clusters into a disjoint `test_novel` 4th split for novel-lineage generalization),
then removed at the user's call: its utility is narrow (it removes training data,
rarely triggers on RefSeq, and needs a post-training eval consumer that doesn't
exist — `finetune_head.py` reads only train/val/test), whereas `--cluster-aware-split`
is broadly useful. Not worth carrying unused infra. If novel-lineage generalization
becomes a needed metric on denser (GenBank) data, resurrect from commit `81c58ce`.

**Still open (minor follow-ups):** reuse capacity-pass reads / cache per-genome
sketches to avoid re-reading genomes for the genome-level clustering.

Files: `core/_orchestration/_cluster.py`, `core/_orchestration/_splits.py`,
`core/generation_orchestrator.py`, `cli/generate.py`.

---

## Cross-repo — PhyloCascadeGLM

Inference/evaluation items live in the PhyloCascadeGLM repo's own `docs/BACKLOG.md`.
As of 2026-06-21 the decision policy is implemented there (depth-normalized "mean"
ranking + dominant-reject termination); what remains is end-to-end validation on
the reject-trained pilot, theta/temperature calibration, the two weak heads
(kingdom; collapsed `694009`), and a hierarchy-aware eval metric. The non-virus
domain gate (P4 Phase 2) is the TaxoTreeSet-side follow-up.
