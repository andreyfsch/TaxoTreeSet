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
  exclude whole taxa from `discover`, confirm the cascade rejects them). Mostly
  PhyloCascadeGLM-side (inference/GPU).

**Phase 2 — non-virus domain gate — DONE (2026-07-23).** The intra-clade reject
covers "not in this clade"; the gate adds "not a virus at all" for the root/shallow
heads (which have no intra-tree "outside", so the root binary head previously had no
reject signal — it was excluded from the pilot). `--reject-cross-domain
bacteria,archaea,eukaryotes` (+ `--reject-cross-domain-sample N`,
`--reject-cross-domain-depth D`): during the sync a **bounded** set of RefSeq
*reference* genomes per domain is fetched (`DiscoveryOrchestrator.stream_reference_reports`
— `--reference`, stops after N, terminates the subprocess early) and registered as
**pending accessions tagged `cross_domain` with no lineage**, so Stage-1 downloads them
but tree building never places them / schedules heads for them. After download they are
materialised into lightweight sequence leaves (`_build_cross_domain_pool`) and
`sample_reject_leaves` appends them to the `far` pool for heads at
`depth <= reject_cross_domain_depth` (for the whole-tree head they are its *only*
negatives). Deeper heads keep intra-clade negatives only (a non-virus is a trivial
negative there). Off by default. Tests: `TestCrossDomainGate` (reject_bucket),
`TestCrossDomainNegatives` (acquire tags+no-lineage / pool skips undownloaded), CLI
threading. Suite 1121.

**Effort.** Reject class: done. Non-virus gate: DONE.

Files: `core/generation/reject_bucket.py` (`sample_reject_leaves` gate),
`core/orchestrator.py` (`stream_reference_reports`), `core/_orchestration/_sync.py`
(`_acquire_cross_domain_negatives` / `_build_cross_domain_pool`),
`core/generation_orchestrator.py` (wiring), `cli/generate.py` (flags).

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

## ✅ P9 — Multi-root scope + plasmid datasets (host-taxonomy, tool-free) — code DONE (2026-07-23)

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
2. **Plasmid acquisition — DONE (2026-07-23).** `io/plasmid_release.py`: parse the curated
   standalone RefSeq plasmid release (GenBank flat files), not a Bacteria crawl. Chosen over
   inverting the P3 molecule detector on downloaded assemblies because plasmid records are
   standalone nucleotide accessions (NZ_/NC_), which the assembly-oriented
   `datasets download genome accession` path does not fetch — so the release is both the
   host-taxid source (`/db_xref="taxon:NNN"` in the source feature) and the sequence source.
   `parse_gbff_records` streams records into `PlasmidRecord(accession, host_taxid, organism,
   length, sequence)`; `ingest_records_to_vault` writes each sequence into the LMDB vault with
   the exact downloader contract (key `accession`, value `zlib.compress(seq)`) so plasmids read
   back through `_read_single_sequence` like genomes; `record_to_report` adapts each to the
   synthetic assembly-report shape the registration path consumes; `iter_release_records` streams
   a whole release directory (`.gbff`/`.gbff.gz`, gzip transparent). 12 unit tests.
3. **Accession-set-driven discovery entry point — DONE (2026-07-23).**
   `DiscoveryOrchestrator.discover_from_reports` — the bottom-up counterpart to
   `discover_from_root`: groups the pre-acquired reports by host TaxID (`_group_reports_by_host`),
   reuses `_build_hierarchy` wholesale (lineage resolution + tree build + registration), then
   `_mark_reports_downloaded` flags each registered accession as already present in the vault (the
   sequence was ingested directly; nothing left to fetch), with a single header whose id is the
   vault key. Unresolvable hosts are skipped, not raised (matches the top-down path). Wired into
   the CLI: `discover --plasmid-release <DIR> --vault <DIR>` (parse+ingest → register by host
   lineage); `_run_plasmid_discovery`/`_validate_plasmid_args` in `cli/discover.py`. 6 orchestrator
   + 2 CLI tests. README Stage-1 "Plasmids: bottom-up discovery" subsection.

**Effort.** Moderate — **DONE (2026-07-23)**. Multi-root plumbing (piece 1) + plasmid acquisition
(piece 2) + accession-driven discovery (piece 3) all shipped, CPU-only. **The release download is
automatic** (`fetch_release`: md5-verified + resumable, synced into `<vault>/refseq_plasmid` by
default) — `discover --plasmids --vault <DIR>` fetches + ingests + registers in one command, like
`--taxon-id` does for viruses (no manual pre-download; `--no-fetch` uses a pre-fetched copy). **What
remains is only an actual production run** (`discover --plasmids` → `generate --binary-only` on the
host tree; downloads to `/mnt/f`) — no more code. Suite 1131. Deferred/future (option 2 in the
design): ingest a precomputed PTU/replicon table for intrinsic plasmid typing instead of
host-taxonomy labeling.

Files: `io/plasmid_release.py` (new; release parser + vault ingester), `core/orchestrator.py`
(`discover_from_reports`), `cli/discover.py` (`--plasmid-release`/`--vault`), `cli/generate.py`
(`--root` list, piece 1), `core/generation_orchestrator.py` (empty-root forest, piece 1). Related:
P3 (plasmid detection primitive), P4 (cross-domain negatives), the `--binary-only` machinery.

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

## 🟢 P11 — Clade-holdout open-set benchmark — DONE

**Goal.** Turn TaxoTreeSet from a dataset generator into the *evaluation instrument* for
open-set taxonomic classification: generate training data with whole clades (genera/families)
withheld, plus a matched eval set of reads from those novel clades, and score any classifier
on whether it correctly backs off to the deepest **retained** ancestor rank instead of
over-committing to a wrong in-index label. This is the highest-value research item — it
measures the exact regime (novel-clade generalization + calibrated back-off) where learned
representations are supposed to beat exact matching, and where the published literature is
weakest (split type / calibration / production baselines routinely unreported).

**Why now.** It is the direct, buildable realization of the standardized-benchmark roadmap:
in-index vs open-set × short vs long-noisy tracks, per-rank + per-ANI-bin metrics, PPV anchors,
calibration, compute, and a mandatory **head-to-head against retained-only Kraken2/Centrifuge**.
Reuses existing machinery (scope resolution, MinHash sketches for distance bins, the binary/multi
head + reject-bucket path, the cluster-aware split for the in-index control, extraction).

**Phased plan** (each phase independently useful): **P1 DONE (2026-07-22)**; P2 open-set
eval-set builder; P3 long-noisy read track + error model; P4 scorer (per-rank/per-bin metrics +
calibration + compute); P5 retained-only k-mer baseline runners. **Full design:
`docs/clade_holdout_benchmark.md`.**

**P1 — DONE (2026-07-22): holdout selection + pruning + manifest.** New `benchmark/holdout.py`
(`select_holdout_taxids` — explicit TaxIDs or seeded fraction at a rank, eligibility = has
genomes + leaves the parent still branching, nested selections deduped; `build_holdout_manifest`
— per-clade members, expected commit rank `ρ*` = deepest retained ancestor, nearest retained
relative + MinHash/Mash ANI-proxy bin; `prune_holdout` — detach held-out subtrees). Orchestrator
selects + records on the FULL tree then prunes before the capacity pass (holdout forces a fresh
bottom-up capacity so retained ancestors aren't credited with pruned descendants), and writes
`benchmark_manifest_<scope>.json`. CLI: `--holdout-clades` / `--holdout-rank` +
`--holdout-fraction` / `--holdout-seed` / `--holdout-manifest` (mutually exclusive clades-vs-rank;
pair with `--no-sync`). Tests: 13 unit (eligibility/selection/dedup/`ρ*`/pruning/manifest) +
3 CLI (thread + mutual-exclusion) + 3 integration (held-out clade gets no head; manifest records
`ρ*`=parent). Suite 1050.

Files: `benchmark/holdout.py` (new), `benchmark/__init__.py` (new), `core/generation_orchestrator.py`,
`cli/generate.py`.

**P2 — DONE (2026-07-23): open-set eval-set builder.** New `benchmark/eval_set.py`
(`build_eval_set` / `build_eval_reads`): reads each held-out genome from the vault, samples
fixed-length reads (short/Illumina-like track via `extract_subseqs` with `min_len==max_len`),
and labels every read with its true lineage (from the registry), true leaf taxid, held-out clade,
expected commit rank `ρ*`, and divergence bin — the ground truth a scorer needs to grade back-off
vs over-commitment. New `taxotreeset benchmark build-eval` subcommand (`--manifest` / `--registry`
/ `--output` / `--read-length` / `--reads-per-genome` / `--seed`). Deterministic; skips
unreadable/too-short genomes. Tests: 6 unit (header index, labels, determinism, skips, parquet
round-trip) + 3 CLI (parse/dispatch/run) + 1 integration (P1→P2: the synthetic holdout manifest
→ labeled novel reads that back off to the parent). Suite 1060.

Files: `benchmark/eval_set.py` (new), `cli/benchmark.py` (new), `__main__.py`.

**P4 — DONE (2026-07-23): open-set scorer.** New `benchmark/scorer.py` (`classify_outcome` +
`score_reads`): grades a classifier's per-read predictions against `ρ*` into five outcomes —
**correct** (commits at `ρ*`), **over_commit** (deeper than `ρ*` = the dangerous confident-wrong
call), **too_shallow** (a proper ancestor of `ρ*`), **misroute** (off-path at `ρ*`'s level or
shallower), **abstain**. On-path depth uses the read's true lineage; off-path over-commit-vs-
misroute uses `ranks.rank_depth`. Aggregates overall + per `ρ*`-rank + per divergence bin (counts
+ rates). New `taxotreeset benchmark score --eval-set … --predictions … --output report.json`
(`--csv` optional); predictions are a parquet/(t)sv of `read_id, predicted_taxid, predicted_rank`
(empty taxid = abstain) — how a classifier produces them is out of scope. Tests: 11 unit (all five
outcome cases + aggregation + CSV) + 2 CLI + 1 integration (full P1→P2→P4 loop: a back-off-to-`ρ*`
classifier scores 1.0 correct, a deeper-wrong one scores 1.0 over-commit). Suite 1074.

Files: `benchmark/scorer.py` (new), `cli/benchmark.py`, `benchmark/__init__.py`.

**P5 — DONE (2026-07-23, kraken2): retained-only baseline glue.** New `benchmark/baselines.py`:
`export_retained_reference` writes the reference genomes with **held-out clades excluded** (a
taxid-labeled FASTA `>seq|kraken:taxid|<taxid>` + seqid->taxid map) so the baseline's index faces
the same open-set condition as the model; `parse_kraken2_output` converts the tool's per-read
`C|U <read_id> <taxid> …` output into the same `read_id -> (taxid, rank)` predictions the scorer
grades (unclassified / taxid 0 -> abstain). New `taxotreeset benchmark export-refs` and
`benchmark parse-baseline` subcommands; the tool's index build + classify run in between (user-side,
documented in the spec). So the k-mer baseline's native LCA back-off is scored on exactly the same
`ρ*` footing as the model — the head-to-head the literature skips. Tests: 6 unit (rank map, export
excludes held-out / FASTA format / skips, parser classified-vs-abstain/malformed) + 3 CLI. Suite
1082.

Files: `benchmark/baselines.py` (new), `cli/benchmark.py`, `benchmark/__init__.py`.

**P3 — DONE (2026-07-23): long-noisy read track.** New `ErrorModel` + `apply_errors` in
`eval_set.py`: indel-dominated, homopolymer-aware (per-base sub/ins/del rates, indel rates boosted
inside homopolymer runs; all-zero = identity). `build_eval_reads` / `build_eval_set` refactored to
`min_len`/`max_len` + an optional `error_model` + a `track` column, so the same builder emits the
short (fixed-length, no error) and long (variable-length, noised) tracks with the *same* labels for
direct cross-regime comparison. `benchmark build-eval --track long` (+ `--min/max-read-length`,
`--sub/ins/del-rate`, `--homopolymer-factor`). Tests: error-model cases (identity / full sub / full
del / full ins / determinism) + long-track (longer + labeled) + short-track columns. Suite 1090.

Files: `benchmark/eval_set.py`, `cli/benchmark.py`, `benchmark/__init__.py`.

**★ P11 COMPLETE (2026-07-23):** the clade-holdout open-set benchmark runs end-to-end —
generate the holdout (P1) → label novel reads, short & long tracks (P2/P3) → score correct-back-off
vs over-commitment per rank × ANI-bin (P4) → head-to-head vs a retained-only k-mer baseline (P5).
Follow-ups (small, optional): Centrifuge parser + a FASTA/FASTQ dump of the eval reads for the
external tools; stratified-by-bin holdout selection; exact ANI (fastANI) vs the MinHash proxy.

---

## ✅ P12 — Per-head reliability annotation (data + training signals) — DONE

**Motivation.** A per-node reliability signal lets a downstream classifier weight or gate its
decisions (e.g. treat a low-reliability node permissively rather than trusting its call). A scan
of the 60 binary pilot heads found **39/60 (65%) have < 14 belongs genomes**, which forces
`_stratified_cuts` to put **exactly 1 genome in val** (`int(N*0.15) < 2`) — so their val/test
metrics are noisy *by construction*, and reliability tracks genome-richness (trustworthy at the
genome-rich trunk, collapsing at the genome-poor tips). Lavidaviridae (1914302; 6 divergent
genomes, one a GC outlier) is the canonical low-reliability node: val f1 peaked at epoch 0.33 then
oscillated while eval_loss rose — overfitting on too-few belongs genomes, not a training bug and
not fixable by the split tooling.

**Reliability is two-source (order matters):**
- **Training behavior DETERMINES it (a-posteriori):** val f1 stability/variance, val↔test f1 gap,
  eval_loss trajectory (overfitting), the learning-audit verdict (learned / degraded / collapsed),
  best-epoch position. These are the ground truth for whether a head can be trusted.
- **Data properties PREDICT/explain it (a-priori, generation-time):** belongs / val / test genome
  counts, GC spread across splits, k-mer separability. Cheap, available *before* training.

**Plan.** (1) TaxoTreeSet-side (CPU, generation-time): emit the a-priori data properties into each
head's `label_map.json` (a `reliability` block: belongs/val/test genome counts, per-split GC
spread). (2) A reliability annotator that merges those with the training metrics (already in the
fine-tune `metrics.json` / audit / `progress.json`) into a per-head reliability score/flag,
primarily driven by the training behavior. (3) Expose it where the downstream classifier reads it;
the *policy* (how reliability gates a decision) stays with the classifier, not TaxoTreeSet.

Files: `benchmark/reliability.py` (new; annotator) + a `reliability` block in
`core/_orchestration/_manifest.py` (`_write_label_maps`). Related: the split machinery
(`_splits.py`), the composition audit (`dataset/composition.py`).

**Done (2026-07-23).** Two parts shipped:
- **A-priori (generation-time).** `_scheduler._head_reliability(pos_split)` records
  `belongs_genomes` / `val_belongs_genomes` / `test_belongs_genomes` / `split_mode` and an
  `a_priori_flag` (`"low"` when belongs < `_RELIABLE_MIN_GENOMES=14`, i.e. val gets 1 genome by
  construction). Wired into the binary `master_manifest` and emitted as a `reliability` block in
  `_manifest._write_label_maps`.
- **Merge with training (a-posteriori).** `benchmark/reliability.py::annotate_reliability(apriori,
  training)` — training behavior *determines* the verdict when present (`verdict_source="training"`):
  `unreliable` if it never learned; `noisy-metrics` if val-f1 pstdev > 0.08 or |test−final-val| >
  0.10; else `reliable`. Without training metrics it falls back to the a-priori flag
  (`verdict_source="a_priori"`). Exposed via `taxotreeset benchmark reliability --heads <dir>
  [--training-metrics m.json] [--write] [--summary out.csv]`. The *policy* (how the classifier acts
  on a verdict) stays downstream. Tests: `test_reliability.py`, `TestHeadReliability` in
  `test_scheduler_single_level.py`, `test_label_maps_carry_apriori_reliability` in
  `test_synthetic_pipeline.py`, CLI tests in `test_cli.py`.

---

## 🟡 P13 — Block-stratify large negative genomes (finish the volume-split fix)

**Context.** P-fix `b5ea511` made the genome-level split volume-aware
(`_assign_stratified` bin-packs whole genomes by window volume, not genome count) —
which resolved the inverted-prior catastrophe found on head `3044732` Homochaacvirus
(val was 82% not_belongs → the head was untrainable). But whole-genome assignment is
**leakage-safe by keeping each genome in one split**, so when a class's window volume is
dominated by **one** giant genome, that genome is indivisible and lands wholly in a
single split. **Residual (measured on the regenerated pilot):** heads whose *negatives*
are dominated by a single large external genome still show a prior *shift* (not the old
inversion) — `694009` SARS-CoV-2 and `864596` Bat coronavirus at val ~68% belongs,
`2499674` similar. They are trainable (both classes present in every split, val properly
sized) but not 50/50.

**Fix.** Block-stratify large negative genomes the same way the *positives* already are
(`_block_stratified_windows`): when a single genome exceeds a split's target volume, cut
it into interleaved positional blocks and spread its windows across train/val/test
(leakage-safe — windows confined to disjoint blocks), instead of assigning the whole
genome to one split. Detect the dominant-genome case inside `_assign_stratified` (or route
oversized genomes through the block path) while keeping the small-genome whole-assignment.
Preserves the ≥1-per-split guarantee. Regenerate the 3 residual heads + re-verify.

Files: `core/_orchestration/_splits.py` (`_assign_stratified` / `_block_stratified_windows`).
Related: P10 (cluster-aware split), the [[datagen-confounds]] Bug-3 note.

---

## 🟢 P14 — Plasmid discovery performance (host-lineage resolution + GBFF parse)

**Context.** The P9 plasmid run (`discover --plasmids` / `generate --plasmids`) works
end-to-end — validated 2026-07-23: fetched the 9-file RefSeq plasmid release (~2.7 GB),
ingested **135,994 plasmids** into the vault, registering by host lineage. But two stages
are slow at that scale:
1. **Host-lineage registration** — hosts newer than the taxoniq snapshot hit the NCBI-CLI
   fallback (`_fetch_lineage_via_ncbi`), one network subprocess **per host**; across 136k
   plasmids / many distinct hosts this is the long tail (checkpointed, but hours).
2. **GBFF parsing** — pure-Python line iteration over multi-GB flat files
   (`parse_gbff_records`), single-threaded.

**Fix (additive, no behavior change).** (1) Batch/cache host-lineage resolution: dedup host
TaxIDs up front and resolve each once (the registration already groups by host, but the
NCBI fallback is per-taxon and un-cached across runs) — optionally a single bulk
`datasets summary taxonomy taxon <ids>` call instead of one-per-host. (2) Parallelize the
GBFF parse across the release's files (they are independent) — a pool over
`iter_release_records`'s per-file streams, merging the vault writes. Neither changes the
resulting registry/vault, only throughput.

Files: `io/plasmid_release.py` (parse), `core/orchestrator.py`
(`_resolve_lineage` / `_fetch_lineage_via_ncbi` caching), `core/_orchestration/_sync.py`.
Related: P9, [[datasets-taxonomy-output]] (the NCBI-fallback all-ranks gap).

---

## Cross-repo — PhyloCascadeGLM

Inference/evaluation items live in the PhyloCascadeGLM repo's own `docs/BACKLOG.md`.
As of 2026-06-21 the decision policy is implemented there (depth-normalized "mean"
ranking + dominant-reject termination); what remains is end-to-end validation on
the reject-trained pilot, theta/temperature calibration, the two weak heads
(kingdom; collapsed `694009`), and a hierarchy-aware eval metric. The non-virus
domain gate (P4 Phase 2) is the TaxoTreeSet-side follow-up.
