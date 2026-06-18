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

## 🟠 P2 — Reproducible NCBI snapshot

**Problem.** `discover` captures "the current state of RefSeq", which is a moving
target — re-running on the same TaxID at different dates yields different
datasets, so results are not citable/reproducible.

**Approach.** Record the NCBI `datasets` CLI version in the registry; emit an
accession manifest with versioned accessions (e.g. `GCF_000857325.2`); optional
content hash. Allow pinning so a published benchmark can re-fetch the exact
snapshot.

**Effort.** Low–moderate — the accessions are already in the registry; this is
mainly version capture plus a manifest export.

Files: `io/registry.py`, the discover/download path.

---

## 🟠 P3 — Molecule-type filter (plasmids / contamination)

**Problem.** No molecule-type filter at ingestion, so plasmids (horizontally
transferred across distant taxa, low phylogenetic signal) enter the vault mixed
with chromosomal sequences. The effect is larger for Bacteria than viruses.

**Approach.** Filter by molecule/sequence type at ingestion (keep chromosomal),
or expose it as a parameter. Decide before any Bacteria expansion.

**Status.** Already captured in `docs/TODOs/data_scope_questions.md` (item 1);
this backlog entry just tracks its priority.

**Effort.** Moderate.

---

## 🟡 P4 — OOD routing validation protocol

**Problem.** The virtual buckets (`virtual_rare_taxa`, `virtual_misc`) are
intended to absorb rare/unplaceable taxa, but whether a genuinely novel
(held-out) taxon is actually routed there — rather than forced into a real
class — is not measured. The separability diagnostic only assesses
in-distribution separability within a head. (Note: the current README makes no
"OOD-aware" claim; this would substantiate one if desired.)

**Approach.** Held-out-taxa protocol: exclude whole taxa from `discover`, then
check the cascade routes them to the correct bucket. Spans generation (bucket
construction, TaxoTreeSet) and routing (inference, PhyloCascadeGLM).

**Effort.** Moderate.

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

## 🟢 P7 — Decompose `capacity.py` / `generation_orchestrator.py`

**Problem.** `capacity.py` (~2300 lines) and the generation orchestrator are
large relative to the "modular pipeline" framing, hindering granular review and
testing.

**Approach.** Split into submodules mirroring the glossary concepts (exact vs
approximate/Bloom capacity, GPU encoding, disk spill, checkpointing). Already
under way: `_BottomUpCapacityComputer` was extracted from
`compute_all_capacities` and `run_preflight` was split into helpers.

**Effort.** Moderate, low-risk (covered by the 810-test suite).

---

## Cross-repo — PhyloCascadeGLM (separate repo; tracked here only so they are not lost)

These belong to the inference/evaluation project and should migrate to that
repo's backlog.

- **Hierarchy-aware evaluation metric.** Replace per-head accuracy with a
  metric that accounts for taxonomic distance (e.g. distance to the lowest
  common ancestor, or hierarchical F1): confusing two species of one genus is
  not the same error as missing the whole family.
  - Note: the related "error propagation in a greedy cascade" critique is
    largely moot — inference is already a best-first search with cumulative
    uncertainty pruning (`_traverser.py`), not a greedy argmax.
- **OOD routing validation at inference** — the inference half of P4 above.
