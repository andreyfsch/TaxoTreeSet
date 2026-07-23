# Clade-holdout open-set benchmark — design spec

## 1. Purpose and scientific rationale

Standard train/val/test splits — including TaxoTreeSet's cluster-aware split — measure
**in-distribution** generalization: the classifier has seen the target clade during training
and is tested on held-out *genomes of the same clade*. This says nothing about the regime that
actually motivates learned representations over exact matching: **open-set novelty**, where a
read comes from a clade with *no representative in the training/reference set*.

This benchmark generates datasets and an evaluation protocol that measure open-set behavior
directly, along the two axes that govern outcomes in practice:

- **reference coverage** — in-index (clade retained) vs open-set (clade withheld);
- **read context/quality** — short & accurate vs long & noisy.

The headline question it answers: *when a read comes from a novel clade, does the classifier
(a) avoid over-committing to a wrong in-index label, and (b) back off to the correct
deepest retained ancestor rank?* This is the failure mode community benchmarks (CAMI) and
the open-set literature flag, and the one most FM-taxonomy studies never test.

## 2. Definitions

- **Retained set** `R` — the taxa whose subtrees are kept in training (the label space the
  classifier is trained on).
- **Held-out set** `H` — whole clades (genera/families) removed from training entirely; their
  genomes appear *only* in the evaluation set.
- **Expected commit rank** `ρ*(x)` — for an eval read `x` drawn from a held-out clade, the
  rank of the **deepest ancestor of that clade that is still in `R`**. Example: a read from a
  held-out *genus* under a *retained family* has `ρ* = family` — the correct behavior is to
  abstain at genus and commit at family. `ρ*` is the ground truth for open-set scoring.
- **Distance bin** — nearest-neighbor divergence (ANI, or a k-mer/MinHash proxy when ANI is
  impractical) between each held-out clade and its closest *retained* relative, bucketed into
  `≥95%`, `90–95%`, `85–90%`, `<85%`. All open-set metrics are reported per bin.

## 3. Holdout construction

Inputs: a scope (`--root`), a holdout rank, and a budget.

1. **Select `H`.** Two modes:
   - explicit: `--holdout-clades <taxid,…>`;
   - stratified auto: `--holdout-rank {genus|family|…} --holdout-fraction f --holdout-seed s`
     — sample a fraction `f` of clades at that rank such that (i) each has a *retained* ancestor
     (so `ρ*` is defined), (ii) removing it does not empty its parent, and (iii) the selection
     spans all distance bins (stratified, not uniform).
2. **Compute distance bins** for each held-out clade vs its nearest retained relative (reuse the
   MinHash sketch machinery from the cluster-aware split as the ANI proxy; exact ANI optional).
3. **Freeze a manifest** (`benchmark_manifest.json`): for each held-out clade — its taxid, rank,
   member genomes, nearest retained ancestor (taxid + rank = `ρ*`), nearest retained *sibling*,
   distance + bin, and the RNG seed. This is the reproducibility contract.

Determinism: identical `--root`, registry snapshot, holdout rank/fraction/seed ⇒ identical `H`,
identical manifest.

## 4. Generated artifacts

1. **Training dataset** — produced by the existing generation pipeline with `H` excluded at
   tree-build time (held-out subtrees are pruned before head scheduling). Everything downstream
   (binary/multi heads, reject bucket, cluster-aware split, extraction) is unchanged; the reject
   negatives are drawn only from `R`, so the label space is exactly `R`.
2. **In-index eval set (control)** — held-out *genomes of retained clades* (the normal test
   split): measures in-distribution accuracy, the baseline the open-set numbers are compared to.
3. **Open-set eval set** — reads/windows from `H` only, each row carrying: true lineage, the
   held-out clade taxid, `ρ*` (expected commit rank), and distance bin. This is the novel-clade
   probe.

## 5. Read-length tracks

From the *same* held-out genomes, emit two parallel eval tracks so any advantage attributable to
context length is separable from read-quality confounders:

- **short/accurate** — fixed-length windows (e.g. 150 bp), no error model (Illumina-like);
- **long/noisy** — longer windows (e.g. 3–30 kbp) passed through a configurable error model
  (indel-dominated, homopolymer-aware, ONT/PacBio-like), with the error profile recorded in the
  manifest.

Both tracks share labels and `ρ*`, so results are directly comparable across read regimes.

## 6. Evaluation protocol and metrics

The scorer consumes a classifier's per-read predictions (`read_id → predicted lineage +
per-rank confidence/abstention`) plus the manifest, and reports, **per rank and per distance
bin**:

- **Species precision at fixed PPV anchor** (PPV ≥ 95%) and the corresponding recall; repeated
  at genus and family.
- **Correct back-off rate** — fraction of held-out reads committed *exactly* at `ρ*` (not
  deeper, not shallower). The core open-set success metric.
- **Over-commitment rate** — fraction of held-out reads assigned a wrong label at a rank
  *deeper* than `ρ*` (i.e. a confident in-index call on a novel read). The dangerous failure.
- **Calibrated recall by rank under abstention** and **abstention/unclassified rate by rank**.
- **Open-set robustness vs distance** — all of the above as a function of the `≥95 / 90–95 /
  85–90 / <85%` bins (expect graceful degradation as divergence grows).
- **Calibration** — ECE and reliability diagrams for the confidence used to abstain.
- **Compute** — wall-clock per 150 Mbp, peak RAM, peak VRAM, model/index size.

All metrics are defined on generic prediction files; the benchmark is agnostic to how the
classifier under evaluation produces them.

## 7. Baselines (head-to-head)

The benchmark must ship a baseline slot so learned methods are compared against production
exact-match tools on the *identical* reads and *identical* retained references:

- build Kraken2/Centrifuge indexes from the **retained** references only (so the baselines face
  the same open-set condition — the held-out clades are absent from their index too);
- run them on both read-length tracks;
- score their output with the same scorer (their native LCA back-off maps naturally onto `ρ*`).

This directly closes the "no production-baseline" gap and produces the in-index vs open-set,
short vs long comparison table.

## 8. Reproducibility artifacts

Bundle, per benchmark instance: the registry/reference snapshot id, `benchmark_manifest.json`,
read-generation params (lengths, error model, seeds), retained-only baseline index build
scripts, and a scorer with pinned dependencies. A benchmark run is fully reconstructable from
these.

## 9. Implementation plan (phased; each phase independently useful)

- **P1 — Holdout generation.** Add holdout selection + tree pruning to the generation path
  (reuse `_resolve_scope_taxids` / `_build_target_tree`; add a "prune `H` before scheduling"
  step). Emit the manifest + `ρ*`. Training datasets exclude `H`; nothing else changes.
- **P2 — Open-set eval-set builder.** A `benchmark build-eval` subcommand that extracts reads
  from `H` with true-lineage + `ρ*` + distance-bin labels; short track first.
- **P3 — Read-length tracks.** Add the long-noisy track + configurable error model.
- **P4 — Scorer.** A `benchmark score` subcommand computing §6 from a predictions file +
  manifest; emits per-rank / per-bin tables + calibration outputs + compute fields.
- **P5 — Baseline runners (DONE, kraken2).** TaxoTreeSet owns the two ends; the tool run is the
  user's. The flow:

  ```
  # 1. export the retained-only reference (held-out clades excluded)
  taxotreeset benchmark export-refs --manifest benchmark_manifest_<scope>.json \
      --registry registry.json --out-fasta retained.fasta --out-map seqid2taxid.tsv
  # 2. build + classify with the tool (user-side)
  kraken2-build --add-to-library retained.fasta --db DB && kraken2-build --build --db DB
  kraken2 --db DB --output k2.out eval_reads.parquet  # (reads exported to FASTA)
  # 3. convert the tool output to scorer predictions, then score with the same harness
  taxotreeset benchmark parse-baseline --tool kraken2 --input k2.out \
      --registry registry.json --output baseline_preds.parquet
  taxotreeset benchmark score --eval-set eval_reads.parquet \
      --predictions baseline_preds.parquet --output baseline_report.json
  ```

  The retained-only index gives the baseline the *same* open-set condition as the model, and its
  native LCA back-off is graded on exactly the same `ρ*` footing. Centrifuge + a FASTA/FASTQ dump
  of the eval reads are the natural follow-ups.

Reuse: scope resolution, MinHash sketches (distance bins), the binary/multi head + reject-bucket
machinery, the cluster-aware split (for the in-index control), and the extraction pipeline.

## 10. Open design questions

- **Distance metric.** MinHash-ANI proxy vs exact ANI (fastANI) — proxy is cheaper and already
  in-tree; validate the proxy against exact ANI on a sample before committing.
- **Holdout granularity.** Genus-level holdout is the default probe; family/order holdouts test
  deeper novelty but shrink the retained label space — expose the rank as a knob.
- **Abstention interface.** The scorer needs a per-rank confidence or an explicit abstention
  token from the classifier; define a minimal predictions schema so any classifier can plug in.
- **Read simulation fidelity.** Whether to use a full read simulator vs an in-house error model;
  start with a documented in-house model and leave a simulator adapter as an extension point.
