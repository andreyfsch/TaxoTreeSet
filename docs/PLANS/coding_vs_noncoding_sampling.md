# Coding vs Non-Coding Sampling for Larger Genomes

> **Recorded 2026-07-02.** A standing design note for the eventual expansion of
> TaxoTreeSet beyond viruses (prokaryotes, then eukaryotes). It captures the
> problem, why it is **domain-graded**, the constraint that decides whether any
> fix is needed, a graded ladder of interventions, and a *measure-before-building*
> experiment. **Nothing here is implemented or scheduled** — it is scoping for when
> non-viral domains are on the table.

## 1. The problem

TaxoTreeSet samples training windows by **random sliding window** over each
genome. For viruses this is fine: viral genomes are **compact** (RNA viruses are
~90%+ coding; even large DNA viruses are gene-dense), so a random window almost
always lands on functional, phylogenetically informative sequence.

As genome size grows, the **non-coding fraction grows** — intergenic regions,
introns, regulatory sequence, and repeats. Much non-coding sequence is under weak
or no purifying selection and therefore **mutates fast**, while genes stay
conserved. Run unchanged on large genomes, the pipeline would produce datasets
**dominated by fast-evolving non-coding sequence**, potentially diluting the
conserved signal a classifier needs.

**The severity is graded by domain** — this matters for prioritisation:

| domain | coding fraction | severity |
|---|---|---|
| Viruses | ~90%+ | negligible |
| Bacteria / Archaea | ~85–90% (small intergenic, no introns) | **mild** |
| Eukaryotes (large) | ~1–2% exonic in extreme cases; vast introns/intergenic/repeats | **severe** |

So the concern is real but does **not** bite until eukaryotes. Prokaryotes — the
natural next expansion — are gene-dense enough that random sampling likely remains
adequate.

## 2. Refinement: informativeness is depth-dependent

Non-coding is **not uniformly uninformative**. Its value depends on taxonomic
depth:

- **Deep ranks (kingdom/phylum):** fast-evolving non-coding loses homology across
  distant taxa → noise. Conserved genes carry the signal.
- **Shallow ranks (genus/species/strain):** that same fast variation is a
  **fingerprint** that separates close relatives.

Therefore a blanket "always sample genes" policy could **hurt** shallow-rank
discrimination. Any intervention must preserve some non-coding signal for the
shallow decisions, not just chase genes for the deep ones.

## 3. The deciding constraint: train must match inference

**You can only bias sampling toward genes at training time if inference is also
gene-aware.** A metagenomic read is whatever fragment was sequenced — for a large
eukaryote it is *mostly non-coding*, and you cannot pre-select a gene from an
unknown short read. Training only on genes while inferring on random reads creates
a **train/inference distribution mismatch**: the model would excel on the coding
reads and fail on the non-coding reads that dominate real input.

This splits into two coherent architectures, and the "extract from genes" idea
only belongs to the second:

1. **Whole-genome-fragment (Kraken2-style).** Train on random windows (incl.
   non-coding); classify any read. The correct response to an uninformative read
   is the cascade's **graceful degradation** (return a coarse answer / reject) —
   *not* excluding it at train time.
2. **Gene / marker-based (MetaPhlAn-style).** Detect genes in **both** training and
   query; classify only gene-derived sequence. Cleaner deep signal, but requires
   gene detection **at inference** — expensive, and impossible on short reads that
   do not span a recognisable gene.

**Consequence:** the necessity of gene-aware sampling is downstream of the (still
open) inference-scenario decision. See `PhyloCascadeGLM/docs/viability_analysis.md`
§5 (inference length) — the same decision governs both.

## 4. A graded ladder of interventions (climb only as far as needed)

The originally-proposed version — *de novo* detection of promoters and alternative
splice sites with external tools — is the **hardest and most error-prone** rung,
and is largely **unnecessary** because RefSeq genomes ship with annotation.

| rung | intervention | cost | needs |
|---|---|---|---|
| 0 | none (current random windows) | — | adequate for viruses/prokaryotes |
| 1 | **mask low-complexity / repeats** (dustmasker / RepeatMasker) | cheap | nothing — removes the *worst* eukaryotic noise (repeats) without any gene concept |
| 2 | **annotation-guided sampling**: parse the **GFF / feature table that already ships with RefSeq** for CDS/gene coordinates, upweight (not exclusively) gene windows | moderate | annotation available (RefSeq usually has it) |
| 3 | *de novo* gene prediction: Prodigal (prokaryotes — fast, reliable, ORF-based, no splicing) / AUGUSTUS+RNA-seq (eukaryotes — slow, error-prone) | high / erratic | external tools |

**Key feasibility point:** RefSeq is **already annotated** — rung 2 is *parsing a
file*, not running a gene-finder. The user's proposed rung-3 detectors are avoidable.
For prokaryotes lacking annotation, Prodigal is cheap and reliable; only eukaryotic
*de novo* annotation is genuinely hard, and there the existing annotation solves it.

**Where the real complexity lands** (calibrating the "considerable complexity"
worry): not the gene detection — the **data plumbing**. The vault stores
`header_id → sequence`; annotation-guided sampling needs a parallel store
`header_id → gene coordinates` and coordinate mapping in the sampler. Real,
incremental work — not a rewrite.

## 4b. Two cheaper handles the ladder understates: uniqueness + a coding floor

Before climbing the ladder, two lighter guarantees cover most of the concern — and
should be the default.

**(a) Within-class sequence uniqueness dissolves most of the *repetitive* problem.**
Repeats hurt mainly by **duplication**: the same element (transposon, satellite,
low-complexity tract) is sampled many times, filling a class with near-identical,
low-information windows. **Guaranteeing every window in a class is unique** (exact
and near-duplicate dedup) removes that inflation directly — no gene concept, no
repeat model, cheaper than rung 1 for the duplication aspect — and, as a bonus,
keeps identical windows from leaking across the train/val/test split. This is the
first thing to enforce, and it is domain-agnostic.

**(b) Keep whole-genome sampling, but floor the coding fraction.**
For the residual **dilution** concern (unique but fast-evolving intron/intergenic
sequence), the fix is *not* gene-only sampling — that recreates the train/inference
mismatch of §3 — but **whole-genome sampling with a minimum-coding floor**: require
that at least some fraction of each class's windows be coding, so a class can never
end up ~100% non-coding and lose the conserved deep signal. The floor preserves the
shallow-rank non-coding fingerprint (§2) while insuring the deep signal — a
light-touch variant of rung 2 (a floor, not a bias), driven by the RefSeq
annotation the genomes already carry. The floor value is itself a
measure-before-building question (§5).

## 5. Measure before building

Whether — and where — this is worth the complexity is **testable cheaply**, the
same way as the k-mer decidability map:

> On a bacterial and a small annotated eukaryotic genome, measure **k-mer LR and
> DNABERT-2 separability on coding vs non-coding windows, separately.** If
> non-coding windows are near chance while coding windows are decidable, the
> concern is **validated and quantified** for that domain. If they are comparable,
> it is over-stated there.

Run this **per domain, prokaryotes first.** Do not build eukaryote-grade
gene-aware sampling before a stratified measurement shows it is needed.

## 6. The cheaper alternative: fix it in the decision, not the sampling

The right response to a non-coding, low-signal read may not be to exclude it at
training time, but to let the cascade **not force a deep classification** —
return a coarse rank or reject via the uncertainty it already models. An
uninformative read *should* stop shallow, not be confidently mis-placed deep.
This reuses existing machinery (decision rule + reject) and is far cheaper than a
gene-detection module. Much of the perceived problem may dissolve into
**calibration** rather than **sampling**.

## 7. Recommendation (phased)

1. **Do not build this now.** For the next expansion (prokaryotes), genomes are
   gene-dense; random sampling likely suffices.
2. **Before eukaryotes, measure** (§5) to quantify necessity per domain rather than
   assume it.
3. **If needed, climb the ladder only as far as the data demands:** rung 1
   (repeat/low-complexity masking) probably captures most of the benefit cheaply;
   rung 2 (existing annotation) if signal still lags. **Skip rung 3** — RefSeq
   annotation makes *de novo* detection unnecessary.
4. **Resolve the inference architecture first** (random-read vs gene-aware); it
   determines whether biasing training toward genes helps or creates a mismatch.
5. Consider whether **decision-side calibration** (§6) already addresses the
   symptom before paying for a sampling module.
6. **Cheap defaults first, before any of the above (§4b):** guarantee
   within-class window **uniqueness** (kills the repeat-duplication inflation) and
   a **minimum-coding floor** on whole-genome sampling (prevents ~100%-non-coding
   classes). Both are cheap, domain-agnostic, and avoid the train/inference
   mismatch of gene-only sampling.

## Related
- `PhyloCascadeGLM/docs/viability_analysis.md` §5 (inference length; the shared
  deciding constraint) and §7 (CAMI; marker-gene vs whole-fragment framing).
- `docs/PLANS/caudoviricetes_cardinality.md` (the same *measure-before-shaping*
  discipline applied to head cardinality).
