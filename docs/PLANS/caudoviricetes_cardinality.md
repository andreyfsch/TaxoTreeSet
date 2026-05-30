# High-Cardinality Heads and the Rare-Taxa Threshold

This document records the empirical investigation that motivated the
leaf-count cardinality threshold with a rare-taxa fallback bucket
(`--min-leaves-per-class`, `--rare-taxa-strategy`). `Caudoviricetes` is used
throughout as the clearest worked example, but the underlying problem — and
the mechanism that addresses it — is general across the viral tree. This file
is a standing reference for the dissertation and for anyone auditing why the
pipeline shapes high-cardinality heads the way it does.

## 1. The general problem: long-tailed, low-diversity heads

A cascaded classifier works only if each decision point (head) has a tractable
number of classes, each with enough distinct training examples to generalize.
Several taxonomic nodes violate this badly. Measured on the full
15000-accession RefSeq viruses build without any threshold (`keep` behavior,
equivalent to the v0.1-balanced baseline), the largest heads were:

| Head                          | Labels |
|-------------------------------|--------|
| Unclassified_Viruses_Fallback | 1360   |
| Caudoviricetes                | 729    |
| Potyvirus                     | 193    |
| Betasatellite                 | 126    |
| Cheoctovirus                  | 118    |
| Orthobunyavirus               | 109    |
| Skunavirus                    | 96     |
| Fromanvirus                   | 83     |
| Badnavirus                    | 76     |
| Phlebovirus                   | 68     |

Six heads exceeded 100 labels and fifteen exceeded 50. These span the whole
virosphere — bacteriophages (`Caudoviricetes`, `Skunavirus`, `Cheoctovirus`),
plant viruses (`Potyvirus`, `Badnavirus`, `Carlavirus`), animal viruses
(`Orthobunyavirus`, `Phlebovirus`), and subviral agents (`Betasatellite`) —
so the problem is structural, not specific to any one clade.

The common failure mode is not merely a large class count but a large class
count combined with a long tail of single-sequence classes. A softmax head in
which half the classes carry one training example cannot learn to generalize:
with a single example a model memorizes the exact sequence, and inference on
any diverged member of the same taxon then fails. The apparent class coverage
is illusory.

## 2. Caudoviricetes as the worked example

`Caudoviricetes` is the most extreme and best-documented instance, which is
why it drove the investigation.

In August 2022 the ICTV restructured bacteriophage taxonomy under this class,
abolishing the three long-standing morphology-based families `Siphoviridae`,
`Podoviridae`, and `Myoviridae`, which had absorbed most tailed dsDNA phages.
Their removal left hundreds of genera with no family assignment: in NCBI
Taxonomy these genera now attach directly to the class node, with no
intermediate rank between genus and class.

Before the threshold, the `Caudoviricetes` head carried **729 labels**: 726
direct-child genera plus three rank-aware virtual buckets (`virtual_species`,
`virtual_family`, `virtual_order`). A `ranked_lineage` query on every one of
the 726 genera confirms they are **all genuinely orphaned** — no family rank
sits between genus and class. They are the direct fallout of the 2022 family
removals. Their sequence-leaf distribution is pathological:

- median 2 leaves per genus, mean 3.9
- **362 genera (50%) have exactly one leaf**
- 590 genera (81%) have fewer than five leaves
- only 57 genera have ten or more leaves

`Unclassified_Viruses_Fallback` (1360 labels) is the same phenomenon from a
different cause: it is the curated catch-all into which the mapping layer
routes accessions with unresolved lineage, so it naturally accumulates a huge
long tail of singletons.

Zhu et al. (2022), "Phage family classification under Caudoviricetes: A review
of current tools using the latest ICTV classification framework"
(Front. Microbiol. 13:1032186), review the consequences for automated
classification. Two findings are directly relevant:

- The reorganized families are more internally conserved than the old ones,
  making family-level classification more feasible. Even so, pairwise Dashing
  similarity among the four largest families remains low (roughly 0.017 to
  0.075): families are distinguishable but not by a wide margin.
- Tools that explicitly detect out-of-distribution (OOD) sequences handle the
  orphaned material best. vConTACT 2.0 routes about 98% of unclassified
  sequences to "outlier"/"singleton"/independent-cluster states rather than
  forcing them into a known family, at the cost of low prediction rate on
  short contigs.

The practical lesson: for taxa riddled with orphaned, under-sampled members, a
classifier that can say "rare/novel" is more useful than one that forces every
input into a specific class.

## 3. Why k-mer clustering does not rescue it

A natural alternative to a flat threshold is to sub-cluster orphaned taxa by
sequence composition and give each cluster its own head. We tested whether
naive k-mer composition carries enough signal.

Method: 4-mer frequency vectors (256-dim) over a central 4000 bp window,
cosine similarity, on representative sequences.

- **Family level (43 absorbed families, top 15 by size):** mean intra-family
  cosine 0.88, mean inter-family cosine 0.76, ratio 1.16x. Some pairs are
  clearly separable (Vilmaviridae vs Salasmaviridae 0.47), others nearly
  indistinguishable (Aliceevansviridae vs Demerecviridae 0.90). Weak but
  present structure.
- **Genus level (top 50 orphaned genera):** k-means silhouette peaks at 0.33
  for k=5 and degrades for larger k (0.19 at k=10, 0.14 at k=20). No discrete
  cluster structure.

This is consistent with the low Dashing similarities reported by Zhu et al.:
naive nucleotide composition is too weak a feature to recover phage family or
genus structure. The tools that succeed use protein clustering, HMM profiles,
or gene-sharing networks, not raw k-mers. Sub-clustering by k-mers would
manufacture arbitrary groups rather than biologically coherent ones, so we
rejected it.

## 4. Decision: leaf-count threshold with rare-taxa fallback

We adopted an OOD-aware cardinality threshold, governed by two CLI options and
applied uniformly to every head in the cascade (not to `Caudoviricetes`
alone):

- `--min-leaves-per-class` (default 3): a child must have at least this many
  sequence leaves to remain a standalone training label.
- `--rare-taxa-strategy` (default `fallback`): under `fallback`, children
  below the floor are absorbed into a single `virtual_rare_taxa` bucket per
  parent; under `keep`, every child is retained regardless of leaf count
  (reproducing v0.1-balanced).

The leaf-count floor is deliberately independent of the existing capacity
cutoff. Capacity guards the *quantity* of extractable subsequences (a single
long genome has high capacity); the leaf-count floor guards the *diversity* of
source sequences. A genus represented by one 100 kbp genome has high capacity
yet only one leaf, and it is precisely such taxa that the floor must catch.

The diversion is gated: it applies only when at least two children clear the
floor (decision A). This prevents a head from degenerating into a single
rare_taxa label when almost everything under a parent is sparse.

The `virtual_rare_taxa` bucket is a cascade terminator and a protected rank,
like the other virtual buckets, so it becomes one fallback label in the parent
head and does not spawn a sub-cascade. A classifier trained on the head learns
to route rare or novel inputs to this label rather than guessing a specific
under-supported class — the same philosophy that makes vConTACT 2.0 effective
on orphaned phages.

## 5. Measured impact

Full 15000-accession RefSeq viruses build, `fallback` with `min-leaves=3`,
versus the `keep` baseline:

| Metric                     | keep (baseline) | fallback | Delta  |
|----------------------------|-----------------|----------|--------|
| Largest head (labels)      | 1360            | 222      | -84%   |
| Caudoviricetes head labels | 729             | 222      | -70%   |
| Total heads                | 1061            | 837      | -21%   |
| Virtual buckets            | 76              | 155      | +79    |
| Passthroughs               | 5317            | 3755     | -1562  |
| Output size                | 39 GB           | 34 GB    | -13%   |
| Runtime                    | 14m18s          | 11m18s   | -21%   |

### Distribution of head sizes

The aggregate counts above hide how the threshold reshapes the *distribution*
of head sizes. It compresses both tails at once — it cuts the high-cardinality
heads and collapses the large mass of trivial two-label heads — pulling the
whole dataset toward a trainable middle band:

| Labels per head | without (baseline) | with (fallback) |
|-----------------|--------------------|-----------------|
| 2               | 425                | 201             |
| 3               | 155                | 174             |
| 4               | 87                 | 91              |
| 5-9             | 214                | 212             |
| 10-19           | 113                | 106             |
| 20-49           | 51                 | 41              |
| 50-99           | 10                 | 8               |
| 100+            | 6                  | 4               |
| **total heads** | **1061**           | **837**         |

(No head has a single label in either build; single-child nodes are handled by
the passthrough mechanism, not as heads.)

The summary statistics make the compression explicit:

| Statistic                | without | with  |
|--------------------------|---------|-------|
| mean labels per head     | 8.9     | 7.8   |
| median labels per head   | 3       | 4     |
| max labels per head      | 1360    | 222   |
| standard deviation       | 48.6    | 14.5  |
| heads with <=2 labels    | 425 (40%) | 201 (24%) |
| total labels (all heads) | 9451    | 6497  |

Two effects stand out. The standard deviation falls from 48.6 to 14.5: head
sizes become far more uniform, which is what makes a per-head LoRA training
budget predictable. And the median *rises* from 3 to 4 even though the mean
falls — because many trivial two-label heads collapse (their sparse children
are absorbed into the parent's rare_taxa bucket), shifting the central mass
upward while the high tail is cut. The total label count across the dataset
drops 31% (9451 to 6497), removing classes that carried no learnable signal.

The threshold acts across the whole tree, not at a single node. It created
**83 rare_taxa buckets** absorbing **2617 taxa** in total. The distribution of
those buckets shows the breadth of the problem:

- 3 buckets absorbed more than 100 taxa (`Unclassified_Viruses_Fallback` 1365,
  `Caudoviricetes` 508, `Betasatellite` 114)
- 9 buckets absorbed 20 to 100 taxa (`Mastadenovirus`, `Mammarenavirus`,
  `Alphasatellitidae`, `Papillomaviridae`, `Picornavirales`, and others)
- 9 buckets absorbed 10 to 19 taxa
- 62 buckets absorbed fewer than 10 taxa

Within `Caudoviricetes` specifically, 508 orphaned genera collapsed into one
`virtual_rare_taxa` label while the 218 genera with three or more leaves
remained as standalone classes. After the threshold, the largest head in the
entire dataset is 222 labels, down from 1360.

## 6. Scope and reproducibility

The tool is intended to apply both to isolate genomes and to metagenomic
contigs. The DNABERT-2 backbone consumes ~2000 bp windows natively, so a long
isolate genome is split into many windows and aggregated by majority vote,
while a short contig yields a few windows with proportionally lower confidence.
The rare-taxa fallback supports both use cases: rare or novel inputs surface as
"rare" rather than as confident misclassifications.

For exact reproduction of the v0.1-balanced dataset, run with
`--rare-taxa-strategy keep`. The threshold behavior is the new default because
it produces trainable heads; the `keep` path is retained for completeness
studies and for regenerating the historical baseline.

## References

Zhu Y, Shang J, Peng C, Sun Y (2022). Phage family classification under
Caudoviricetes: A review of current tools using the latest ICTV classification
framework. Frontiers in Microbiology 13:1032186.
doi:10.3389/fmicb.2022.1032186
