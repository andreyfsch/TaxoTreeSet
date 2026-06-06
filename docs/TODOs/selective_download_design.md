# Design: selective download for large scopes

## Problem

The pipeline downloads the entire scope (Stage 1), builds the tree
(Stage 2), balances and schedules (Stage 3, where `n_per_class` per label
is decided), then extracts (Stage 4). This assumes the whole scope fits
on disk: fine for viruses (~184 MB) and bacteria (~27 GB), impossible for
eukaryota (~3 TB of reference genomes).

For large scopes the decision of *what to download* must happen up front,
during sync, before any sequence is fetched. The rest of the pipeline is
unchanged; it simply operates on the selectively downloaded subset.

## Where the decision lives

In the sync stage. After discovery has populated the registry with
metadata (no sequences), sync checks whether the sequences needed for the
requested taxon are already present. If not, it sums the
`total_sequence_length` of the pending accessions to estimate the volume
of a full download:

- Volume below a threshold: download everything (current behavior).
- Volume at or above the threshold: estimate what actually needs to be
  downloaded to represent each label's diversity, then download only that.

## What "diversity" means here

Balancing extracts `n_per_class` unique subsequences per training label,
equalized across the sibling labels of a head. The diversity that matters
is therefore the aggregate capacity of a label (the union of unique
subsequences across all accessions under it). Intra-label diversity is
the mechanism that serves inter-label discrimination: a head learns to
separate its labels from the internal variety of each one. The two are
one objective, and the unit of accounting is the label.

## The estimation target

The selection aims for each label's real `n_per_class`, not the
`max_n_per_class` ceiling. There is no guarantee a label has the
diversity to reach the ceiling, and for eukaryota (very large genomes)
chasing an unreachable or unnecessary ceiling would download an enormous,
wasted volume. Estimating the real per-label target avoids fetching large
sequences the balancing layer would discard anyway.

## Why capacity can be estimated from size

A node's capacity is the count of unique length-min_len subsequences
extractable via sliding window over its sequence leaves. For a sequence
of length L, there are L - min_len + 1 windows, and for non-repetitive
sequences most are unique, so capacity is approximately L. Genome size
(total_sequence_length, available from NCBI metadata without
downloading) is thus a good proxy for capacity.

The proxy is biased: size is an upper bound on capacity, because
repetitive regions (telomeres, centromeres, transposons, especially
prominent in eukaryotic chromosomes) contribute few unique windows.
Estimating capacity from size therefore overestimates capacity, which
leads the selection to pick fewer accessions than truly needed. This bias
is in the safe direction by construction: under-downloading is detected
and corrected by measuring real capacity after the first download (see
refinement).

## Feasibility

_compute_children_capacities returns a {child_name: capacity} dict; the
rest of balancing operates on that dict of numbers and never touches
sequences. Capacities can therefore be injected. The surgical change is
to let compute_balanced_extraction_plan accept an optional capacity
source: when provided, it uses the supplied (estimated) capacities; when
absent, it computes them from sequences as today.

## Flow

### Phase 1: estimation (in sync, when volume >= threshold)

1. Build the scope's tree from the registry (structure from stored
   lineages, no sequences).
2. Assign each accession an estimated capacity = its
   total_sequence_length.
3. Run balancing with the estimated capacities to obtain an estimated
   n_per_class per label.
4. For each label, select accessions (reference assembly first, then by
   size) accumulating estimated capacity until the label's estimated
   n_per_class is met.
5. Mark only those accessions pending, then download them.

### Phase 2: refinement (after the download)

6. Compute the real aggregate capacity of the downloaded labels (Stage 3
   already does this).
7. For labels that fell short of n_per_class (size overestimated their
   capacity due to repetitive content), mark additional accessions
   pending and download a further batch.
8. Repeat until labels meet their target or their accessions are
   exhausted (a legitimate ceiling).

## Prerequisites (low risk, implement first)

- Discovery records total_sequence_length per accession in the registry.
  The value already flows through the genome-report stream; it is simply
  not stored today.
- compute_balanced_extraction_plan accepts an injectable capacity source,
  defaulting to the current sequence-based computation.

## Out of scope (separate, later)

Intra-genome sampling: downloading only part of a single huge genome
(eukaryotic chromosomes can be hundreds of MB each). This layer works
below accession granularity and depends on partial FASTA retrieval (range
requests or NCBI-specific tooling). It is a distinct investigation,
needed for eukaryota proper, built on top of the accession-level
selection described here.

## Threshold

A configurable size threshold (config and/or CLI) separates "download
everything" from "selective download". Its default value is to be
determined; it should sit comfortably above bacteria-scale volumes so the
common path stays simple, and below eukaryota-scale.
