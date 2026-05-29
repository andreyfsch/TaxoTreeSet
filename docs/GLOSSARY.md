# TaxoTreeSet Glossary

This document defines the technical terminology used throughout the project.
Definitions here are authoritative; any inconsistency between code and this
glossary should be reported and corrected.

## Architectural Concepts

### Rank-Aware Bucketing
Mechanism that handles taxonomic hierarchies where direct children of a node
have heterogeneous ranks (a phenomenon that became common after the ICTV 2022
viral taxonomy reorganization, especially in `Caudoviricetes`).

When a parent node has children with different ranks (e.g., some children are
genera, others are species or families), this mechanism:

1. Identifies the modal (canonical) rank among children
2. Separates children with non-canonical ranks into virtual buckets, grouped
   by their actual rank (e.g., `virtual_species`, `virtual_family`)
3. Merges buckets with fewer than `min_subclades_per_bucket` children into
   a generic `virtual_misc` bucket
4. Marks virtual buckets as cascade terminators — they are training labels
   in the parent head but do not generate their own sub-cascades

Configured via the `--min-subclades-per-bucket` CLI argument.

Previously called "Op3" during development.

### Low-Capacity Bucketing
Mechanism that handles taxonomic heads where some child classes lack sufficient
genomic material to meet the per-class subsequence threshold.

When a head's per-class capacity falls below `min_num_seqs`:

1. Computes a percentile cutoff (default 98%) over child capacities
2. Children above the cutoff remain as eligible training classes
3. Children below the cutoff are absorbed into a `virtual_low_capacity` bucket
4. The bucket itself becomes a training label in the parent head

Configured via `--min-num-seqs` and `--cutoff-percentage` CLI arguments.

Previously called "Op_B" or "Opção B" during development.

## Bucket Types in `virtual_id_registry`

| Rank string             | Created by                  | Purpose                                                |
|-------------------------|-----------------------------|--------------------------------------------------------|
| `virtual_bucket`        | RankAwareBucketing          | Groups children whose rank differs from the modal rank |
| `virtual_misc`          | RankAwareBucketing          | Generic bucket for rare ranks with < min subclades     |
| `virtual_low_capacity`  | LowCapacityBucketing        | Absorbs children with insufficient genomic material    |
| `virtual_cluster`       | MolecularClusterBucket      | (Legacy) K-means clustering of similar taxa            |
| `realm_group`           | CuratedRealmFallback        | Pre-configured semantic fallbacks (999000-999003)      |

All bucket taxids follow the pattern `9XXXXXXXX` (9 digits starting with 9),
deterministically generated from `sha256(parent_taxid + purpose)`.

## Balancing Scenarios

The `compute_balanced_extraction_plan` function returns one of these scenarios:

| Scenario string         | Trigger                                    | Behavior                                          |
|-------------------------|--------------------------------------------|--------------------------------------------------|
| `level_all`             | min_capacity >= min_num_seqs               | All children retained; n_per_class = min_cap     |
| `level_all_capped`      | level_all + min_cap > max_n_per_class      | n_per_class clamped to max_n_per_class           |
| `cutoff_applied`        | min_capacity < min_num_seqs                | Percentile cutoff + LowCapacityBucket created    |

## Other Terms

### Cascade Terminator
A node that exists as a training label in its parent's head but does not
recurse into its own sub-cascade. Used for virtual buckets to prevent
double-counting of classes during the cascaded BFS inference.

### Passthrough
A node with a single taxonomic child whose own head is redirected to the
child. The parent does not have a trainable head; instead, its head label
is the child's head. Stored in `passthroughs_viruses.json`.

### Head
A trainable classifier corresponding to a decision point in the taxonomic
cascade. Each head produces one Parquet dataset (train/val/test) and trains
one LoRA adapter on top of DNABERT-2.

### Capacity
The number of unique subsequences of length `min_len` extractable from all
genome sequences under a taxonomic node via sliding window. Can be computed
exactly (set union) or approximately (Bloom filter).