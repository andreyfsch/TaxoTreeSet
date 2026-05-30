# TaxoTreeSet

TaxoTreeSet builds balanced, hierarchically structured training datasets from
NCBI Virus RefSeq for cascaded LoRA fine-tuning of genomic language models. It
turns a raw catalog of viral genome sequences into a tree of
per-decision-point training shards (one per classifier head), each ready to
train a LoRA adapter on top of a foundation model backbone such as DNABERT-2.

## Overview

A single flat classifier over thousands of viral taxa is impractical to train
and to interpret. TaxoTreeSet instead mirrors the NCBI taxonomy as a cascade of
small classifiers: each internal taxonomic node becomes a *head* that
discriminates only among its direct children. At inference time, a sequence is
routed down the tree head by head until it reaches the most specific
confidently predicted taxon.

Producing such a cascade from real taxonomy is not mechanical. NCBI Taxonomy is
irregular -- sibling nodes can carry different ranks, clades vary in sampling
depth by orders of magnitude, and the post-ICTV-2022 reorganization left many
viral genera orphaned without a family. TaxoTreeSet contains the machinery to
turn this messy input into balanced, trainable heads: rank-aware bucketing,
capacity-based balancing, a leaf-count cardinality threshold, and curated
semantic fallbacks. Each mechanism is documented in `docs/GLOSSARY.md`.

The output format (Parquet shards of subsequence/label pairs plus JSON
manifests) is model-agnostic; DNABERT-2 is the reference backbone but the
datasets can train any sequence classifier.

## Requirements

- Python 3.11
- `bigtree` (taxonomic tree construction)
- `taxoniq` (NCBI Taxonomy lineage resolution)
- `numpy` (vectorized capacity estimation)
- `pyarrow` (Parquet output)
- `lmdb` (sequence vault storage)
- `zstandard` (sequence compression in the vault)

## Workflow

TaxoTreeSet runs in two stages, each with its own entry point.

### Stage 1: Discovery

`main_discovery.py` queries NCBI from a biological root TaxID, applies the
configured scope mapping, and writes an inventory (`registry.json`) plus the
downloaded sequences into the LMDB vault.

```
python main_discovery.py --taxon-id 10239 --mapping configs/mapping.json --registry data/registry.json
```

Key options:

| Option            | Default               | Purpose                                          |
|-------------------|-----------------------|--------------------------------------------------|
| `--taxon-id, -t`  | 10239 (Viruses)       | NCBI TaxID of the biological root                |
| `--mapping, -m`   | configs/mapping.json  | Scope and fallback redirection rules             |
| `--registry, -r`  | data/registry.json    | Destination inventory file                       |
| `--reset, -f`     | off                   | Delete the old registry before a fresh discovery |

### Stage 2: Generation

`main_generation.py` builds the taxonomic tree from the registry, runs the
decision-point cascade to decide heads, buckets, and passthroughs, and writes
the balanced Parquet shards plus the sidecar manifests.

```
python main_generation.py --rank viruses --output data/datasets --approximate-capacity
```

Key options:

| Option                   | Default       | Purpose                                                        |
|--------------------------|---------------|----------------------------------------------------------------|
| `--rank, -g`             | viruses       | Target biological domain scope                                 |
| `--output, -o`           | data/datasets | Output directory for shards and manifests                      |
| `--max-subseq-len, -w`   | 2000          | Sliding-window size (bp) for subsequence extraction            |
| `--approximate-capacity` | off           | Use the Bloom filter for capacity (~12MB) instead of exact     |
| `--min-num-seqs`         | 1000          | Below this per-class capacity, the cutoff scenario triggers    |
| `--cutoff-percentage`    | 98.0          | Percentile of children retained when cutoff applies            |
| `--max-n-per-class`      | 20000         | Hard ceiling on subseqs per class                              |
| `--min-leaves-per-class` | 3             | Minimum sequence leaves for a child to stay a standalone class |
| `--rare-taxa-strategy`   | fallback      | `fallback` (divert rare taxa) or `keep` (retain all classes)   |

## Output

Stage 2 produces, under the output directory:

- `train.parquet` / `val.parquet` / `test.parquet` per head, each with columns
  `seq` (string) and `class_idx` (int32), under a directory tree mirroring the
  taxonomy.
- `manifest_<domain>.json`: every head with its labels, scenario, per-class
  count, and leaf count.
- `passthroughs_<domain>.json`: single-child nodes redirected to their child.
- `virtual_id_registry_<domain>.json`: catalog of synthetic buckets, their
  parents, and absorbed taxa.

These three JSON files are the contract with downstream training and evaluation
code.

## Architecture

The cascade is a recursive top-down traversal of the taxonomy. For each node it
classifies children by rank, estimates each child's capacity, computes a
balanced extraction plan, materializes any virtual buckets, distributes the
per-class sample budget across leaves, stratifies into train/val/test, records
the head in the manifest, and recurses into canonical children. The core terms
(head, bucket, passthrough, capacity, cascade terminator) are defined in
`docs/GLOSSARY.md`.

## Documentation

- `docs/GLOSSARY.md` -- authoritative definitions of all technical terms
- `docs/PLANS/caudoviricetes_cardinality.md` -- diagnosis and rationale for the
  rare-taxa cardinality threshold
- `docs/PLANS/cami_evaluation_plan.md` -- evaluation plan against CAMI II and
  external tools
- `configs/README.md` -- configuration file reference
