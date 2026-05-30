# Evaluation Plan (Computational Realism)

This plan scopes the experiments that validate the cascaded classifier within
a realistic compute budget. It favors a small number of high-value experiments
over a full ablation matrix, and marks which design components are justified
theoretically rather than empirically.

## Experiment 1 (top priority): External validation on CAMI II
- Cascade vs Kraken2, Kaiju, MetaPhlAn, geNomad, vConTACT3
- Datasets: CAMI II marine, strain madness
- Metrics: purity, completeness, macro F1, weighted UniFrac, coverage
- Cost: ~1-2 weeks GPU + 2 weeks analysis

## Experiment 2 (zero cost): Threshold-coverage analysis
- Same model, varying the pruning threshold
- Precision vs coverage curves per taxonomic level
- Demonstrates the cascade's graceful degradation
- Cost: a few hours of additional inference

## Experiment 3 (selective ablation): Balancing on a subtree
- Regenerate Caudovirales (~150 heads) with and without balancing
- Train both, compare macro F1
- Acknowledged limitation: a representative subset, not the full tree
- Cost: ~3-5 days GPU

## Experiment 4 (structural analysis): Justifying rank-aware bucketing
- Statistics: what would happen without rank-aware bucketing
  - Maximum head cardinality without the bucketing pass
  - Heads that would become intractable (>1000 classes)
- A structural argument, not an empirical one
- Cost: a few hours of analysis

## Experiment 5 (ambitious): Public submission to the CAMI portal
- Permanently citable result
- Cost: time to prepare outputs in the CAMI format

## Experiment 6 (zero cost): Cardinality threshold ablation
- Same subtree, varying `--min-leaves-per-class` (e.g. 2, 3, 5) and
  `--rare-taxa-strategy` (fallback vs keep)
- Compare head-size distribution, total label count, and downstream macro F1
  on the trainable classes vs recall of the rare_taxa fallback
- Justifies the default leaf-count floor empirically
- See `docs/PLANS/caudoviricetes_cardinality.md` for the diagnosis
- Cost: hours of regeneration + inference

## Components justified theoretically (no ablation)
- NoiseFilter: NCBI administrative containers are not biological clades
- Absolute cap: prevents dataset explosion outside the DNABERT-2 tested range
- Proportional per-leaf distribution: documented adaptation from the master's
  thesis
- Rare-taxa fallback: addresses single-sequence orphan classes that cannot be
  learned regardless of model (structural, supported by the head-size
  distribution analysis)

Total estimated GPU time: 3-4 weeks (vs 12+ weeks for a full ablation).
