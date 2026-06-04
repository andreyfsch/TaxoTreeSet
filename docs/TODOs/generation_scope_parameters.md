# Backlog: generation scope parameters

Three related parameters that give the user control over *what* part of
the taxonomy gets materialized into Parquet shards, and *how deep*. They
share a common foundation (a well-defined rank ordering) and partly
overlap, so they are grouped here. None is implemented yet.

## 1. Parametrizable depth boundary

Today the cascade descends to a fixed floor at genus (the run banner
prints "Depth Boundary: GENUS (Fixed Floor)"). The user should be able to
choose the rank at which shard generation stops (e.g. species, genus,
family, order).

Implementation notes:
- The genus floor is embedded in the generation cascade
  (generation_orchestrator scheduling/recursion). It must become a
  parameter threaded from the CLI down to the scheduler.
- Non-canonical ranks (no_rank and similar) complicate "stop at family":
  the system needs a well-defined canonical rank ordering to decide when
  a node is at or below the requested floor. The rank-aware bucketing
  layer already reasons about canonical ranks, so that notion exists and
  can be the basis.

## 2. Arbitrary root (TaxID or clade name)

Today generation's `--rank` accepts only {viruses, bacteria, archaea,
eukaryotes, all}, each mapped to a fixed domain TaxID. The user should be
able to pass an arbitrary TaxID (e.g. 2731619) or a clade name (e.g.
"Caudoviricetes") and have generation build the tree from that node.

Motivation: generate a single branch with different parameterizations to
compare training runs -- a legitimate research workflow.

Implementation notes:
- Discovery already starts from an arbitrary `--taxon-id`; the gap is on
  the generation side, which is hardwired to the four domain scopes.
- Need to resolve a clade name to a TaxID (taxoniq may already support
  this; otherwise an NCBI lookup).
- The scope mapping (mapping.json) carries domain-specific fallback
  redirections. Its interaction with an arbitrary root must be defined --
  likely a neutral fallback when the root is not one of the known scopes.

## 3. Single-level flag (no descent)

A flag that generates shards only for the given node, without recursing
into descendants. Combined with parameter 2, the user can pass a clade
and this flag to get only that clade's shards.

Implementation notes:
- Conceptually this is "depth = the root rank only" -- a special case of
  parameter 1. It can be expressed as a minimal depth or as a dedicated
  `--single-level` flag; the dedicated flag is more ergonomic.
- Technically a short-circuit in the scheduler recursion: process the
  root node, do not schedule its children.

## Dependencies and suggested order

- Parameter 2 (arbitrary root) is the foundation; it unlocks the
  single-branch workflows the other two serve.
- Parameter 1 (depth) is orthogonal but related, and both 1 and 3 rely on
  a well-defined canonical rank ordering.
- Parameter 3 is a special case of parameter 1 and can ride on its
  implementation.

These should land before the full test-coverage effort, since they change
the generation interface and would otherwise force test rewrites.

## Volume pre-check and selective download (the structural change)

Distinct from the scope parameters above, and the most substantial change
of this stage. The current pipeline downloads the entire scope first
(Stage 1), then builds the tree and runs balancing (Stage 3), which is
where the number of sequences actually needed per class is decided. That
order assumes the whole scope fits on disk: fine for viruses (~184MB) and
bacteria (~27GB), impossible for eukaryota (~3TB of reference genomes).

For large scopes the pipeline order must invert: discover structure ->
estimate/plan what balancing will need -> download only that subset ->
extract. The structure-without-download half already exists: incremental
sync's discover_from_root populates the registry with metadata only. What
is missing is using that registry to bound the download to what balancing
will actually consume, instead of download_all_pending fetching the whole
scope.

Open design question: balancing decides via capacity (unique k-mers),
which needs the sequence in hand. Two candidate approaches:
- Estimate capacity from genome size (NCBI metadata exposes assembly
  length) without downloading, to choose what to fetch, then refine with
  the real sequence.
- Demand-driven download: fetch per class incrementally until n_per_class
  is met, stopping once sufficient.
Either interacts with the capacity modes (exact vs bloom) and
max_n_per_class.

Two separable layers, do not conflate:
(a) download only the needed ACCESSIONS (a subset of genomes) -- this
    stage's structural change.
(b) download only PART of each huge genome (intra-genome sampling) -- a
    deeper, later layer needed for eukaryota's multi-GB genomes.

This is the change that the user considers leaves the code "done" with no
substantial changes afterward, so it precedes the test-coverage effort.
The "all" target group and arbitrary-root handling (scope parameters
above) should be resolved together with this work, not as separate items.

## Observed inefficiency: tree build enumerates the whole registry

Concrete instance of the download-everything-then-filter problem, seen
while validating arbitrary roots. Generating from Caudoviricetes (1208
children) still runs "Resolving Lineage Vectors" over all 15042 vault
accessions: _enumerate_accession_tasks flattens the entire registry, and
_process_accession decides per-accession whether each falls under the
chosen domain_taxid, discarding the rest only after resolving its
lineage. The result is correct (the tree is anchored at the chosen root)
but the whole scope is walked to keep a fraction.

Tolerable for viruses (~6s), prohibitive for bacteria/eukaryota where the
registry holds far more. The selective-download / volume pre-check work
should also prune the tree-build input to the chosen root's subtree
before resolving lineages, not after. The registry's taxon hierarchy
already records parent links, so the subtree under a root TaxID can be
selected up front.
