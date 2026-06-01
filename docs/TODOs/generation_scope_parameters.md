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
