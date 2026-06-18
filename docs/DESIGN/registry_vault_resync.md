# Issue: registry and LMDB vault drift

> **Status: resolved.** The incremental-sync resolution below is implemented
> (Stage 0 sync, vault reconciliation, `--no-sync`). The "Reverse mapping"
> note remains a deferred optimization for multi-TB vaults.

## Background
The registry tracks assembly accessions (e.g. GCF_000857325.2) and a
per-accession downloaded flag. The LMDB vault stores sequences keyed by
FASTA sequence IDs (the first token of each header, e.g. NC_001802.1).
These two identifier spaces are related but distinct: a vault key does
not contain the assembly accession, so the vault cannot be reverse-mapped
to the registry by inspecting keys alone. This drove the original drift
incident, where a registry reset left every accession downloaded=False
while the vault still held valid sequences.

## Resolution (incremental sync)
The generate run folds discovery into a Stage 0 sync: on each execution
it re-runs discover_from_root for the requested scope. Because discovery
is idempotent (existing accessions keep their downloaded status; only new
ones are added as pending), re-running it is the delta detection -- new
NCBI entries enter the registry as pending and the existing download
stage fetches them. A --no-sync flag skips this for fast iteration on an
already-populated vault.

### Two reconciliation cases
- Case 1 (vault degraded): an accession is downloaded=True but its
  recorded headers are missing from the LMDB. This is fully detectable
  and is reconciled: the sync checks each downloaded accession's headers
  against the vault and resets downloaded=False when they are absent, so
  the download stage re-fetches them.

- Case 2 (registry reset, vault intact): an accession is downloaded=False
  yet its sequences are already in the vault (the accident that created
  this issue). A pending accession has no headers recorded, and the FASTA
  sequence IDs in the vault do not reverse-map to the assembly accession,
  so there is no cheap local way to recognize it as present. This case is
  not auto-reconciled. Incremental sync prevents it from recurring in
  normal operation (registry and vault evolve together); it only arises
  from disaster (a lost registry), where re-downloading the delta is the
  correct path -- NCBI Datasets fetches only what is missing.

### Reverse mapping (deferred optimization)
Case 2 is technically solvable by querying NCBI to map vault sequence IDs
back to assembly accessions (datasets summary --report sequence, or
e-utilities elink). It is not worth it now: confirming each assembly is
present would cost roughly one NCBI query per assembly, slower than a
batched re-download for the modest vaults in play (viruses ~184MB,
bacteria ~27GB). Revisit only if the vault grows large enough that
re-downloading a recovery delta becomes prohibitive (eukaryotes, multi-TB
vault), where avoiding re-download would justify the reverse-mapping
queries.
