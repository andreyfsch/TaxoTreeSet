# Issue: registry and LMDB vault are out of sync

## State
After multiple test runs, the registry has 15000 accessions with
downloaded=False, but the LMDB vault at data/vault/sequences.lmdb
still holds valid sequences (the 39GB parquet dataset was generated
from those sequences).

## Resolution paths
1. Run main_discovery.py --reset + main_generation.py to re-fetch
   everything (1-3h, but guarantees consistency).
2. Write a one-shot script that iterates the LMDB keys, parses them
   into per-accession header lists, and writes them back to the
   registry with downloaded=True.

Option 2 is preferred but requires solving the LMDB-key-to-accession
mapping (using manifest_viruses.json as a reverse index could work).

## Planned resolution (incremental sync)

The unified entry point planned for the packaging/integration work will
fold discovery into every run as an incremental sync: on each execution
the tool reconciles the vault against what NCBI currently offers for the
requested scope, downloading only the delta and marking present
accessions as downloaded. That reconciliation makes the
"downloaded=False but present in the vault" state self-correcting -- the
sync pass detects the sequences already in the vault and updates the
registry accordingly -- so neither manual resolution path above will be
needed once it lands. This TODO is kept until that integration is in
place; do not invest in the one-shot resync script in the meantime, as
it would be superseded.
