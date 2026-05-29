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
