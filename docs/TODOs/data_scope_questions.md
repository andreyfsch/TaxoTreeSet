# Data scope questions (open)

Distinct from the sync mechanics: these are about *what* sequences are
valid to ingest and *which* top-level groups the tool should cover. They
shape dataset quality and the multi-domain expansion, and should be
resolved before (or as part of) expanding beyond viruses.

## 1. Plasmid sequences in Bacteria (quality risk)

The download path applies an assembly-level filter
(_DEFAULT_ASSEMBLY_LEVELS = "complete,chromosome") but no molecule-type
filter. A bacterial assembly's FASTA bundles the chromosome together with
its plasmids as separate sequences, and every sequence in the file is
ingested into the vault indiscriminately. Plasmids are horizontally
transferred across distant taxa, so the same plasmid sequence can appear
under many classes, carrying little reliable phylogenetic signal of the
host. For a taxonomic-classification objective this likely degrades the
training signal once Bacteria is in scope.

Open question: filter plasmids out at ingestion (e.g. by inspecting the
sequence/molecule type the NCBI report exposes, keeping only chromosomal
sequences), keep them, or make it a parameter. Needs a decision before
the Bacteria expansion. Note Kraken2 reference databases do include
plasmid sequences, so there is precedent either way depending on goal.

## 2. Viroids as an additional top-level group

NCBI exposes Viroids (taxid 12884) as a high-level group separate from
Viruses (10239). Viroids are small circular RNAs without a capsid, mostly
plant pathogens. Adding them is mechanically trivial (one more entry in
the domain-to-taxid map), but the scientific call is whether a Viroids
head adds useful signal: there are only tens to low-hundreds of species
with very short genomes (~250-400 nt), so the head would be tiny. Decide
based on the experiment design, not mechanics.

## 3. The "all" target group (resolved)

Confirmed via the NCBI Taxonomy Browser: there is no single taxid above
Viruses, Bacteria, Archaea, and Eukaryotes (and Viroids) to use as a
universal root. The "all" target therefore cannot map to one taxid; it must
iterate the known top-level groups.

Resolved: `--root all` now resolves to a `None` domain anchor, so the whole
registry is in scope via the existing "everything" path, and sync iterates
the top-level groups. `_domains_to_sync` re-discovers the domains already
present in the registry, falling back to all four superkingdoms when it is
empty. (`_resolve_root_taxid` -- formerly `_resolve_domain_taxid` -- returns
None for "all".)
