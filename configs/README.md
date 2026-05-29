# TaxoTreeSet Configuration Files

This directory contains the configuration files that control taxonomic tree
construction and dataset generation.

## Files

| File                  | Purpose                                                    |
|-----------------------|------------------------------------------------------------|
| `mapping.json`        | Domain scope definitions and taxon redirection rules.      |
| `mapping.schema.json` | JSON Schema for validating `mapping.json` structure.       |
| `noise_patterns.json` | Regex patterns and rank filters for the NoiseFilter.       |

## mapping.json

Defines how NCBI taxa are routed during tree construction. Each top-level
biological domain (kingdom/superkingdom) is configured as a *scope*, which
specifies:

- The taxa that are preserved as canonical training labels
- The taxa that are absorbed into semantic fallback groups
- The catalog of fallback groups (curated virtual TaxIDs in the
  999000-999999 range)

### Why redirections exist

The NCBI Taxonomy is incomplete and inconsistent for higher-level ranks,
particularly for viruses and bacteria. Many taxa lack a kingdom or realm
assignment, or belong to candidate phyla with insufficient genomic material.
Without curated routing, these taxa would generate noisy heads in the
cascaded classifier (heads with a single class, or heads with one class
representing thousands of unrelated genomes).

The redirection mechanism solves this by:

1. **Preserving well-represented clades** as canonical training labels.
   These are taxa with ample sequenced genomes and stable ICTV/LPSN
   placement (e.g., `Orthornavirae`, `Pseudomonadota`).

2. **Grouping unclear or sparse taxa** into semantic fallback buckets
   (`Archaeal_Viruses_Group`, `Subviral_Agents_Group`, etc).
   These buckets become single training labels in the parent head,
   representing the "I belong to this scope, but my placement is unclear"
   class.

3. **Catching everything else** in a default scope-level fallback
   (`Unclassified_<Scope>_Fallback`) when no specific rule applies.

### Structure

```jsonc
{
  "project_name": "TaxoTreeSet",
  "schema_version": "1.0",
  "global_fallback_id": "1",
  "global_fallback_label": "Unknown_Life_Form",
  "scopes": {
    "<superkingdom_taxid>": {
      "label": "<Scope_Name>",
      "default_id": "<999xxx>",
      "virtual_id_labels": {
        "<999xxx>": "<Group_Label>"
      },
      "redirections": {
        "<source_taxid>": {
          "target_id": "<destination_taxid>",
          "label": "<Source_Label>"
        }
      }
    }
  }
}
```

A `target_id` equal to the source TaxID means the taxon is preserved as
canonical. A `target_id` pointing to a 999xxx virtual ID means the taxon
is absorbed into that semantic fallback group.

### Current scope coverage

| Scope        | TaxID  | Redirections | Tested  | Notes                                  |
|--------------|--------|--------------|---------|----------------------------------------|
| Viruses      | 10239  | 19           | Yes     | Validated end-to-end, all heads        |
| Bacteria     | 2      | 51           | No      | Most extensive, includes Candidatus    |
| Archaea      | 2157   | 8            | No      | Smaller; rare phyla in fallback        |
| Eukaryota    | 2759   | 30           | No      | Phyla-level granularity                |

When you run the pipeline on an untested scope for the first time,
expect to iterate on the redirections based on the resulting head
cardinality distribution. Use `scripts/census_with_filter.py` to
preview the structure before generating the full dataset.

### Virtual ID conventions

Each scope reserves a block of 999xxx TaxIDs for its semantic fallbacks:

| Scope     | Reserved range |
|-----------|----------------|
| Viruses   | 999000-999099  |
| Bacteria  | 999100-999199  |
| Archaea   | 999200-999299  |
| Eukaryota | 999300-999399  |

The lowest ID in each range (e.g., 999000, 999100) is always the
`default_id` catch-all fallback.

### Adding a new scope

To enable a new biological domain:

1. Identify the NCBI superkingdom TaxID for the domain.
2. Reserve a 999xxx block and define the `virtual_id_labels`.
3. Survey the top-level clades under the domain
   (kingdom/phylum/realm/etc) and decide which should be preserved as
   canonical and which should be absorbed.
4. Add entries under `redirections` accordingly.
5. Validate the file against `mapping.schema.json` and run a discovery
   census before full generation.


## noise_patterns.json

Defines two filters used during taxonomic tree construction to remove
administrative containers from the NCBI Taxonomy.

### Why noise filtering exists

The NCBI Taxonomy contains many nodes that are not biologically meaningful
taxonomic ranks but rather administrative groupings: 'unclassified X',
'environmental samples', clinical isolates, serological subgroups, and so
on. Without filtering, these would generate spurious training heads with
no biological value (e.g., a head containing only 'isolate 1' vs 'isolate
2' of the same species).

The NoiseFilter applies two complementary filtering mechanisms:

1. **Name-based regex patterns** match scientific names against
   case-insensitive patterns. Matching nodes are skipped, and their
   sequences are reassigned to the next valid ancestor.

2. **Rank blacklist** filters nodes by their rank string regardless of
   name. This catches all serotype, serogroup, subtype, isolate, strain,
   and genotype entries — administrative subdivisions below the species
   level that are not part of formal biological taxonomy.

### Adding a new pattern

When you identify a new noise container during a census:

1. Add an entry to `name_patterns` with the regex and a description
   explaining what the pattern matches and why it should be filtered.
2. Include at least one concrete example in the description.
3. Test the regex with `scripts/test_noise_filter.py` before committing.
4. Run a census with `scripts/census_with_filter.py` to measure the
   impact.

### Structure

```jsonc
{
  "schema_version": "1.0",
  "metadata": {
    "description": "...",
    "policy": "...",
    "last_updated": "YYYY-MM-DD"
  },
  "name_patterns": [
    {
      "regex": "<python_regex>",
      "description": "<rationale_with_example>"
    }
  ],
  "rank_blacklist": {
    "description": "...",
    "ranks": ["serotype", "serogroup", "..."]
  }
}
```

### Pattern conventions

- Use `^` and `$` anchors to constrain matching to the start or end of
  the name (avoids accidental substring matches).
- Use `\\b` word boundaries to match whole words.
- Make patterns case-insensitive by relying on the NoiseFilter's
  default mode (no inline `(?i)` flag needed).
- Escape backslashes properly (`\\b` not `\b`) since JSON requires it.

### Current pattern coverage

The default configuration removes approximately 9,000 administrative
containers from the NCBI Viruses tree (~9% of reachable nodes),
including:

- 'unclassified X' containers across all viral ranks
- 'incertae sedis' clades
- 'environmental samples' and 'uncultured' entries
- Clinical and serological subdivisions (isolates, genogroups, CRFs)
- The Shi et al. invertebrate virome containers
- Malformed name entries

## Type Checking

The project uses Pyright (via Pylance) in `basic` mode. Two specific
diagnostics are silenced globally because they conflict with our use
of the `bigtree` library:

1. `reportPrivateImportUsage`: bigtree exposes `Node` via the top-level
   `__init__.py` but does not declare it in `__all__`. We follow the
   library's documented import path (`from bigtree import Node`).

2. `reportAttributeAccessIssue`: bigtree's `Node` extends `BaseNode`
   which accepts arbitrary attributes at runtime. We exploit this to
   attach taxonomic metadata (`rank`, `scientific_name`, `header_id`,
   `fasta_path`) directly to nodes. This is the documented usage
   pattern.

If we ever migrate away from bigtree or upstream fixes its type stubs,
these settings can be removed from `pyrightconfig.json`.

## See also

- `docs/GLOSSARY.md`: Definitions of all architectural terms used in the
  project (rank-aware bucketing, low-capacity bucketing, etc.).