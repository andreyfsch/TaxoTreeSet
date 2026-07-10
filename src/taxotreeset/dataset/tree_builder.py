"""Taxonomic tree construction from NCBI accessions and mapping rules.

This module provides the function that builds the in-memory bigtree
representation of the cascaded taxonomic hierarchy. The tree is the
foundational data structure consumed by the generation orchestrator;
each internal node represents a taxonomic rank (kingdom, phylum, class,
etc.), and leaves under each node represent the individual sequence
headers available for that taxon.

The construction process integrates four sources of information:

1. **The accession registry** (data/registry.json), which lists all
   downloaded accessions and their headers.

2. **The registry's cached lineages**, used to resolve
   the full lineage of each accession from its TaxID up to the root.

3. **The scope mapping configuration** (configs/mapping.json), which
   defines redirection rules for taxa that lack proper kingdom or
   realm placement in NCBI. Redirections route problematic taxa into
   curated semantic fallback groups (see CuratedRealmFallback in
   docs/GLOSSARY.md).

4. **The noise patterns** (configs/noise_patterns.json), applied via
   the NoiseFilter to skip administrative containers that should not
   become trainable nodes.

The output is a single root Node from which the entire taxonomic
hierarchy hangs. Each node carries a ``rank`` attribute (e.g., 'genus',
'realm_group', 'sequence') and a ``scientific_name`` attribute. Sequence
leaves additionally carry ``header_id`` and ``fasta_path`` attributes
pointing back to their location in the LMDB vault.

Typical usage::

    from taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree

    tree_root = generate_seqs_by_taxon_tree(
        registry_path="data/registry.json",
        vault_path="data/vault",
        domain_taxid="10239",  # Viruses
        mapping_path="configs/mapping.json",
        noise_patterns_path="configs/noise_patterns.json",
    )
"""

import json
import logging
import os
from typing import Any

from bigtree import Node
from tqdm import tqdm

from taxotreeset.io.noise_filter import NoiseFilter

logger = logging.getLogger("TaxoTreeSet.Dataset.TreeBuilder")

_DEFAULT_MAPPING_PATH = "configs/mapping.json"
_DEFAULT_NOISE_PATTERNS_PATH = "configs/noise_patterns.json"
_LMDB_FILE_NAME = "sequences.lmdb"
_VIRTUAL_RANK_LABEL = "realm_group"
_UNKNOWN_RANK_LABEL = "unknown"


def generate_seqs_by_taxon_tree(
    registry_path: str,
    vault_path: str,
    domain_taxid: str | None = None,
    mapping_path: str = _DEFAULT_MAPPING_PATH,
    noise_patterns_path: str = _DEFAULT_NOISE_PATTERNS_PATH,
    all_ranks: bool = False,
) -> Node:
    """Build the in-memory taxonomic tree from the accession registry.

    Iterates over every (taxon, accession) pair in the registry,
    resolves each accession's full lineage via NCBI Taxonomy, applies
    noise filtering to skip administrative containers, applies scope
    redirections to route unplaced taxa into curated fallback groups,
    and stitches together a single rooted tree of bigtree nodes.

    Each sequence header in the registry becomes a leaf node with
    rank='sequence', linked to the LMDB vault via the ``fasta_path``
    and ``header_id`` attributes.

    Args:
        registry_path: Path to the registry JSON produced by the
            discovery phase.
        vault_path: Directory hosting the LMDB vault. Used to set the
            ``fasta_path`` attribute on each sequence leaf.
        domain_taxid: NCBI TaxID of the root domain to anchor the tree
            (e.g., '10239' for Viruses). If None, the tree spans every
            accession in the registry without domain anchoring.
        mapping_path: Path to the scope mapping configuration JSON.
            Missing files are tolerated; an empty mapping is used.
        noise_patterns_path: Path to the noise patterns configuration
            JSON consumed by the NoiseFilter.
        all_ranks: When True the lineages carry full NCBI granularity
            (clade, realm, subfamily, ...), so the redirectable top-level
            group may sit below intermediate ranks. Scope redirections then
            scan the lineage for the first matching key instead of only
            inspecting the domain's direct child. Canonical mode (False)
            keeps the direct-child-only behaviour.

    Returns:
        The root Node of the constructed tree. The actual taxonomic
        subtree of interest hangs from the child with name equal to
        ``domain_taxid``.

    Example:
        >>> root = generate_seqs_by_taxon_tree(
        ...     registry_path="data/registry.json",
        ...     domain_taxid="10239",
        ... )
        >>> # root.children[0] will be the domain anchor (Viruses)
    """
    registry_data = _load_json(registry_path)
    mapping_data = _load_optional_json(mapping_path)
    noise_filter = NoiseFilter(noise_patterns_path)

    scope_config = _resolve_scope_config(mapping_data, domain_taxid)
    scope_config["all_ranks"] = all_ranks

    root = Node("root", rank="root")
    accession_tasks = _enumerate_accession_tasks(registry_data, domain_taxid)

    logger.info(
        f"Spawning phylogenetic tree workers for {len(accession_tasks)} "
        "metadata entries."
    )

    accessions_dict = registry_data.get("accessions", {})
    lineages = registry_data.get("lineages", {})

    for taxid_str, accession_id in tqdm(
        accession_tasks, desc="Resolving Lineage Vectors"
    ):
        accession_info = accessions_dict.get(accession_id, {})
        if not accession_info:
            continue

        _process_accession(
            root=root,
            taxid_str=taxid_str,
            accession_id=accession_id,
            accession_info=accession_info,
            domain_taxid=domain_taxid,
            scope_config=scope_config,
            noise_filter=noise_filter,
            vault_path=vault_path,
            lineages=lineages,
        )

    _log_noise_filter_summary(noise_filter)
    return root


def _load_json(path: str) -> dict[str, Any]:
    """Load a required JSON file, raising on absence.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        Parsed JSON contents as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


def _load_optional_json(path: str) -> dict[str, Any]:
    """Load an optional JSON file, returning an empty dict on absence.

    Args:
        path: Filesystem path to the JSON file.

    Returns:
        Parsed JSON contents, or an empty dict if the file is missing.
    """
    if not os.path.exists(path):
        return {}
    return _load_json(path)


def _resolve_scope_config(
    mapping_data: dict[str, Any],
    domain_taxid: str | None,
) -> dict[str, Any]:
    """Extract the scope configuration for a given domain.

    Reads the per-domain fallback ID from the mapping configuration
    rather than assuming a global default. Each biological domain
    reserves its own 999xxx block (Viruses: 999000-999099,
    Bacteria: 999100-999199, Archaea: 999200-999299,
    Eukaryota: 999300-999399).

    Args:
        mapping_data: Full mapping configuration dictionary.
        domain_taxid: TaxID of the domain whose scope to retrieve.

    Returns:
        A dictionary with three keys:
            - 'default_id': the catch-all fallback virtual TaxID for
              this specific domain, or None if the scope is missing
              from the mapping (which means redirections cannot route
              taxa without explicit rules).
            - 'redirections': map of source TaxID -> redirection rule
            - 'virtual_labels': map of virtual TaxID -> human label
    """
    scopes = mapping_data.get("scopes", {})
    group_config = scopes.get(str(domain_taxid), {})

    if not group_config and domain_taxid:
        logger.info(
            f"No dedicated scope for root TaxID '{domain_taxid}' in "
            "mapping.json; using neutral routing (no redirections or "
            "fallback). Expected for arbitrary roots outside the "
            "configured domains."
        )

    return {
        "default_id": group_config.get("default_id"),
        "redirections": group_config.get("redirections", {}),
        "virtual_labels": group_config.get("virtual_id_labels", {}),
    }


def _enumerate_accession_tasks(
    registry_data: dict[str, Any],
    domain_taxid: str | None,
) -> list[tuple[str, str]]:
    """Flatten the taxon-to-accessions map into a sequential task list.

    When ``domain_taxid`` is given, only taxa whose cached lineage
    contains that TaxID are included, so generation processes just the
    requested root's subtree instead of the whole registry. Taxa with no
    cached lineage are excluded (they cannot be placed in the tree).

    Args:
        registry_data: Full registry dictionary loaded from disk.
        domain_taxid: Root TaxID to restrict to, or None for everything.

    Returns:
        List of (taxid, accession_id) tuples enumerating every in-scope
        accession exactly once per taxon it belongs to.
    """
    taxons_dict = registry_data.get("taxons", {})
    lineages = registry_data.get("lineages", {})

    def _in_scope(taxid: str) -> bool:
        if domain_taxid is None:
            return True
        stored = lineages.get(taxid)
        if not stored:
            return False
        return any(
            ancestor["taxid"] == str(domain_taxid) for ancestor in stored
        )

    return [
        (taxid, accession_id)
        for taxid, accessions in taxons_dict.items()
        if _in_scope(taxid)
        for accession_id in accessions
    ]
def _maybe_append_target_taxid(
    filtered_lineage: list[str],
    target_taxid: str | int,
    noise_filter: NoiseFilter,
    taxon_info: dict[str, tuple[str, str]],
) -> None:
    """Conditionally append the accession's target_taxid to its lineage.

    The accession's target_taxid is appended ONLY when it survives the
    noise filter. Otherwise the lineage remains anchored to its nearest
    valid ancestor (typically the species rank), and the accession's
    sequences will be attached there.

    This guards against creating administrative nodes (serotype,
    isolate, etc.) under species, which would inflate the tree with
    spurious heads when species ancestors gain multiple children.

    TaxIDs absent from the index are appended defensively (we cannot
    prove they are noise, so we keep them).

    Args:
        filtered_lineage: Noise-filtered lineage to append into (mutated).
        target_taxid: The accession's target TaxID.
        noise_filter: Configured NoiseFilter instance.
        taxon_info: Map of TaxID to (name, rank) for this lineage.
    """
    target_str = str(target_taxid)
    info = taxon_info.get(target_str)
    if info is None:
        filtered_lineage.append(target_str)
        return
    target_name, target_rank = info
    if noise_filter.is_noise(target_name, target_rank):
        logger.debug(
            f"[NOISE-SKIP-TARGET] target_taxid={target_str} "
            f"name='{target_name}' rank={target_rank} "
            "filtered out, accession anchored to nearest ancestor."
        )
        return
    filtered_lineage.append(target_str)


def _process_accession(
    root: Node,
    taxid_str: str,
    accession_id: str,
    accession_info: dict[str, Any],
    domain_taxid: str | None,
    scope_config: dict[str, Any],
    noise_filter: NoiseFilter,
    vault_path: str,
    lineages: dict[str, list[dict[str, str]]],
) -> None:
    """Stitch a single accession into the tree at the correct lineage path.

    Resolves the NCBI lineage for the accession, applies noise filters
    to remove administrative ancestors, applies scope redirections to
    route unplaced taxa, walks the lineage creating nodes as needed,
    and attaches all sequence headers as leaves under the final node.

    Args:
        root: Root node of the tree under construction.
        taxid_str: TaxID under which the accession was discovered.
        accession_id: NCBI accession identifier.
        accession_info: Accession metadata from the registry.
        domain_taxid: Optional domain anchor for the tree.
        scope_config: Resolved scope configuration dict.
        noise_filter: Configured NoiseFilter instance.
        vault_path: LMDB vault directory.
        lineages: The registry's ``lineages`` map (taxid to ancestry).
    """
    target_taxid = accession_info.get("taxid") or taxid_str
    stored_lineage = lineages.get(str(target_taxid))
    if not stored_lineage:
        logger.debug(
            f"[NO-LINEAGE] acc={accession_id} taxid={target_taxid} has no "
            "cached lineage; skipping."
        )
        return
    # taxid -> (name, rank) for every ancestor on this accession's path,
    # so noise filtering and node labeling read names and ranks from the
    # registry instead of re-resolving them.
    taxon_info: dict[str, tuple[str, str]] = {
        ancestor["taxid"]: (ancestor["name"], ancestor["rank"])
        for ancestor in stored_lineage
    }
    raw_lineage = _lineage_ids_from_registry(target_taxid, lineages)
    filtered_lineage = _apply_noise_filter_to_lineage(
        raw_lineage, noise_filter, taxon_info
    )

    if not filtered_lineage:
        logger.debug(f"[NOISE-ORPHAN] acc={accession_id} entire lineage filtered out.")
        return

    target_str = str(target_taxid)
    if target_str not in filtered_lineage:
        _maybe_append_target_taxid(
            filtered_lineage, target_taxid, noise_filter, taxon_info
        )

    anchored_lineage = _anchor_lineage_to_domain(filtered_lineage, domain_taxid)
    final_lineage = _apply_scope_redirections(
        anchored_lineage, domain_taxid, scope_config
    )

    leaf_taxon_node = _build_lineage_path(
        root=root,
        lineage_ids=final_lineage,
        virtual_labels=scope_config["virtual_labels"],
        taxon_info=taxon_info,
    )

    _attach_sequence_leaves(
        taxon_node=leaf_taxon_node,
        accession_info=accession_info,
        vault_path=vault_path,
    )


def _lineage_ids_from_registry(
    target_taxid: str | int,
    lineages: dict[str, list[dict[str, str]]],
) -> list[str]:
    """Read a taxon's cached lineage from the registry, root to leaf.

    The registry stores each lineage species-to-root as dicts resolved
    during discovery (taxoniq with an NCBI fallback). Tree construction
    wants TaxID strings root-to-leaf, so the stored order is reversed.

    Args:
        target_taxid: TaxID whose lineage to read.
        lineages: The registry's ``lineages`` map.

    Returns:
        List of TaxID strings from root to leaf, or empty list when the
        TaxID has no cached lineage.
    """
    stored = lineages.get(str(target_taxid))
    if not stored:
        return []
    return [ancestor["taxid"] for ancestor in stored][::-1]


def _apply_noise_filter_to_lineage(
    lineage_ids: list[str],
    noise_filter: NoiseFilter,
    taxon_info: dict[str, tuple[str, str]],
) -> list[str]:
    """Remove administrative containers from a taxonomic lineage.

    For each TaxID in the lineage, reads its scientific name and rank
    from the registry-derived index, then queries the NoiseFilter.
    TaxIDs flagged as noise are skipped; the accession effectively
    "climbs" to the next valid ancestor in its lineage.

    TaxIDs absent from the index are kept in the lineage as a defensive
    default (the absence of metadata is not evidence of administrative
    status).

    Args:
        lineage_ids: Root-to-leaf list of TaxID strings.
        noise_filter: Configured NoiseFilter instance.
        taxon_info: Map of TaxID to (name, rank) for this lineage.

    Returns:
        Filtered lineage with administrative TaxIDs removed.
    """
    filtered: list[str] = []
    for taxid in lineage_ids:
        info = taxon_info.get(taxid)
        if info is None:
            filtered.append(taxid)
            continue
        name, rank = info
        if noise_filter.is_noise(name, rank):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[NOISE-SKIP] taxid={taxid} name='{name}' "
                    f"rank={rank} reason={noise_filter.explain(name, rank)}"
                )
            continue
        filtered.append(taxid)
    return filtered


def _anchor_lineage_to_domain(
    lineage_ids: list[str],
    domain_taxid: str | None,
) -> list[str]:
    """Ensure a lineage is anchored under the requested domain.

    If ``domain_taxid`` is present in the lineage, truncates the
    lineage to start at that point. If absent, prepends the domain
    TaxID. When no anchor is requested, returns the lineage unchanged.

    Args:
        lineage_ids: Filtered lineage from _apply_noise_filter_to_lineage.
        domain_taxid: Optional domain anchor.

    Returns:
        Lineage starting with the domain TaxID when an anchor is set.
    """
    if not domain_taxid:
        return lineage_ids

    domain_str = str(domain_taxid)
    if domain_str in lineage_ids:
        anchor_index = lineage_ids.index(domain_str)
        return lineage_ids[anchor_index:]

    return [domain_str, *lineage_ids]


def _apply_scope_redirections(
    lineage_ids: list[str],
    domain_taxid: str | None,
    scope_config: dict[str, Any],
) -> list[str]:
    """Apply scope-level redirections to the recognised top-level group.

    The redirection key is the domain's direct child in canonical mode;
    in all-ranks mode (``scope_config['all_ranks']``) it is the first
    lineage node that carries a rule, since clade/realm ranks can sit
    above the recognised group. There are three possible outcomes:

    1. **Explicit rule found**: applies the rule's target_id. A self-
       redirect (target_id equals source) preserves the lineage; a
       virtual redirect (target_id differs) inserts the virtual group
       directly above the matched taxon (keeping intermediate ranks
       resolved above it in all-ranks mode).

    2. **No rule found, default fallback available**: routes the taxon
       to the scope's default fallback (e.g., 999000 for Viruses),
       collapsing the lineage to ``[domain, default, leaf]``.

    3. **No rule found, no default available**: leaves the lineage
       unchanged. This branch is exercised when the domain has no
       scope configuration; the taxon retains its NCBI placement.

    Args:
        lineage_ids: Domain-anchored lineage from
            _anchor_lineage_to_domain.
        domain_taxid: Domain anchor TaxID.
        scope_config: Resolved scope configuration. Its 'default_id'
            entry may be None when no scope is defined for the domain;
            its 'all_ranks' flag selects direct-child vs. scan matching.

    Returns:
        Possibly modified lineage with redirections applied.
    """
    if not domain_taxid or len(lineage_ids) <= 1:
        return lineage_ids

    domain_str = str(domain_taxid)
    if lineage_ids[0] != domain_str:
        return lineage_ids

    redirections = scope_config["redirections"]
    all_ranks = scope_config.get("all_ranks", False)

    # Locate the redirectable node. Canonical lineages place the recognised
    # top-level group (kingdom) as the domain's direct child, so only
    # lineage_ids[1] is inspected. All-ranks lineages may interpose clade /
    # realm nodes above it (e.g. Viruses -> clade Riboviria -> kingdom
    # Orthornavirae), pushing the key deeper -- scan for the first match so the
    # recognised group is still found instead of dumping the whole subtree into
    # the default fallback. When the match is at index 1 both paths behave
    # identically, so canonical routing is unchanged.
    match_index = None
    if all_ranks:
        for idx in range(1, len(lineage_ids)):
            if lineage_ids[idx] in redirections:
                match_index = idx
                break
    elif lineage_ids[1] in redirections:
        match_index = 1

    if match_index is not None:
        source_id = lineage_ids[match_index]
        target_id = str(redirections[source_id].get("target_id"))
        if target_id != source_id:
            logger.debug(
                f"[VIRTUAL-INSERT] taxid={source_id} -> virtual group {target_id}"
            )
            # Insert the virtual group directly above the matched taxon,
            # preserving any intermediate ranks resolved above it.
            return [
                *lineage_ids[:match_index], target_id, *lineage_ids[match_index:]
            ]
        return lineage_ids

    next_level_id = lineage_ids[1]
    default_id = scope_config["default_id"]
    if default_id is None:
        logger.debug(
            f"[FALLBACK-SKIP] taxid={next_level_id} has no redirection "
            f"rule and domain '{domain_taxid}' has no default fallback "
            "configured; preserving original lineage."
        )
        return lineage_ids

    logger.debug(f"[FALLBACK-DEFAULT] taxid={next_level_id} no rule -> {default_id}")
    return [domain_str, str(default_id), lineage_ids[-1]]


def _build_lineage_path(
    root: Node,
    lineage_ids: list[str],
    virtual_labels: dict[str, str],
    taxon_info: dict[str, tuple[str, str]],
) -> Node:
    """Walk the lineage from root creating nodes as needed.

    For each TaxID in the lineage, locates the matching child of the
    current node or creates a new one. The returned node is the leaf
    of the lineage path, ready to host sequence headers.

    Node metadata (rank and scientific_name) is set based on three
    sources, in order of preference:
        1. Virtual labels from the scope configuration.
        2. The registry-derived (name, rank) index.
        3. Defaults (rank='unknown', name=taxid string) when the TaxID
           is absent from the index.

    Args:
        root: Tree root.
        lineage_ids: Final lineage to materialize.
        virtual_labels: Map of virtual TaxID to human-readable label.
        taxon_info: Map of TaxID to (name, rank) for this lineage.

    Returns:
        The leaf node of the lineage path.
    """
    current = root
    for taxid_str in lineage_ids:
        existing_child = _find_child_by_name(current, taxid_str)
        if existing_child is not None:
            current = existing_child
            continue
        new_node = Node(taxid_str, parent=current)
        _annotate_node_metadata(
            new_node, taxid_str, virtual_labels, taxon_info
        )
        current = new_node

    return current


def _find_child_by_name(node: Node, name: str) -> Node | None:
    """Locate a direct child of a node by name attribute.

    Args:
        node: Parent node to search.
        name: Child name to find.

    Returns:
        The matching child Node, or None if no child has that name.
    """
    for child in node.children:
        if child.name == name:
            return child
    return None


def _annotate_node_metadata(
    node: Node,
    taxid_str: str,
    virtual_labels: dict[str, str],
    taxon_lookup: dict[str, tuple[str, str]],
) -> None:
    """Set rank and scientific_name on a freshly created lineage node.

    Reads the node's name and rank from the registry-derived index.
    Nodes whose TaxID is absent from the index (or is a virtual label)
    are annotated defensively.

    Args:
        node: The newly created Node to annotate.
        taxid_str: The TaxID string this node represents.
        virtual_labels: Map of virtual TaxID to human-readable label.
        taxon_lookup: Map of TaxID to (name, rank) for this lineage.
    """
    if taxid_str in virtual_labels:
        node.rank = _VIRTUAL_RANK_LABEL
        node.scientific_name = virtual_labels[taxid_str]
        return
    info = taxon_lookup.get(taxid_str)
    if info is None:
        node.rank = _UNKNOWN_RANK_LABEL
        node.scientific_name = taxid_str
        return
    name, rank = info
    node.rank = rank.lower().strip()
    node.scientific_name = name


def _attach_sequence_leaves(
    taxon_node: Node,
    accession_info: dict[str, Any],
    vault_path: str,
) -> None:
    """Attach all sequence headers of an accession as leaf nodes.

    For each valid header in the accession metadata, creates a leaf
    Node with rank='sequence', linked to the LMDB vault via the
    ``fasta_path`` and ``header_id`` attributes. Headers that already
    exist as children are skipped (idempotent across calls).

    Args:
        taxon_node: Parent taxon node to host the sequence leaves.
        accession_info: Accession metadata containing the headers list.
        vault_path: LMDB vault directory.
    """
    lmdb_path = os.path.join(vault_path, _LMDB_FILE_NAME)
    headers_list = accession_info.get("headers", [])

    for header_entry in headers_list:
        if not isinstance(header_entry, dict) or not header_entry.get("id"):
            continue

        header_id = header_entry["id"]
        if _find_child_by_name(taxon_node, header_id) is not None:
            continue

        sequence_node = Node(header_id, parent=taxon_node)
        sequence_node.rank = "sequence"
        sequence_node.header_id = str(header_id)
        sequence_node.fasta_path = lmdb_path
        sequence_node.scientific_name = accession_info.get("organism") or ""


def _log_noise_filter_summary(noise_filter: NoiseFilter) -> None:
    """Emit a summary log line with the noise filter's statistics.

    Args:
        noise_filter: NoiseFilter instance after tree construction.
    """
    stats = noise_filter.stats()
    total_hits = stats["name_hits"] + stats["rank_hits"]
    evaluated = stats["evaluated"]
    hit_percentage = 100.0 * total_hits / max(evaluated, 1)
    logger.info(
        f"NoiseFilter: evaluated {evaluated} nodes, "
        f"filtered {stats['name_hits']} by name + "
        f"{stats['rank_hits']} by rank "
        f"= {total_hits} total ({hit_percentage:.1f}%)"
    )
