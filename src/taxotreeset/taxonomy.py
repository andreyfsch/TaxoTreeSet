"""Shared taxonomy identifier resolution.

Resolves a user-supplied scope reference -- a numeric NCBI TaxID or a
clade scientific name -- to a TaxID string. Tries the local taxoniq
snapshot first and falls back to a live NCBI Datasets CLI lookup for
names or IDs newer than the snapshot, mirroring the lineage-resolution
strategy used during discovery.
"""
import json
import logging
import subprocess

import taxoniq

logger = logging.getLogger("TaxoTreeSet.Taxonomy")


def resolve_to_taxid(reference: str) -> str:
    """Resolve a TaxID or clade name to a TaxID string.

    Args:
        reference: A numeric NCBI TaxID (e.g. "2731619") or a clade
            scientific name (e.g. "Caudoviricetes").

    Returns:
        The resolved NCBI TaxID as a string.

    Raises:
        ValueError: If the reference cannot be resolved by either
            taxoniq or the NCBI taxonomy fallback.
    """
    reference = reference.strip()
    if reference.isdigit():
        return reference

    # A non-numeric reference is a clade name. Try taxoniq, then NCBI.
    try:
        taxon = taxoniq.Taxon(scientific_name=reference)
        return str(taxon.tax_id)
    except (KeyError, taxoniq.TaxoniqException):
        pass

    taxid = _resolve_name_via_ncbi(reference)
    if taxid is None:
        raise ValueError(
            f"Could not resolve {reference!r} to a TaxID via taxoniq or "
            "the NCBI taxonomy fallback. Pass a numeric TaxID or a valid "
            "clade scientific name."
        )
    return taxid


def _resolve_name_via_ncbi(name: str) -> str | None:
    """Resolve a clade name to a TaxID via the NCBI Datasets CLI.

    Args:
        name: Clade scientific name to resolve.

    Returns:
        The TaxID as a string, or None if the lookup yields nothing.
    """
    command = [
        "datasets",
        "summary",
        "taxonomy",
        "taxon",
        name,
        "--as-json-lines",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.debug("NCBI name resolution failed for %r: %s", name, exc)
        return None

    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        taxonomy = payload.get("taxonomy", {})
        taxid = taxonomy.get("tax_id")
        if taxid is not None:
            return str(taxid)
    return None
