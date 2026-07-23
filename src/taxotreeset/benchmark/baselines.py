"""Retained-only exact-match baseline glue for the open-set benchmark (P11-P5).

The head-to-head the FM-taxonomy literature routinely skips is a *production
k-mer baseline* facing the *same* open-set condition. TaxoTreeSet owns the two
ends of that comparison; the tool run in between (Kraken2 / Centrifuge index build
+ classify) is the user's:

1. ``export_retained_reference`` writes the reference genomes **with the held-out
   clades removed** — so the baseline's index, like the trained model, has never
   seen the novel clades — as a taxid-labeled FASTA (+ seqid->taxid map) ready for
   ``kraken2-build`` / ``centrifuge-build``.
2. ``parse_kraken2_output`` converts the tool's per-read output into the same
   ``read_id -> (taxid, rank)`` predictions the scorer (P11-P4) grades, so the
   baseline's native LCA back-off is scored on exactly the same footing as the
   model. Unclassified reads map to abstention.

See ``docs/clade_holdout_benchmark.md`` for the surrounding design.
"""

from typing import Any

from taxotreeset.dataset.utils import _read_single_sequence


def taxid_rank_map(lineages: dict) -> dict[str, str]:
    """Map every taxid appearing in any lineage to its rank."""
    ranks: dict[str, str] = {}
    for lineage in lineages.values():
        for node in lineage:
            ranks[str(node.get("taxid", ""))] = node.get("rank", "")
    return ranks


def export_retained_reference(
    held_out_taxids: set[str],
    accessions: dict,
    lineages: dict,
    out_fasta: str,
    out_seqid2taxid: str,
) -> int:
    """Write the retained genomes (held-out clades excluded) as a labeled FASTA.

    A genome is *excluded* when any taxid in its lineage is a held-out clade — i.e.
    it lives under a withheld clade. The rest are written as
    ``>{header}|kraken:taxid|{leaf_taxid}`` records plus a ``{header}\\t{taxid}``
    seqid->taxid map (both consumed by kraken2-build / centrifuge-build).

    Returns the number of genomes written.
    """
    held_out = {str(t) for t in held_out_taxids}
    written = 0
    with open(out_fasta, "w", encoding="utf-8") as fasta, \
            open(out_seqid2taxid, "w", encoding="utf-8") as seqmap:
        for acc in accessions.values():
            taxid = str(acc.get("taxid", ""))
            lineage_taxids = {
                str(node.get("taxid", "")) for node in lineages.get(taxid, [])
            }
            if lineage_taxids & held_out:
                continue  # under a held-out clade -> keep the baseline open-set
            path = acc.get("local_path", "")
            for header in acc.get("headers", []):
                hid = header.get("id")
                if not hid or not path:
                    continue
                seq = _read_single_sequence(path, hid)
                if not seq:
                    continue
                fasta.write(f">{hid}|kraken:taxid|{taxid}\n{seq}\n")
                seqmap.write(f"{hid}\t{taxid}\n")
                written += 1
    return written


def parse_kraken2_output(
    lines: Any, taxid_rank: dict[str, str]
) -> dict[str, tuple[str | None, str | None]]:
    """Parse Kraken2 per-read output into scorer predictions.

    Each Kraken2 line is ``C|U <read_id> <taxid> <length> <lca-map>``; an
    unclassified read (``U``) or taxid ``0`` becomes an abstention
    (``(None, None)``). The assigned taxid's rank is looked up in ``taxid_rank``.
    """
    predictions: dict[str, tuple[str | None, str | None]] = {}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        status, read_id, taxid = parts[0], parts[1], parts[2].strip()
        if status != "C" or taxid in ("0", ""):
            predictions[read_id] = (None, None)
        else:
            predictions[read_id] = (taxid, taxid_rank.get(taxid))
    return predictions
