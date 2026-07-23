"""Open-set eval-set builder (benchmark, P11-P2).

Turns the held-out clades recorded by the clade-holdout manifest (P11-P1) into a
labeled set of *novel* reads: fixed-length reads sampled from each held-out
genome, tagged with the genome's true lineage, its held-out clade, the expected
commit rank ``rho*`` (the deepest retained ancestor a classifier should back off
to), and the divergence bin. This is the ground truth a scorer (P11-P4) needs to
grade open-set back-off vs over-commitment.

Short track only (fixed-length, Illumina-like); the long-noisy track is P11-P3.
See ``docs/clade_holdout_benchmark.md``.
"""

import json
import random
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from taxotreeset.dataset.sequence_utils import extract_subseqs
from taxotreeset.dataset.utils import _read_single_sequence

_DEFAULT_READ_LENGTH = 150
_DEFAULT_READS_PER_GENOME = 200


def _header_index(accessions: dict) -> dict[str, tuple[str, str]]:
    """Map each genome header id to ``(leaf_taxid, lmdb_path)``."""
    index: dict[str, tuple[str, str]] = {}
    for acc in accessions.values():
        taxid = str(acc.get("taxid", ""))
        path = acc.get("local_path", "")
        for header in acc.get("headers", []):
            hid = header.get("id")
            if hid:
                index[hid] = (taxid, path)
    return index


def build_eval_reads(
    manifest_entries: list[dict],
    accessions: dict,
    lineages: dict,
    *,
    read_length: int = _DEFAULT_READ_LENGTH,
    reads_per_genome: int = _DEFAULT_READS_PER_GENOME,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build labeled eval-read rows from the held-out clades' genomes.

    Deterministic given ``seed`` (manifest entries and their member headers are
    iterated in sorted order). A genome shorter than ``read_length``, or one that
    cannot be read, is skipped.
    """
    header_index = _header_index(accessions)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for entry in sorted(manifest_entries, key=lambda e: str(e.get("taxid", ""))):
        held_out = str(entry.get("taxid", ""))
        rho_taxid = entry.get("expected_commit_taxid")
        rho_rank = entry.get("expected_commit_rank")
        distance_bin = entry.get("distance_bin")
        for header in sorted(entry.get("member_headers", [])):
            taxid, path = header_index.get(header, ("", ""))
            if not taxid or not path:
                continue
            seq = _read_single_sequence(path, header)
            if not seq:
                continue
            reads = extract_subseqs(
                seq, reads_per_genome, read_length, read_length, rng
            )
            lineage_json = json.dumps(
                [[str(n.get("taxid", "")), n.get("rank", "")]
                 for n in lineages.get(taxid, [])]
            )
            for j, read in enumerate(reads):
                rows.append(
                    {
                        "read_id": f"{header}:{j}",
                        "seq": read,
                        "source_header": header,
                        "true_leaf_taxid": taxid,
                        "true_lineage": lineage_json,
                        "held_out_taxid": held_out,
                        "expected_commit_taxid": rho_taxid,
                        "expected_commit_rank": rho_rank,
                        "distance_bin": distance_bin,
                    }
                )
    return rows


def write_eval_set(rows: list[dict], output_path: str) -> int:
    """Write eval rows to a parquet file; return the row count."""
    pq.write_table(pa.Table.from_pylist(rows), output_path)
    return len(rows)


def build_eval_set(
    manifest_path: str,
    accessions: dict,
    lineages: dict,
    output_path: str,
    *,
    read_length: int = _DEFAULT_READ_LENGTH,
    reads_per_genome: int = _DEFAULT_READS_PER_GENOME,
    seed: int = 0,
) -> tuple[int, int]:
    """Read the holdout manifest, build the eval reads, and write the parquet.

    Returns ``(n_reads, n_clades)``.
    """
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest.get("holdout", [])
    rows = build_eval_reads(
        entries, accessions, lineages,
        read_length=read_length, reads_per_genome=reads_per_genome, seed=seed,
    )
    write_eval_set(rows, output_path)
    return len(rows), len(entries)
