"""Open-set eval-set builder (benchmark, P11-P2 short track + P11-P3 long track).

Turns the held-out clades recorded by the clade-holdout manifest (P11-P1) into a
labeled set of *novel* reads: reads sampled from each held-out genome, tagged with
the genome's true lineage, its held-out clade, the expected commit rank ``rho*``
(the deepest retained ancestor a classifier should back off to), and the
divergence bin — the ground truth a scorer (P11-P4) needs to grade open-set
back-off vs over-commitment.

Two tracks share those labels so results are directly comparable:

- **short/accurate** (Illumina-like): fixed-length reads, no error model;
- **long/noisy** (ONT/PacBio-like): longer, variable-length reads passed through an
  indel-dominated, homopolymer-aware :class:`ErrorModel`.

See ``docs/clade_holdout_benchmark.md``.
"""

import json
import random
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from taxotreeset.dataset.sequence_utils import extract_subseqs
from taxotreeset.dataset.utils import _read_single_sequence

_BASES = "ACGT"


@dataclass(frozen=True)
class ErrorModel:
    """Indel-dominated, homopolymer-aware read error model (ONT/PacBio-like).

    Rates are per-base probabilities. Within a homopolymer run (a base equal to
    the previous one in the source), the insertion/deletion rates are multiplied by
    ``homopolymer_factor`` — the dominant long-read error mode. Substitutions are
    uniform over the other three bases. All-zero rates make the model the identity.
    """

    sub_rate: float = 0.01
    ins_rate: float = 0.02
    del_rate: float = 0.02
    homopolymer_factor: float = 2.0


def apply_errors(seq: str, model: ErrorModel, rng: random.Random) -> str:
    """Return ``seq`` with substitutions, insertions, and deletions applied."""
    out: list[str] = []
    prev = ""
    for base in seq:
        boost = model.homopolymer_factor if base == prev else 1.0
        prev = base
        if rng.random() < model.del_rate * boost:
            continue  # deletion
        emit = base
        if rng.random() < model.sub_rate:
            alt = [b for b in _BASES if b != base]
            if alt:
                emit = rng.choice(alt)
        out.append(emit)
        if rng.random() < model.ins_rate * boost:
            out.append(rng.choice(_BASES))
    return "".join(out)


def build_eval_reads(
    manifest_entries: list[dict],
    accessions: dict,
    lineages: dict,
    *,
    min_len: int = 150,
    max_len: int = 150,
    error_model: ErrorModel | None = None,
    track: str = "short",
    reads_per_genome: int = 200,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build labeled eval-read rows from the held-out clades' genomes.

    Reads are sampled at length in ``[min_len, max_len]`` (equal for the fixed
    short track); when ``error_model`` is given each read is passed through it (the
    long track). Deterministic given ``seed`` (manifest entries and member headers
    are iterated in sorted order). Genomes shorter than ``min_len``, or unreadable,
    are skipped.
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
            if len(seq) < min_len:
                continue
            reads = extract_subseqs(
                seq, reads_per_genome, min_len, min(max_len, len(seq)), rng
            )
            lineage_json = json.dumps(
                [[str(n.get("taxid", "")), n.get("rank", "")]
                 for n in lineages.get(taxid, [])]
            )
            for j, read in enumerate(reads):
                if error_model is not None:
                    read = apply_errors(read, error_model, rng)
                rows.append(
                    {
                        "read_id": f"{header}:{j}",
                        "seq": read,
                        "read_length": len(read),
                        "track": track,
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
    min_len: int = 150,
    max_len: int = 150,
    error_model: ErrorModel | None = None,
    track: str = "short",
    reads_per_genome: int = 200,
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
        min_len=min_len, max_len=max_len, error_model=error_model,
        track=track, reads_per_genome=reads_per_genome, seed=seed,
    )
    write_eval_set(rows, output_path)
    return len(rows), len(entries)
