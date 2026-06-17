"""Shared fixtures for TaxoTreeSet integration tests.

The synthetic vault is a minimal LMDB database containing three
deterministic pseudo-random ACGT sequences, wired to a registry JSON
that describes a small but structurally complete viral hierarchy:

    Viruses (10239, superkingdom)
    ├── Coronaviridae (11118, family)
    │   ├── SARS-CoV-2 (2697049, species)  → NC_045512
    │   └── Mouse hepatitis virus (11234, species)  → NC_001846
    └── Adenoviridae (12227, family)
        └── Human mastadenovirus C (10509, species)  → NC_001407

Sequences are 2000 bp each, generated from a deterministic LCG so
capacity tests are stable across runs.
"""

import json
import zlib

import lmdb
import pytest

_DOMAIN_TAXID = "10239"

_SPECIES = {
    "2697049": {
        "name": "Severe acute respiratory syndrome coronavirus 2",
        "rank": "species",
        "organism": "SARS-CoV-2",
        "accession": "GCF_001",
        "header": "NC_045512",
        "lineage": [
            {
                "taxid": "2697049",
                "rank": "species",
                "name": "Severe acute respiratory syndrome coronavirus 2",
            },
            {"taxid": "11118", "rank": "family", "name": "Coronaviridae"},
            {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
        ],
    },
    "11234": {
        "name": "Mouse hepatitis virus",
        "rank": "species",
        "organism": "MHV",
        "accession": "GCF_002",
        "header": "NC_001846",
        "lineage": [
            {"taxid": "11234", "rank": "species", "name": "Mouse hepatitis virus"},
            {"taxid": "11118", "rank": "family", "name": "Coronaviridae"},
            {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
        ],
    },
    "10509": {
        "name": "Human mastadenovirus C",
        "rank": "species",
        "organism": "HAdV-C",
        "accession": "GCF_003",
        "header": "NC_001407",
        "lineage": [
            {"taxid": "10509", "rank": "species", "name": "Human mastadenovirus C"},
            {"taxid": "12227", "rank": "family", "name": "Adenoviridae"},
            {"taxid": "10239", "rank": "superkingdom", "name": "Viruses"},
        ],
    },
}


def _deterministic_sequence(seed: int, length: int = 2000) -> str:
    """LCG-based deterministic ACGT sequence, stable across platforms."""
    bases = "ACGT"
    x = seed & 0xFFFFFFFF
    result = []
    for _ in range(length):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        result.append(bases[x & 3])
    return "".join(result)


_SEQUENCES: dict[str, str] = {
    "NC_045512": _deterministic_sequence(seed=1, length=2000),
    "NC_001846": _deterministic_sequence(seed=2, length=2000),
    "NC_001407": _deterministic_sequence(seed=3, length=2000),
}


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory):
    """Create vault + registry on disk; return paths and sequences."""
    base = tmp_path_factory.mktemp("integration")
    vault_dir = base / "vault"
    vault_dir.mkdir()
    lmdb_dir = vault_dir / "sequences.lmdb"
    lmdb_dir.mkdir()

    env = lmdb.open(str(lmdb_dir), map_size=32 * 1024 * 1024, max_dbs=0)
    with env.begin(write=True) as txn:
        for header_id, seq in _SEQUENCES.items():
            txn.put(header_id.encode("utf-8"), zlib.compress(seq.encode("utf-8")))
    env.close()

    lmdb_path = str(lmdb_dir)
    registry_data = _build_registry(lmdb_path)
    registry_path = base / "registry.json"
    registry_path.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    mapping_path = base / "mapping.json"
    mapping_path.write_text(json.dumps({"scopes": {}}), encoding="utf-8")

    return {
        "vault_dir": str(vault_dir),
        "lmdb_path": lmdb_path,
        "registry_path": str(registry_path),
        "mapping_path": str(mapping_path),
        "registry_data": registry_data,
        "sequences": _SEQUENCES,
        "domain_taxid": _DOMAIN_TAXID,
    }


def _build_registry(lmdb_path: str) -> dict:
    taxons = {}
    accessions = {}
    lineages = {}

    for species_taxid, spec in _SPECIES.items():
        acc_id = spec["accession"]
        header = spec["header"]

        taxons[species_taxid] = [acc_id]
        accessions[acc_id] = {
            "taxid": species_taxid,
            "organism": spec["organism"],
            "is_reference": True,
            "total_sequence_length": len(_SEQUENCES[header]),
            "downloaded": True,
            "download_deferred": False,
            "local_path": lmdb_path,
            "headers": [{"id": header, "name": f"{spec['organism']}, complete genome"}],
        }
        lineages[species_taxid] = spec["lineage"]

    return {
        "last_update": None,
        "taxons": taxons,
        "accessions": accessions,
        "lineages": lineages,
        "capacities": {},
    }
