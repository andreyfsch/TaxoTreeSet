"""Integration test for the domain gate (P4) on the synthetic viral tree.

With the default ``reject_cross_domain="auto"`` and a ``viruses`` root, the domain
node (10239) must get its own binary head whose negatives are the cross-domain
genomes only, split evenly by domain; the deeper heads must not see them. The
cross-domain accessions are written into a private copy of the synthetic vault and
registry as already acquired and downloaded, so no network is touched.
"""

import json
import zlib
from pathlib import Path
from unittest.mock import patch

import lmdb
import pandas as pd
import pytest

from tests.integration.conftest import (
    _SEQUENCES,
    _build_registry,
    _deterministic_sequence,
)

_MIN_LEN = 10

# Two bacteria (one with two sequences) and one scaffold-level eukaryote with more
# sequences than the per-genome cap: by leaf or by length the eukaryote would take
# almost every negative window.
_CROSS = {
    "GCF_B1": ("bacteria", {"B1_chr": 3000}),
    "GCF_B2": ("bacteria", {"B2_chr": 2500, "B2_pl": 800}),
    "GCF_E1": ("eukaryotes", {f"E1_scaf{i}": 1500 for i in range(30)}),
}


@pytest.fixture(scope="module")
def gate_output(tmp_path_factory):
    from taxotreeset.core.generation_orchestrator import GenerationOrchestrator
    from taxotreeset.io.registry import NCBIRegistry

    base = tmp_path_factory.mktemp("gate")
    lmdb_dir = base / "vault" / "sequences.lmdb"
    lmdb_dir.mkdir(parents=True)
    env = lmdb.open(str(lmdb_dir), map_size=64 * 1024 * 1024, max_dbs=0)
    seed = 100
    with env.begin(write=True) as txn:
        for header, seq in _SEQUENCES.items():
            txn.put(header.encode(), zlib.compress(seq.encode()))
        for _acc, (_dom, seqs) in _CROSS.items():
            for header, length in seqs.items():
                seed += 1
                txn.put(header.encode(),
                        zlib.compress(_deterministic_sequence(seed, length).encode()))
    env.close()

    reg = _build_registry(str(lmdb_dir))
    reg["cross_domain_negatives"] = list(_CROSS)
    for acc, (dom, seqs) in _CROSS.items():
        reg["accessions"][acc] = {
            "taxid": "1", "organism": acc, "is_reference": True,
            "total_sequence_length": sum(seqs.values()),
            "downloaded": True, "download_deferred": False,
            "local_path": str(lmdb_dir), "cross_domain": True,
            "cross_domain_group": dom,
            "headers": [{"id": h, "name": h} for h in seqs],
        }
    registry_path = base / "registry.json"
    registry_path.write_text(json.dumps(reg), encoding="utf-8")
    mapping_path = base / "mapping.json"
    mapping_path.write_text(json.dumps({"scopes": {}}), encoding="utf-8")

    output_dir = str(base / "out")
    orch = GenerationOrchestrator(
        registry=NCBIRegistry(registry_path=str(registry_path)),
        vault_path=str(base / "vault"),
        output_dir=output_dir,
        config_path=str(mapping_path),
        min_subseq_len=_MIN_LEN,
        max_subseq_len=200,
        max_n_per_class=100,
        min_leaves_per_class=1,
        rare_taxa_strategy="keep",
        n_gpu_workers=0,
        n_workers=1,
        binary_only=True,
        binary_budget=60,
    )
    # Already acquired (written above): the acquisition step must not reach NCBI.
    with patch.object(GenerationOrchestrator, "_acquire_cross_domain_negatives") as acq:
        orch.run_pipeline(target_group="viruses", sync=False, abundance_threshold=1)
    return {"output_dir": output_dir, "acquire_calls": acq.call_count}


def _manifest(out):
    return json.loads((Path(out) / "manifest_viruses.json").read_text())


class TestDomainGate:
    def test_auto_resolves_and_acquires_without_sync(self, gate_output):
        meta = json.loads(
            (Path(gate_output["output_dir"]) / "run_metadata_viruses.json").read_text())
        assert meta["parameters"]["reject_cross_domain"] == [
            "bacteria", "archaea", "eukaryotes"]
        assert gate_output["acquire_calls"] == 1

    def test_domain_node_gets_a_binary_head(self, gate_output):
        manifest = _manifest(gate_output["output_dir"])
        assert "10239" in manifest
        head = Path(manifest["10239"]["directory_path"])
        # The synthetic capacities give a 4-window budget, too small for a val
        # split (the deeper heads lack one too); train must hold both classes.
        train = pd.read_parquet(head / "train.parquet")["class_idx"].value_counts()
        assert train.get(0, 0) > 0 and train.get(1, 0) > 0
        assert len(pd.read_parquet(head / "test.parquet")) > 0

    def test_gate_negatives_are_cross_domain_only_and_even(self, gate_output):
        prov = _manifest(gate_output["output_dir"])["10239"]["reject_provenance"]
        assert prov["near"]["windows"] == 0
        cross = prov["cross_domain"]
        assert set(cross) == {"cross_domain:bacteria", "cross_domain:eukaryotes"}
        assert cross["cross_domain:bacteria"]["genomes"] == 2
        assert cross["cross_domain:eukaryotes"]["genomes"] == 1
        wb = cross["cross_domain:bacteria"]["windows"]
        we = cross["cross_domain:eukaryotes"]["windows"]
        assert wb + we == prov["far"]["windows"] and abs(wb - we) <= 1
        # the 30-scaffold eukaryote is capped per genome, not by leaf count
        assert prov["far"]["leaves_used"] <= 3 + 20

    def test_deeper_heads_keep_intra_tree_negatives(self, gate_output):
        manifest = _manifest(gate_output["output_dir"])
        deeper = [t for t in manifest if t != "10239"]
        assert deeper
        for t in deeper:
            assert "cross_domain" not in manifest[t]["reject_provenance"], t
