"""End-to-end acceptance tests with live NCBI data.

These tests require network access, the NCBI Datasets CLI, and the NCBI
FTP/HTTPS endpoints. They are marked ``network`` and should be run
explicitly:

    python -m pytest tests/integration/test_ncbi_acceptance.py -m network -v

Scope: SARS-CoV-2 (TaxID 2697049), a single complete genome (~30 KB
compressed) that is stable, reference-quality, and fast to download.
The test exercises the full pipeline:

    discover_from_root → download_pending → generate_seqs_by_taxon_tree
        → compute_all_capacities → DatasetBuilder.build_node_dataset

The acceptance criteria mirror a production run but at tiny scale,
verifying that each stage receives valid input from the previous one.
"""

import json
import os

import pyarrow.parquet as pq
import pytest

from taxotreeset.core.generation.capacity import compute_all_capacities
from taxotreeset.dataset.builder import DatasetBuilder
from taxotreeset.dataset.tree_builder import generate_seqs_by_taxon_tree
from taxotreeset.io.downloader import NCBIDownloader
from taxotreeset.io.registry import NCBIRegistry
from taxotreeset.core.orchestrator import DiscoveryOrchestrator


_SARS_COV2_TAXID = 2697049
_SARS_COV2_ACCESSION = "GCF_009858895.2"
_SARS_COV2_HEADER = "NC_045512.2"
_SARS_COV2_GENOME_LEN = 29903
_MIN_SUBSEQ_LEN = 100
_MAX_SUBSEQ_LEN = 500


# ---------------------------------------------------------------------------
# module-scoped pipeline fixture
#
# Each stage runs once for the entire module so the download only happens
# once. Fixtures are chained: each stage receives the previous stage's
# output.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _acceptance_base(tmp_path_factory):
    """Shared temp directory for all module-scoped acceptance fixtures."""
    return tmp_path_factory.mktemp("acceptance")


@pytest.fixture(scope="module")
def sars_mapping_path(_acceptance_base):
    p = _acceptance_base / "mapping.json"
    p.write_text(json.dumps({"scopes": {}}), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def sars_registry(_acceptance_base, sars_mapping_path):
    registry_path = str(_acceptance_base / "registry.json")
    return NCBIRegistry(registry_path=registry_path, config_path=sars_mapping_path)


@pytest.fixture(scope="module")
def discovered_registry(sars_registry, sars_mapping_path):
    """Stage 1: discover SARS-CoV-2 from NCBI."""
    with open(sars_mapping_path, encoding="utf-8") as fh:
        mapping_config = json.load(fh)
    orchestrator = DiscoveryOrchestrator(
        registry=sars_registry,
        mapping_config=mapping_config,
    )
    orchestrator.discover_from_root(_SARS_COV2_TAXID)
    return sars_registry


@pytest.fixture(scope="module")
def vault_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("vault"))


@pytest.fixture(scope="module")
def downloaded_registry(discovered_registry, vault_dir):
    """Stage 2: download the SARS-CoV-2 genome from NCBI."""
    downloader = NCBIDownloader(
        registry=discovered_registry,
        vault_path=vault_dir,
    )
    downloader.download_all_pending()
    downloader.reconcile_with_vault()
    return discovered_registry


@pytest.fixture(scope="module")
def tree_root(downloaded_registry, vault_dir, sars_mapping_path):
    """Stage 3: build the taxonomic tree from the downloaded data."""
    return generate_seqs_by_taxon_tree(
        registry_path=downloaded_registry.registry_path,
        vault_path=vault_dir,
        domain_taxid="10239",
        mapping_path=sars_mapping_path,
    )


@pytest.fixture(scope="module")
def all_capacities(tree_root):
    """Stage 4: compute all node capacities via LMDB."""
    return compute_all_capacities(tree_root, min_len=_MIN_SUBSEQ_LEN)


@pytest.fixture(scope="module")
def sars_dataset_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("datasets"))


# ---------------------------------------------------------------------------
# Stage 1: discovery
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcceptanceDiscovery:
    def test_reference_accession_discovered(self, discovered_registry):
        assert _SARS_COV2_ACCESSION in discovered_registry.registry["accessions"]

    def test_accession_has_genome_length(self, discovered_registry):
        info = discovered_registry.registry["accessions"][_SARS_COV2_ACCESSION]
        assert info["total_sequence_length"] >= _SARS_COV2_GENOME_LEN

    def test_lineage_stored_for_species(self, discovered_registry):
        assert str(_SARS_COV2_TAXID) in discovered_registry.registry["lineages"]

    def test_lineage_contains_viruses(self, discovered_registry):
        lineage = discovered_registry.registry["lineages"][str(_SARS_COV2_TAXID)]
        taxids = {e["taxid"] for e in lineage}
        assert "10239" in taxids


# ---------------------------------------------------------------------------
# Stage 2: download
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcceptanceDownload:
    def test_accession_marked_downloaded(self, downloaded_registry):
        info = downloaded_registry.registry["accessions"][_SARS_COV2_ACCESSION]
        assert info["downloaded"] is True

    def test_lmdb_vault_exists(self, downloaded_registry, vault_dir):
        lmdb_path = os.path.join(vault_dir, "sequences.lmdb")
        assert os.path.exists(lmdb_path)

    def test_genome_readable_from_vault(self, downloaded_registry, vault_dir):
        from taxotreeset.dataset.utils import _read_single_sequence
        lmdb_path = os.path.join(vault_dir, "sequences.lmdb")
        seq = _read_single_sequence(lmdb_path, _SARS_COV2_HEADER)
        assert len(seq) >= _SARS_COV2_GENOME_LEN

    def test_genome_contains_only_iupac_bases(self, downloaded_registry, vault_dir):
        from taxotreeset.dataset.utils import _read_single_sequence
        lmdb_path = os.path.join(vault_dir, "sequences.lmdb")
        seq = _read_single_sequence(lmdb_path, _SARS_COV2_HEADER)
        valid = set("ACGTURYSWKMBDHVN")
        assert set(seq.upper()).issubset(valid)

    def test_pending_volume_is_zero_after_download(self, downloaded_registry):
        pending = downloaded_registry.get_pending_volume()
        assert pending == 0


# ---------------------------------------------------------------------------
# Stage 3: tree construction
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcceptanceTreeConstruction:
    def test_tree_root_exists(self, tree_root):
        assert tree_root is not None
        assert tree_root.name == "root"

    def test_viruses_domain_is_in_tree(self, tree_root):
        child_names = {c.name for c in tree_root.children}
        assert "10239" in child_names

    def test_sars_species_is_in_tree(self, tree_root):
        all_names = {n.name for n in tree_root.descendants}
        assert str(_SARS_COV2_TAXID) in all_names

    def test_sequence_leaf_attached_with_correct_header(self, tree_root):
        all_leaves = list(tree_root.leaves)
        seq_leaves = [leaf for leaf in all_leaves if getattr(leaf, "rank", "") == "sequence"]
        header_ids = {getattr(leaf, "header_id", None) for leaf in seq_leaves}
        assert _SARS_COV2_HEADER in header_ids


# ---------------------------------------------------------------------------
# Stage 4: capacity computation
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcceptanceCapacities:
    def test_capacities_dict_is_non_empty(self, all_capacities):
        assert len(all_capacities) > 0

    def test_sars_species_has_positive_capacity(self, all_capacities):
        cap = all_capacities.get(str(_SARS_COV2_TAXID))
        assert cap is not None
        assert cap > 0

    def test_species_capacity_within_genome_length_bounds(self, all_capacities):
        cap = all_capacities[str(_SARS_COV2_TAXID)]
        upper_bound = _SARS_COV2_GENOME_LEN - _MIN_SUBSEQ_LEN + 1
        assert cap <= upper_bound

    def test_domain_capacity_ge_species_capacity(self, all_capacities):
        domain_cap = all_capacities.get("10239", 0)
        species_cap = all_capacities.get(str(_SARS_COV2_TAXID), 0)
        assert domain_cap >= species_cap


# ---------------------------------------------------------------------------
# Stage 5: dataset materialization
# ---------------------------------------------------------------------------


@pytest.mark.network
class TestAcceptanceDatasetMaterialization:
    def _get_species_node(self, tree_root):
        for node in tree_root.descendants:
            if node.name == str(_SARS_COV2_TAXID):
                return node
        return None

    def test_can_compute_balanced_plan(self, tree_root, all_capacities):
        species_node = self._get_species_node(tree_root)
        assert species_node is not None
        seq_leaves = [
            leaf for leaf in species_node.leaves
            if getattr(leaf, "rank", "") == "sequence"
        ]
        assert len(seq_leaves) >= 1

    def test_builder_produces_parquet_files(self, tree_root, all_capacities, sars_dataset_dir):
        species_node = self._get_species_node(tree_root)
        assert species_node is not None

        output_dir = os.path.join(sars_dataset_dir, str(_SARS_COV2_TAXID))
        os.makedirs(output_dir, exist_ok=True)

        builder = DatasetBuilder(
            output_dir=sars_dataset_dir,
            max_subseq_len=_MAX_SUBSEQ_LEN,
            seed=42,
            output_format="parquet",
        )
        seq_leaves = [
            leaf for leaf in species_node.leaves
            if getattr(leaf, "rank", "") == "sequence"
        ]
        leaf = seq_leaves[0]
        # Single-leaf fraction split (the orchestrator does the real splitting;
        # here we just feed the builder a valid train/val/test task set).
        fractions = {"train": (0.0, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.0)}
        tasks = {
            split_name: [
                {
                    "fasta_path": leaf.fasta_path,
                    "header_id": leaf.header_id,
                    "start_pct": start,
                    "end_pct": end,
                    "n": 3,
                    "class_idx": 0,
                }
            ]
            for split_name, (start, end) in fractions.items()
        }

        job = (str(_SARS_COV2_TAXID), output_dir, tasks, _MAX_SUBSEQ_LEN, 42, "parquet")
        results = builder.build_node_dataset([job], parallel=False)
        assert results == [True]

        train_path = os.path.join(output_dir, "train.parquet")
        assert os.path.exists(train_path)

    def test_parquet_sequences_are_within_length_bounds(
        self, tree_root, all_capacities, sars_dataset_dir
    ):
        output_dir = os.path.join(sars_dataset_dir, str(_SARS_COV2_TAXID))
        train_path = os.path.join(output_dir, "train.parquet")
        if not os.path.exists(train_path):
            pytest.skip("train.parquet not produced by earlier test")

        table = pq.read_table(train_path)
        seqs = table.column("seq").to_pylist()
        for seq in seqs:
            assert len(seq) <= _MAX_SUBSEQ_LEN
            assert len(seq) >= _MIN_SUBSEQ_LEN
