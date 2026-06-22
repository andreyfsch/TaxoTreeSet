"""Real on-disk LMDB vault fixture for behavioral capacity tests.

Lets tests drive the I/O-coupled capacity functions (``_capacity_exact``,
``_capacity_approximate``, ``compute_node_capacity``, and eventually the
bottom-up computer) through a genuine temporary vault instead of patching
``capacity._read_sequence_cached`` by path. Decoupling the tests from where a
symbol lives is the groundwork that lets the I/O core be moved out of
``capacity.py`` later without migrating the ~45 patch-by-path tests.

The vault is written exactly the way ``io/downloader.py`` writes it —
``zlib.compress(seq.encode("utf-8"))`` stored under the UTF-8-encoded
``header_id`` key in a directory-mode LMDB environment — so the production
reader ``dataset/utils._read_single_sequence`` consumes it unchanged.
"""

import zlib

import lmdb
from bigtree import Node

# 256 MiB is far more than any test sequence needs and well within the sparse
# allocation LMDB makes on disk (only written pages are materialized). The
# directory + ``max_dbs=0`` layout mirrors the reader's ``lmdb.open`` in
# ``dataset/utils._get_lmdb_env``.
_TEST_VAULT_MAP_SIZE = 256 * 1024 * 1024


def make_test_vault(tmp_path, sequences: dict[str, str]) -> str:
    """Write sequences to a temporary LMDB vault and return its path.

    Mirrors the downloader's on-disk format (zlib-compressed UTF-8 values
    keyed by the UTF-8 ``header_id``) so the production readers
    (``_read_single_sequence`` / ``_read_sequence_cached``) consume the vault
    with no monkeypatching.

    Args:
        tmp_path: A pytest ``tmp_path`` (or any ``pathlib.Path``) under which
            the ``vault`` directory is created.
        sequences: Mapping of ``header_id`` -> nucleotide sequence string.

    Returns:
        Filesystem path (str) to the LMDB vault directory, suitable for use as
        a sequence leaf's ``fasta_path``.
    """
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(vault_dir), map_size=_TEST_VAULT_MAP_SIZE, max_dbs=0)
    try:
        with env.begin(write=True) as txn:
            for header_id, sequence in sequences.items():
                txn.put(
                    header_id.encode("utf-8"),
                    zlib.compress(sequence.encode("utf-8")),
                )
    finally:
        env.sync()
        env.close()
    return str(vault_dir)


def make_vault_leaf(header_id: str, fasta_path: str, name: str | None = None) -> Node:
    """Return a sequence-rank leaf bound to a vault entry.

    Args:
        header_id: LMDB key previously written via :func:`make_test_vault`.
        fasta_path: Vault path returned by :func:`make_test_vault`.
        name: Optional node name; defaults to ``header_id``.

    Returns:
        A bigtree ``Node`` with ``rank="sequence"``, ``header_id`` and
        ``fasta_path`` set — the shape the capacity readers expect.
    """
    node = Node(name or header_id)
    node.rank = "sequence"
    node.header_id = header_id
    node.fasta_path = fasta_path
    return node


def true_unique_kmers(sequences, min_len: int) -> int:
    """Exact count of distinct ``min_len`` sliding-window substrings.

    Independent ground truth for the exact-capacity path: a plain Python set
    over every window of every sequence. Valid for arbitrary alphabets (pure
    ACGT or IUPAC-ambiguous), since exact capacity is, by definition, exact.

    Args:
        sequences: Iterable of sequence strings.
        min_len: Sliding-window size in bases.

    Returns:
        Number of distinct windows across all sequences.
    """
    seen: set[str] = set()
    for seq in sequences:
        for start in range(len(seq) - min_len + 1):
            seen.add(seq[start : start + min_len])
    return len(seen)
