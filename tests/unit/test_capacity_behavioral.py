"""Behavioral (vault-driven) tests for the I/O-coupled capacity functions.

Unlike ``test_capacity.py`` — which patches ``capacity._read_sequence_cached`` /
``capacity._read_single_sequence`` by path in ~45 places — these tests exercise
``_capacity_exact``, ``_capacity_approximate`` and ``compute_node_capacity``
against a *real* temporary LMDB vault written by
:func:`tests.unit._vault_fixture.make_test_vault`. Because nothing is patched by
symbol name, these tests keep passing even when the I/O core is later moved out
of ``capacity.py`` — that is the point: they are the groundwork that unblocks the
full decomposition (P7 Part C).
"""

import random

from taxotreeset.core.generation.capacity import (
    _capacity_approximate,
    _capacity_exact,
    _read_sequence_cached,
    compute_node_capacity,
)
from taxotreeset.dataset.utils import _read_single_sequence

from tests.unit._vault_fixture import (
    make_test_vault,
    make_vault_leaf,
    true_unique_kmers,
)

# Deterministic pure-ACGT sequences with internal repeats (so unique-window
# counts are strictly below the raw window count, exercising dedup).
_SEQ_A = "ACGTACGTACGTTTGGCCAAACGTACGTAACCGGTTACGT"
_SEQ_B = "TTGGCCAAACGTACGTACGTGATTACAGATTACAGATTACA"
# A sequence carrying an ambiguity code (N): forces the exact path's string-set
# branch alongside the 2-bit-packed branch.
_SEQ_N = "ACGTACGTNNACGTACGTACGTRYACGTACGT"

_MIN_LEN = 8


def _nonrepetitive_sequence() -> str:
    """A 300-base ACGT sequence whose 8-mer windows are all distinct.

    ``random.Random(2)`` is a fixed, platform-stable PRNG seed verified to
    produce no repeated ``_MIN_LEN``-windows, so the exact unique-window count
    equals the raw window count. With no intra-chunk duplicates, the Bloom
    estimator has nothing to over-count, which lets the approximate path be
    asserted exactly (see ``_consume_sequence_into_bloom_vectorized``: the
    documented drift comes purely from duplicate windows within a 2048-window
    chunk).
    """
    rnd = random.Random(2)
    return "".join(rnd.choice("ACGT") for _ in range(300))


# ─────────────────────────────────────────────────────────────────────────────
# Fixture sanity: the temp vault round-trips through the production readers
# ─────────────────────────────────────────────────────────────────────────────


def test_vault_roundtrips_through_production_readers(tmp_path):
    """make_test_vault output is read back verbatim by the real readers."""
    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A, "NC_B": _SEQ_B})

    assert _read_single_sequence(vault, "NC_A") == _SEQ_A
    assert _read_sequence_cached(vault, "NC_B") == _SEQ_B


def test_vault_missing_key_reads_empty(tmp_path):
    """A header absent from the vault yields the empty-string error sentinel."""
    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A})

    assert _read_single_sequence(vault, "NC_MISSING") == ""


# ─────────────────────────────────────────────────────────────────────────────
# _capacity_exact over a real vault (no patching)
# ─────────────────────────────────────────────────────────────────────────────


def test_capacity_exact_single_leaf_matches_ground_truth(tmp_path):
    """Exact capacity equals the independent unique-window count."""
    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A})
    leaf = make_vault_leaf("NC_A", vault)

    expected = true_unique_kmers([_SEQ_A], _MIN_LEN)
    assert _capacity_exact([leaf], _MIN_LEN) == expected


def test_capacity_exact_dedups_across_leaves(tmp_path):
    """Windows shared by two leaves are counted once (set union semantics)."""
    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A, "NC_B": _SEQ_B})
    leaves = [make_vault_leaf("NC_A", vault), make_vault_leaf("NC_B", vault)]

    expected = true_unique_kmers([_SEQ_A, _SEQ_B], _MIN_LEN)
    # The two sequences share the "TTGGCCAAACGTACGT..." prefix region, so the
    # union is strictly smaller than the sum of the per-leaf capacities.
    per_leaf_sum = true_unique_kmers([_SEQ_A], _MIN_LEN) + true_unique_kmers(
        [_SEQ_B], _MIN_LEN
    )
    assert expected < per_leaf_sum
    assert _capacity_exact(leaves, _MIN_LEN) == expected


def test_capacity_exact_handles_ambiguous_bases(tmp_path):
    """Ambiguous (non-ACGT) windows are counted exactly via the string set."""
    vault = make_test_vault(tmp_path, {"NC_N": _SEQ_N})
    leaf = make_vault_leaf("NC_N", vault)

    expected = true_unique_kmers([_SEQ_N], _MIN_LEN)
    assert _capacity_exact([leaf], _MIN_LEN) == expected


def test_capacity_exact_sequence_shorter_than_window(tmp_path):
    """A sequence shorter than the window contributes no capacity."""
    vault = make_test_vault(tmp_path, {"NC_SHORT": "ACGTAC"})
    leaf = make_vault_leaf("NC_SHORT", vault)

    assert _capacity_exact([leaf], _MIN_LEN) == 0


# ─────────────────────────────────────────────────────────────────────────────
# _capacity_approximate over a real vault (no patching)
# ─────────────────────────────────────────────────────────────────────────────


def test_capacity_approximate_exact_for_nonrepetitive_sequence(tmp_path):
    """With no duplicate windows, the Bloom estimate equals the exact count.

    At this fill (293 items) the filter's false-positive rate is effectively
    zero and there are no intra-chunk duplicates to over-count, so the
    estimator is exact — the strongest assertion the approximate path admits.
    """
    seq = _nonrepetitive_sequence()
    vault = make_test_vault(tmp_path, {"NC_U": seq})
    leaf = make_vault_leaf("NC_U", vault)

    n_windows = len(seq) - _MIN_LEN + 1
    exact = true_unique_kmers([seq], _MIN_LEN)
    assert exact == n_windows  # sanity: the fixed seed really is repeat-free

    assert _capacity_approximate([leaf], _MIN_LEN) == exact


def test_capacity_approximate_bounded_for_repetitive_input(tmp_path):
    """Bloom never undercounts and stays at or below the raw window total.

    For short, repetitive sequences the vectorized estimator over-counts
    duplicate windows that fall in the same chunk, so it sits between the exact
    unique count and the raw window count rather than hitting exact. This pins
    that documented direction and bound.
    """
    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A, "NC_B": _SEQ_B})
    leaves = [make_vault_leaf("NC_A", vault), make_vault_leaf("NC_B", vault)]

    exact = true_unique_kmers([_SEQ_A, _SEQ_B], _MIN_LEN)
    total_windows = (len(_SEQ_A) - _MIN_LEN + 1) + (len(_SEQ_B) - _MIN_LEN + 1)
    approx = _capacity_approximate(leaves, _MIN_LEN)
    assert exact <= approx <= total_windows


# ─────────────────────────────────────────────────────────────────────────────
# compute_node_capacity dispatch over a real vault (no patching)
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_node_capacity_exact_via_node_leaves(tmp_path):
    """Exact dispatch gathers descendant sequence leaves from the tree."""
    from bigtree import Node

    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A, "NC_B": _SEQ_B})
    parent = Node("genus")
    parent.rank = "genus"
    for leaf in (make_vault_leaf("NC_A", vault), make_vault_leaf("NC_B", vault)):
        leaf.parent = parent

    expected = true_unique_kmers([_SEQ_A, _SEQ_B], _MIN_LEN)
    assert compute_node_capacity(parent, _MIN_LEN, {}, mode="exact") == expected


def test_compute_node_capacity_uses_leaf_cache_when_present(tmp_path):
    """A populated leaf_cache short-circuits the tree scan with the same result."""
    from bigtree import Node

    vault = make_test_vault(tmp_path, {"NC_A": _SEQ_A, "NC_B": _SEQ_B})
    parent = Node("genus")
    parent.rank = "genus"
    leaves = [make_vault_leaf("NC_A", vault), make_vault_leaf("NC_B", vault)]
    leaf_cache = {str(parent.name): leaves}

    expected = true_unique_kmers([_SEQ_A, _SEQ_B], _MIN_LEN)
    assert compute_node_capacity(parent, _MIN_LEN, leaf_cache, mode="exact") == expected


def test_compute_node_capacity_approximate_matches_exact(tmp_path):
    """Approximate dispatch equals exact for a repeat-free node (no drift)."""
    from bigtree import Node

    seq = _nonrepetitive_sequence()
    vault = make_test_vault(tmp_path, {"NC_U": seq})
    parent = Node("genus")
    parent.rank = "genus"
    leaf_cache = {"genus": [make_vault_leaf("NC_U", vault)]}

    exact = true_unique_kmers([seq], _MIN_LEN)
    approx = compute_node_capacity(parent, _MIN_LEN, leaf_cache, mode="approximate")
    assert approx == exact


def test_compute_node_capacity_no_sequence_leaves_returns_zero(tmp_path):
    """A node with no sequence-rank descendants has zero capacity."""
    from bigtree import Node

    parent = Node("empty_genus")
    parent.rank = "genus"
    child = Node("species_no_seqs", parent=parent)
    child.rank = "species"

    assert compute_node_capacity(parent, _MIN_LEN, {}, mode="exact") == 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_all_capacities: the parallel bottom-up pass over a real vault. Guards
# the _bottomup.py extraction — the workers run in spawn subprocesses, so a broken
# pickle/module-global would surface here (and can't be mocked). Parallel must
# equal serial and the exact ground truth.
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_all_capacities_parallel_matches_serial_and_truth(tmp_path):
    from bigtree import Node

    from taxotreeset.core.generation.capacity import compute_all_capacities

    seqs = {
        "NC_A": _SEQ_A, "NC_B": _SEQ_B,
        "NC_C": _SEQ_A[::-1], "NC_D": _SEQ_B[::-1],   # 4 leaves -> 2 workers busy
    }
    vault = make_test_vault(tmp_path, seqs)

    def build_tree():
        root = Node("root")
        root.rank = "superkingdom"
        for i, hid in enumerate(seqs):
            sp = Node(f"sp{i}", parent=root)
            sp.rank = "species"
            make_vault_leaf(hid, vault).parent = sp
        return root

    spill_serial = tmp_path / "serial"
    spill_serial.mkdir()
    spill_parallel = tmp_path / "parallel"
    spill_parallel.mkdir()

    serial = compute_all_capacities(
        build_tree(), _MIN_LEN, spill_dir=str(spill_serial),
        n_workers=1, n_gpu_workers=0,
    )
    parallel = compute_all_capacities(
        build_tree(), _MIN_LEN, spill_dir=str(spill_parallel),
        n_workers=2, n_gpu_workers=0,
    )

    assert parallel == serial                       # spawn workers agree with serial
    assert serial["root"] == true_unique_kmers(seqs.values(), _MIN_LEN)
