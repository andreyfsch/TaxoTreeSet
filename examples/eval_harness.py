#!/usr/bin/env python3
"""Score PhyloCascadeGLM and the three reference baselines on the same eval reads.

This script is NOT part of the taxotreeset package and is not installed with it.
Like examples/finetune_head.py it manages its own dependencies: it needs
`phylocascadeglm` on the path for the cascade half (the baseline half needs only
taxotreeset). Paths default to the viral pilot layout; override them on the CLI.


Feeds taxotreeset.benchmark.scorer.score_reads, which grades every read against the
expected commit rho* and already aggregates by divergence bin. This module only
produces the `read_id -> (taxid, rank)` maps it consumes.

Three things this harness gets right that a naive driver does not:

1. ENTRY POINTS. The pilot bundle is a forest: its heads form disjoint subtrees with
   no packed node above them (there is no head at 10239, and none is needed — the
   root head is permissive by design, passing everything to its children). A single
   `root_taxid` therefore reaches exactly one subtree, and pack()'s default of
   "10239" makes _search_binary return `no_adapter` for every read. We emulate the
   permissive root faithfully: run every forest root, let each subtree self-verify,
   and keep the accepting ones.

2. ONE CALL PER READ. Classifier.classify(reads=[...]) pools the windows of ALL
   reads and fuses them by mean logits — it is built for many reads *from one
   organism*. The eval set is 12,350 independent reads, so each gets its own call.

3. BUNDLE-EXHAUSTED vs GENUINE TERMINAL. In a thin-slice bundle a taxon whose real
   children were never trained has no entry in _meta.tree, so registry.children()
   is empty and the node is reported stop_reason="leaf" — indistinguishable from
   "this is the answer". We flag it, because for a 60-head slice of a 16,481-head
   tree that distinction changes the interpretation of most deep calls.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path("/mnt/f/taxotreeset_viruses")
EVAL_PARQUET = BASE / "baselines" / "eval_reads.parquet"


# --------------------------------------------------------------------------- #
# taxid -> rank, needed because the scorer grades (taxid, rank) pairs
# --------------------------------------------------------------------------- #
def build_rank_map(eval_rows: list[dict]) -> dict[str, str]:
    """Ranks for every taxon that appears on any true lineage, plus the manifest."""
    ranks: dict[str, str] = {}
    for row in eval_rows:
        lineage = row["true_lineage"]
        if isinstance(lineage, str):
            lineage = json.loads(lineage)
        for taxid, rank in lineage:
            ranks.setdefault(str(taxid), rank)
    man = BASE / "datasets_binary_allranks" / "manifest_viruses.json"
    if man.exists():
        for taxid, entry in json.loads(man.read_text()).items():
            if isinstance(entry, dict) and entry.get("rank"):
                ranks.setdefault(str(taxid), entry["rank"])
    return ranks


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def kraken2_predictions(path: Path, ranks: dict[str, str]) -> dict:
    """kraken2.out: <C|U>\tread_id\ttaxid\tlen\tkmer-map. U == abstain."""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 3 or f[0] != "C":
            continue
        taxid = f[2].strip()
        if taxid in ("0", ""):
            continue
        out[f[1]] = (taxid, ranks.get(taxid))
    return out


def kaiju_predictions(path: Path, ranks: dict[str, str]) -> dict:
    """kaiju.out: <C|U>\tread_id\ttaxid\t... U == abstain."""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 3 or f[0] != "C":
            continue
        taxid = f[2].strip()
        if taxid in ("0", ""):
            continue
        out[f[1]] = (taxid, ranks.get(taxid))
    return out


def centrifuge_predictions(path: Path, ranks: dict[str, str]) -> dict:
    """centrifuge.out is one row per HIT, so a read can appear several times.

    Keep the best-scoring hit per read; ties break on the longer match, then on
    the smaller taxid so the result is deterministic across runs.
    """
    best: dict[str, tuple[tuple[float, float, int], str]] = {}
    lines = path.read_text().splitlines()
    if lines and lines[0].lower().startswith("readid"):
        lines = lines[1:]
    for line in lines:
        f = line.split("\t")
        if len(f) < 6:
            continue
        read_id, taxid = f[0], f[2].strip()
        if taxid in ("0", "", "unclassified"):
            continue
        try:
            score, hitlen = float(f[3]), float(f[5])
        except ValueError:
            continue
        # Negated taxid so the smaller one wins the final tie -> deterministic.
        key = (score, hitlen, -int(taxid) if taxid.isdigit() else 0)
        if read_id not in best or key > best[read_id][0]:
            best[read_id] = (key, taxid)
    return {rid: (t, ranks.get(t)) for rid, (_, t) in best.items()}


# --------------------------------------------------------------------------- #
# PhyloCascadeGLM
# --------------------------------------------------------------------------- #
def forest_roots(registry) -> list[str]:
    """Heads that no other head in the bundle sits above — the entry points.

    Emulates the permissive root: it would push all of these and let each
    self-verify, so running them all is the faithful equivalent.
    """
    tree = registry.meta.tree or {}
    children = {c for kids in tree.values() for c in kids}
    return [t for t in tree if t not in children]


def phylocascade_predictions(
    bundle_path: Path,
    eval_rows: list[dict],
    entry_points: list[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
    log_every: int = 250,
    inferer_factory=None,
) -> tuple[dict, list[dict]]:
    """One classify() call per read, per entry point; deepest accepting commit wins.

    Returns (predictions, diagnostics). Diagnostics carry the entry point used, the
    stop_reason, and whether the commit was bundle-exhausted rather than terminal.

    Args:
        inferer_factory: Optional ``registry -> NodeInferer``. When given, that
            inferer replaces the DNABERT-2 one and EVERYTHING else — registry,
            reject margin, false-reject weighting, deepest-survivor rule, true-path
            consistency — stays identical, so the run isolates the head model as
            the only variable. Used by the GC-only baseline.
    """
    from phylocascadeglm._registry import AdapterRegistry
    from phylocascadeglm.classify import Classifier

    registry = AdapterRegistry(bundle_path / "adapter_registry.json")
    roots = entry_points or forest_roots(registry)
    if not roots:
        raise RuntimeError(f"{bundle_path}: no entry points — is _meta.tree empty?")

    # A commit is "bundle exhausted" when the bundle simply has nothing below it.
    def exhausted(taxid: str) -> bool:
        return not registry.children(str(taxid))

    # One inferer instance is shared across entry points: it is stateless per call
    # and the real one caches an expensive backbone.
    shared = inferer_factory(registry) if inferer_factory else None
    classifiers = {
        r: Classifier(bundle_path, device=device, root_taxid=r, inferer=shared)
        for r in roots
    }

    preds: dict[str, tuple[str | None, str | None]] = {}
    diags: list[dict] = []
    rows = eval_rows[:limit] if limit else eval_rows
    for i, row in enumerate(rows):
        if log_every and i and i % log_every == 0:
            print(f"  {i}/{len(rows)} reads", flush=True)
        best = None
        for root, clf in classifiers.items():
            res = clf.classify(row["seq"], query_id=row["read_id"])
            path = res.classification or []
            if not path:
                continue
            depth = len(path)
            if best is None or depth > best[0]:
                best = (depth, root, res)
        if best is None:
            preds[row["read_id"]] = (None, None)
            diags.append({"read_id": row["read_id"], "entry": None,
                          "stop_reason": "no_commit", "bundle_exhausted": False})
            continue
        _, root, res = best
        leaf = res.classification[-1]
        taxid = str(getattr(leaf, "taxid", leaf))
        rank = getattr(leaf, "rank", None)
        preds[row["read_id"]] = (taxid, rank)
        diags.append({
            "read_id": row["read_id"], "entry": root, "commit": taxid, "rank": rank,
            "stop_reason": getattr(res, "stop_reason", None),
            "bundle_exhausted": exhausted(taxid),
        })
    return preds, diags


# --------------------------------------------------------------------------- #
def load_eval_rows() -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(EVAL_PARQUET).to_pylist()


def print_report(name: str, report: dict) -> None:
    o = report["overall"]
    print(f"\n=== {name} ===")
    print(f"  overall  n={o['n']}  " + "  ".join(
        f"{k}={o.get(k+'_rate', 0):.3f}" for k in
        ("correct", "over_commit", "too_shallow", "misroute", "abstain")))
    print("  by divergence bin:")
    print(f"    {'bin':<14}{'n':>6}{'correct':>9}{'over':>8}{'shallow':>9}"
          f"{'misroute':>10}{'abstain':>9}")
    for b, r in report["by_distance_bin"].items():
        print(f"    {b:<14}{r['n']:>6}"
              f"{r.get('correct_rate',0):>9.3f}{r.get('over_commit_rate',0):>8.3f}"
              f"{r.get('too_shallow_rate',0):>9.3f}{r.get('misroute_rate',0):>10.3f}"
              f"{r.get('abstain_rate',0):>9.3f}")


def main() -> None:
    import argparse

    from taxotreeset.benchmark.scorer import score_reads

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-parquet", type=Path, default=EVAL_PARQUET)
    p.add_argument("--baselines-dir", type=Path, default=BASE / "baselines")
    p.add_argument("--bundle", type=Path, default=None,
                   help="PhyloCascadeGLM bundle. Omitted: score the baselines only.")
    p.add_argument("--entry-points", nargs="*", default=None,
                   help="Override the forest roots used as cascade entry points. "
                        "Default: every head with no packed ancestor, which is what "
                        "a permissive root would push to.")
    p.add_argument("--limit", type=int, default=None,
                   help="Score only the first N reads (smoke-testing the cascade).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--diagnostics", type=Path, default=None,
                   help="Write the per-read cascade diagnostics as JSONL.")
    p.add_argument("--gc-baseline", action="store_true",
                   help="Also run a GC-only cascade: identical traversal, every head "
                        "replaced by a logistic on GC content fitted on that head's own "
                        "train split. If the real cascade only ties this, the language "
                        "model is not what is doing the work.")
    p.add_argument("--datasets-root", type=Path,
                   default=BASE / "datasets_binary_allranks",
                   help="Dataset tree the GC heads are fitted on.")
    args = p.parse_args()

    import pyarrow.parquet as pq
    rows = pq.read_table(args.eval_parquet).to_pylist()
    ranks = build_rank_map(rows)
    print(f"eval reads: {len(rows)}   taxids with a known rank: {len(ranks)}")

    b = args.baselines_dir
    for name, fn, path in [
        ("Kraken2", kraken2_predictions, b / "kraken2.out"),
        ("Centrifuge", centrifuge_predictions, b / "centrifuge.out"),
        ("Kaiju", kaiju_predictions, b / "kaiju.out"),
    ]:
        if not path.exists():
            print(f"  (skipping {name}: {path} missing)")
            continue
        preds = fn(path, ranks)
        unranked = sum(1 for t, r in preds.values() if r is None)
        print(f"\n{name}: {len(preds)} reads committed, {unranked} with unknown rank")
        print_report(name, score_reads(rows, preds))

    if args.bundle:
        preds, diags = phylocascade_predictions(
            args.bundle, rows, entry_points=args.entry_points,
            limit=args.limit, device=args.device,
        )
        scored = rows[:args.limit] if args.limit else rows
        exhausted = sum(1 for d in diags if d.get("bundle_exhausted"))
        print(f"\nPhyloCascadeGLM: {sum(1 for v in preds.values() if v[0])} commits; "
              f"{exhausted} of them bottomed out on a taxon with no packed children "
              f"(bundle exhausted, NOT a genuine terminal)")
        print_report("PhyloCascadeGLM", score_reads(scored, preds))
        if args.diagnostics:
            with open(args.diagnostics, "w") as fh:
                for d in diags:
                    fh.write(json.dumps(d) + "\n")
            print(f"  per-read diagnostics -> {args.diagnostics}")

    if args.gc_baseline:
        if not args.bundle:
            p.error("--gc-baseline needs --bundle (it reuses the bundle's tree)")
        from gc_cascade import GCInferer, build_dataset_index, load_or_fit

        from phylocascadeglm._registry import AdapterRegistry

        registry = AdapterRegistry(args.bundle / "adapter_registry.json")
        taxids = list((registry.meta.tree or {}).keys())
        print(f"\nfitting {len(taxids)} GC heads on their own train splits…")
        index = build_dataset_index(args.datasets_root)
        coefs = load_or_fit(args.bundle / "gc_heads.json", taxids, index)
        missing = sum(1 for t in taxids if coefs.get(str(t), (0.0, 0.0)) == (0.0, 0.0))
        if missing:
            print(f"  {missing}/{len(taxids)} heads got an uninformative fit "
                  f"(no dataset, one class, or no GC variance) — they emit p=0.5")

        preds, diags = phylocascade_predictions(
            args.bundle, rows, entry_points=args.entry_points,
            limit=args.limit, device=args.device,
            inferer_factory=lambda reg: GCInferer(reg, coefs),
        )
        scored = rows[:args.limit] if args.limit else rows
        print(f"\nGC-only cascade: {sum(1 for v in preds.values() if v[0])} commits")
        print_report("GC-only cascade", score_reads(scored, preds))


if __name__ == "__main__":
    main()
