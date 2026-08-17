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
# Reparado: 6.950 dos 12.350 reads do arquivo original carregam true_lineage VAZIA,
# efeito de um `lineages.get(taxid, [])` silencioso no construtor. Uma verdade vazia
# nao soma ao recall mas cobra o denominador da precisao de quem responde, o que
# penalizava exatamente a ferramenta que nunca se abstem. Ver
# evaluation/repair_eval_lineages.py.
EVAL_PARQUET = BASE / "baselines" / "eval_reads_repaired.parquet"


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

    Enumerates every PACKED head, not the keys of ``meta.tree``. ``tree`` is built
    as a parent -> children map, so a head with no children never becomes a key —
    and a single-node component is neither a key nor anyone's child, so iterating
    keys drops it entirely. On the pilot bundle that silently lost 5 of 7 entry
    points (1032474, 10474 Fuselloviridae, 154834, 1611875, 2872567) and with them
    1,350 of the 12,350 eval reads, 10.9%, whose expected commit is at one of
    those heads. They would have scored as unexplained abstentions.
    """
    tree = registry.meta.tree or {}
    children = {c for kids in tree.values() for c in kids}
    return [t for t in registry.taxids if t not in children]


def _subtree_size(registry, root: str) -> int:
    """Number of packed heads reachable from ``root``, itself included.

    Used as a prior in arbitration: a read is likelier to belong to a large clade
    than to a lone leaf. Cached on the registry because it is asked once per offer
    per read, i.e. tens of thousands of times.
    """
    cache = getattr(registry, "_subtree_size_cache", None)
    if cache is None:
        cache = {}
        registry._subtree_size_cache = cache
    if root in cache:
        return cache[root]
    tree = registry.meta.tree or {}
    seen, stack = set(), [str(root)]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(str(c) for c in tree.get(n, []))
    cache[root] = len(seen)
    return cache[root]


def phylocascade_predictions(
    bundle_path: Path,
    eval_rows: list[dict],
    entry_points: list[str] | None = None,
    limit: int | None = None,
    device: str = "cpu",
    log_every: int = 250,
    inferer_factory=None,
    arbitration: str = "confidence",
    belonging_margin: float = 0.0,
    reject_margin: float = 0.0,
    consensus_agreement: float | None = None,
    record_descent: bool = False,
    descent_out: Path | None = None,
) -> tuple[dict, list[dict]]:
    """One classify() call per read, per entry point; an arbiter picks among them.

    Returns (predictions, diagnostics). Diagnostics carry the entry point used, the
    stop_reason, whether the commit was bundle-exhausted rather than terminal, and
    EVERY entry point's offer, so arbitration policies can be compared offline
    without paying for another neural pass.

    Arbitration matters more than it looks. The entry points are DISJOINT subtrees,
    so this is not `prefer_longest_survivor` — that rule eliminates within one tree,
    where a deeper survivor really did survive more rejections. Across independent
    forests a deeper path only means that subtree happens to be taller, and with a
    per-head false-accept rate near 0.5 several subtrees accept the same read.

    Picking by depth is then actively harmful, in a way that is not a matter of
    degree: with a strict `>`, ties go to whichever root is enumerated first, so
    every single-node subtree after the first can NEVER win a read. On the pilot
    bundle (subtree sizes 1,1,1,1,1,14,41) that silently disenfranchised four
    heads holding 1,300 of 12,350 reads — including head 154834, the one with val
    f1 1.000.

    Policies:
        "confidence" — highest acceptance probability at the entry node (default).
        "unanimous"  — abstain unless exactly one subtree accepts. Under a
                       hierarchical F with beta<1, abstaining beats misrouting.
        "deepest"    — the original rule, kept so the regression is reproducible.

    Args:
        arbitration: One of "confidence", "unanimous", "deepest".
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
    #
    # This used to fall through to None whenever no inferer_factory was given, i.e.
    # in every ordinary run, and Classifier then built its OWN SingleSequenceInferer
    # per entry point. Seven DNABERT-2 backbones on a 4 GiB card: the process
    # reached 3.9 of 4.0 GiB, the desktop compositor was left with nothing, and the
    # machine froze for seconds at a time during every harness run. The comment
    # above described the intent; the code did not implement it.
    if inferer_factory is not None:
        shared = inferer_factory(registry)
    else:
        from phylocascadeglm._inferer import SingleSequenceInferer
        shared = SingleSequenceInferer(bundle_path, device=device)
    classifiers = {
        r: Classifier(bundle_path, device=device, root_taxid=r, inferer=shared,
                      belonging_margin=belonging_margin,
                      reject_margin=reject_margin,
                      consensus_agreement=consensus_agreement)
        for r in roots
    }

    preds: dict[str, tuple[str | None, str | None]] = {}
    diags: list[dict] = []
    descent: list[dict] = []
    rows = eval_rows[:limit] if limit else eval_rows
    for i, row in enumerate(rows):
        if log_every and i and i % log_every == 0:
            print(f"  {i}/{len(rows)} reads", flush=True)
        # A precomputing inferer serves logits from a table keyed by (read, taxid),
        # and NodeInferer.infer(windows, taxid) carries no read identity, so tell it
        # which read this is. Harmless for the normal inferer, which has no such
        # attribute. This is what lets a run do 60 adapter loads instead of ~172,900
        # -- measured at 53 ms each on NVMe, that is the difference between minutes
        # and the ~5 hours a full run currently takes.
        if shared is not None and hasattr(shared, "current"):
            shared.current = i
        offers = []
        for root, clf in classifiers.items():
            if record_descent:
                # Every expanded node, INCLUDING the ones that rejected. The
                # consensus is derived from the same search, so this costs no extra
                # inference -- classify() itself is search() followed by consensus().
                #
                # Why record it: 71% of the loss is reads that stop too early because
                # a child head rejects them, and each node self-verdicts ALONE against
                # an absolute threshold, trained 50/50 and deployed at roughly 1:250.
                # Siblings never compete. Whether that is fixable offline depends on a
                # number nothing currently records -- when the correct child is
                # rejected, where does it RANK among its siblings' scores. With this
                # dump, any descent policy can be simulated without a GPU.
                all_res = clf.classify_all(row["seq"], query_id=row["read_id"])
                res = clf._traverser.consensus(all_res) if all_res else None
                if res is None:
                    continue
                res.query_id = row["read_id"]
                # Per RESULT, not flattened per node. Survival is a property of the
                # PATH -- `reject_penalty == 0` means no head on it said "not mine" --
                # and a flattened node list cannot express it: an offline replay of
                # the current rule then disagreed with the harness by 4 points of
                # correct and 10 of misroute, and the simulator's own validation gate
                # caught it. Recording the rule's actual input costs nothing here.
                descent.append({
                    "read_id": row["read_id"], "entry": root,
                    "results": [
                        {"reject_penalty": round(float(r.reject_penalty), 6),
                         "stop_reason": r.stop_reason,
                         "partial": bool(r.partial),
                         "path": [{"taxid": str(getattr(e, "taxid", e)),
                                   "p": round(float(getattr(e, "p", 0.0)), 6)}
                                  for e in (r.classification or [])]}
                        for r in all_res
                    ],
                })
            else:
                res = clf.classify(row["seq"], query_id=row["read_id"])
            path = res.classification or []
            if not path:
                continue
            offers.append({
                "entry": root, "depth": len(path),
                "subtree_size": _subtree_size(registry, root),
                "p_entry": float(getattr(path[0], "p", 0.0)),
                "p_min": min(float(getattr(e, "p", 0.0)) for e in path),
                "commit": str(getattr(path[-1], "taxid", path[-1])),
                "_res": res,
            })
        if not offers or (arbitration == "unanimous" and len(offers) > 1):
            preds[row["read_id"]] = (None, None)
            diags.append({
                "read_id": row["read_id"], "entry": None,
                "stop_reason": "ambiguous" if offers else "no_commit",
                "bundle_exhausted": False,
                "offers": [{k: v for k, v in o.items() if k != "_res"}
                           for o in offers],
            })
            continue
        if arbitration == "deepest":
            best = max(offers, key=lambda o: o["depth"])
        elif arbitration == "subtree":
            # Prefer the LARGER subtree, breaking ties on confidence. Measured on
            # hierarchical F(0.5), the project's metric, over 12,350 reads:
            #     confidence  0.191    depth  0.217    subtree  0.283
            # The mechanism: heads that steal reads are single-node subtrees that
            # simply shout louder. For reads taken by 2872567 the correct entry
            # still scored p_entry 0.67 against the thief's 0.93 -- the right answer
            # was in the offers, just not first. Size acts as a prior: a read is
            # likelier to belong to a large clade than to a lone leaf.
            #
            # Caveat worth keeping: subtree size is a property of THIS bundle's
            # shape. On the full tree the sizes differ, so this should be re-checked
            # rather than assumed to carry over.
            best = max(offers, key=lambda o: (o.get("subtree_size", 1),
                                              o["p_entry"]))
        else:                                  # "confidence" and "unanimous"
            best = max(offers, key=lambda o: (o["p_entry"], o["depth"]))
        root, res = best["entry"], best["_res"]
        leaf = res.classification[-1]
        taxid = str(getattr(leaf, "taxid", leaf))
        rank = getattr(leaf, "rank", None)
        preds[row["read_id"]] = (taxid, rank)
        diags.append({
            "read_id": row["read_id"], "entry": root, "commit": taxid, "rank": rank,
            "stop_reason": getattr(res, "stop_reason", None),
            "bundle_exhausted": exhausted(taxid),
            "offers": [{k: v for k, v in o.items() if k != "_res"} for o in offers],
        })
    if record_descent and descent_out is not None:
        with open(descent_out, "w") as fh:
            for rec in descent:
                fh.write(json.dumps(rec) + "\n")
        print(f"  descida por read -> {descent_out}  ({len(descent)} registros)")

    return preds, diags


# --------------------------------------------------------------------------- #
def load_eval_rows() -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(EVAL_PARQUET).to_pylist()


def _print_with_f(name: str, report: dict, rows=None, preds=None) -> None:
    """print_report mais o F(0.5) do scorer -- a medida de topo do projeto.

    Ate 2026-08-17 o harness so mencionava F(0.5) em comentario e nunca o calculava,
    e por isso o placar do projeto vinha de um codigo que ninguem podia auditar.
    """
    prf = None
    if rows is not None and preds is not None:
        from taxotreeset.benchmark.scorer import hierarchical_prf
        try:
            prf = hierarchical_prf(rows, preds)
        except ValueError as exc:            # verdade vazia: falha alto, nao zera
            print(f"  [F(0.5) indisponivel] {exc}")
    print_report(name, report, prf)


def print_report(name: str, report: dict, prf: dict | None = None) -> None:
    o = report["overall"]
    if prf:
        print(f"\n=== {name} — F({prf['beta']}) hierarquico ===")
        print(f"  precisao {prf['precision']:.3f}  recall {prf['recall']:.3f}  "
              f"F({prf['beta']}) {prf['f_beta']:.3f}   n={prf['n_reads']}")
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
                   help="Score only the FIRST N reads. Rarely what you want — the "
                        "head of this eval set is far easier than the whole; prefer "
                        "--sample-n.")
    p.add_argument("--sample-n", type=int, default=None,
                   help="Score a RANDOM sample of N reads. Baselines are scored on "
                        "the same subset, so the comparison stays valid.")
    p.add_argument("--sample-seed", type=int, default=0,
                   help="Seed for --sample-n, so a configuration sweep compares "
                        "runs on identical reads.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--reject-margin", type=float, default=0.0,
                   help="Evidence a head needs to REJECT: it prunes when "
                        "p_reject >= p_belong + reject_margin. NEGATIVE values make "
                        "ACCEPTANCE harder, which is the lever --belonging-margin was "
                        "mistaken for: that one only charges a score penalty the "
                        "arbitration never reads, so sweeping it over 0.10/0.25 "
                        "reproduced the baseline to three decimals in every bin.")
    p.add_argument("--belonging-margin", type=float, default=0.0,
                   help="How decisively a head must prefer belonging before the "
                        "traversal descends into it. 0.0 (the default since always) "
                        "accepts on any advantage however small, which is what lets "
                        "a biased node act as a sink: 2559587 absorbed 724 reads and "
                        "got none right. Measured false-accept runs 0.24-0.51 by "
                        "lineage distance where the cascade needs ~0.01.")
    p.add_argument("--arbitration", default="subtree",
                   choices=["subtree", "confidence", "unanimous", "deepest"],
                   help="How to pick among entry points that all accept a read. "
                        "The entry points are disjoint subtrees, so 'deepest' is "
                        "NOT prefer_longest_survivor — it rewards tall subtrees and "
                        "gives every tie to whichever root is enumerated first.")
    p.add_argument("--consensus-agreement", type=float, default=None,
                   help="Relax the LCA's unanimity to this fraction of surviving "
                        "paths agreeing on the next step. The LCA is 1.0 and returns "
                        "0.000 accuracy at genus on the pilot; 0.50 measured offline "
                        "at 0.108. None keeps the current behaviour.")
    p.add_argument("--record-descent", type=Path, default=None,
                   help="Dump every expanded node per read (including the ones "
                        "that rejected) as JSONL, so descent policies can be "
                        "simulated offline. Costs no extra inference.")
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
    p.add_argument("--gc-only", action="store_true",
                   help="Skip the DNABERT-2 cascade and score only the GC one. The "
                        "bundle is still needed for its tree and calibration; this "
                        "just avoids paying for a neural pass already measured.")
    args = p.parse_args()

    import pyarrow.parquet as pq
    rows = pq.read_table(args.eval_parquet).to_pylist()
    if args.sample_n:
        # RANDOM subset, not the head of the file. --limit takes the first N, and
        # the first N of this eval set are wildly unrepresentative: the first 200
        # score 0.835 where the full 12,350 score 0.096. Every smoke test run
        # against the head of the file (60 reads -> 0.883, 200 -> 0.835) was
        # measuring an easy prefix, which is how a dead parameter survived a
        # 10-hour sweep before anyone noticed it changed nothing.
        import random
        rng = random.Random(args.sample_seed)
        rows = rng.sample(rows, min(args.sample_n, len(rows)))
        print(f"amostra aleatoria: {len(rows)} reads (seed {args.sample_seed})")
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
        _print_with_f(name, score_reads(rows, preds), rows, preds)

    if args.bundle and not args.gc_only:
        preds, diags = phylocascade_predictions(
            args.bundle, rows, entry_points=args.entry_points,
            limit=args.limit, device=args.device,
            arbitration=args.arbitration,
            belonging_margin=args.belonging_margin,
            reject_margin=args.reject_margin,
            consensus_agreement=args.consensus_agreement,
            record_descent=args.record_descent is not None,
            descent_out=args.record_descent,
        )
        scored = rows[:args.limit] if args.limit else rows
        exhausted = sum(1 for d in diags if d.get("bundle_exhausted"))
        print(f"\nPhyloCascadeGLM: {sum(1 for v in preds.values() if v[0])} commits; "
              f"{exhausted} of them bottomed out on a taxon with no packed children "
              f"(bundle exhausted, NOT a genuine terminal)")
        _print_with_f("PhyloCascadeGLM", score_reads(scored, preds), scored, preds)
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
        # Every packed head, not meta.tree's keys — tree is a parent -> children
        # map, so a head with no children is absent from it. On the pilot bundle
        # that was 18 of 47, and the 29 missing ones would have silently fitted
        # no GC rule and emitted p=0.5 at every node they decide.
        taxids = list(registry.taxids)
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
        _print_with_f("GC-only cascade", score_reads(scored, preds), scored, preds)


if __name__ == "__main__":
    main()
