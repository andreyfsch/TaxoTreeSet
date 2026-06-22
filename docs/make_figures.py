#!/usr/bin/env python3
"""Generate the documentation figures for the TaxoTreeSet README.

Each figure is a self-contained matplotlib schematic (no external data), so the
docs stay reproducible:

    python docs/make_figures.py

Figures are written to docs/figures/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Rectangle)

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

BLUE, GREEN, ORANGE, RED, GREY = "#3b6ea5", "#2e7d32", "#e08214", "#c0392b", "#666666"
PINK = "#c2185b"
LIGHT = {"#3b6ea5": "#dce6f1", "#2e7d32": "#dcecdc", "#e08214": "#fae6d0",
         "#c0392b": "#f6dad7", "#666666": "#e6e6e6"}


def _box(ax, x, y, w, h, text, ec=BLUE, fontsize=9, weight="normal",
         dashed=False, double=False, fc=None):
    fc = fc or LIGHT.get(ec, "#eeeeee")
    style = "round,pad=0.02,rounding_size=0.6"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec,
                                lw=1.6, ls="--" if dashed else "-"))
    if double:
        ax.add_patch(FancyBboxPatch((x + 0.25, y + 0.25), w - 0.5, h - 0.5,
                                    boxstyle=style, fc="none", ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color="#222222")
    return (x, y, w, h)


def _arrow(ax, p1, p2, color="#333333", ls="-", lw=1.8):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14,
                                 color=color, ls=ls, lw=lw,
                                 shrinkA=2, shrinkB=2))


def _canvas(w=12, h=7):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------------------
# fig 6 - TaxoTreeSet pipeline
# ---------------------------------------------------------------------------

def fig_taxotreeset() -> None:
    fig, ax = _canvas(12, 6)
    ax.text(50, 95, "TaxoTreeSet - balanced hierarchical dataset generation",
            ha="center", fontsize=13, weight="bold")

    cy, by, bh = 67, 54, 26          # shared centre so every arrow is horizontal
    _box(ax, 2, by, 18, bh,
         "Input\n\nNCBI taxonomy\n+ RefSeq genomes",
         ec=GREY, fontsize=8.5, fc="#f2f2f2")
    _box(ax, 25, by, 17, bh,
         "discover\n\nscan NCBI →\nregistry +\nsequence vault",
         ec=BLUE, fontsize=8.5)
    _box(ax, 47, by, 23, bh,
         "generate\n\nbuild the cascade of\nbalanced heads,\nextract the shards",
         ec=BLUE, fontsize=8.5)
    _box(ax, 74, by, 23, bh,
         "per-head datasets\n\none balanced\ntrain / val / test set\n"
         "per head (+ label_map)", ec=GREEN, fontsize=8)

    _arrow(ax, (20, cy), (25, cy))
    _arrow(ax, (42, cy), (47, cy))
    _arrow(ax, (70, cy), (74, cy))

    ax.text(50, 30, "One balanced classifier dataset per taxonomic node "
            "(family → genera, genus → species, ...)",
            ha="center", fontsize=9.5, style="italic")
    ax.text(50, 22, "Virtual buckets (misc / rare) absorb low-capacity and "
            "long-tail taxa so every head stays class-balanced.",
            ha="center", fontsize=8.5, color="#555555")

    fig.savefig(FIG / "pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _node(ax, cx, cy, label, color=BLUE, r=3.2, dashed=False, fc=None):
    fc = fc or LIGHT.get(color, "#eeeeee")
    ax.add_patch(Circle((cx, cy), r, fc=fc, ec=color, lw=1.6,
                        ls="--" if dashed else "-"))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=7)


def _stack(ax, cx, base_y, count, color, kept=None, unit=2.1, w=5.2):
    """Draw a capacity stack: ``count`` unit blocks; blocks above ``kept`` are
    hatched to show they are discarded by balancing."""
    kept = count if kept is None else kept
    for i in range(count):
        y = base_y + i * unit
        keep = i < kept
        ax.add_patch(Rectangle((cx - w / 2, y), w, unit * 0.78,
                     fc=LIGHT.get(color, "#eee") if keep else "#ffffff",
                     ec=color if keep else "#bbbbbb", lw=0.8,
                     hatch=None if keep else "////"))
    return base_y + count * unit  # top y


def _curve(ax, p1, p2, color, rad=-0.35):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                 color=color, lw=1.4, shrinkA=2, shrinkB=2,
                 connectionstyle=f"arc3,rad={rad}"))


def _capacity_inset(ax) -> None:
    """Top strip: genome + sliding window → fragments = a capacity stack."""
    ax.add_patch(FancyBboxPatch((16, 82), 68, 13,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fbfbfb", ec="#cfcfcf", lw=1.1))
    ax.text(20, 91.5, "What is\ncapacity?", ha="center", va="center",
            fontsize=8, weight="bold", color="#444444")
    # genome bar with a sliding window (solid) and its next position (faded)
    ax.add_patch(Rectangle((27, 89), 18, 2.2, fc="#e9e9e9", ec=GREY, lw=1.0))
    ax.text(36, 92.2, "genome", ha="center", fontsize=6.5, color="#666666")
    ax.add_patch(Rectangle((28, 88.6), 3, 3.0, fc="none", ec=BLUE, lw=1.3))
    ax.add_patch(Rectangle((31.4, 88.6), 3, 3.0, fc="none", ec=BLUE, lw=1.0,
                 ls="--", alpha=0.6))
    ax.annotate("", xy=(34, 90.1), xytext=(31.5, 90.1),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0))
    ax.text(36, 87.0, "sliding window (min_len)", ha="center", fontsize=6,
            color=BLUE)
    # arrow to fragments
    ax.annotate("", xy=(50, 90), xytext=(46, 90),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.4))
    for i in range(4):  # unique subsequences
        ax.add_patch(Rectangle((50.5, 88.2 + i * 1.0), 5, 0.8,
                     fc=LIGHT[GREEN], ec=GREEN, lw=0.7))
    ax.text(53, 87.0, "unique\nsubsequences", ha="center", va="top",
            fontsize=5.8, color="#555555")
    ax.text(60.5, 90.3, "=", ha="center", fontsize=11)
    for i in range(4):  # the stack icon used throughout the figure
        ax.add_patch(Rectangle((63, 88.2 + i * 1.0), 5, 0.8,
                     fc=LIGHT[GREEN], ec=GREEN, lw=0.7))
    ax.text(71.5, 90.3, "capacity\n(stack height)", ha="left", va="center",
            fontsize=7, weight="bold", color="#2e7d32")


def fig_generate_visual() -> None:
    """Low-text visual schematic: tree transformation across generate."""
    fig, ax = _canvas(14, 7.4)
    ax.text(50, 98, "How generate turns a taxonomy into balanced heads",
            ha="center", fontsize=14, weight="bold")

    _capacity_inset(ax)

    panels = [(2, 5, 30, 73, "1  Capacity per child"),
              (36, 5, 30, 73, "2  Balance + bucket"),
              (70, 5, 28, 73, "3  Balanced head → dataset")]
    for x, y, w, h, t in panels:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.0",
                     fc="#fcfcfc", ec="#cccccc", lw=1.2))
        ax.text(x + w / 2, y + h - 4, t, ha="center", fontsize=10, weight="bold")
    _arrow(ax, (32, 42), (36, 42), lw=2.6)
    _arrow(ax, (66, 42), (70, 42), lw=2.6)

    base, unit, fam_y, arr_y = 18, 2.1, 64, 60
    # ---- Panel 1: capacity per child, aggregated bottom-up ----
    p1cx = [8, 14.5, 21, 27.5]
    _node(ax, 16, fam_y, "parent\ntaxon", BLUE, r=4)
    for cx, n in zip(p1cx, [7, 4, 2, 1]):
        top = _stack(ax, cx, base, n, GREEN, unit=unit)
        _arrow(ax, (cx, top + 1), (16, arr_y), color="#9bb9d6", lw=1.0)
    ax.text(17, 13.5, "stack height = capacity", ha="center", fontsize=7.5,
            style="italic", color="#555555")

    # ---- Panel 2: percentile cutoff → long tail routed into the bucket ----
    # Children sorted by capacity; keep the top p-th percentile, divert the
    # smallest tail (would-be-discarded clades) into the bucket instead of
    # dropping them. p is configurable (--cutoff-percentage, default 98).
    _node(ax, 50, fam_y, "parent\ntaxon", BLUE, r=4)
    p2cx = [39, 42.6, 46.2, 49.8, 53.4, 57.0]
    p2h = [9, 7, 5, 4, 2, 1]          # sorted by capacity (descending)
    cut_idx = 4                        # retain first 4; divert the smallest tail
    cut_x = (p2cx[cut_idx - 1] + p2cx[cut_idx]) / 2
    top_y = base + 9 * unit
    for i, (cx, n) in enumerate(zip(p2cx, p2h)):
        retained = i < cut_idx
        _stack(ax, cx, base, n, GREEN if retained else GREY, unit=unit, w=3.2)
        if retained:
            _arrow(ax, (cx, base + n * unit + 1), (50, arr_y),
                   color="#9bb9d6", lw=0.9)
    ax.plot([cut_x, cut_x], [base - 1, top_y], ls="--", color=RED, lw=1.3)
    ax.text(cut_x, top_y + 1.5, "p-th percentile\ncutoff", ha="center",
            fontsize=7, color=RED)
    ax.text(44, top_y + 1.5, "keep top p%", ha="center", fontsize=6.8,
            color="#2e7d32")
    bucket_cx = 62
    _stack(ax, bucket_cx, base, 3, ORANGE, unit=unit)
    _node(ax, bucket_cx, base + 3 * unit + 4, "misc/\nrare", ORANGE, r=4.2,
          dashed=True)
    for cx in p2cx[cut_idx:]:
        _curve(ax, (cx, base + 1), (bucket_cx - 3, base + 2), ORANGE, rad=-0.3)
    _arrow(ax, (bucket_cx, base + 3 * unit + 8), (50, arr_y), color="#f0b070",
           lw=0.9)
    ax.text(50, 13.5, "smallest clades (below the p-th percentile)\n"
            "→ routed to the bucket, not dropped", ha="center", fontsize=7,
            color="#555555")
    ax.text(50, 8.5, "(p = --cutoff-percentage, default 98)", ha="center",
            fontsize=6.3, style="italic", color="#888888")

    # ---- Panel 3: balance to a common level, then dataset ----
    p3cx = [76, 82, 88]
    p3cols = [GREEN, GREEN, ORANGE]
    keep = 4   # n_per_class = min capacity among retained (a real child here)
    _node(ax, 82, fam_y, "parent\ntaxon", BLUE, r=4)
    for cx, n, c in zip(p3cx, [6, 4, 5], p3cols):
        _stack(ax, cx, base, n, c, kept=keep, unit=unit)
        _arrow(ax, (cx, base + keep * unit + 1), (82, arr_y), color="#9bb9d6",
               lw=1.0)
    ax.plot([73, 91], [base + keep * unit, base + keep * unit], ls="--",
            color="#2e7d32", lw=1.2)
    ax.text(97.5, base + keep * unit, "n_per_class", fontsize=6.5,
            color="#2e7d32", ha="right", va="center", weight="bold")
    ax.text(82, 5.5, "level = min capacity of retained classes", ha="center",
            fontsize=6, style="italic", color="#777777")
    for i in range(3):  # dataset sheets (distinct colour for emphasis)
        ax.add_patch(FancyBboxPatch((74 + i * 0.9, 8), 9, 6,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     fc="#f7d4e1", ec=PINK, lw=1.1))
    ax.text(78.5, 11, "train/val/test\ndataset", ha="center", va="center",
            fontsize=6.8, color="#7a1437", weight="bold")

    # legend
    ax.add_patch(Rectangle((3, 1), 3, 2.0, fc=LIGHT[GREEN], ec=GREEN, lw=0.8))
    ax.text(7, 2.0, "kept", fontsize=7, va="center")
    ax.add_patch(Rectangle((14, 1), 3, 2.0, fc="#fff", ec="#bbb", lw=0.8,
                 hatch="////"))
    ax.text(18, 2.0, "discarded by balancing", fontsize=7, va="center")
    ax.text(48, 2.0, "dashed node = virtual bucket", fontsize=7, va="center")

    fig.savefig(FIG / "generate_balancing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _mini(ax, cx, cy, label, color, r=1.9):
    ax.add_patch(Circle((cx, cy), r, fc=LIGHT.get(color, "#eee"), ec=color, lw=1.0))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6)


def fig_virtual_buckets() -> None:
    """The three virtual-bucket mechanisms for children that can't be classes."""
    fig, ax = _canvas(14, 6.4)
    ax.text(50, 96, "Virtual buckets: children that don't become clean classes",
            ha="center", fontsize=13.5, weight="bold")
    cols = [(2, 5, 30, 84, "Rank-aware  (virtual_misc)"),
            (36, 5, 30, 84, "Rare taxa  (virtual_rare_taxa)"),
            (70, 5, 28, 84, "Low capacity  (percentile)")]
    for x, y, w, h, t in cols:
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.3,rounding_size=1.0",
                     fc="#fcfcfc", ec="#cccccc", lw=1.2))
        ax.text(x + w / 2, y + h - 4, t, ha="center", fontsize=10, weight="bold")

    # ---- Col 1: rank-aware → baseline rank kept, off-rank grouped ----
    # The parent's baseline child rank (here: order) is kept as individual
    # classes; children attached at other ranks are grouped, one misc bucket
    # per rank.
    _node(ax, 16, 74, "parent\ntaxon", BLUE, r=4.2)
    for cx, lab in [(7, "order\nA"), (15, "order\nB")]:
        _node(ax, cx, 58, lab, GREEN, r=3.4)
        _arrow(ax, (cx, 61.4), (16, 70), color="#9bb9d6", lw=1.0)

    def rank_bucket(x0, label, letter):
        ax.add_patch(FancyBboxPatch((x0, 38), 11, 10,
                     boxstyle="round,pad=0.2,rounding_size=0.6",
                     fc="#fdf0e3", ec=ORANGE, ls="--", lw=1.3))
        for i in range(2):
            _mini(ax, x0 + 3.3 + i * 3.4, 43, letter, ORANGE, r=1.7)
        ax.text(x0 + 5.5, 35.8, label, ha="center", fontsize=6.5, color="#9a5a12")

    rank_bucket(3.5, "misc: genera", "G")
    rank_bucket(17, "misc: species", "S")
    _arrow(ax, (9, 48), (14.5, 70), color="#f0b070", lw=1.1)
    _arrow(ax, (22.5, 48), (17.5, 70), color="#f0b070", lw=1.1)
    ax.text(16, 12.5, "baseline child rank = order (kept as classes);\n"
            "children attached at other ranks →\none misc bucket per rank",
            ha="center", fontsize=6.6, color="#555555")

    # ---- Col 2: rare taxa → few distinct genomes (low diversity) ----
    _node(ax, 50, 72, "parent\ntaxon", BLUE, r=4)

    def genomes(cx, lengths, color):  # each bar = one source genome (a leaf)
        for j, L in enumerate(lengths):
            w = 0.8 + L * 0.42
            ax.add_patch(FancyBboxPatch((cx - w / 2, 50.5 - j * 1.5), w, 0.95,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         fc=LIGHT.get(color, "#eee"), ec=color, lw=0.7))

    kids = [(39, [2, 2, 2, 2]), (45, [2, 2, 2]), (53, [9]), (58, [2])]
    for cx, gl in kids:
        keep = len(gl) >= 2          # enough distinct genomes (leaves)
        _node(ax, cx, 56, "", GREEN if keep else GREY, r=2.2)
        genomes(cx, gl, GREEN if keep else GREY)
        if keep:
            _arrow(ax, (cx, 58), (50, 68), color="#9bb9d6", lw=0.9)
    _node(ax, 59, 31, "rare\ntaxa", ORANGE, r=4.2, dashed=True)
    for cx, gl in kids:
        if len(gl) < 2:
            _curve(ax, (cx, 49), (57, 33), ORANGE, rad=-0.3)
    _arrow(ax, (59, 35.5), (50, 68), color="#f0b070", lw=0.9)
    ax.text(53, 44.5, "1 genome:\nbig but rare", fontsize=5.8, color=RED,
            ha="center")
    ax.text(50, 12.5, "diversity = # distinct genomes (bars);\n"
            "below the floor → rare-taxa (even if large)",
            ha="center", fontsize=6.6, color="#555555")

    # ---- Col 3: low capacity → percentile cutoff ----
    _node(ax, 82, 72, "parent\ntaxon", BLUE, r=4)
    lc = [(75, 6), (79, 4), (83, 2), (87, 1)]
    base3, unit3, cut_x = 40, 1.6, 81
    for cx, n in lc:
        retained = cx < cut_x
        _stack(ax, cx, base3, n, GREEN if retained else GREY, unit=unit3, w=3.0)
        if retained:
            _arrow(ax, (cx, base3 + n * unit3 + 1), (82, 68),
                   color="#9bb9d6", lw=0.9)
    ax.plot([cut_x, cut_x], [base3 - 1, base3 + 7 * unit3], ls="--",
            color=RED, lw=1.2)
    ax.text(cut_x, base3 + 7 * unit3 + 1, "p-th pct", ha="center",
            fontsize=6.5, color=RED)
    _node(ax, 90, 30, "low-\ncap.", ORANGE, r=4, dashed=True)
    for cx, n in lc:
        if cx > cut_x:
            _curve(ax, (cx, base3 + 1), (88, 32), ORANGE, rad=-0.3)
    _arrow(ax, (90, 34), (82, 68), color="#f0b070", lw=0.9)
    ax.text(84, 12.5, "capacity = subsequence volume;\n"
            "< p-th percentile → low-capacity\n(regardless of # genomes)",
            ha="center", fontsize=6.6, color="#555555")

    ax.text(50, 2.5, "All bucket types become routable classes on the head "
            "(misc / rare / low-capacity) instead of being dropped.",
            ha="center", fontsize=7.5, style="italic", color="#444444")

    fig.savefig(FIG / "virtual_buckets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_capacity_bottomup() -> None:
    """Capacity aggregates leaves (genomes) -> root as a deduplicated union."""
    import math
    fig, ax = _canvas(12, 6.4)
    ax.text(50, 96, "Capacity is computed bottom-up (genomes → root)",
            ha="center", fontsize=13.5, weight="bold")

    def arrow_to(src, center, r=4, color="#9bb9d6", lw=1.0):
        dx, dy = center[0] - src[0], center[1] - src[1]
        d = math.hypot(dx, dy) or 1.0
        end = (center[0] - dx / d * r, center[1] - dy / d * r)
        _arrow(ax, src, end, color=color, lw=lw)

    def badge(cx, cy, label, n):
        _node(ax, cx, cy, label, BLUE, r=4)
        ax.add_patch(FancyBboxPatch((cx + 4.6, cy - 2.3), 8, 4.6,
                     boxstyle="round,pad=0.1,rounding_size=0.5",
                     fc=LIGHT[GREEN], ec=GREEN, lw=1.0))
        ax.text(cx + 8.6, cy, f"cap {n}", ha="center", va="center",
                fontsize=7, color="#2e7d32", weight="bold")

    base, unit, w = 22, 1.4, 3.4
    # (cx, capacity, index of the block shared with its sibling = top block)
    leaves = [(12, 3, 2), (33, 4, 3), (64, 2, 1), (87, 3, 2)]
    top_of = {}
    for cx, n, sh in leaves:
        ax.add_patch(FancyBboxPatch((cx - 5, 18.5), 10, 1.8,
                     boxstyle="round,pad=0.05,rounding_size=0.4",
                     fc="#e9e9e9", ec=GREY, lw=0.9))
        ax.text(cx, 16, "genome", ha="center", fontsize=5.6, color="#777777")
        top = _stack(ax, cx, base, n, GREEN, unit=unit, w=w)
        ax.add_patch(Rectangle((cx - w / 2, base + sh * unit), w, unit * 0.78,
                     fc="#ffe08a", ec="#e67e22", lw=1.7))  # the shared block
        ax.text(cx - w / 2 - 1.2, base + n * unit / 2, str(n), ha="right",
                va="center", fontsize=6.5, color="#2e7d32", weight="bold")
        top_of[cx] = top

    def shared_link(cx1, sh1, cx2, sh2, midx):
        y1 = base + (sh1 + 0.4) * unit
        y2 = base + (sh2 + 0.4) * unit
        ax.annotate("", xy=(cx2 - w / 2, y2), xytext=(cx1 + w / 2, y1),
                    arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.3,
                                    connectionstyle="arc3,rad=-0.3"))
        ax.text(midx, min(y1, y2) - 3.2, "same subsequence\n(counted once)",
                ha="center", fontsize=6, color="#e67e22")

    shared_link(12, 2, 33, 3, 22.5)
    shared_link(64, 1, 87, 2, 75.5)

    # cross-subtree shared subsequence → justifies the root's -1
    mag = "#d6249f"
    for cx in (33, 64):
        ax.add_patch(Rectangle((cx - w / 2, base), w, unit * 0.78,
                     fc="#f9d6ee", ec=mag, lw=1.7))
    ax.annotate("", xy=(64 - w / 2, base + 0.4 * unit),
                xytext=(33 + w / 2, base + 0.4 * unit),
                arrowprops=dict(arrowstyle="<->", color=mag, lw=1.3,
                                connectionstyle="arc3,rad=-0.4"))
    ax.text(48.5, 19, "same subsequence across subtrees\n"
            "(counted once at the root)", ha="center", fontsize=6, color=mag)

    badge(24, 52, "node\nA", 6)
    badge(72, 52, "node\nB", 4)
    badge(46, 80, "root", 9)
    arrow_to((12, top_of[12] + 0.5), (24, 52))
    arrow_to((33, top_of[33] + 0.5), (24, 52))
    arrow_to((64, top_of[64] + 0.5), (72, 52))
    arrow_to((87, top_of[87] + 0.5), (72, 52))
    arrow_to((24, 56), (46, 80))
    arrow_to((72, 56), (46, 80))

    from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

    def eq(x, y, green, tail, tail_color):
        a = TextArea(green, textprops=dict(color="#2e7d32", weight="bold",
                                           fontsize=6.8))
        b = TextArea(tail, textprops=dict(color=tail_color, weight="bold",
                                          fontsize=6.8))
        pack = HPacker(children=[a, b], align="center", pad=0, sep=2)
        ax.add_artist(AnnotationBbox(pack, (x, y), frameon=False,
                                     box_alignment=(0.5, 0.5)))

    eq(24, 45.5, "6 = 3 + 4", "- 1", "#e67e22")
    eq(72, 45.5, "4 = 2 + 3", "- 1", "#e67e22")
    eq(46, 71.5, "9 = 6 + 4", "- 1", mag)

    ax.text(50, 7, "A node's capacity = how many DISTINCT subsequences its "
            "genomes yield together.\nA subsequence found in two genomes is "
            "counted once, so capacities merge as a UNION - they do not simply "
            "add up.", ha="center", fontsize=7.5, style="italic", color="#444444")
    fig.savefig(FIG / "capacity_bottomup.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_distribution_split() -> None:
    """Split is by whole genome when >=3 leaves; slice a single genome otherwise."""
    fig, ax = _canvas(12, 6.2)
    ax.text(50, 97, "Train / val / test split: by whole genome when possible",
            ha="center", fontsize=13, weight="bold")
    ax.text(50, 92, "each genome's sliding-window samples (count proportional to "
            "genome length) all land in its assigned split",
            ha="center", fontsize=7, style="italic", color="#555555")

    gc, bc, pc = GREEN, BLUE, PINK
    lgt = {gc: "#dcecdc", bc: "#dce6f1", pc: "#f7d4e1"}

    # ---- Left panel: >= 3 genomes → assign whole genomes ----
    ax.add_patch(FancyBboxPatch((2, 6), 60, 82,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fcfcfc", ec="#cccccc", lw=1.2))
    ax.text(32, 84, ">= 3 genomes (default): each WHOLE genome → one split",
            ha="center", fontsize=8.5, weight="bold")
    genomes = [(34, gc, "genome 1 → train"), (28, gc, "genome 2 → train"),
               (22, gc, "genome 3 → train"), (30, bc, "genome 4 → val"),
               (20, pc, "genome 5 → test")]
    ys = [78, 72, 66, 60, 54]
    for (length, c, lab), y in zip(genomes, ys):
        ax.add_patch(FancyBboxPatch((20, y), length, 3.6,
                     boxstyle="round,pad=0.05,rounding_size=0.4",
                     fc=lgt[c], ec=c, lw=1.1))
        ax.text(19, y + 1.8, lab, ha="right", va="center", fontsize=6.2, color=c)
    for j, (lab, c) in enumerate([("train", gc), ("val", bc), ("test", pc)]):
        bx = 5 + j * 18
        ax.add_patch(FancyBboxPatch((bx, 26), 15, 6,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     fc=lgt[c], ec=c, lw=1.1))
        ax.text(bx + 7.5, 29, lab, ha="center", fontsize=7,
                color=c, weight="bold")
    for (length, c), y, j in [((22, gc), 66, 0), ((30, bc), 60, 1), ((20, pc), 54, 2)]:
        _curve(ax, (20 + length / 2, y), (5 + j * 18 + 7.5, 32), c, rad=-0.2)
    ax.text(32, 16, "each genome goes entirely to ONE split  →  no sliding "
            "window\nis shared across splits (no leakage)", ha="center",
            fontsize=6.8, style="italic", color="#444444")

    # ---- Right panel: < 3 genomes → slice EACH sequence 70/15/15 (fallback) ----
    ax.add_patch(FancyBboxPatch((65, 6), 33, 82,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fdfbf7", ec="#dddddd", lw=1.2))
    ax.text(81.5, 81, "< 3 genomes (fallback):\nslice EACH sequence 70/15/15",
            ha="center", fontsize=8.2, weight="bold")
    x0 = 68
    for total, gy, lab in [(22, 71, "g1"), (15, 62, "g2")]:  # length proportional
        cur = x0
        for frac, c in [(0.70, gc), (0.15, bc), (0.15, pc)]:
            wseg = total * frac
            ax.add_patch(Rectangle((cur, gy), wseg, 4.0, fc=lgt[c], ec=c, lw=1.0))
            cur += wseg
        ax.text(x0 - 0.8, gy + 2, lab, ha="right", va="center", fontsize=6,
                color="#777777")
    for frac_mid, pct, c in [(0.35, "70%", gc), (0.78, "15%", bc), (0.93, "15%", pc)]:
        ax.text(x0 + 22 * frac_mid, 76.5, pct, ha="center", fontsize=5.6, color=c)
    for j, (lab, c, frac_mid) in enumerate(
            [("train", gc, 0.35), ("val", bc, 0.78), ("test", pc, 0.93)]):
        by = 42 - j * 9
        ax.add_patch(FancyBboxPatch((80, by), 16, 6,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     fc=lgt[c], ec=c, lw=1.1))
        ax.text(88, by + 3, lab, ha="center", fontsize=6.8,
                color=c, weight="bold")
        _curve(ax, (x0 + 22 * frac_mid, 71), (80, by + 3), c, rad=-0.2)   # g1
        _curve(ax, (x0 + 15 * frac_mid, 62), (80, by + 3), c, rad=-0.12)  # g2
    ax.text(81.5, 13, "each of the (1-2) genomes is sliced 70/15/15 by position\n"
            "(longer genome → more samples); intra-genome leakage accepted",
            ha="center", fontsize=6.2, style="italic", color="#444444")

    fig.savefig(FIG / "split_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_selective_download() -> None:
    """Selective download (above threshold) + capacity-driven refinement loop."""
    fig, ax = _canvas(13, 7.0)
    ax.text(50, 97.5, "Selective download + capacity-driven refinement",
            ha="center", fontsize=13, weight="bold")
    ax.text(50, 93.5, "driven by per-label capacity targets at the requested "
            "taxonomic rank", ha="center", fontsize=8, style="italic",
            color="#555555")

    def db_cylinder(cx, cy, w, h, color, label, fill):
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc=fill, ec=color, lw=1.5))
        ax.add_patch(Ellipse((cx, cy - h / 2), w, h * 0.30, fc=fill, ec=color, lw=1.5))
        ax.add_patch(Ellipse((cx, cy + h / 2), w, h * 0.30, fc="#ffffff", ec=color, lw=1.5))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=7.5,
                weight="bold", color=color)

    def box(cx, cy, w, h, txt, ec, fc, fs=6.3):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.2,rounding_size=0.6", fc=fc, ec=ec, lw=1.2))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs, color="#222222")

    def carrow(p1, p2, color, rad=0.0, lw=1.3):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>",
                    color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))

    # ===== LEFT panel: what to download =====
    ax.add_patch(FancyBboxPatch((2, 5), 46, 87, boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fcfcfc", ec="#cccccc", lw=1.2))
    ax.text(25, 88, "What to download (per label, at the requested rank)",
            ha="center", fontsize=8.5, weight="bold")
    ax.text(6, 82.5, "pending sequences for this label", fontsize=6.2,
            color="#5a6b7b", ha="left")
    cur = 6
    for wseg in [5, 3, 6, 2, 4, 5, 3, 4]:    # different sequences, varied sizes
        ax.add_patch(Rectangle((cur, 78.5), wseg - 0.4, 3, fc="#e2e6ea",
                     ec="#7f8c9a", lw=0.8))
        cur += wseg
    ax.plot([26, 26], [76.8, 82.8], ls="--", color=RED, lw=1.4)
    ax.text(26, 75, "threshold (default 50 GiB)", fontsize=5.8, color=RED,
            ha="center")
    ax.text(25, 71, "volume > threshold → select per label\n"
            "(otherwise: download all)", fontsize=6.3, style="italic",
            color="#444444", ha="center")
    ax.text(6, 65, "target ≈ genome SIZE (heuristic);\nREFERENCE genomes first, "
            "then larger", fontsize=6.4, weight="bold", ha="left", color="#222222")
    rows = [("acc A", True, 6, True), ("acc B", True, 4, True),
            ("acc C", False, 7, True), ("acc D", False, 5, True),
            ("acc E", False, 3, False), ("acc F", False, 2, False)]
    y0, dy = 56, 4.6
    for i, (name, ref, size, sel) in enumerate(rows):
        y = y0 - i * dy
        c = GREEN if sel else GREY
        ax.text(8, y, name, ha="right", va="center", fontsize=5.7)
        if ref:
            ax.add_patch(FancyBboxPatch((9.5, y - 1.3), 6, 2.6,
                         boxstyle="round,pad=0.1,rounding_size=0.4",
                         fc="#ffe08a", ec="#e67e22", lw=0.9))
            ax.text(12.5, y, "REF", ha="center", va="center", fontsize=4.9,
                    color="#9a5a12", weight="bold")
        ax.add_patch(Rectangle((17, y - 1.2), size * 2.4, 2.4,
                     fc=LIGHT.get(c, "#eee"), ec=c, lw=0.9))
    ty = y0 - 3.5 * dy
    ax.plot([6, 38], [ty, ty], ls="--", color=BLUE, lw=1.1)
    ax.text(39, ty, "← target met:\ndefer the rest", fontsize=5.8, color=BLUE,
            ha="left", va="center")

    # ===== RIGHT panel: download → measure → refine (cycle) =====
    ax.add_patch(FancyBboxPatch((50, 5), 48, 87, boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fbfdfd", ec="#cccccc", lw=1.2))
    ax.text(74, 88, "Download → measure → refine (loop)", ha="center",
            fontsize=8.5, weight="bold")

    vault = "#0f8a8a"
    db_cylinder(75, 78, 18, 11, vault, "LMDB\nvault", "#d6eeee")
    box(91, 58, 13, 12, "extract →\nREAL\ncapacity", "#34699a", "#dce6f1")
    box(75, 37, 26, 9, "real capacity ≥ target ?", "#444444", "#f0f0f0")
    box(59, 58, 15, 13, "Refinement:\nundefer more\nsequences", ORANGE, LIGHT[ORANGE])
    box(75, 17, 14, 8, "done", GREEN, LIGHT[GREEN])

    carrow((83, 74), (90, 64), vault, rad=-0.25)             # vault → extract
    carrow((89, 52), (84, 42), "#34699a", rad=-0.25)         # extract → decision
    carrow((64, 40), (61, 52), ORANGE, rad=-0.25)            # decision → refinement
    ax.text(59, 45, "no", fontsize=6.2, color=RED, ha="center")
    carrow((61, 64), (69, 74), ORANGE, rad=-0.25)            # refinement → vault (loop)
    ax.text(56, 67, "another round", fontsize=5.6, color="#9a5a12", ha="center",
            va="center")
    carrow((75, 32.5), (75, 21), GREEN)                      # decision → done
    ax.text(77, 27, "yes", fontsize=6.2, color="#2e7d32", ha="left")

    # brace covering the selected (green) accessions; arrow leaves its tip
    bx, st, sb = 36, y0 + 2, y0 - 3 * dy - 2
    mid = (st + sb) / 2
    ax.plot([bx, bx], [sb, st], color=GREEN, lw=1.3)
    ax.plot([bx, bx - 1.5], [st, st], color=GREEN, lw=1.3)
    ax.plot([bx, bx - 1.5], [sb, sb], color=GREEN, lw=1.3)
    ax.plot([bx, bx + 2.5], [mid, mid], color=GREEN, lw=1.3)
    carrow((bx + 2.5, mid), (67, 80), GREEN, rad=-0.18, lw=1.6)  # download into vault
    ax.text(42, 64, "download\nselected", fontsize=6, color="#2e7d32",
            ha="center", va="center")

    ax.text(50, 8.5, "the genome-size estimate can OVER-count capacity (repetitive "
            "genomes);\nrefinement tops up until real capacity ≥ target", ha="center",
            fontsize=7, style="italic", color="#444444")
    fig.savefig(FIG / "selective_download.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_tree_of_heads() -> None:
    """Parameterised cascade: --root (where to start) and --stop-at (how deep)."""
    fig, ax = _canvas(13, 7)
    ax.text(50, 97, "Parameterising the cascade: --root (where to start) and "
            "--stop-at (how deep)", ha="center", fontsize=12, weight="bold")

    ranks = [("superkingdom", 84), ("kingdom", 75), ("phylum", 66),
             ("class", 57), ("order", 48), ("family", 39), ("genus", 30),
             ("species", 21)]
    for name, y in ranks:
        ax.plot([11, 50], [y, y], color="#eeeeee", lw=0.8, zorder=0)
        ax.text(10, y, name, ha="right", va="center", fontsize=6.2,
                color="#999999")

    nodes = {
        "V": (30, 84), "K1": (22, 75), "K2": (40, 75),
        "P1": (16, 66), "P2": (28, 66), "P3": (40, 66),
        "C1": (16, 57), "C2": (28, 57), "C3": (40, 57),
        "O1": (16, 48), "O2": (28, 48), "O3": (40, 48),
        "Fa": (16, 39), "Fb": (28, 39), "Fc": (40, 39),
        "g1": (24, 30), "g2": (32, 30), "s1": (21, 21), "s2": (27, 21),
    }
    edges = [("V", "K1"), ("V", "K2"), ("K1", "P1"), ("K1", "P2"), ("K2", "P3"),
             ("P1", "C1"), ("P2", "C2"), ("P3", "C3"), ("C1", "O1"), ("C2", "O2"),
             ("C3", "O3"), ("O1", "Fa"), ("O2", "Fb"), ("O3", "Fc"),
             ("Fb", "g1"), ("Fb", "g2"), ("g1", "s1"), ("g1", "s2")]
    labels = {"g1", "g2", "s1", "s2"}
    for a, b in edges:
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                color="#cfcfcf", lw=0.9, zorder=1)
    for k, (x, y) in nodes.items():
        if k in labels:
            ax.add_patch(Circle((x, y), 2.3, fc="#ffffff", ec="#9e9e9e",
                         lw=1.1, zorder=2))
        else:
            ax.add_patch(Circle((x, y), 2.3, fc=LIGHT[BLUE], ec=BLUE,
                         lw=1.3, zorder=2))

    # --root marker
    ax.annotate("", xy=(27.5, 84), xytext=(20, 90),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.4))
    ax.text(15, 91.5, "--root  (start here)", fontsize=7, color="#c0392b",
            weight="bold", ha="center")
    ax.text(37, 89.5, "default: viruses; or a clade,\ne.g. --root Caudoviricetes",
            fontsize=6, color="#777777", ha="left", va="center")

    # --stop-at boundary (between family and genus)
    ax.plot([11, 50], [34.5, 34.5], ls="--", color=ORANGE, lw=1.6, zorder=3)
    ax.text(50.5, 33, "--stop-at\n= family", fontsize=6.3, color="#9a5a12",
            va="center", ha="left", weight="bold")

    # legend
    ax.add_patch(Circle((14, 12), 2.0, fc=LIGHT[BLUE], ec=BLUE, lw=1.2))
    ax.text(17, 12, "head: a balanced classifier", fontsize=6.5, va="center")
    ax.add_patch(Circle((14, 6.5), 2.0, fc="#fff", ec="#9e9e9e", lw=1.2))
    ax.text(17, 6.5, "label only (deeper than --stop-at)", fontsize=6.5,
            va="center")

    # parameters box (right, top)
    ax.add_patch(FancyBboxPatch((57, 67), 41, 21,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fcfcfc", ec="#cccccc", lw=1.1))
    ax.text(77.5, 84.5, "Root & depth parameters", ha="center", fontsize=8.5,
            weight="bold")
    ax.text(59, 81,
            "--root        viruses | bacteria | TaxID | clade\n"
            "--stop-at     rank where heads stop;\n"
            "              deeper taxa become labels\n"
            "--single-level   only the root's head\n"
            "(default)     full depth, down to species",
            ha="left", va="top", fontsize=6.3, color="#333333", family="monospace")

    # callout: what a head is
    ax.add_patch(FancyBboxPatch((57, 33), 41, 28,
                 boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc="#fbfdff", ec="#cccccc", lw=1.1))
    ax.text(77.5, 58, "What each head is", ha="center", fontsize=8.5,
            weight="bold")
    _node(ax, 63, 50, "head", BLUE, r=3.8)
    chips = [(73, 54, "child"), (73, 50, "child"), (73, 46, "misc")]
    for cx, cy, t in chips:
        ax.add_patch(FancyBboxPatch((cx - 4, cy - 1.4), 8, 2.8,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     fc=LIGHT[GREEN], ec=GREEN, lw=0.9))
        ax.text(cx, cy, t, ha="center", va="center", fontsize=5.6)
        ax.annotate("", xy=(cx - 4, cy), xytext=(66, 50),
                    arrowprops=dict(arrowstyle="-|>", color="#9bb9d6", lw=0.9))
    for i in range(3):                       # balanced dataset shards
        ax.add_patch(FancyBboxPatch((85 + i * 0.7, 47), 8, 5,
                     boxstyle="round,pad=0.1,rounding_size=0.4",
                     fc="#f7d4e1", ec=PINK, lw=1.0))
    ax.annotate("", xy=(85, 50), xytext=(78, 50),
                arrowprops=dict(arrowstyle="-|>", color=PINK, lw=1.1))
    ax.text(89.5, 49.5, "dataset", ha="center", va="center", fontsize=5.6,
            color="#7a1437", weight="bold")
    ax.text(77.5, 38, "every head classifies its children (the next rank)\n"
            "into one balanced train / val / test dataset", ha="center",
            fontsize=6.6, style="italic", color="#444444")

    fig.savefig(FIG / "parameterization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_generate_detail() -> None:
    """Roadmap of generate's four stages, pointing to the detail figures."""
    fig, ax = _canvas(13, 7.6)
    ax.text(50, 97, "generate: the four stages (overview & map to the detail "
            "figures)", ha="center", fontsize=12.5, weight="bold")

    def stage_box(y, h, title, desc, ref, ec, fc=None):
        ax.add_patch(FancyBboxPatch((8, y), 84, h,
                     boxstyle="round,pad=0.3,rounding_size=0.8",
                     fc=fc or LIGHT.get(ec, "#f4f4f4"), ec=ec, lw=1.3))
        ax.text(11, y + h - 3, title, ha="left", va="center", fontsize=9,
                weight="bold", color="#222222")
        if desc:
            ax.text(11, y + (h - 6) / 2 + 1, desc, ha="left", va="center",
                    fontsize=7.2, color="#444444")
        if ref:
            ax.text(89, y + h / 2, ref, ha="right", va="center", fontsize=6.6,
                    style="italic", color="#777777")
        return y

    # optional sync pre-step
    ax.add_patch(FancyBboxPatch((26, 89), 48, 6,
                 boxstyle="round,pad=0.3,rounding_size=0.8",
                 fc="#eeeeee", ec=GREY, lw=1.1, ls="--"))
    ax.text(50, 92, "Sync (optional): registry + vault  ↔  NCBI", ha="center",
            va="center", fontsize=7.5, color="#555555")

    stages = [
        (76, 8, "1 · Download", "selective fetch, with capacity-driven refinement",
         "▸ see: Selective download", BLUE),
        (64, 8, "2 · Tree + Capacity",
         "build the taxonomy tree; capacity computed bottom-up",
         "▸ see: Capacity (bottom-up)", BLUE),
        (46, 13, "3 · Scheduling  (at every node)",
         "balance classes · virtual buckets · distribute n_per_class",
         "▸ see: Virtual buckets\n▸ see: Capacity → balance → dataset\n"
         "▸ see: Distribution + split", GREEN),
        (34, 8, "4 · Extraction",
         "sliding-window subsequences  →  train / val / test",
         "▸ see: Distribution + split", BLUE),
    ]
    for y, h, title, desc, ref, ec in stages:
        stage_box(y, h, title, desc, ref, ec)

    stage_box(20, 8, "Output · per-head datasets",
              "parquet / csv  +  label_map  +  k-mer separability", "", GREEN)

    # arrows down the centre between consecutive boxes
    for top_b, bot_b in [(89, 84), (76, 72), (64, 59), (46, 42), (34, 28)]:
        _arrow(ax, (50, top_b), (50, bot_b), lw=1.6)

    ax.text(50, 12.5, "scope set by --root / --stop-at      "
            "▸ see: Parameterising the cascade", ha="center", fontsize=7.5,
            style="italic", color="#444444")

    fig.savefig(FIG / "generate_stages.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_reject_bucket() -> None:
    """The reject class: a per-head label trained on out-of-subtree negatives.

    Unlike the other virtual buckets (which absorb a parent's own
    under-supported children), the reject class is fed by sequences sampled
    from *outside* the head's subtree — near (nearest-ancestor sibling clades)
    and far (clades elsewhere in the tree) — shown here on the taxonomy tree
    with arrows feeding the head's reject label.
    """
    fig, ax = _canvas(13.5, 7.4)
    ax.text(50, 96, "Reject class: out-of-subtree clades feed a none-of-these "
            "label on the head", ha="center", fontsize=12.5, weight="bold")

    def edge(p1, p2, color="#cccccc", lw=1.3):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, zorder=0)

    # ---- taxonomy tree backbone (head's lineage on the left, a distant
    #      branch on the right; only the head's own parent is labelled) ----
    root = (50, 90)
    par, mid = (22, 78), (72, 78)            # par = head's parent; mid = unrelated branch
    edge(root, par); edge(root, mid)
    _node(ax, *root, "root", BLUE, r=3.0)
    _node(ax, *par, "parent", BLUE, r=3.6)
    _node(ax, *mid, "", BLUE, r=3.0)

    head, sibA, sibB = (9, 61), (22, 61), (32, 61)
    m1, m2 = (60, 61), (83, 61)
    for child in (head, sibA, sibB):
        edge(par, child)
    for child in (m1, m2):
        edge(mid, child)
    _node(ax, *m1, "", BLUE, r=2.8)
    _node(ax, *m2, "", BLUE, r=2.8)
    _node(ax, *sibA, "", GREY, r=2.8)
    _node(ax, *sibB, "", GREY, r=2.8)
    _node(ax, *head, "head", BLUE, r=4.8)

    f1, f2, f3, f4 = (52, 45), (66, 45), (76, 45), (90, 45)
    edge(m1, f1); edge(m1, f2); edge(m2, f3); edge(m2, f4)
    for fc_ in (f1, f2, f3, f4):
        _node(ax, *fc_, "", GREY, r=2.5)

    ax.text(27, 67, "near — sibling clades", ha="center", fontsize=7, color="#555555")
    ax.text(71, 51.5, "far — distant clades", ha="center", fontsize=7, color="#555555")

    # ---- head's real classes (genuine, in-subtree) ----
    for cx, lab in [(3, "A"), (9, "B"), (15, "C")]:
        _node(ax, cx, 45, lab, GREEN, r=2.5)
        edge(head, (cx, 47), color="#bcd0e6")
    ax.add_patch(FancyBboxPatch((0.5, 31), 18.5, 35.5,
                 boxstyle="round,pad=0.3,rounding_size=0.8",
                 fc="none", ec=GREEN, ls="--", lw=1.1))
    ax.text(9.5, 35, "head's real classes\n(genuine, in-subtree)",
            ha="center", va="center", fontsize=6.6, color="#2e7d32")

    # ---- the reject label on the head, fed by near + far clades ----
    reject = (39, 21)
    edge(head, reject, color="#f0b070", lw=1.5)   # it is one of the head's labels
    _node(ax, *reject, "virtual_\nreject", ORANGE, r=5.2, dashed=True)
    ax.text(39, 13.5, "none-of-these", ha="center", fontsize=6.8, color="#9a5a12")

    _curve(ax, sibA, (35, 24), "#e0902c", rad=-0.22)
    _curve(ax, sibB, (37, 25), "#e0902c", rad=-0.15)
    _curve(ax, f2, (43, 22), "#e0902c", rad=0.34)
    _curve(ax, f3, (44, 20), "#e0902c", rad=0.40)

    ax.text(74, 26, "windows sampled from the\nnear + far clades  →  reject\n\n"
            "--reject-near-far-ratio  sets near:far\n"
            "size ≈ n_per_class  (--reject-fraction)",
            ha="center", fontsize=7, color="#555555")

    ax.text(50, 4, "The reject class is a normal head label whose windows come "
            "from sequences OUTSIDE the head's subtree, giving the head an "
            "explicit “none of these” option.   Opt-in: --reject-class.",
            ha="center", fontsize=7.2, style="italic", color="#444444")

    fig.savefig(FIG / "reject_bucket.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_taxotreeset()
    fig_generate_detail()
    fig_generate_visual()
    fig_virtual_buckets()
    fig_reject_bucket()
    fig_capacity_bottomup()
    fig_distribution_split()
    fig_selective_download()
    fig_tree_of_heads()
    print(f"Figures written to {FIG}/")
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.name}")
