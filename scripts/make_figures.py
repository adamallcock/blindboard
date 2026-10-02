#!/usr/bin/env python3
"""Paper figures (SVG+PDF+PNG) from stored run artifacts.

Fig 1  effort x tier accuracy matrix, OpenAI family, retained harness
Fig 2  deducible vs converted per episode (identity-line scatter), retained
Fig 3  accuracy and output tokens under increasing load, both harnesses:
       two stacked shared-x panels (NOT dual-axis)
Fig 4  what retention changes: paired retained - visible deltas per cell

ONE population (fix for M5). This module does NO selection or aggregation of
its own: it imports scripts/build_paper_assets.canonical_selection() -- the
same call that builds the tables -- and draws the cells it returns. A plotted
cell and its table cell are therefore the same number by construction, not by
coincidence. Everything a figure needs is derived in build_paper_assets
(plotted_matrix_cells / plotted_cost_cells / plotted_scatter_episodes) so the
cross-output equality is unit-testable without matplotlib installed.

The freeze switch also lives there: build_paper_assets.enforce_freeze_if_frozen
is the single gate, replacing the second, independently-set SNAPSHOT_FROZEN /
SNAPSHOT_CUTOFF pair this module used to carry.

Style follows the dataviz method: validated categorical palette in fixed
model order (nano/luna/terra/sol), sequential single-hue ramp for
magnitude, thin marks, direct labels for series whose hue is low-contrast
(relief rule), recessive grid. Relabel-tier conversions are recomputed by
replay (the stored t10 diagnostics predate the addressing fix)."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blindboard.tiers import TIERS
from blindboard.tracker import BeliefTracker
from blindboard.viz import (  # vendored benchkit style constants
    GRID,
    SURFACE,
    TEXT,
    TEXT2,
    categorical_map,
    seq_cmap,
    style_axes,
)
from build_paper_assets import (  # THE selection; never re-derived here
    HARNESS_FIG_LADDER,
    HARNESS_FIG_LADDER_MODELS,
    HARNESS_FIG_SMALL_TIERS,
    LEGACY_TREE,
    PRIMARY_TREE,
    canonical_selection,
    enforce_freeze_if_frozen,
    harness_pairs,
    load_calibration_best,
    plotted_cost_cells,
    plotted_matrix_cells,
    plotted_scatter_episodes,
)
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / PRIMARY_TREE  # retained reasoning: the paper's population
LEGACY = ROOT / LEGACY_TREE    # the July visible-transcript comparison arm
OUT = ROOT / "paper" / "figures"

# Fixed categorical order, validated (palette slots from blindboard.viz).
MODEL_COLORS = categorical_map(
    ["gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
)
# NOTE: the snapshot scope switch used to live here as a SECOND, independently
# set SNAPSHOT_FROZEN plus a model allowlist and a hardcoded date cutoff. That
# let the figures be frozen to a different population than the tables (K3). It
# is gone: build_paper_assets.SNAPSHOT_FROZEN is the only switch, and
# enforce_freeze_if_frozen() below is the only gate.
MODEL_SHORT = {
    "gpt-5.4-nano": "nano",
    "gpt-5.6-luna": "luna",
    "gpt-5.6-terra": "terra",
    "gpt-5.6-sol": "sol",
    "claude-haiku-4-5": "haiku",
    "claude-sonnet-5": "sonnet",
    "claude-opus-4-8": "opus",
    "gemini/gemini-3-flash-preview": "g3-flash",
    "gemini/gemini-3.1-pro-preview": "g3.1-pro",
    "openrouter/deepseek/deepseek-v4-flash": "ds-flash",
    "openrouter/deepseek/deepseek-v4-pro": "ds-pro",
    "openrouter/qwen/qwen3.6-27b": "q27b",
    "openrouter/qwen/qwen3.6-35b-a3b": "q35b",
    "openrouter/qwen/qwen3.6-plus": "q-plus",
    "gpt-6-luna": "gpt6-luna",
    "gpt-6-sol": "gpt6-sol",
    "gpt-6-astra": "gpt6-astra",
    "claude-opus-5-5": "opus5.5",
    "gemini/gemini-3.8-flash": "g3.8-flash",
    "openrouter/deepseek/deepseek-v4.1-flash": "ds4.1-flash",
}
# Size-class icons for the luna / terra / sol names of both generations
# (moon, earth, sun): the same glyphs the paper's tables print.
MODEL_ICON = {
    "gpt-5.6-luna": "\u263e", "gpt-6-luna": "\u263e",
    "gpt-5.6-terra": "\u2295",
    "gpt-5.6-sol": "\u263c", "gpt-6-sol": "\u263c",
}


def model_name(model: str) -> str:
    """Short name, with its icon where the model has one."""
    icon = MODEL_ICON.get(model)
    return f"{icon} {MODEL_SHORT[model]}" if icon else MODEL_SHORT[model]


# Canonical column order for the appendix's full matrix (mirrors
# scripts/build_paper_assets.py's MODEL_ORDER/EFFORT_ORDER, plus qwen-plus
# which postdates that table).
MODEL_ORDER = {
    "gpt-5.4-nano": 0,
    "gpt-5.6-luna": 1,
    "gpt-5.6-terra": 2,
    "gpt-5.6-sol": 3,
    "gpt-6-luna": 4,
    "gpt-6-sol": 5,
    "gpt-6-astra": 6,
    "claude-haiku-4-5": 7,
    "claude-sonnet-5": 8,
    "claude-opus-4-8": 9,
    "claude-opus-5-5": 10,
    "gemini/gemini-3-flash-preview": 11,
    "gemini/gemini-3.1-pro-preview": 12,
    "gemini/gemini-3.8-flash": 13,
    "openrouter/deepseek/deepseek-v4-flash": 14,
    "openrouter/deepseek/deepseek-v4-pro": 15,
    "openrouter/deepseek/deepseek-v4.1-flash": 16,
    "openrouter/qwen/qwen3.6-27b": 17,
    "openrouter/qwen/qwen3.6-35b-a3b": 18,
    "openrouter/qwen/qwen3.6-plus": 19,
}
# "single-mode" is the merged label build_paper_assets gives a cell whose route
# ignores graded effort; it sorts last, exactly as it does in the table.
EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2, "max": 3, "single-mode": 4}
REFERENCE = {  # solver-v3 planner certificates (60 seeds, PYTHONHASHSEED=0)
    "t0_frozen": 1.0, "t2_announced": 1.0, "t3_observer": 0.867, "t6_gauntlet": 1.0,
    "t6_observer26": 0.897, "t7_blitz": 1.0, "t7_perpetual": 0.935,
    "t9_numrow": 0.969, "t9_macbook": 0.967, "t9_dual": 1.0,
    "t10_relabel": 0.944, "t11_maelstrom": 0.973,
}


def corrected_conversion(tier: str, episode: dict) -> tuple[int, int] | None:
    """(deducible, converted) with relabel-aware addressing, via replay."""
    config = TIERS[tier]
    if config.dual:
        return None  # dual episodes aggregate differently; skip in fig 2
    board = config.board_obj()
    groups = {label: idx for label, idx, _ in board.column_groups}
    addr_to_phys = list(range(board.n))
    tracker = BeliefTracker(board)
    for r in episode["records"]:
        if r["press_positions"]:
            tracker.observe_chord(r["press_positions"], r["results"])
        rot = r["rotation"]
        if rot["mode"] == "announced_full":
            tracker.apply_rotation_mapping({int(k): v for k, v in rot["permutation"].items()})
        elif rot["mode"] in ("announced", "observer", "storm") and rot["positions"]:
            tracker.apply_rotation(rot["positions"])
        rl = r.get("relabel")
        if rl:
            ia, ib = groups[rl["a"]], groups[rl["b"]]
            for x, y in zip(ia, ib):
                addr_to_phys[x], addr_to_phys[y] = addr_to_phys[y], addr_to_phys[x]
    guess = episode["result"]["guess"]
    deduced = tracker.deduced()
    cur_addr = {addr_to_phys[i]: board.addresses[i] for i in range(board.n)}
    converted = sum(1 for p, letter in deduced.items() if guess.get(cur_addr[p]) == letter)
    return len(deduced), converted


def load_cells(tree: Path = RESULTS) -> dict[tuple[str, str, str], dict]:
    """The canonical selection, straight from the table builder.

    No filtering, no re-averaging, no second snapshot switch: whatever the
    tables score is what the figures plot.
    """
    enforce_freeze_if_frozen(RESULTS)
    cells, anomalies = canonical_selection(tree)
    for note in anomalies:
        print(f"  note: {note}", file=sys.stderr)
    return cells


MATRIX_TIERS = ["t0_frozen", "t2_announced", "t3_observer", "t6_gauntlet", "t6_observer26",
                "t7_blitz", "t7_perpetual", "t9_numrow", "t9_macbook", "t9_dual",
                "t10_relabel", "t11_maelstrom"]
# The leaderboard columns are exactly the (model, effort) cells that make up
# the two pinned boards (see paper/blindboard.tex Discussion): Easy (t2 +
# gauntlet at medium) and Medium (perpetual + maelstrom at max). Route-broken
# cells (qwen3.6-27b's inert dial, qwen3.6-35b's completion-ceiling censoring)
# and the untested gemini-3.1-pro column are deliberately not board members
# and belong in the appendix's full matrix instead, not here.
LEADERBOARD_MEDIUM_MODELS = (
    "gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
    "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8",
    "gemini/gemini-3-flash-preview",
    "openrouter/deepseek/deepseek-v4-pro",
)
LEADERBOARD_MAX_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")




def _draw_matrix(tiers, cols, acc, *, figsize, title, cell_fontsize=8, tick_fontsize=8):
    cmap = seq_cmap()
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, tier in enumerate(tiers):
        for j, (model, effort) in enumerate(cols):
            # `acc` is plotted_matrix_cells(): the table's own per-cell means,
            # already selected and averaged. Nothing is averaged here.
            val = REFERENCE.get(tier) if model == "reference" else acc.get((tier, model, effort))
            if val is None:
                ax.text(j, i, "–", ha="center", va="center", color=GRID,
                        fontsize=cell_fontsize + 1)
                continue
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(val),
                                       edgecolor=SURFACE, linewidth=2))
            ax.text(j, i, f"{val:.2f}".lstrip("0") or "0", ha="center", va="center",
                    fontsize=cell_fontsize, color="white" if val > 0.55 else TEXT)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(tiers) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(
        ["reference"] + [f"{MODEL_SHORT[m]}\n{e}" for m, e in cols[1:]],
        fontsize=tick_fontsize, color=TEXT,
    )
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels([t.split("_", 1)[1] + f"  ({t.split('_')[0]})" for t in tiers],
                       fontsize=tick_fontsize, color=TEXT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, color=TEXT, pad=12, loc="left")
    fig.tight_layout()
    return fig


# Routes whose requested medium is one configuration with every other label
# (build_paper_assets.SAME_CONFIG_MERGES): their Easy-board column is the
# merged single-mode cell.
LEADERBOARD_SINGLE_MODE = ("openrouter/deepseek/deepseek-v4-pro",)


# The one family with an effort sweep and the full load stack under the
# retained harness: nano, luna's four effort levels, terra and sol at two.
FAMILY_COLUMNS = (
    ("gpt-5.4-nano", "medium"),
    ("gpt-5.6-luna", "low"), ("gpt-5.6-luna", "medium"),
    ("gpt-5.6-luna", "high"), ("gpt-5.6-luna", "max"),
    ("gpt-5.6-terra", "medium"), ("gpt-5.6-terra", "max"),
    ("gpt-5.6-sol", "medium"), ("gpt-5.6-sol", "max"),
)


# Figure 2: effort on the gauntlet, then load at maximum effort.
EFFORT_TIER = "t6_gauntlet"
EFFORTS = ("low", "medium", "high", "max")
LOAD_TIERS = ("t6_gauntlet", "t6_observer26", "t7_blitz", "t7_perpetual", "t9_numrow",
              "t9_macbook", "t9_dual", "t10_relabel", "t11_maelstrom")
FAMILY = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
# Figure 2's rows, grouped by the load each adds.
LOAD_GROUPS = (
    ("26 keys", (("t6_gauntlet", "gauntlet: rotations stop"),
                 ("t7_blitz", "blitz: fewer turns"),
                 ("t6_observer26", "observer26: pressing moves keys"),
                 ("t7_perpetual", "perpetual: rotations never stop"))),
    ("added to perpetual churn", (("t9_numrow", "numrow: 39 keys"),
                                  ("t9_macbook", "macbook: 47 keys"),
                                  ("t9_dual", "dual: two boards"),
                                  ("t10_relabel", "relabel: addresses swap"),
                                  ("t11_maelstrom", "maelstrom: 47 keys + relabel"))),
)
LOAD_MARKERS = {"gpt-5.6-luna": "o", "gpt-5.6-terra": "D", "gpt-5.6-sol": "s"}
LOAD_DODGE = {"gpt-5.6-luna": 0.17, "gpt-5.6-terra": 0.0, "gpt-5.6-sol": -0.17}
REF_INK = "#898781"  # muted ink: the reference is context, not a series


def _tier_label(tier: str) -> str:
    return f"{tier.split('_', 1)[1]}  ({tier.split('_')[0]})"


def fig_load_profile(cells):
    """The GPT-5.6 models at maximum effort on every tier they played there,
    reasoning retained, as a dot plot: one row per tier, grouped by the load
    it adds; a shape per model, so identity never rests on hue alone; the
    reference score as a gray tick. Values are the table's own cell means;
    reference scores come from the calibration loader the tables use."""
    acc = plotted_matrix_cells(cells)
    cert = load_calibration_best(ROOT / "results" / "calibration")
    fig, ax = plt.subplots(figsize=(6.6, 3.15), facecolor=SURFACE)
    style_axes(ax)
    ax.tick_params(length=0, labelsize=8.5, colors=TEXT2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", visible=False)
    y, ticks, labels = 0.0, [], []
    for g, (title, rows) in enumerate(LOAD_GROUPS):
        if g:
            y -= 0.55
        ax.text(0.006, y + 0.05, title, fontsize=8.5, color=TEXT2, style="italic",
                ha="left", va="bottom", transform=ax.get_yaxis_transform())
        y -= 0.75
        for tier, label in rows:
            ticks.append(y)
            labels.append(label)
            ax.scatter([cert[tier]], [y], marker="|", s=140, color=REF_INK, linewidths=1.7, zorder=2)
            for model in FAMILY:
                val = acc.get((tier, model, "max"))
                if val is not None:
                    ax.scatter([val], [y + LOAD_DODGE[model]], marker=LOAD_MARKERS[model], s=46,
                               color=MODEL_COLORS[model], edgecolor=SURFACE, linewidth=1.0, zorder=3)
            y -= 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, color=TEXT, fontsize=8.5)
    ax.set_ylim(y + 0.4, 0.45)
    ax.set_xlim(0.4, 1.03)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels([".4", ".5", ".6", ".7", ".8", ".9", "1.0"])
    ax.set_xlabel("board accuracy at maximum effort (axis starts at .4)", fontsize=8.5, color=TEXT)
    handles = [Line2D([], [], marker=LOAD_MARKERS[m], linestyle="", markersize=7, color=MODEL_COLORS[m],
                      markeredgecolor=SURFACE, label=model_name(m)) for m in FAMILY]
    handles.append(Line2D([], [], marker="|", linestyle="", markersize=10, color=REF_INK,
                          markeredgewidth=1.7, label="reference score"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.55, 1.02), handletextpad=0.3, columnspacing=1.6, labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig1_full_matrix(cells):
    """Appendix matrix: every (model, effort) column with data, including the
    route-pathology (qwen) and untested (gemini-3.1-pro) cells dropped from
    the leaderboard figure."""
    acc = plotted_matrix_cells(cells)
    present = sorted(
        {(m, e) for (t, m, e) in acc if t in MATRIX_TIERS},
        key=lambda me: (MODEL_ORDER.get(me[0], 999), EFFORT_ORDER.get(me[1], 999)),
    )
    cols = [("reference", None)] + present
    return _draw_matrix(MATRIX_TIERS, cols, acc, figsize=(13, 4.8),
                         cell_fontsize=6.5, tick_fontsize=6.5,
                         title="Board accuracy by tier, model, and reasoning effort (full matrix)")


SCATTER_MODELS = ("gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")


SCATTER_AREA = 15  # marker area per episode; a mark's area counts the episodes it stands for


def fig2_scatter(cells):
    """Acquisition against conversion, one small panel per GPT-5.x model, so
    luna's many episodes do not bury the others. Every scored single-board
    episode is counted, including those with nothing deducible (at the
    origin); episodes that land on the same spot share one mark whose area
    grows with their number. Triangles are maximum effort. A point on the diagonal lost nothing to conversion; the
    distance below it is the conversion gap."""
    fig, axes = plt.subplots(1, len(SCATTER_MODELS), figsize=(6.6, 2.35), facecolor=SURFACE,
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.12})
    points: dict[str, Counter] = {m: Counter() for m in SCATTER_MODELS}
    for record in plotted_scatter_episodes(cells, MODEL_COLORS):
        episode = json.loads(record.path.read_text())
        pair = corrected_conversion(record.tier, episode)
        if pair is None:
            continue
        n = TIERS[record.tier].board_obj().n
        points[record.model][(pair[0] / n, pair[1] / n, record.effort == "max")] += 1
    for ax, model in zip(axes, SCATTER_MODELS):
        style_axes(ax)
        ax.tick_params(length=0, labelsize=8, colors=TEXT2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.fill_between([0, 1], [0, 0], [0, 1], color=GRID, alpha=0.35, linewidth=0, zorder=0)
        ax.plot([0, 1], [0, 1], color=TEXT2, linewidth=0.9, zorder=1)
        color = MODEL_COLORS[model]
        # Larger stacks first, so a small mark is never hidden under a big one.
        for (x, y, at_max), count in sorted(points[model].items(), key=lambda kv: -kv[1]):
            ax.scatter(x, y, s=SCATTER_AREA * count, marker="^" if at_max else "o", color=color,
                       edgecolor=SURFACE, linewidth=0.7, alpha=0.85, zorder=3, clip_on=False)
        ax.set_title(f"{model_name(model)}  ({sum(points[model].values())})", fontsize=9,
                     color=TEXT, loc="left", pad=4)
        ax.set_xlim(0, 1.03)
        ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", ".5", "1"])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"])
        ax.set_aspect("equal")
    axes[0].set_ylabel("determined and\nguessed right", fontsize=8.5, color=TEXT)
    axes[0].text(0.97, 0.06, "conversion\nloss", fontsize=8, color=TEXT2, ha="right", va="bottom")
    axes[0].text(0.04, 0.97, "diagonal:\nno conversion\nloss", fontsize=8, color=TEXT2, ha="left", va="top")
    fig.supxlabel("fraction of the board the replay tracker determined",
                  fontsize=8.5, color=TEXT, y=0.03)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=TEXT2, markersize=5, label="below max effort"),
        Line2D([], [], marker="^", linestyle="", color=TEXT2, markersize=5.5, label="max effort"),
        Line2D([], [], marker="o", linestyle="", color=TEXT2, alpha=0.6,
               markersize=(SCATTER_AREA * 10) ** 0.5, label="10 episodes"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.99, 1.06), handletextpad=0.3, columnspacing=1.2, labelcolor=TEXT)
    return fig


def fig3_cost(cells, legacy_cells):
    """Both harnesses on the same seeds: solid lines keep the model's reasoning
    across turns, dashed lines resend only the visible transcript."""
    tiers = ["t6_gauntlet", "t7_blitz", "t7_perpetual", "t9_numrow", "t9_macbook",
             "t11_maelstrom"]
    labels = ["gauntlet\n26k/26t", "blitz\n26k/16t", "perpetual\n26k/24t",
              "numrow\n39k/30t", "macbook\n47k/36t", "maelstrom\n47k+relabel"]
    # Both panels read the canonical cells: accuracy is the table's own mean,
    # and tokens-per-episode divides the SELECTED episodes' output tokens by
    # the selected n (the old version summed run-level tokens over every
    # attempt while averaging accuracy over the same attempts -- a different
    # population from the table's).
    def series(source):
        acc_by_cell = plotted_matrix_cells(source)
        ktok_by_cell = plotted_cost_cells(source)
        return {
            (model, tier): {"acc": value, "ktok": ktok_by_cell[(tier, model, effort)]}
            for (tier, model, effort), value in acc_by_cell.items()
            if effort == "max" and tier in tiers
        }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 5.6), sharex=True,
                                   facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [1, 1]})
    for ax in (ax1, ax2):
        style_axes(ax)
    x = range(len(tiers))
    for stats, style, label_it in ((series(legacy_cells), (0, (3, 2)), False),
                                   (series(cells), "-", True)):
        for model in ("gpt-5.6-luna", "gpt-5.6-sol"):
            xs = [i for i, t in enumerate(tiers) if (model, t) in stats]
            if not xs:
                continue
            accs = [stats[(model, tiers[i])]["acc"] for i in xs]
            toks = [stats[(model, tiers[i])]["ktok"] for i in xs]
            color = MODEL_COLORS[model]
            for ax, ys in ((ax1, accs), (ax2, toks)):
                ax.plot(xs, ys, color=color, linewidth=2, linestyle=style, marker="o",
                        markersize=4.5, markeredgecolor=SURFACE,
                        alpha=1.0 if label_it else 0.7)
            if label_it:
                ax1.annotate(model_name(model), (xs[-1], accs[-1]), xytext=(6, 0),
                             textcoords="offset points", fontsize=8.5, color=TEXT,
                             va="center")
    from matplotlib.lines import Line2D

    ax2.legend([Line2D([], [], color=TEXT2, linewidth=2, linestyle="-"),
                Line2D([], [], color=TEXT2, linewidth=2, linestyle=(0, (3, 2)))],
               ["reasoning retained", "visible transcript only"],
               loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT)
    ax1.set_ylabel("board accuracy", fontsize=9, color=TEXT)
    ax1.set_ylim(0.3, 1.03)
    # The plotted series is result.usage.output_tokens over the selected
    # episodes; the game records carry no thinking-token field, so the axis,
    # the caption and the prose all say "output tokens".
    ax2.set_ylabel("output tokens / episode (k, log)", fontsize=9, color=TEXT)
    ax2.set_yscale("log")
    ticks = [20, 50, 100, 200, 500]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([str(t) for t in ticks])
    ax2.minorticks_off()
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8, color=TEXT)
    ax1.set_title("Max effort under increasing load, under both harnesses",
                  fontsize=11, color=TEXT, pad=10, loc="left")
    fig.align_ylabels((ax1, ax2))
    fig.tight_layout()
    return fig


NEUTRAL = "#898781"  # rows outside the four-model palette (DeepSeek), and context
# The figure's selection lives with the caption's count, in build_paper_assets.
SMALL_TIERS = HARNESS_FIG_SMALL_TIERS
LADDER = HARNESS_FIG_LADDER
LADDER_MODELS = HARNESS_FIG_LADDER_MODELS
# Shape as well as colour tells the two load series apart (grayscale, CVD).
LADDER_MARKERS = {"gpt-5.6-luna": "o", "gpt-5.6-sol": "s"}
EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "max": 3, "single-mode": 4}


def _arrow_rows(ax, rows, label):
    """One row per paired cell: a hollow dot at the visible-transcript score,
    a filled dot at the retained score, and an arrow between them. Where the
    two coincide, the filled dot sits inside a larger ring, so no change is
    drawn rather than hidden."""
    for y, r in enumerate(rows):
        color = MODEL_COLORS.get(r["model"], NEUTRAL)
        same = abs(r["retained"] - r["visible"]) <= 1e-9
        if not same:
            ax.annotate("", xy=(r["retained"], y), xytext=(r["visible"], y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7,
                                        shrinkA=4, shrinkB=4, mutation_scale=9), zorder=2)
        ax.scatter([r["visible"]], [y], s=70 if same else 36, facecolor=SURFACE, edgecolor=color,
                   linewidth=1.4, zorder=3)
        ax.scatter([r["retained"]], [y], s=22 if same else 40, color=color, edgecolor=SURFACE,
                   linewidth=1.0, zorder=4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label(r) for r in rows], fontsize=8.5, color=TEXT)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"])
    ax.grid(axis="y", visible=False)


def fig4_harness_deltas(cells, legacy_cells):
    """Retained reasoning against the visible transcript, on the same seeds:
    every paired cell on the 12-key tiers and on the gauntlet, as an arrow
    from its visible-transcript score to its retained score. The arrows read
    the paired means, the harness table's own numbers."""
    pairs = harness_pairs(cells, legacy_cells)
    order = lambda r: (MODEL_ORDER.get(r["model"], 99), EFFORT_RANK.get(r["effort"], 9),
                       SMALL_TIERS.index(r["tier"]) if r["tier"] in SMALL_TIERS else 0)
    small = sorted((r for r in pairs if r["tier"] in SMALL_TIERS), key=order)
    gauntlet = sorted((r for r in pairs if r["tier"] == "t6_gauntlet"), key=order)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.6, 3.1), facecolor=SURFACE,
                                     gridspec_kw={"wspace": 0.95})
    for ax in (ax_a, ax_b):
        style_axes(ax)
        ax.tick_params(length=0, labelsize=8.5, colors=TEXT2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("board accuracy", fontsize=8.5, color=TEXT)
    _arrow_rows(ax_a, small, lambda r: f"{model_name(r['model'])} · {r['tier'].split('_', 1)[1]}")
    _arrow_rows(ax_b, gauntlet, lambda r: model_name(r["model"]) + (
        "" if r["effort"] == "single-mode" else f" · {r['effort']}"))
    ax_a.set_title("(a) 12-key tiers", fontsize=9.5, color=TEXT, loc="left", pad=6)
    ax_b.set_title("(b) 26-key gauntlet", fontsize=9.5, color=TEXT, loc="left", pad=6)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=SURFACE,
               markeredgecolor=TEXT2, label="visible transcript"),
        Line2D([], [], marker="o", linestyle="", markersize=6.5, color=TEXT2,
               markeredgecolor=SURFACE, label="reasoning retained"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.03), handletextpad=0.3, columnspacing=1.6, labelcolor=TEXT)
    return fig


def fig_load_curves(cells, legacy_cells):
    """Appendix: luna and sol at maximum effort as load rises, under both
    harnesses: (a) board accuracy and (b) output tokens per episode, log
    scale. Hollow is the visible transcript, filled the retained harness; a
    shape per model. The panels read the canonical cells, as the prose does."""
    acc_r, acc_v = plotted_matrix_cells(cells), plotted_matrix_cells(legacy_cells)
    tok_r, tok_v = plotted_cost_cells(cells), plotted_cost_cells(legacy_cells)
    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(6.6, 2.9), facecolor=SURFACE,
                                     gridspec_kw={"wspace": 0.42})
    xs = range(len(LADDER))
    for ax in (ax_c, ax_d):
        style_axes(ax)
        ax.tick_params(length=0, labelsize=8.5, colors=TEXT2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([t.split("_", 1)[1] for t in LADDER], rotation=35, ha="right")
        ax.set_xlim(-0.3, len(LADDER) - 0.4)
    for model in LADDER_MODELS:
        color = MODEL_COLORS[model]
        marker = LADDER_MARKERS[model]
        for ax, src_r, src_v in ((ax_c, acc_r, acc_v), (ax_d, tok_r, tok_v)):
            r = [src_r.get((t, model, "max")) for t in LADDER]
            v = [src_v.get((t, model, "max")) for t in LADDER]
            ax.plot(xs, v, color=color, linewidth=1.4, alpha=0.45, zorder=2)
            ax.scatter(xs, v, s=28, marker=marker, facecolor=SURFACE, edgecolor=color, linewidth=1.2, zorder=3)
            ax.plot(xs, r, color=color, linewidth=2, zorder=4)
            ax.scatter(xs, r, s=34, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=5)
        ax_c.annotate(model_name(model), (len(LADDER) - 1, acc_r[(LADDER[-1], model, "max")]),
                      xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5, color=TEXT)
    ax_c.set_ylim(0.3, 1.05)
    ax_c.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax_c.set_yticklabels([".4", ".6", ".8", "1.0"])
    ax_c.set_ylabel("board accuracy", fontsize=8.5, color=TEXT)
    ax_c.set_title("(a) Accuracy", fontsize=9.5, color=TEXT, loc="left", pad=6)
    ax_d.set_yscale("log")
    ax_d.set_yticks([20, 50, 100, 200, 500])
    ax_d.set_yticklabels(["20k", "50k", "100k", "200k", "500k"])
    ax_d.minorticks_off()
    ax_d.set_ylabel("output tokens per episode", fontsize=8.5, color=TEXT)
    ax_d.set_title("(b) Output tokens, log scale", fontsize=9.5, color=TEXT, loc="left", pad=6)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor=SURFACE,
               markeredgecolor=TEXT2, label="visible transcript"),
        Line2D([], [], marker="o", linestyle="", markersize=6.5, color=TEXT2,
               markeredgecolor=SURFACE, label="reasoning retained"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.04), handletextpad=0.3, columnspacing=1.6, labelcolor=TEXT)
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    legacy_cells = load_cells(LEGACY)
    for name, fig in [
        ("fig_load_profile", fig_load_profile(cells)),
        ("fig1_full_matrix", fig1_full_matrix(cells)),
        ("fig2_deducible_converted", fig2_scatter(cells)),
        ("fig3_cost_before_accuracy", fig3_cost(cells, legacy_cells)),
        ("fig4_harness_deltas", fig4_harness_deltas(cells, legacy_cells)),
        ("fig_load_curves", fig_load_curves(cells, legacy_cells)),
    ]:
        for ext in ("svg", "pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=170, facecolor=SURFACE,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote paper/figures/{name}.{{svg,pdf,png}}")


if __name__ == "__main__":
    main()
