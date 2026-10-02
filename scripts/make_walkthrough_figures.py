#!/usr/bin/env python3
"""Appendix walkthrough figures: a real 3-turn episode drawn step by step,
plus one mini-diagram per rotation mode.

figA_walkthrough: four panels of an actual fingers12/t2 episode (played by
the reference agent) showing, per turn: which keys were pressed, the
unordered result set, which keys the rotation notice names, and what a
sound player can deduce at that point. Truth letters are printed small and
gray for the READER; the player never sees them.

figA_modes: before/after strips for each rotation mode with one line of
"what the player is told"."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.agents import ReferenceAgent
from blindboard.env import Episode
from blindboard.tiers import TIERS
from blindboard.tracker import BeliefTracker

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
PRESS = "#2a78d6"   # pressed keys: blue outline
ROTATE = "#eb6834"  # rotated keys: orange fill tint
DEDUCE = "#008300"  # deduced letters: green
KEYFACE = "#efeeea"

FINGERS = ["pinky", "ring", "middle", "index"]
ROWS = ["top", "home", "bottom"]


ROTATE_FACE = "#fbe3d6"   # tint of the moved-key orange
KEY_EDGE = "#d4d3ce"
INK_MUTED = "#898781"


def _key(ax, x, y, w, h, face, edge, lw, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=lw,
        linestyle=(0, (2.2, 1.6)) if dashed else "solid"))


def draw_board(ax, board, truth, deduced, pressed, rotated, show_truth=True, axis_labels=False):
    """12-key board: columns = fingers, rows = top/home/bottom. A pinned
    symbol is a large black letter; an unknown key is blank. Blue outlines
    the pressed chord, orange tints the keys a rotation notice names, and
    the small gray letter is the hidden truth (the reader's, never the
    player's). ``axis_labels`` names the rows and finger columns once."""
    ax.set_xlim(-0.08, 4.08)
    ax.set_ylim(-0.08, 3.08)
    ax.set_aspect("equal")
    ax.axis("off")
    for pos, _addr in enumerate(board.addresses):
        col, row = divmod(pos, 3)
        x, y = col, 2 - row
        face, edge, lw = KEYFACE, KEY_EDGE, 0.9
        if pos in rotated:
            face, edge, lw = ROTATE_FACE, ROTATE, 1.5
        if pos in pressed:
            edge, lw = PRESS, 2.4
        _key(ax, x + 0.06, y + 0.06, 0.88, 0.88, face, edge, lw)
        if pos in deduced:
            ax.text(x + 0.47, y + 0.52, deduced[pos], fontsize=12.5, color=TEXT,
                    ha="center", va="center", fontweight="bold")
        if show_truth:
            ax.text(x + 0.86, y + 0.12, truth[pos], fontsize=6.5, color=INK_MUTED,
                    ha="right", va="bottom")
    if axis_labels:
        for row, name in enumerate(ROWS):
            ax.text(-0.14, 2 - row + 0.5, name, fontsize=6.5, color=TEXT2, ha="right", va="center")
        # Staggered one line apart, so the finger names never run together.
        for col, name in enumerate(FINGERS):
            ax.text(col + 0.5, -0.1 - 0.24 * (col % 2), name, fontsize=6.5, color=TEXT2,
                    ha="center", va="top")


def _pedagogy_score(seed: int) -> int:
    """Prefer seeds whose first three turns produce visible deduction."""
    config = TIERS["t2_announced"]
    ep = Episode(config, seed)
    agent = ReferenceAgent(config, seed, samples=30)
    tracker = BeliefTracker(ep.board)
    for _ in range(3):
        action = agent.act(ep.observation())
        if action.guess is not None:
            return -1
        ep.apply(action)
        rec = ep.records[-1]
        tracker.observe_chord(rec["press_positions"], rec["results"])
        if rec["rotation"]["positions"]:
            tracker.apply_rotation(rec["rotation"]["positions"])
    return len(tracker.deduced())


def walkthrough():
    """The core inference on the selected episode, in four steps: the first
    chord, a second chord sharing exactly one key, the pin their results
    imply, and the rotation that undoes it. Every key, chord, result, and
    notice is the episode's own; the hidden layout is not drawn."""
    config = TIERS["t2_announced"]
    seed = max(range(20), key=_pedagogy_score)
    ep = Episode(config, seed)
    agent = ReferenceAgent(config, seed, samples=30)
    tracker = BeliefTracker(ep.board)
    turns = []
    for _ in range(2):
        ep.apply(agent.act(ep.observation()))
        rec = ep.records[-1]
        tracker.observe_chord(rec["press_positions"], rec["results"])
        turns.append((rec, dict(tracker.deduced())))
        if rec["rotation"]["positions"]:
            tracker.apply_rotation(rec["rotation"]["positions"])
    (first, _), (second, pinned) = turns
    shared = set(first["press_positions"]) & set(second["press_positions"])
    common = set(first["results"]) & set(second["results"])
    assert len(shared) == 1 and len(common) == 1, "the walkthrough needs a one-key, one-symbol overlap"
    key, symbol = shared.pop(), common.pop()
    assert pinned.get(key) == symbol, "the tracker must pin the shared symbol"
    unpinned = set(second["rotation"]["positions"])
    assert key in unpinned, "the next rotation must move the pinned key"
    address = ep.board.addresses[key]

    def fmt(results):
        return "{" + ", ".join(results) + "}"

    panels = [
        ("1  First chord", dict(pressed=set(first["press_positions"]),
                                notice=set(first["rotation"]["positions"])),
         f"result {fmt(first['results'])}"),
        ("2  Second chord", dict(pressed=set(second["press_positions"]), shared={key}),
         f"result {fmt(second['results'])}"),
        (f"3  Overlap pins {symbol}", dict(letters={key: symbol}, shared={key}),
         f"{fmt(first['results'])} $\\cap$\n{fmt(second['results'])} = {{{symbol}}}"),
        (f"4  Rotation unpins {symbol}", dict(notice=unpinned, maybe={p: symbol for p in unpinned}),
         f"{symbol} is now on one\nof the moved keys"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.75), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.22})
    for k, (ax, (title, marks, note)) in enumerate(zip(axes, panels)):
        _teaching_board(ax, ep.board, axis_labels=(k == 0), **marks)
        ax.set_title(title, fontsize=9, color=TEXT, pad=5, loc="left")
        ax.text(2.0, -0.78, note, fontsize=8.5, color=TEXT, ha="center", va="top", linespacing=1.3)
    # No baked-in caption: the LaTeX \caption carries it.
    return fig


def _teaching_board(ax, board, pressed=frozenset(), notice=frozenset(), shared=frozenset(),
                    letters=None, maybe=None, axis_labels=False):
    """12-key board for the walkthrough. Pressed keys have a solid blue
    outline, keys a rotation notice names a dashed orange outline on a light
    tint, the shared key a heavier outline; a pinned symbol is a large black
    letter, and a symbol that is somewhere in a set a small gray one."""
    letters, maybe = letters or {}, maybe or {}
    ax.set_xlim(-0.08, 4.08)
    ax.set_ylim(-0.08, 3.08)
    ax.set_aspect("equal")
    ax.axis("off")
    for pos, _addr in enumerate(board.addresses):
        col, row = divmod(pos, 3)
        x, y = col, 2 - row
        face, edge, lw, dashed = KEYFACE, KEY_EDGE, 0.9, False
        if pos in notice:
            face, edge, lw, dashed = ROTATE_FACE, ROTATE, 1.6, True
        if pos in pressed:
            edge, lw, dashed = PRESS, 2.2, False
        if pos in shared:
            edge, lw, dashed = PRESS, 3.4, False
        _key(ax, x + 0.06, y + 0.06, 0.88, 0.88, face, edge, lw, dashed=dashed)
        if pos in letters:
            ax.text(x + 0.5, y + 0.5, letters[pos], fontsize=14, color=TEXT,
                    ha="center", va="center", fontweight="bold")
        elif pos in maybe:
            ax.text(x + 0.5, y + 0.5, maybe[pos] + "?", fontsize=10, color=INK_MUTED,
                    ha="center", va="center")
    if axis_labels:
        for row, name in enumerate(ROWS):
            ax.text(-0.14, 2 - row + 0.5, name, fontsize=7.5, color=TEXT2, ha="right", va="center")
        # Staggered one line apart, so the finger names never run together.
        for col, name in enumerate(FINGERS):
            ax.text(col + 0.5, -0.1 - 0.27 * (col % 2), name, fontsize=7.5, color=TEXT2,
                    ha="center", va="top")


MODE_STRIPS = [
    dict(mode="none", perm=[0, 1, 2, 3, 4, 5], moved=[], pressed=[], named=False,
         told_moves=False, told="Nothing moves."),
    dict(mode="announced_full", perm=[3, 0, 2, 1, 4, 5], moved=[0, 1, 3], pressed=[],
         named=True, told_moves=True, told="Told every move."),
    dict(mode="announced", perm=[3, 0, 2, 1, 4, 5], moved=[0, 1, 3], pressed=[],
         named=True, told_moves=False, told="Told which keys moved, not where."),
    dict(mode="observer", perm=[2, 1, 0, 3, 4, 5], moved=[0, 2], pressed=[0, 2],
         named=True, told_moves=False, told="The pressed keys reshuffle."),
    dict(mode="unannounced", perm=[0, 3, 1, 2, 4, 5], moved=[1, 2, 3], pressed=[],
         named=False, told_moves=False, told="Keys move; no notice."),
    dict(mode="storm", perm=[4, 1, 0, 3, 2, 5], moved=[0, 2, 4], pressed=[0, 2],
         named=True, told_moves=False, told="Pressed and other keys reshuffle; set named."),
]
LETTERS = "ABCDEF"


def _mode_panel(ax, spec):
    ax.set_xlim(-1.05, 6.05)
    ax.set_ylim(-0.25, 3.25)
    ax.axis("off")
    perm, moved, pressed = spec["perm"], set(spec["moved"]), set(spec["pressed"])
    named = spec["named"]
    rows = ((1.55, "before", list(LETTERS)), (0.0, "after", [LETTERS[p] for p in perm]))
    for y, label, letters in rows:
        ax.text(-0.12, y + 0.36, label, fontsize=7, color=TEXT2, ha="right", va="center")
        for i, letter in enumerate(letters):
            face, edge, lw, dashed = KEYFACE, KEY_EDGE, 0.9, False
            if i in moved and named:
                face, edge, lw = ROTATE_FACE, ROTATE, 1.4
            elif i in moved:
                dashed, edge, lw = True, INK_MUTED, 1.1
            if y > 1 and i in pressed:
                edge, lw = PRESS, 2.2
            _key(ax, i + 0.07, y, 0.86, 0.72, face, edge, lw, dashed=dashed)
            ax.text(i + 0.5, y + 0.36, letter, fontsize=9, color=TEXT, ha="center", va="center")
    # Where each symbol went: a solid orange arrow when the notice reports
    # the move, a dashed gray one when only the reader sees it.
    for i, src in enumerate(perm):
        if src == i:
            continue
        told = spec["told_moves"]
        ax.annotate("", xy=(i + 0.5, 0.74), xytext=(src + 0.5, 1.53),
                    arrowprops=dict(arrowstyle="-|>", color=ROTATE if told else INK_MUTED, lw=1.2,
                                    linestyle="solid" if told else (0, (2.5, 1.8)), mutation_scale=7,
                                    shrinkA=1, shrinkB=1, connectionstyle="arc3,rad=0.0"))
    ax.text(-1.0, 3.06, spec["mode"], fontsize=8.5, color=TEXT, fontweight="bold",
            family="monospace", va="center")
    ax.text(-1.0, 2.66, spec["told"], fontsize=8, color=TEXT2, va="center")


def modes_figure():
    fig, axes = plt.subplots(3, 2, figsize=(6.6, 4.5), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.18, "wspace": 0.08})
    for ax, spec in zip(axes.flat, MODE_STRIPS):
        _mode_panel(ax, spec)
    legend = [
        Patch(facecolor=KEYFACE, edgecolor=PRESS, linewidth=2.0, label="pressed chord"),
        Patch(facecolor=ROTATE_FACE, edgecolor=ROTATE, linewidth=1.4, label="keys the notice names"),
        Patch(facecolor=KEYFACE, edgecolor=INK_MUTED, linewidth=1.1, linestyle=(0, (2.2, 1.6)),
              label="moved, not named"),
        Line2D([], [], color=ROTATE, linewidth=1.2, label="move reported"),
        Line2D([], [], color=INK_MUTED, linewidth=1.2, label="move not reported"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=7.2,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.1,
               handlelength=1.6, labelcolor=TEXT)
    # No baked-in caption; see walkthrough().
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fig in (("figA_walkthrough", walkthrough()), ("figA_modes", modes_figure())):
        for ext in ("svg", "pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=170, facecolor=SURFACE,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"wrote paper/figures/{name}.{{svg,pdf,png}}")


if __name__ == "__main__":
    main()
