#!/usr/bin/env python3
"""Reasoning screen: turn-by-turn view of an episode — observation, the
model's reasoning summary, its action, and what a sound tracker could
deduce at that point. For eyeballing WHERE a model's beliefs diverge.

  uv run python scripts/show_reasoning.py <seedNNNN.json> --tier t6_gauntlet [--turns 5-9] [--chars 600]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.tiers import TIERS
from blindboard.tracker import BeliefTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_json")
    parser.add_argument("--tier", required=True, choices=list(TIERS))
    parser.add_argument("--turns", default=None, help="e.g. 5-9 or 12")
    parser.add_argument("--chars", type=int, default=700, help="max reasoning chars/turn")
    args = parser.parse_args()

    data = json.loads(Path(args.episode_json).read_text(encoding="utf-8"))
    config = TIERS[args.tier]
    board = config.board_obj()
    lo, hi = 1, 10**9
    if args.turns:
        parts = args.turns.split("-")
        lo = int(parts[0])
        hi = int(parts[-1])

    # Deducible-count timeline from the records via the sound tracker.
    tracker = BeliefTracker(board)
    deducible_by_turn: dict[int, int] = {}
    for r in data["records"]:
        if r["press_positions"]:
            tracker.observe_chord(r["press_positions"], r["results"])
        rot = r["rotation"]
        if rot["mode"] == "announced_full":
            tracker.apply_rotation_mapping({int(k): v for k, v in rot["permutation"].items()})
        elif rot["mode"] in ("announced", "observer", "storm"):
            tracker.apply_rotation(rot["positions"])
        deducible_by_turn[r["turn"]] = len(tracker.deduced())

    result = data["result"]
    print(
        f"{result.get('model', '?')} effort={result.get('effort', '?')} seed={result['seed']} "
        f"tier={args.tier}: correct {result['correct']}/{result['board_n']}, "
        f"locks +{result['lock_correct']}/-{result['lock_wrong']}\n"
    )
    turn = 0
    for msg in data["transcript"]:
        if msg["role"] == "user" and msg["text"].startswith("TURN"):
            turn += 1
        if not (lo <= turn <= hi):
            continue
        if msg["role"] == "user":
            label = "OBS" if msg["text"].startswith("TURN") else "ENV"
            print(f"--- {label} (turn {turn}) " + "-" * 40)
            print(textwrap.indent(msg["text"], "  "))
        elif msg["role"] == "assistant":
            reasoning = " ".join(msg.get("reasoning", []))
            if reasoning:
                shown = reasoning[: args.chars] + ("…" if len(reasoning) > args.chars else "")
                print("  [thinking]")
                print(textwrap.indent(textwrap.fill(shown, 100), "  | "))
            print(f"  [action] {msg['text'].strip()[:300]}")
            ded = deducible_by_turn.get(turn)
            if ded is not None:
                print(f"  [sound tracker would know: {ded}/{board.n}]")
    print()


if __name__ == "__main__":
    main()
