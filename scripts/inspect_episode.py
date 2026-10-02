#!/usr/bin/env python3
"""Episode autopsy: deterministic replay + belief-curve + lock/guess audit.

  uv run python scripts/inspect_episode.py results/llm/<run>/seed0000.json --tier t2_announced

Always uses canonical board ordering (transcripts are saved with
sort_keys=True, so NEVER derive position indices from saved dict order)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.env import Action, Episode
from blindboard.tiers import TIERS
from blindboard.tracker import BeliefTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_json")
    parser.add_argument("--tier", required=True, choices=list(TIERS))
    args = parser.parse_args()

    data = json.loads(Path(args.episode_json).read_text(encoding="utf-8"))
    records = data["records"]
    result = data["result"]
    seed = result["seed"]
    config = TIERS[args.tier]
    ep = Episode(config, seed)
    board = ep.board
    tracker = BeliefTracker(board)

    print(f"seed {seed}  tier {args.tier}  model {result.get('model', '?')}")
    print(f"initial layout: {dict(zip(board.addresses, ep.layout))}\n")
    lock_log: list[tuple[int, str, str, bool]] = []
    for r in records:
        turn = ep.turn
        act = Action(presses=list(r["presses"]), locks=dict(r["locks_accepted"]))
        if ep.validate(act) is not None:
            print(f"turn {turn}: RECORDED ACTION NOW INVALID (env drift?) — stopping")
            return
        pre_layout = list(ep.layout)
        for addr, letter in r["locks_accepted"].items():
            ok = pre_layout[board.index_of(addr)] == letter
            lock_log.append((turn, addr, letter, ok))
        ep.apply(act)
        rec = ep.records[-1]
        drift = rec["results"] != r["results"] or rec["rotation"] != r["rotation"]
        positions = [board.index_of(a) for a in r["presses"]]
        if positions:
            tracker.observe_chord(positions, r["results"])
        rot = rec["rotation"]
        if rot["mode"] == "announced_full":
            tracker.apply_rotation_mapping({int(k): v for k, v in rot["permutation"].items()})
        elif rot["mode"] in ("announced", "observer", "storm"):
            tracker.apply_rotation(rot["positions"])
        rotated = ",".join(board.addresses[p] for p in rot["positions"]) or "-"
        print(
            f"turn {turn:2d}: pressed {', '.join(r['presses']) or '-':60s} "
            f"-> {','.join(r['results']) or '-':8s} rotated: {rotated:40s} "
            f"deducible: {len(tracker.deduced()):2d}/{board.n}"
            + ("  [REPLAY DRIFT]" if drift else "")
        )
    print("\nlocks (evaluated at lock time):")
    for turn, addr, letter, ok in lock_log:
        print(f"  turn {turn:2d}: {addr} = {letter}  {'CORRECT' if ok else 'WRONG'}")
    guess = result.get("guess", {})
    final_layout = {addr: ep.layout[i] for i, addr in enumerate(board.addresses)}
    deduced = tracker.deduced()
    print("\nguess audit (canonical order):")
    for i, addr in enumerate(board.addresses):
        g, t = guess.get(addr, "-"), final_layout[addr]
        ded = deduced.get(i)
        mark = "ok " if g == t else "MISS"
        extra = " (was deducible)" if ded is not None and g != t else ""
        print(f"  {addr:18s} guess {g}  truth {t}  {mark}{extra}")
    n_correct = sum(1 for a in board.addresses if guess.get(a) == final_layout[a])
    print(f"\ncorrect {n_correct}/{board.n}  deducible_at_guess {len(deduced)}")


if __name__ == "__main__":
    main()
