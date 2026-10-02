"""Stored diagnostics must equal a fresh replay (red-team M-5).

The relabel addressing bug was fixed in code and the paper, but the
stored run artifacts served the pre-fix numbers until 2026-07-21. This
gate replays every stored relabel episode's records through the current
tracker with relabel-aware addressing and requires the stored
diagnostics to match, so artifact and paper cannot diverge again."""

from __future__ import annotations

import glob
import json
from pathlib import Path

from blindboard.tiers import TIERS
from blindboard.tracker import BeliefTracker


def replay_conversion(tier: str, episode: dict) -> tuple[int, int]:
    config = TIERS[tier]
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


def test_stored_relabel_diagnostics_match_replay() -> None:
    root = Path(__file__).resolve().parents[1]
    dirs = sorted(glob.glob(str(root / "results/llm/*t10_relabel*")))
    checked = 0
    for d in dirs:
        if not (Path(d) / "summary.json").exists():
            continue
        for f in sorted(Path(d).glob("seed*.json")):
            ep = json.loads(f.read_text())
            ded, conv = replay_conversion("t10_relabel", ep)
            diag = ep["result"]["diagnostics"]
            assert diag["deducible_at_guess"] == ded, f.name
            assert diag["deduced_correct_in_guess"] == conv, f.name
            checked += 1
    assert checked >= 9, "expected the two stored relabel runs to be present"
