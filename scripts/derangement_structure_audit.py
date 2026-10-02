#!/usr/bin/env python3
"""Does the rotation generator carry structure a player could bet on?

On t7_perpetual and t11_maelstrom every turn ends with a derangement of
three announced keys, and only two arrangements are possible. A player
who knows the three symbols still faces a coin flip at the guess, and the
tier ceilings assume the flip is fair and fresh. A marginal balance
statistic cannot establish that: a sequence can be 50/50 overall and still
perfectly predictable from its past. This audit simulates episodes with
the public generator (synthetic seeds and salts, never an official salt)
and tests the direction of each rotation against everything a player
sees: the previous rotation, the running tally of directions (the rule a
model's reasoning bet on), symbol order, key geometry, relabel turns,
overlap with the previous rotated set, and the turn index.

Direction is "forward" when the symbol on the first-listed (lowest)
position moves to the second-listed position. The generator draws each
derangement by rejection sampling from ``random.Random.shuffle``
(``blindboard/env.py``), so every test should sit near 50%.

Usage: uv run python scripts/derangement_structure_audit.py [--episodes 1500]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.env import Episode, with_salt  # noqa: E402
from blindboard.protocol import Action  # noqa: E402
from blindboard.tiers import TIERS  # noqa: E402

AUDIT_TIERS = ("t11_maelstrom", "t7_perpetual")
FIRST_SEED = 100_000  # clear of every seed a model or the calibration played


def _forward(rotation: dict) -> bool:
    positions = rotation["positions"]
    perm = {int(k): v for k, v in rotation["permutation"].items()}
    return perm[positions[0]] == positions[1]


def simulate(tier: str, episodes: int, salt_prefix: str) -> list[list[dict]]:
    """Per episode, one record per three-key rotation, in turn order."""
    config = TIERS[tier]
    actions = random.Random(f"derangement-audit:{tier}:{salt_prefix}")
    out = []
    for i in range(episodes):
        cfg = with_salt(config, f"{salt_prefix}{i}") if salt_prefix else config
        ep = Episode(cfg, FIRST_SEED + i)
        seq: list[dict] = []
        while not ep.done:
            if ep.observation().phase == "guess" or ep.turn >= cfg.horizon:
                ep.apply(Action(guess={a: ep.board.symbols[0] for a in ep.board.addresses}))
                break
            before = list(ep.layout)
            ep.apply(Action(presses=actions.sample(ep.board.addresses, cfg.chord_size)))
            record = ep.records[-1]
            rotation = record["rotation"]
            if len(rotation.get("positions", [])) == 3:
                p = rotation["positions"]
                seq.append({
                    "forward": _forward(rotation),
                    "positions": p,
                    "symbols": [before[q] for q in p],
                    "relabel": record.get("relabel") is not None,
                })
        out.append(seq)
    return out


def _test(name: str, tier: str, outcomes: list[bool]) -> dict:
    n = len(outcomes)
    rate = sum(outcomes) / n if n else float("nan")
    z = (rate - 0.5) / math.sqrt(0.25 / n) if n else float("nan")
    return {"test": name, "tier": tier, "n": n, "rate": rate, "z": z}


def tier_tests(tier: str, eps: list[list[dict]], n_keys: int) -> list[dict]:
    flat = [r for s in eps for r in s]
    tests = [_test("P(forward)", tier, [r["forward"] for r in flat])]
    tests.append(_test("same direction as the previous rotation", tier,
                       [s[i]["forward"] == s[i - 1]["forward"] for s in eps for i in range(1, len(s))]))
    tally = []
    for s in eps:
        for i in range(4, len(s)):
            ups = sum(r["forward"] for r in s[:i])
            if ups * 2 != i:
                tally.append((ups * 2 > i) == s[i]["forward"])
    tests.append(_test("running-tally majority predicts the next direction", tier, tally))
    last = []
    for s in eps:
        if len(s) >= 5:
            ups, i = sum(r["forward"] for r in s[:-1]), len(s) - 1
            if ups * 2 != i:
                last.append((ups * 2 > i) == s[-1]["forward"])
    tests.append(_test("running tally predicts the final rotation", tier, last))
    tests.append(_test("P(forward | first symbol sorts before second)", tier,
                       [r["forward"] for r in flat if r["symbols"][0] < r["symbols"][1]]))
    tests.append(_test("P(forward | first symbol sorts after second)", tier,
                       [r["forward"] for r in flat if r["symbols"][0] > r["symbols"][1]]))
    tests.append(_test("P(forward | first gap shorter than second)", tier,
                       [r["forward"] for r in flat
                        if r["positions"][1] - r["positions"][0] < r["positions"][2] - r["positions"][1]]))
    half = n_keys // 2
    tests.append(_test("P(forward | all three keys in one half)", tier,
                       [r["forward"] for r in flat if r["positions"][2] < half or r["positions"][0] >= half]))
    relabel = [r["forward"] for r in flat if r["relabel"]]
    if relabel:
        tests.append(_test("P(forward | relabel on this turn)", tier, relabel))
    tests.append(_test("P(forward | overlaps the previous rotated set)", tier,
                       [s[i]["forward"] for s in eps for i in range(1, len(s))
                        if set(s[i]["positions"]) & set(s[i - 1]["positions"])]))
    for lo, hi in ((0, 8), (8, 16), (16, 99)):
        tests.append(_test(f"P(forward | turn {lo + 1}-{min(hi, 36)})", tier,
                           [s[i]["forward"] for s in eps for i in range(len(s)) if lo <= i < hi]))
    return tests


def run_audit(episodes: int = 1500) -> dict:
    """``episodes`` unsalted plus ``episodes`` salted episodes per tier."""
    tests: list[dict] = []
    rotations = 0
    for tier in AUDIT_TIERS:
        eps = simulate(tier, episodes, "") + simulate(tier, episodes, "audit")
        rotations += sum(len(s) for s in eps)
        tests.extend(tier_tests(tier, eps, TIERS[tier].board_obj().n))
    tally = [t for t in tests if t["test"] == "running-tally majority predicts the next direction"]
    return {
        "rotations": rotations,
        "tests": tests,
        "max_abs_z": max(abs(t["z"]) for t in tests),
        "tally_hit_rate": sum(t["rate"] * t["n"] for t in tally) / sum(t["n"] for t in tally),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    args = parser.parse_args()
    audit = run_audit(args.episodes)
    for t in audit["tests"]:
        print(f"{t['tier']:14s} {t['test']:52s} n={t['n']:6d} rate={t['rate']:.4f} z={t['z']:+.2f}")
    print(f"\n{audit['rotations']} rotations, {len(audit['tests'])} tests, "
          f"max |z| {audit['max_abs_z']:.2f}, tally rule hits {audit['tally_hit_rate']:.3f}")


if __name__ == "__main__":
    main()
