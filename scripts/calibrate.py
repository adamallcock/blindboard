#!/usr/bin/env python3
"""Local play tests: run scripted agents over tiers to calibrate difficulty.

Usage:
  python scripts/calibrate.py --seeds 200
  python scripts/calibrate.py --tiers t2_announced t3_observer --agents reference random
  # Held-out check of the certificates (which use seeds 0..59) on fresh seeds:
  python scripts/calibrate.py --agents planner reference --seeds 60 --seed-start 5000 \
      --out results/calibration-heldout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Certificates are defined under a pinned hash seed: BeliefTracker's
# backtracking iterates a set before its seeded shuffle, so reference play
# is PYTHONHASHSEED-dependent across processes (found during solver-v3
# calibration, 2026-07-18). Re-exec with the pin rather than silently
# producing a per-process realization.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.agents import AGENTS
from blindboard.runner import run_scripted_episode, summarize
from blindboard.tiers import TIERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="*", default=list(TIERS))
    parser.add_argument("--agents", nargs="*", default=["reference", "assumed_order", "random"])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="first seed; certificates use 0, held-out checks a fresh range")
    parser.add_argument("--privileged-rotation", action="store_true",
                        help="feed reference the rotated sets even in unannounced modes (upper bound)")
    parser.add_argument("--out", default="results/calibration")
    args = parser.parse_args()

    table: dict[str, dict[str, dict]] = {}
    for tier_name in args.tiers:
        config = TIERS[tier_name]
        table[tier_name] = {}
        for agent_name in args.agents:
            t0 = time.time()
            results = []
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                agent = AGENTS[agent_name](config, seed)
                results.append(
                    run_scripted_episode(
                        agent, config, seed, privileged_rotation=args.privileged_rotation
                    )
                )
            summary = summarize(results)
            diag = [r["diagnostics"] for r in results if r.get("diagnostics")]
            if diag:
                summary["deducible_at_guess_mean"] = round(
                    sum(d["deducible_at_guess"] for d in diag) / len(diag), 2
                )
            summary["wall_seconds"] = round(time.time() - t0, 1)
            table[tier_name][agent_name] = summary
            print(
                f"{tier_name:18s} {agent_name:14s} "
                f"acc={summary['board_accuracy_mean']:.3f}±{summary['board_accuracy_sem']:.3f} "
                f"solve={summary['solve_rate']:.2f} "
                f"guess_turn={summary['guess_turn_mean']:5.2f} "
                f"lock={summary['lock_score_mean']:6.2f} "
                f"deducible={summary.get('deducible_at_guess_mean', '-')} "
                f"({summary['wall_seconds']}s)"
            )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"calibration-{stamp}.json"
    out_path.write_text(
        json.dumps({"seeds": args.seeds, "seed_start": args.seed_start, "table": table},
                   indent=1, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
