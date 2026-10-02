#!/usr/bin/env python3
"""Watch an agent play (or play yourself) — the feel-the-game script.

  python scripts/play.py --tier t2_announced --seed 0 --agent reference
  python scripts/play.py --tier t0_frozen --seed 3 --agent human
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.agents import AGENTS, ReferenceAgent
from blindboard.env import Episode
from blindboard.prompts import build_system_prompt, format_observation
from blindboard.protocol import parse_action
from blindboard.tiers import TIERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default="t2_announced")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent", default="reference", choices=[*AGENTS, "human"])
    args = parser.parse_args()

    config = TIERS[args.tier]
    ep = Episode(config, args.seed)
    agent = None if args.agent == "human" else AGENTS[args.agent](config, args.seed)
    if args.agent == "human":
        print(build_system_prompt(config))
    print(f"\n[secret layout: {dict(zip(ep.board.addresses, ep.layout))}]\n" if agent else "")

    while not ep.done:
        obs = ep.observation()
        print("-" * 60)
        print(format_observation(obs, config))
        if agent is None:
            while True:
                raw = input("> ")
                action, err = parse_action(raw, ep.board)
                if err is None and action is not None:
                    err = ep.validate(action)
                if err is None:
                    break
                print(f"INVALID: {err}")
        else:
            action = agent.act(obs)
            err = ep.validate(action)
            if err:
                raise RuntimeError(err)
            if action.guess:
                print(f"[{args.agent}] GUESS")
            else:
                print(f"[{args.agent}] presses {action.presses} locks {action.locks or ''}")
                if isinstance(agent, ReferenceAgent):
                    print(f"    deduced so far: {len(agent.tracker.deduced())}/{ep.board.n}")
        ep.apply(action)

    result = ep.result or {}
    print("=" * 60)
    print(
        f"correct {result['correct']}/{result['board_n']}  "
        f"lock_score {result['lock_score']} "
        f"(+{result['lock_correct']}/-{result['lock_wrong']})  "
        f"guessed at turn {result['guess_turn']}"
    )
    print("final layout:", json.dumps(result["final_layout"]))


if __name__ == "__main__":
    main()
