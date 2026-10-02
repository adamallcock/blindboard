#!/usr/bin/env python3
"""Run the static finger-typing probe (single-turn, cheap).

  secret run copybench-openai --env OPENAI_API_KEY -- \
    python scripts/run_static.py --model gpt-5.4-nano --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.anthropic_client import make_adapter
from blindboard.static_probe import run_static_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="*", default=["encode_word", "decode", "encode_answer"])
    parser.add_argument("--n", type=int, default=20, help="items per task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--service-tier", default="flex")
    parser.add_argument("--out", default="results/static")
    parser.add_argument("--parallel", type=int, default=16)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / f"{args.model}-{args.effort}-{stamp}"
    adapter = make_adapter(
        model=args.model, effort=args.effort, service_tier=args.service_tier or None
    )
    outcome = run_static_probe(
        adapter, args.tasks, args.n, args.seed, out_dir=out_dir, parallel=args.parallel
    )
    summary = outcome["summary"]
    summary["usage"] = dict(adapter.usage_totals)
    cost = adapter.cost_usd()
    summary["cost_usd_list_price"] = round(cost, 4) if cost is not None else None
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
