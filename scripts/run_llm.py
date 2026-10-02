#!/usr/bin/env python3
"""Run an LLM on blindboard tiers (interactive runner world, sync).

  secret run copybench-openai --env OPENAI_API_KEY -- \
    python scripts/run_llm.py --model gpt-5.4-nano --tier t0_frozen --seeds 5
  secret run anthropic --env ANTHROPIC_API_KEY -- \
    python scripts/run_llm.py --model claude-haiku-4-5 --tier t0_frozen --seeds 5

Provider dispatch is by model name (claude-* -> Anthropic Messages, else
OpenAI Responses). Costs are reported at undiscounted list prices (house
rule) even though collection uses flex (OpenAI) — Anthropic has no flex."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.anthropic_client import make_adapter
from blindboard.env import with_salt
from blindboard.retained_adapter import HARNESS as RETAINED_HARNESS, RetainedReasoningAdapter
from blindboard.llm_runner import run_llm_episode
from blindboard.runner import summarize
from blindboard.tiers import TIERS


def _cost_record(model: str, harness: str | None) -> dict[str, str]:
    """How this run's cost was produced, for the cost-integrity audit. The
    retained harness prices every request at its day's list (the route's own
    bill on OpenRouter); visible-transcript runs are priced from usage totals."""
    record = {"collected_on": time.strftime("%Y-%m-%d", time.gmtime())}
    if harness == "retained":
        record["cost_basis"] = (
            "route_reported" if model.startswith("openrouter/") else "per_request_list"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tier", required=True, choices=list(TIERS))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--seed-list", default=None,
                        help="comma-separated explicit seeds (overrides --seeds/--seed-offset); for gap-filling interrupted runs")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--salt", default="",
                        help="secret per-evaluation salt (official runs; see docs/release-policy.md). "
                             "Empty (default) reproduces the published unsalted environment.")
    parser.add_argument("--allow-unbound-effort", action="store_true",
                        help="collect from a route that cannot bind graded effort; the run is "
                             "stamped effort_binding=False (default: refuse before any billed call)")
    parser.add_argument("--harness", choices=["retained"], default=None,
                        help="retained: provider-native session that carries the model's own "
                             "reasoning across turns (provider-native replay of its opaque "
                             "reasoning state; see blindboard/retained_adapter.py for supported "
                             "providers). Required for multi-turn episodes; the legacy text "
                             "adapters fail closed.")
    parser.add_argument("--compact-threshold", type=int, default=None,
                        help="enable server-side compaction at this many tokens (retained harness only)")
    parser.add_argument("--service-tier", default="flex")
    parser.add_argument("--out", default="results/llm")
    parser.add_argument("--parallel", type=int, default=16, help="concurrent episodes")
    parser.add_argument("--verbose", action="store_true", help="per-turn progress lines")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_slug = args.model.replace("/", "-")
    harness_tag = f"-{args.harness}" if args.harness else ""
    run_dir = Path(args.out) / f"{model_slug}-{args.effort}{harness_tag}-{args.tier}-{stamp}"
    adapters: dict[int, object] = {}

    failed: dict[int, str] = {}

    def run_one(seed: int) -> dict | None:
        try:
            return _run_one(seed)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not sink the cell
            failed[seed] = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"seed {seed}: FAILED {failed[seed]}", flush=True)
            return None

    def _run_one(seed: int) -> dict:
        if args.harness == "retained":
            adapter = RetainedReasoningAdapter(
                model=args.model,
                effort=args.effort,
                service_tier=args.service_tier or None,
                compact_threshold=args.compact_threshold,
            )
        else:
            adapter = make_adapter(
                model=args.model,
                effort=args.effort,
                service_tier=args.service_tier or None,
                **({"require_effort_binding": False} if args.allow_unbound_effort else {}),
            )
        adapters[seed] = adapter
        t0 = time.time()
        config = with_salt(TIERS[args.tier], args.salt) if args.salt else TIERS[args.tier]
        result = run_llm_episode(
            adapter, config, seed, transcript_dir=run_dir, verbose=args.verbose
        )
        d = result.get("diagnostics") or {}
        print(
            f"seed {seed}: correct {result['correct']}/{result['board_n']} "
            f"guess_turn={result['guess_turn']} lock_score={result['lock_score']} "
            f"deducible={d.get('deducible_at_guess', '-')} "
            f"retries={result['total_retries']} violations={len(result['violations'])} "
            f"({time.time() - t0:.0f}s, {result['api_calls']} calls)",
            flush=True,
        )
        return result

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",") if s.strip() != ""]
    else:
        seeds = list(range(args.seed_offset, args.seed_offset + args.seeds))
    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(run_one, seeds))
    else:
        results = [run_one(s) for s in seeds]
    results = [r for r in results if r is not None]
    if not results:
        # Still billed: record what the failed attempts cost before exiting.
        # failed_run.json is not a summary, so no cell ever scores this dir.
        spent: dict[str, int] = {}
        for adapter in adapters.values():
            for k, v in adapter.usage_totals.items():
                spent[k] = spent.get(k, 0) + v
        costs = [adapter.cost_usd() for adapter in adapters.values()]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "failed_run.json").write_text(json.dumps({
            "model": args.model, "effort": args.effort, "tier": args.tier,
            "harness": RETAINED_HARNESS if args.harness == "retained" else "visible-transcript",
            "failed_seeds": {str(k): v for k, v in sorted(failed.items())},
            **_cost_record(args.model, args.harness),
            "usage": spent,
            "cost_usd_list_price": (
                round(sum(c for c in costs if c is not None), 4)
                if any(c is not None for c in costs) else None
            ),
        }, indent=1, sort_keys=True), encoding="utf-8")
        raise SystemExit(f"every episode failed: {failed}")
    summary = summarize(results)
    if failed:
        # Recorded, never hidden: gap-fill these with --seed-list.
        summary["failed_seeds"] = {str(k): v for k, v in sorted(failed.items())}
    diag = [r["diagnostics"] for r in results if r.get("diagnostics")]
    if diag:
        summary["deducible_at_guess_mean"] = round(
            sum(d["deducible_at_guess"] for d in diag) / len(diag), 2
        )
        summary["deduced_correct_in_guess_mean"] = round(
            sum(d["deduced_correct_in_guess"] for d in diag) / len(diag), 2
        )
        summary["known_presses_total"] = sum(d["known_presses"] for d in diag)
    summary["total_retries"] = sum(r["total_retries"] for r in results)
    summary["model"] = args.model
    summary["effort"] = args.effort
    from blindboard.prompts import PROMPT_VERSION
    summary["prompt_version"] = PROMPT_VERSION
    bindings = {getattr(a, "effort_binding", None) for a in adapters.values()}
    if bindings == {False}:
        summary["effort_binding"] = False  # requested effort cannot bind on this route
    summary["tier"] = args.tier
    summary["harness"] = RETAINED_HARNESS if args.harness == "retained" else "visible-transcript"
    summary.update(_cost_record(args.model, args.harness))
    if args.harness == "retained":
        summary["reasoning_context"] = "all_turns"
        summary["compact_threshold"] = args.compact_threshold
    if args.salt:
        from blindboard.env import salt_commitment
        summary["salt_commitment"] = salt_commitment(args.salt)
    usage: dict[str, int] = {}
    for adapter in adapters.values():
        for k, v in adapter.usage_totals.items():
            usage[k] = usage.get(k, 0) + v
    summary["usage"] = usage
    costs = [adapter.cost_usd() for adapter in adapters.values()]
    summary["cost_usd_list_price"] = (
        round(sum(c for c in costs if c is not None), 4) if any(c is not None for c in costs) else None
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
