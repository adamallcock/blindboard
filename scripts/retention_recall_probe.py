#!/usr/bin/env python3
"""Behavioural check that a model can USE its retained reasoning.

The live retention checks show that a turn's reasoning re-enters the next
prompt (the billed prompt grows by it). This probe asks whether the model
can act on it, with the modular-secret design from the Gemini session notes
(benchmark-kit docs/2026-09-25-gemini-native-sessions.md): a direct "what
did you pick?" question is a poor probe, because models often reason that
earlier thoughts are dropped and answer UNKNOWN.

One trial, all through the harness's own native session:

1. The model privately picks N in [1000, 9999] and replies only N mod 97.
2. The session is forked twice (same native state, fresh request keys):
   R1 asks for N mod 89 at once; R2 first answers an unrelated question,
   then asks for N mod 83.
3. A visible-only control (the legacy text harness, which drops reasoning)
   gets the same visible transcripts and both questions.

Two variants. ``escape`` lets the model answer UNKNOWN, which measures
whether it acts on its retained reasoning unprompted; ``forced`` requires a
number, which measures whether the information is there to use at all.

N is never in any visible text. The primary test is whether both retained
branches answer for the N that the first turn's own reasoning summary
states (the summary is checked against the visible residue). Joint
consistency of the three residues is reported too, but it is weak: a
model without memory can pass it by reconstructing a canonical N from the
visible residue (the smallest four-digit N with that residue, say). The
visible-only control's match rate to the first turn's N is the baseline
for stereotyped choices and reconstruction. Trials whose first reply shows
a four-digit number are marked leaked and excluded.

Every attempted trial is kept and counted: a trial whose first-turn summary
states no N cannot be scored against N, and a branch that fails records its
exception, so an infrastructure failure is never counted as an abstention.

Only public facts are written (visible texts, reasoning summaries, usage,
list cost); the provider's opaque state never leaves memory.

Usage (keys via the secret wrapper, one provider per process):
  secret run copybench-openai --env OPENAI_API_KEY -- \\
    uv run python scripts/retention_recall_probe.py --models gpt-5.6-sol gpt-6-astra
  secret run openrouter --env OPENROUTER_API_KEY -- \\
    uv run python scripts/retention_recall_probe.py --effort high \\
    --models openrouter/deepseek/deepseek-v4-pro
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard import openrouter_client  # noqa: E402
from blindboard.anthropic_client import make_adapter  # noqa: E402
from blindboard.anthropic_native import AnthropicNativeSession  # noqa: E402
from blindboard.client import Message  # noqa: E402
from blindboard.gemini_native import GeminiNativeSession  # noqa: E402
from blindboard.native_session import today_utc  # noqa: E402
from blindboard.openrouter_native import OpenRouterNativeSession  # noqa: E402
from blindboard.pricing import price_call  # noqa: E402
from blindboard.responses_native import OpenAINativeSession  # noqa: E402
from blindboard.retained_adapter import factory_for  # noqa: E402

SYSTEM = (
    "You are taking part in a short test of working memory. Follow each "
    "instruction exactly and keep every reply to the bare answer."
)
T1 = (
    "Choose a random whole number N with 1000 <= N <= 9999 and keep it to "
    "yourself. Do not write N anywhere in your reply. Reply with only the "
    "value of N mod 97, as a bare number."
)
ESCAPE = " If you cannot determine it, reply UNKNOWN."
FORCED = " You must reply with a number."
Q89 = "Reply with only the value of N mod 89, as a bare number." + ESCAPE
INTERVENING = "Unrelated question: reply with only the value of 38 + 47."
Q83 = (
    "Back to your number N from the start: reply with only the value of "
    "N mod 83, as a bare number." + ESCAPE
)
VARIANTS = {
    "escape": (Q89, Q83),
    "forced": (Q89.replace(ESCAPE, FORCED), Q83.replace(ESCAPE, FORCED)),
}
LOW, HIGH = 1000, 9999


def parse_residue(text: str | None) -> int | None:
    """The reply's number, or None for UNKNOWN / no number."""
    if not text or "UNKNOWN" in text.upper():
        return None
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def leaked(text: str | None) -> bool:
    """The first reply shows a four-digit number (N may be visible)."""
    return bool(text) and re.search(r"(?<!\d)\d{4}(?!\d)", text) is not None


def candidates(r97: int, r89: int) -> list[int]:
    return [n for n in range(LOW, HIGH + 1) if n % 97 == r97 and n % 89 == r89]


def jointly_consistent(r97: int | None, r89: int | None, r83: int | None) -> bool:
    """Some N in range matches all three residues."""
    if r97 is None or r89 is None or r83 is None:
        return False
    return any(n % 83 == r83 for n in candidates(r97, r89))


def summary_numbers(summaries: list[str], r97: int | None) -> list[int]:
    """Four-digit numbers in the reasoning summaries that fit the visible residue."""
    if r97 is None:
        return []
    found = {int(m) for s in summaries for m in re.findall(r"(?<!\d)\d{4}(?!\d)", s or "")}
    return sorted(n for n in found if LOW <= n <= HIGH and n % 97 == r97)


def fork(session: Any) -> Any:
    """An independent copy of a native session's state; the transport is shared."""
    if isinstance(session, OpenAINativeSession):
        child = copy.copy(session)
        inner = copy.copy(session._session)
        inner._mutation_lock = threading.Lock()
        inner._replay_items = copy.deepcopy(session._session._replay_items)
        inner._state_digests = copy.deepcopy(session._session._state_digests)
        inner._pending = None
        # A fresh logical id gives every branch its own idempotency keys.
        inner.logical_session_id = str(uuid.uuid4())
        child._session = inner
        return child
    if isinstance(session, AnthropicNativeSession):
        child = copy.copy(session)
        child._messages = copy.deepcopy(session._messages)
        child._lock = threading.Lock()
        return child
    if isinstance(session, GeminiNativeSession):
        child = copy.copy(session)
        child._contents = copy.deepcopy(session._contents)
        child._lock = threading.Lock()
        return child
    if isinstance(session, OpenRouterNativeSession):
        child = copy.copy(session)
        child._history = copy.deepcopy(session._history)
        child._avoided_hosts = list(session._avoided_hosts)
        child._lock = threading.Lock()
        return child
    raise TypeError(f"no fork for {type(session).__name__}")


def turn_record(turn: Any) -> dict[str, Any]:
    return {
        "text": turn.text,
        "usage": dict(turn.usage),
        "cost_usd_list": turn.cost_usd,
        "reasoning_summaries": list(turn.reasoning_summaries),
    }


def control_cost(model: str, usage: dict[str, int], on: str, reported: float | None = None) -> float | None:
    """List cost of a visible-only call; all input priced as uncached. An
    OpenRouter call is reported at OpenRouter's own per-request bill."""
    if model.startswith("openrouter/") and reported is not None:
        return reported
    inp = sum(int(usage.get(k, 0) or 0) for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    out = int(usage.get("output_tokens", 0) or 0)
    key = f"anthropic/{model}" if model.startswith("claude") else model
    return price_call(key, {"input_tokens": inp, "output_tokens": out}, on=on)


_normalize_openrouter = openrouter_client.normalize_openrouter_response


def _normalize_with_cost(resp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """OpenRouter's reported ``usage.cost``, which the text adapter drops."""
    text, meta = _normalize_openrouter(resp)
    cost = (resp.get("usage") or {}).get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        meta["reported_cost_usd"] = float(cost)
    return text, meta


def session_tier(model: str) -> str | None:
    """The campaign's tiers: flex for OpenAI, and for Gemini flex with a
    fall-back to standard after three refusals."""
    return "flex" if model.startswith(("gpt-", "gemini/")) else None


def run_trial(model: str, effort: str, variant: str, trial: int, on: str) -> dict[str, Any]:
    q89, q83 = VARIANTS[variant]
    record: dict[str, Any] = {"model": model, "effort": effort, "variant": variant, "trial": trial}
    try:
        root = factory_for(model)(
            model=model, instructions=SYSTEM, effort=effort,
            service_tier=session_tier(model),
            compact_threshold=None, api_key=None, transport=None, price_date=None,
        )
        t1 = root.turn([T1])
        record["t1"] = turn_record(t1)
        r1_session, r2_session = fork(root), fork(root)
        results: dict[str, Any] = {}

        def r1() -> None:
            results["r1"] = turn_record(r1_session.turn([q89]))

        def r2() -> None:
            results["r2_intervening"] = turn_record(r2_session.turn([INTERVENING]))
            results["r2"] = turn_record(r2_session.turn([q83]))

        def visible_only(key: str, tail: list[Message]) -> None:
            adapter = make_adapter(
                model=model, effort=effort,
                service_tier="flex" if model.startswith("gpt-") else None,
                allow_lossy_multi_turn=True,
            )
            text, meta = adapter.call_with_meta([
                Message("system", SYSTEM), Message("user", T1), Message("assistant", t1.text),
                *tail,
            ])
            usage = {k: v for k, v in (meta.get("usage") or {}).items() if isinstance(v, int)}
            results[key] = {
                "text": text, "usage": usage,
                "cost_usd_list": control_cost(model, usage, on, meta.get("reported_cost_usd")),
                "reasoning_summaries": list(meta.get("reasoning") or []),
            }

        def c1() -> None:
            visible_only("control", [Message("user", q89)])

        def c2() -> None:
            visible_only("control2", [
                Message("user", INTERVENING), Message("assistant", "85"), Message("user", q83),
            ])

        branch_errors: dict[str, str] = {}

        def guarded(name: str, branch: Any) -> Any:
            def run() -> None:
                try:
                    branch()
                except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                    branch_errors[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return run

        threads = [
            threading.Thread(target=guarded(name, branch))
            for name, branch in (("r1", r1), ("r2", r2), ("control", c1), ("control2", c2))
        ]
        errors: list[str] = []
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        record.update(results)
        if branch_errors:
            record["branch_errors"] = dict(sorted(branch_errors.items()))
        missing = [k for k in ("r1", "r2", "control", "control2") if k not in results]
        if missing:
            errors.append(f"branches failed: {missing}")
        r97 = parse_residue(t1.text)
        r89 = parse_residue(results.get("r1", {}).get("text"))
        r83 = parse_residue(results.get("r2", {}).get("text"))
        c89 = parse_residue(results.get("control", {}).get("text"))
        c83 = parse_residue(results.get("control2", {}).get("text"))
        summary_n = summary_numbers(record["t1"]["reasoning_summaries"], r97)
        retained_complete = "r1" in results and "r2" in results
        record["verdict"] = {
            "r97": r97, "r89": r89, "r83": r83, "control_r89": c89, "control_r83": c83,
            "leaked": leaked(t1.text),
            # Both retained branches returned a reply (a failed branch is an
            # infrastructure failure, never an abstention).
            "retained_complete": retained_complete,
            "retained_answered": r89 is not None and r83 is not None,
            "retained_abstained": retained_complete and (r89 is None or r83 is None),
            "jointly_consistent": jointly_consistent(r97, r89, r83),
            # N as the first turn's own summary states it, when it does.
            "summary_n": summary_n,
            "matches_summary_n": bool(summary_n) and any(
                n % 89 == r89 and n % 83 == r83 for n in summary_n),
            "control_matches_summary_n": bool(summary_n) and any(
                n % 89 == c89 and n % 83 == c83 for n in summary_n),
            "control_answered": c89 is not None and c83 is not None,
            "control_jointly_consistent": jointly_consistent(r97, c89, c83),
        }
        if errors:
            record["error"] = "; ".join(errors)
    except Exception as exc:  # noqa: BLE001 - one bad trial must not sink the probe
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    costs = [
        (record.get(k) or {}).get("cost_usd_list")
        for k in ("t1", "r1", "r2_intervening", "r2", "control", "control2")
    ]
    record["cost_usd_list"] = sum(c for c in costs if c is not None)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--trials", type=int, default=8, help="per variant")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--out", default="results/diagnostics")
    parser.add_argument("--parallel", type=int, default=32)
    args = parser.parse_args()

    openrouter_client.normalize_openrouter_response = _normalize_with_cost
    on = today_utc()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jobs = [(m, v, t) for m in args.models for v in args.variants for t in range(args.trials)]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        records = list(pool.map(lambda job: run_trial(job[0], args.effort, job[1], job[2], on), jobs))

    for model in args.models:
        rows = [r for r in records if r["model"] == model]

        def tally(variant: str) -> dict[str, int]:
            part = [r for r in rows if r["variant"] == variant]
            scored = [r for r in part if "verdict" in r and not r["verdict"]["leaked"]]
            v = [r["verdict"] for r in scored]
            return {
                "trials": len(part),
                "errors": sum(1 for r in part if r.get("error")),
                "leaked": sum(1 for r in part if (r.get("verdict") or {}).get("leaked")),
                "scored": len(scored),
                "branch_failures": sum(1 for x in v if not x.get("retained_complete", True)),
                "summary_n_complete": sum(
                    1 for x in v if x["summary_n"] and x.get("retained_complete", True)),
                "abstained_summary_n": sum(
                    1 for x in v if x["summary_n"] and x.get("retained_abstained", False)),
                "retained_answered": sum(x["retained_answered"] for x in v),
                "jointly_consistent": sum(x["jointly_consistent"] for x in v),
                "summary_shows_n": sum(bool(x["summary_n"]) for x in v),
                "matches_summary_n": sum(x["matches_summary_n"] for x in v),
                "control_matches_summary_n": sum(x["control_matches_summary_n"] for x in v),
                "control_answered": sum(x["control_answered"] for x in v),
                "control_jointly_consistent": sum(x["control_jointly_consistent"] for x in v),
            }

        summary = {
            "purpose": "retention recall probe: can the model act on its retained reasoning?",
            "model": model,
            "effort": args.effort,
            "tier": None,
            "harness": "retained-reasoning (forked native sessions) vs visible-only control",
            "collected_on": on,
            "cost_basis": "per_request_list",
            "variants": {variant: tally(variant) for variant in args.variants},
            "cost_usd_list_price": round(sum(r["cost_usd_list"] for r in rows), 6),
        }
        run_dir = Path(args.out) / f"retention-recall-probe-{model.replace('/', '-')}-{args.effort}-{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trials.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
        )
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in ("model", "variants", "cost_usd_list_price")}),
              flush=True)


if __name__ == "__main__":
    main()
