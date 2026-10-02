"""Interactive LLM episode loop: full-transcript regime (R1).

The whole conversation is resent every turn (prefix-cache friendly), so
the model CAN reread old results — this measures reasoning over an
accumulating, partially-stale log under interference, the realistic agent
condition. A truncated 'memo' regime (R2) is future work.

Illegal/unparseable actions get exactly one corrective retry; a second
failure forfeits the turn (press phase) or scores zero (guess phase).
Retries and violations are first-class metrics: instruction integrity
degrading under cognitive load is signal, not noise."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blindboard.client import (
    APIAdapter,
    LossyMultiTurnStateError,
    Message,
    provenance_rollup,
    require_lossless_multi_turn,
)
from blindboard.dual import make_episode
from blindboard.env import EnvConfig
from blindboard.prompts import build_system_prompt, format_observation
from blindboard.protocol import parse_action
from blindboard.runner import diagnose_any

LEGACY_VISIBLE_HARNESS = "visible-transcript (legacy opt-in)"


def run_llm_episode(
    adapter: APIAdapter,
    config: EnvConfig,
    seed: int,
    transcript_dir: Path | None = None,
    max_retries_per_turn: int = 1,
    verbose: bool = False,
    legacy_visible_transcript: bool = False,
) -> dict[str, Any]:
    if legacy_visible_transcript:
        # Reproduces the pre-September condition: each request is rebuilt
        # from visible text, so the model sees its earlier moves but none of
        # its earlier reasoning. Only for cells labelled as that legacy
        # condition, and only with an adapter the caller built lossy on purpose.
        if getattr(adapter, "allow_lossy_multi_turn", None) is not True:
            raise LossyMultiTurnStateError(
                "legacy_visible_transcript=True needs an adapter built with "
                f"allow_lossy_multi_turn=True; {type(adapter).__name__} is not"
            )
    else:
        require_lossless_multi_turn(
            adapter,
            purpose="Blindboard full-transcript episode",
        )
    ep = make_episode(config, seed)
    usage_before = dict(adapter.usage_totals)
    calls_before = adapter.calls
    messages: list[Message] = [Message("system", build_system_prompt(config))]
    # Parallel transcript with per-assistant-call reasoning summaries + usage.
    transcript: list[dict] = [messages[0].to_dict()]
    total_retries = 0

    def call(msgs: list[Message]) -> str:
        text, meta = adapter.call_with_meta(msgs)
        msgs.append(Message("assistant", text))
        transcript.append({"role": "assistant", "text": text, **meta})
        return text

    while not ep.done:
        obs = ep.observation()
        messages.append(Message("user", format_observation(obs, config)))
        transcript.append(messages[-1].to_dict())
        reply = call(messages)
        action, err = parse_action(reply, ep.board)
        if err is None and action is not None:
            err = ep.validate(action)
        retries = 0
        while err is not None and retries < max_retries_per_turn:
            retries += 1
            total_retries += 1
            messages.append(
                Message(
                    "user",
                    f"INVALID ACTION: {err} Reply with ONLY a corrected action JSON.",
                )
            )
            transcript.append(messages[-1].to_dict())
            reply = call(messages)
            action, err = parse_action(reply, ep.board)
            if err is None and action is not None:
                err = ep.validate(action)
        if err is not None:
            if ep.phase == "guess":
                ep.force_zero_guess(err)
            else:
                ep.skip_turn(err)
            continue
        assert action is not None
        ep.apply(action, meta={"retries": retries})
        if verbose:
            print(
                f"  seed {seed} turn {ep.turn - 1}: pressed {action.presses or 'GUESS'}"
                + (f" locks={action.locks}" if action.locks else ""),
                flush=True,
            )
    result = dict(ep.result or {})
    result["records"] = ep.records
    result.update(diagnose_any(ep))
    usage = {
        k: adapter.usage_totals.get(k, 0) - usage_before.get(k, 0)
        for k in adapter.usage_totals
    }
    result["usage"] = usage
    result["api_calls"] = adapter.calls - calls_before
    result["total_retries"] = total_retries
    result["model"] = adapter.model
    result["effort"] = adapter.effort
    from blindboard.prompts import PROMPT_VERSION
    result["prompt_version"] = PROMPT_VERSION
    if legacy_visible_transcript:
        result["harness"] = LEGACY_VISIBLE_HARNESS
    # Which implementation actually served this episode: distinct routed
    # hosts / model builds / service tiers across its calls (per-call
    # response IDs stay in the transcript). More than one value for a key
    # means the episode was not served by a single implementation.
    result["provenance"] = provenance_rollup(transcript)
    if transcript_dir is not None:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        path = transcript_dir / f"seed{seed:04d}.json"
        path.write_text(
            json.dumps(
                {
                    "result": {k: v for k, v in result.items() if k != "records"},
                    "records": result["records"],
                    "transcript": transcript,
                },
                ensure_ascii=False,
                indent=1,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return result
