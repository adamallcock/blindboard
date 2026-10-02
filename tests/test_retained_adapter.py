"""Retained-reasoning adapter: native replay, transcript guard, public meta.

No network: a scripted transport stands in for the Responses API and
records every payload, so the tests can inspect exactly what would be
sent."""

from __future__ import annotations

import json

import pytest

from blindboard.client import Message
from blindboard.retained_adapter import (
    HARNESS,
    RetainedReasoningAdapter,
    TranscriptDivergenceError,
)


class ScriptedTransport:
    """Returns one completed response per POST, each with an encrypted
    reasoning item, and records the payloads it was sent."""

    def __init__(self, replies=None, reply_fn=None) -> None:
        self.replies = list(replies or [])
        self.reply_fn = reply_fn
        self.payloads: list[dict] = []

    def request(self, method, url, *, payload, idempotency_key, timeout_seconds):
        assert method == "POST"
        self.payloads.append(json.loads(json.dumps(payload)))
        n = len(self.payloads)
        text = self.reply_fn(payload) if self.reply_fn else self.replies[n - 1]
        return {
            "id": f"resp_{n}",
            "status": "completed",
            "model": "gpt-5.6-luna-2026-05-01",
            "service_tier": "flex",
            "reasoning": {"context": "all_turns"},
            "output": [
                {
                    "type": "reasoning",
                    "id": f"rs_{n}",
                    "encrypted_content": f"ENC-SECRET-{n}",
                    "summary": [{"type": "summary_text", "text": f"summary {n}"}],
                },
                {
                    "type": "message",
                    "id": f"msg_{n}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                },
            ],
            "usage": {
                # As observed live: the implicit breakpoint writes whatever
                # is not already cached.
                "input_tokens": 100 * n,
                "input_tokens_details": {
                    "cached_tokens": 50 * (n - 1),
                    "cache_write_tokens": 100 * n - 50 * (n - 1),
                },
                "output_tokens": 40,
                "output_tokens_details": {"reasoning_tokens": 30},
                "total_tokens": 100 * n + 40,
            },
        }


def _adapter(transport, price_date: str = "2026-09-25") -> RetainedReasoningAdapter:
    return RetainedReasoningAdapter(
        model="gpt-5.6-luna", transport=transport, api_key="k", price_date=price_date
    )


def test_second_turn_replays_native_reasoning_before_new_input() -> None:
    transport = ScriptedTransport(replies=["A1", "A2"])
    adapter = _adapter(transport)
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    text1, _ = adapter.call_with_meta(msgs)
    msgs += [Message("assistant", text1), Message("user", "obs 2")]
    adapter.call_with_meta(msgs)

    first, second = transport.payloads
    assert first["instructions"] == "rules"
    assert first["store"] is False
    assert first["reasoning"]["context"] == "all_turns"
    assert [item.get("role") for item in first["input"]] == ["user"]

    replayed = second["input"]
    types = [item.get("type") or item.get("role") for item in replayed]
    assert types == ["user", "reasoning", "message", "user"]
    assert replayed[1]["encrypted_content"] == "ENC-SECRET-1"
    assert replayed[-1]["content"] == "obs 2"
    # The assistant turn travels as the native message item, never as a
    # re-typed text message.
    assert not any(item.get("role") == "assistant" and "type" not in item for item in replayed)


def test_divergent_transcript_fails_before_any_billed_call() -> None:
    transport = ScriptedTransport(replies=["A1", "A2"])
    adapter = _adapter(transport)
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    adapter.call_with_meta(msgs)
    tampered = msgs + [Message("assistant", "NOT WHAT THE MODEL SAID"), Message("user", "obs 2")]
    with pytest.raises(TranscriptDivergenceError):
        adapter.call_with_meta(tampered)
    assert len(transport.payloads) == 1


def test_changed_system_prompt_and_missing_input_both_refuse() -> None:
    transport = ScriptedTransport(replies=["A1"])
    adapter = _adapter(transport)
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    text, _ = adapter.call_with_meta(msgs)
    with pytest.raises(TranscriptDivergenceError, match="system prompt"):
        adapter.call_with_meta([Message("system", "other"), *msgs[1:], Message("assistant", text), Message("user", "x")])
    with pytest.raises(TranscriptDivergenceError, match="no new input"):
        adapter.call_with_meta(msgs + [Message("assistant", text)])
    assert len(transport.payloads) == 1


def test_corrective_retry_is_just_another_native_turn() -> None:
    transport = ScriptedTransport(replies=["bad json", "{\"presses\": []}"])
    adapter = _adapter(transport)
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    bad, _ = adapter.call_with_meta(msgs)
    msgs += [Message("assistant", bad), Message("user", "INVALID ACTION: fix it")]
    good, _ = adapter.call_with_meta(msgs)
    assert good == "{\"presses\": []}"
    assert transport.payloads[1]["input"][-1]["content"] == "INVALID ACTION: fix it"


def test_usage_cost_and_public_meta_never_leak_native_state() -> None:
    transport = ScriptedTransport(replies=["A1", "A2"])
    adapter = _adapter(transport)
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    t1, m1 = adapter.call_with_meta(msgs)
    _, m2 = adapter.call_with_meta(msgs + [Message("assistant", t1), Message("user", "obs 2")])

    assert m2["usage"] == {
        "input_tokens": 200,
        "output_tokens": 40,
        "total_tokens": 240,
        "cache_read_tokens": 50,
        "cache_write_tokens": 150,
        "reasoning_tokens": 30,
    }
    assert adapter.usage_totals["input_tokens"] == 300
    assert adapter.calls == 2
    # September standard tier for luna: write $0.25, read $0.02, out $1.20.
    call1 = (100 * 0.25 + 40 * 1.20) / 1e6
    call2 = (50 * 0.02 + 150 * 0.25 + 40 * 1.20) / 1e6
    assert m1["cost_usd_list"] == pytest.approx(call1)
    assert adapter.cost_usd() == pytest.approx(call1 + call2)
    assert m2["price_date"] == "2026-09-25"
    assert m1["reasoning"] == ["summary 1"]
    assert m2["harness"]["name"] == HARNESS
    assert m2["harness"]["retention"]["reasoning_context"]["returned"] == "all_turns"
    assert m2["provenance"]["routed_model"] == "gpt-5.6-luna-2026-05-01"
    assert "ENC-SECRET" not in json.dumps(m1) + json.dumps(m2)


def test_unsupported_models_are_refused() -> None:
    with pytest.raises(ValueError, match="no native retained-reasoning session"):
        RetainedReasoningAdapter(model="mistral/some-model", api_key="k")


def test_full_episode_runs_through_the_lossless_guard(tmp_path) -> None:
    """End to end: the runner accepts the adapter, plays a frozen board,
    and the saved transcript carries summaries but no encrypted state."""
    from blindboard.llm_runner import run_llm_episode
    from blindboard.tiers import TIERS

    config = TIERS["t0_frozen"]
    board = config.board_obj()
    cursor = {"i": 0}

    def reply(payload) -> str:
        last = payload["input"][-1]["content"]
        if "guess" in last.lower() and "final" in last.lower():
            pairs = {addr: sym for addr, sym in zip(board.addresses, board.symbols)}
            return json.dumps({"guess": pairs})
        chord = [board.addresses[(cursor["i"] + k) % board.n] for k in range(config.chord_size)]
        cursor["i"] += config.chord_size
        return json.dumps({"presses": chord})

    transport = ScriptedTransport(reply_fn=reply)
    adapter = _adapter(transport)
    result = run_llm_episode(adapter, config, seed=0, transcript_dir=tmp_path)
    assert result["api_calls"] == len(transport.payloads) > 1
    saved = (tmp_path / "seed0000.json").read_text()
    assert "ENC-SECRET" not in saved
    turns = [m for m in json.loads(saved)["transcript"] if m["role"] == "assistant"]
    assert all(t["harness"]["name"] == HARNESS for t in turns)
    # Native reasoning accumulates: the last request replays every earlier turn's item.
    last_input = transport.payloads[-1]["input"]
    assert sum(1 for item in last_input if item.get("type") == "reasoning") == len(transport.payloads) - 1


def test_same_tokens_price_at_the_list_in_effect_that_day() -> None:
    """July's list for the July campaign; September's for September runs."""
    costs = {}
    for day in ("2026-07-20", "2026-09-25"):
        adapter = _adapter(ScriptedTransport(replies=["A1"]), price_date=day)
        adapter.call_with_meta([Message("system", "rules"), Message("user", "obs")])
        costs[day] = adapter.cost_usd()
    # 100 written + 40 output: July $1.25/$6.00 vs September $0.25/$1.20
    assert costs["2026-07-20"] == pytest.approx((100 * 1.25 + 40 * 6.0) / 1e6)
    assert costs["2026-09-25"] == pytest.approx((100 * 0.25 + 40 * 1.20) / 1e6)


def test_a_date_between_disagreeing_snapshots_makes_the_run_unpriced() -> None:
    adapter = _adapter(ScriptedTransport(replies=["A1"]), price_date="2026-07-31")
    adapter.call_with_meta([Message("system", "rules"), Message("user", "obs")])
    assert adapter.cost_usd() is None


def test_incomplete_turn_is_an_empty_billed_reply_and_its_input_is_resent() -> None:
    """A turn cut off at the output limit scores like the visible harness's
    empty reply: the runner retries or forfeits, the tokens are billed, the
    attempt is not replayed, and the unanswered observation goes out again."""

    class CutOffSecond(ScriptedTransport):
        def request(self, method, url, *, payload, idempotency_key, timeout_seconds):
            if len(self.payloads) == 1:
                self.payloads.append(json.loads(json.dumps(payload)))
                return {
                    "id": "resp_cut", "status": "incomplete", "model": "gpt-5.6-luna-2026-05-01",
                    "service_tier": "flex", "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [{"type": "reasoning", "id": "rs_cut", "encrypted_content": "ENC-CUT",
                                "summary": []}],
                    "usage": {"input_tokens": 200, "output_tokens": 1000,
                              "output_tokens_details": {"reasoning_tokens": 1000},
                              "total_tokens": 1200},
                }
            return super().request(method, url, payload=payload,
                                   idempotency_key=idempotency_key, timeout_seconds=timeout_seconds)

    transport = CutOffSecond(replies=["A1", "unused", "A3"])
    adapter = _adapter(transport)
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    t1, _ = adapter.call_with_meta(msgs)
    msgs += [Message("assistant", t1), Message("user", "obs 2")]
    t2, m2 = adapter.call_with_meta(msgs)
    assert t2 == ""
    assert m2["harness"]["incomplete"] == "max_output_tokens" and m2["stop_reason"] == "max_output_tokens"
    # September luna: 200 uncached input at $0.20, 1000 output at $1.20.
    assert m2["cost_usd_list"] == pytest.approx((200 * 0.20 + 1000 * 1.20) / 1e6)
    assert adapter.usage_totals["output_tokens"] == 40 + 1000

    msgs += [Message("assistant", t2), Message("user", "INVALID ACTION: fix it")]
    t3, _ = adapter.call_with_meta(msgs)
    assert t3 == "A3"
    replay = transport.payloads[-1]["input"]
    assert [item.get("type") or item.get("role") for item in replay] == [
        "user", "reasoning", "message", "user", "user"]
    assert [item.get("content") for item in replay[-2:]] == ["obs 2", "INVALID ACTION: fix it"]
    assert "ENC-CUT" not in json.dumps(replay)
