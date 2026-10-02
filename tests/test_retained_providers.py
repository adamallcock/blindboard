"""Retained-reasoning bridge wiring for the non-OpenAI native sessions.

The kit tests cover each session's mechanics; these check what the bridge
adds: model-name routing, the provider's replay reaching the second
request, dated standard-tier pricing, and refusal before any billed call
for models without a verified retention verdict. No network."""

from __future__ import annotations

import copy
import json

import pytest

import blindboard.openrouter_native as on
from blindboard.client import Message
from blindboard.native_session import RetentionUnsupportedError
from blindboard.openrouter_native import INERT_TOOL
from blindboard.retained_adapter import RetainedReasoningAdapter

SIGNATURE = "EqQBCkYIBRgCKkB" + "s1" * 200
ENCRYPTED = "eyJlbmMiOiJ4In0=" * 8


class AnthropicFake:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post(self, url, payload, *, timeout_seconds):
        self.payloads.append(copy.deepcopy(dict(payload)))
        n = len(self.payloads)
        writes = 300 if n == 1 else 20
        reads = 0 if n == 1 else 300
        return {
            "id": f"msg_{n}", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": [
                {"type": "thinking", "thinking": f"summary {n}", "signature": SIGNATURE + str(n)},
                {"type": "text", "text": f"A{n}"},
            ],
            "stop_reason": "end_turn", "stop_sequence": None, "stop_details": None,
            "usage": {
                "input_tokens": 10, "cache_creation_input_tokens": writes,
                "cache_read_input_tokens": reads,
                "cache_creation": {"ephemeral_5m_input_tokens": writes, "ephemeral_1h_input_tokens": 0},
                "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 80},
                "service_tier": "standard", "inference_geo": "global",
            },
            "container": None,
        }


def _two_turns(adapter):
    msgs = [Message("system", "rules"), Message("user", "obs 1")]
    t1, m1 = adapter.call_with_meta(msgs)
    t2, m2 = adapter.call_with_meta(msgs + [Message("assistant", t1), Message("user", "obs 2")])
    return (t1, m1), (t2, m2)


def test_claude_replays_signed_thinking_and_prices_cache_classes() -> None:
    fake = AnthropicFake()
    adapter = RetainedReasoningAdapter(
        model="anthropic/claude-sonnet-5", transport=fake, api_key="k", price_date="2026-09-25"
    )
    (t1, m1), (t2, m2) = _two_turns(adapter)
    assert (t1, t2) == ("A1", "A2")
    first, second = fake.payloads
    assert first["model"] == second["model"] == "claude-sonnet-5"
    replayed = second["messages"][1]
    assert replayed["role"] == "assistant"
    assert replayed["content"][0] == {"type": "thinking", "thinking": "summary 1", "signature": SIGNATURE + "1"}
    # Sonnet 5 at $2/$10: 5-minute writes $2.50, reads $0.20.
    call1 = (10 * 2.0 + 300 * 2.50 + 100 * 10.0) / 1e6
    call2 = (10 * 2.0 + 300 * 0.20 + 20 * 2.50 + 100 * 10.0) / 1e6
    assert m1["cost_usd_list"] == pytest.approx(call1)
    assert adapter.cost_usd() == pytest.approx(call1 + call2)
    assert SIGNATURE not in json.dumps(m1) + json.dumps(m2)


def test_claude_without_verified_retention_is_refused_before_any_call() -> None:
    fake = AnthropicFake()
    adapter = RetainedReasoningAdapter(model="claude-haiku-4-5", transport=fake, api_key="k")
    with pytest.raises(RetentionUnsupportedError):
        adapter.call_with_meta([Message("system", "rules"), Message("user", "obs 1")])
    assert fake.payloads == []


INDEX = {
    "deepseek/deepseek-v4.1-flash": {
        "id": "deepseek/deepseek-v4.1-flash",
        "supported_parameters": ["reasoning", "reasoning_effort", "tools", "tool_choice"],
        "reasoning": {"supported_efforts": ["max", "high", "low"], "default_effort": "high"},
    },
    "qwen/qwen3.6-27b": {
        "id": "qwen/qwen3.6-27b",
        "supported_parameters": ["reasoning", "reasoning_effort"],
    },
}


class OpenRouterFake:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def __call__(self, payload, *, timeout_seconds):
        self.payloads.append(copy.deepcopy(dict(payload)))
        n = len(self.payloads)
        details = [{"type": "reasoning.encrypted", "data": ENCRYPTED + str(n), "id": f"enc-{n}",
                    "format": "unknown", "index": 0}]
        prompt = 200 if n == 1 else 200 + 60 + 40
        return {
            "id": f"gen-{n}", "provider": "DeepInfra", "model": "deepseek/deepseek-v4.1-flash",
            "object": "chat.completion", "service_tier": "default",
            "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "stop",
                         "message": {"role": "assistant", "content": f"A{n}", "refusal": None,
                                     "reasoning": f"thought {n}", "reasoning_details": details}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 60, "total_tokens": prompt + 60,
                      "is_byok": False, "completion_tokens_details": {"reasoning_tokens": 30},
                      "cost": 0.0001 * n},
        }


@pytest.fixture
def pinned_index(monkeypatch):
    monkeypatch.setattr(on, "_MODELS_INDEX_CACHE", copy.deepcopy(INDEX))


def test_deepseek_carries_the_inert_tool_and_replays_reasoning_details(pinned_index) -> None:
    fake = OpenRouterFake()
    adapter = RetainedReasoningAdapter(
        model="openrouter/deepseek/deepseek-v4.1-flash", effort="high", transport=fake, api_key="k"
    )
    (_, m1), (_, m2) = _two_turns(adapter)
    first, second = fake.payloads
    assert first["model"] == "deepseek/deepseek-v4.1-flash"
    assert first["tools"] == [INERT_TOOL] and first["tool_choice"] == "auto"
    assistant = [m for m in second["messages"] if m["role"] == "assistant"]
    assert assistant[0]["reasoning_details"][0]["data"] == ENCRYPTED + "1"
    # Cost is OpenRouter's own usage.cost: the bill for the host that served it.
    assert m1["cost_usd_list"] == pytest.approx(0.0001)
    assert adapter.cost_usd() == pytest.approx(0.0003)
    retention = m2["harness"]["retention"]
    assert retention["request_shape"] == "with_tools" and retention["replayed_reasoning_items"] == 1
    assert m2["harness"]["tools_sent"] == ["noop"]
    assert ENCRYPTED not in json.dumps(m1) + json.dumps(m2)


def test_openrouter_model_without_a_verdict_is_refused_before_any_call(pinned_index) -> None:
    fake = OpenRouterFake()
    adapter = RetainedReasoningAdapter(model="openrouter/qwen/qwen3.6-27b", effort="high",
                                       transport=fake, api_key="k")
    with pytest.raises(RetentionUnsupportedError):
        adapter.call_with_meta([Message("system", "rules"), Message("user", "obs 1")])
    assert fake.payloads == []


class GeminiFake:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.urls: list[str] = []

    def post(self, url, payload, *, timeout_seconds):
        self.urls.append(url)
        self.payloads.append(copy.deepcopy(dict(payload)))
        n = len(self.payloads)
        prompt = 120 if n == 1 else 120 + 308 + 30
        return {
            "candidates": [{"index": 0, "finishReason": "STOP", "content": {"role": "model", "parts": [
                {"text": f"thinking {n}", "thought": True},
                {"text": f"A{n}", "thoughtSignature": SIGNATURE + str(n)},
            ]}}],
            "usageMetadata": {"promptTokenCount": prompt, "candidatesTokenCount": 8,
                              "thoughtsTokenCount": 300, "totalTokenCount": prompt + 308,
                              "serviceTier": "flex"},
            "modelVersion": "gemini-3.8-flash",
        }


def test_gemini_replays_every_part_and_prices_flex_at_standard() -> None:
    fake = GeminiFake()
    adapter = RetainedReasoningAdapter(model="gemini/gemini-3.8-flash", effort="medium",
                                       transport=fake, api_key="k", price_date="2026-09-25")
    (t1, m1), (_, m2) = _two_turns(adapter)
    assert t1 == "A1"
    assert "/models/gemini-3.8-flash:generateContent" in fake.urls[0]
    assert fake.payloads[0]["serviceTier"] == "flex"
    model_turn = fake.payloads[1]["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"] == [{"text": "thinking 1", "thought": True},
                                   {"text": "A1", "thoughtSignature": SIGNATURE + "1"}]
    # Standard list although the call ran on flex: output includes thinking.
    assert m1["cost_usd_list"] == pytest.approx((120 * 0.75 + 308 * 3.75) / 1e6)
    assert SIGNATURE not in json.dumps(m1) + json.dumps(m2)


def test_compaction_is_openai_only() -> None:
    adapter = RetainedReasoningAdapter(model="claude-sonnet-5", compact_threshold=100_000,
                                       transport=AnthropicFake(), api_key="k")
    with pytest.raises(ValueError, match="OpenAI-only"):
        adapter.call_with_meta([Message("system", "rules"), Message("user", "obs 1")])
