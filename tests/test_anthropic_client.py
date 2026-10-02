"""Vendored Anthropic adapter: dispatch, pricing wiring, body shapes.

The adapter's internals are tested at the benchkit source; this guards the
consumer-side rendering (import map, registry entries, runner interface).
"""

from __future__ import annotations

import pytest

from blindboard.anthropic_client import (
    AnthropicAdapter,
    build_anthropic_body,
    make_adapter,
    normalize_anthropic_response,
)
from blindboard.client import APIAdapter, Message
from blindboard.pricing import PRICES


def test_registry_has_direct_api_claude_entries() -> None:
    assert PRICES["anthropic/claude-haiku-4-5"] == (1.0, 5.0)
    # $2/$10 on every date: the scheduled $3/$15 was cancelled and never charged.
    assert PRICES["anthropic/claude-sonnet-5"] == (2.0, 10.0)
    assert PRICES["anthropic/claude-opus-4-8"] == (5.0, 25.0)


def test_dispatch_and_cost_interface(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    a = make_adapter(model="claude-haiku-4-5", effort="medium", service_tier="flex")
    assert isinstance(a, AnthropicAdapter)
    a.usage_totals = {"input_tokens": 1_000_000, "output_tokens": 200_000}
    assert a.cost_usd() == pytest.approx(1.0 + 1.0)
    assert isinstance(make_adapter(model="gpt-5.6-luna"), APIAdapter)


def test_multiturn_body_matches_runner_message_shape() -> None:
    convo = [Message("system", "rules"), Message("user", "t1"),
             Message("assistant", "{}"), Message("user", "t2")]
    body = build_anthropic_body(model="claude-opus-4-8", messages=convo, effort="high")
    assert body["system"] == "rules"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["thinking"]["display"] == "summarized"


def test_normalize_meta_shape_matches_call_with_meta_contract() -> None:
    text, meta = normalize_anthropic_response(
        {
            "content": [{"type": "thinking", "thinking": "s"}, {"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    assert text == "ok"
    assert set(meta) >= {"reasoning", "usage", "stop_reason", "provenance"}
    assert meta["usage"]["total_tokens"] == 15


def test_provenance_names_the_snapshot_that_answered() -> None:
    """An alias request resolves to a dated build; the transcript records
    which one, so a later dispute can name the exact implementation."""
    _, meta = normalize_anthropic_response(
        {
            "id": "msg_01ABC",
            "model": "claude-opus-4-8-20260401",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    assert meta["provenance"] == {
        "response_id": "msg_01ABC",
        "routed_model": "claude-opus-4-8-20260401",
        "finish_reason": "end_turn",
    }


def test_openai_responses_provenance_records_the_tier_actually_applied() -> None:
    """Flex is requested; only the response says what was granted."""
    from blindboard.client import extract_provenance

    assert extract_provenance(
        {
            "id": "resp_1",
            "model": "gpt-5.6-luna-2026-05-01",
            "status": "completed",
            "service_tier": "default",
        }
    ) == {
        "response_id": "resp_1",
        "routed_model": "gpt-5.6-luna-2026-05-01",
        "finish_reason": "completed",
        "service_tier": "default",
    }


def test_episode_result_rolls_up_provenance_for_the_run(tmp_path) -> None:
    """The runner stores per-call meta wholesale (so provenance lands in
    the transcript) AND summarizes distinct hosts/builds on the result."""
    import json

    from blindboard.env import EnvConfig  # noqa: F401  (tiers builds these)
    from blindboard.llm_runner import run_llm_episode
    from blindboard.tiers import TIERS

    config = TIERS["t0_frozen"]

    class ProvenanceAdapter:
        """Legal round-robin bot that reports two different routed hosts."""

        supports_lossless_multi_turn = True

        def __init__(self) -> None:
            self.board = config.board_obj()
            self.cursor = 0
            self.calls = 0
            self.usage_totals: dict[str, int] = {}
            self.model = "openrouter/qwen/qwen3.6-27b"
            self.effort = "none"

        def call_with_meta(self, messages) -> tuple[str, dict]:
            self.calls += 1
            last = messages[-1].text
            if "final guess" in last or "GUESS" in last.upper():
                letters = list(self.board.symbols)
                pairs = ", ".join(
                    f'"{a}": "{letters[i]}"' for i, a in enumerate(self.board.addresses)
                )
                reply = f'{{"guess": {{{pairs}}}}}'
            else:
                addrs = [
                    self.board.addresses[(self.cursor + i) % self.board.n]
                    for i in range(config.chord_size)
                ]
                self.cursor += config.chord_size
                reply = '{"presses": ' + str(addrs).replace("'", '"') + "}"
            return reply, {
                "reasoning": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "provenance": {
                    "response_id": f"gen-{self.calls}",
                    "provider": "Fireworks" if self.calls % 2 else "Together",
                    "routed_model": "qwen/qwen3.6-27b",
                },
            }

        def __call__(self, messages) -> str:
            return self.call_with_meta(messages)[0]

    result = run_llm_episode(ProvenanceAdapter(), config, seed=0, transcript_dir=tmp_path)
    assert result["provenance"] == {
        "provider": ["Fireworks", "Together"],  # two hosts served this episode
        "routed_model": ["qwen/qwen3.6-27b"],
    }
    saved = json.loads((tmp_path / "seed0000.json").read_text())
    assert saved["result"]["provenance"] == result["provenance"]
    # Per-call response IDs stay addressable in the transcript itself.
    assistant = [m for m in saved["transcript"] if m["role"] == "assistant"]
    assert assistant and all(m["provenance"]["response_id"] for m in assistant)
