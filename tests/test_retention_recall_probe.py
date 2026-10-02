"""Retention recall probe: residue checks and native-session forks.

No network: scripted transports record every payload, so the tests can
check that a fork replays the parent's native state and that two forks
never share state or request keys."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import retention_recall_probe as probe  # noqa: E402
from blindboard.anthropic_native import AnthropicNativeSession  # noqa: E402
from blindboard.responses_native import OpenAINativeSession  # noqa: E402


def test_residues_parse_and_unknown_is_none() -> None:
    assert probe.parse_residue("23") == 23
    assert probe.parse_residue(" 76\n") == 76
    assert probe.parse_residue("UNKNOWN") is None
    assert probe.parse_residue("unknown.") is None
    assert probe.parse_residue("") is None


def test_joint_consistency_accepts_the_true_residues_only() -> None:
    n = 4729
    assert probe.jointly_consistent(n % 97, n % 89, n % 83)
    assert not probe.jointly_consistent(n % 97, n % 89, (n + 1) % 83)
    assert not probe.jointly_consistent(n % 97, None, n % 83)
    # Two residues always leave a candidate in range: the third is the check.
    assert all(probe.candidates(a, b) for a in range(97) for b in (0, 44, 88))


def test_leak_and_summary_number_detection() -> None:
    assert probe.leaked("N is 4729, mod 97 = 73")
    assert not probe.leaked("73")
    assert probe.summary_numbers(["I picked 4729 and 1234."], 4729 % 97) == [4729]


class ResponsesFake:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.keys: list[str] = []

    def request(self, method, url, *, payload, idempotency_key, timeout_seconds):
        self.payloads.append(json.loads(json.dumps(payload)))
        self.keys.append(idempotency_key)
        n = len(self.payloads)
        return {
            "id": f"resp_{n}", "status": "completed", "model": "gpt-5.6-luna",
            "service_tier": "flex", "reasoning": {"context": "all_turns"},
            "output": [
                {"type": "reasoning", "id": f"rs_{n}", "encrypted_content": f"ENC-{n}",
                 "summary": [{"type": "summary_text", "text": f"summary {n}"}]},
                {"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": f"A{n}", "annotations": []}]},
            ],
            "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 30},
                      "total_tokens": 140},
        }


def test_openai_forks_replay_parent_state_and_stay_independent() -> None:
    fake = ResponsesFake()
    root = OpenAINativeSession(model="gpt-5.6-luna", instructions=probe.SYSTEM,
                               api_key="k", transport=fake, pricer=lambda usage: 0.0)
    root.turn([probe.T1])
    a, b = probe.fork(root), probe.fork(root)
    a.turn([probe.Q89])
    b.turn([probe.INTERVENING])
    b.turn([probe.Q83])
    first, fork_a, fork_b1, fork_b2 = fake.payloads
    parent_items = first["input"] + [{"type": "reasoning"}, {"type": "message"}]
    for payload in (fork_a, fork_b1):
        replayed = payload["input"][: len(parent_items)]
        assert replayed[0] == first["input"][0]
        assert replayed[1]["type"] == "reasoning" and replayed[1]["encrypted_content"] == "ENC-1"
        assert replayed[2]["type"] == "message"
    assert fork_a["input"][-1]["content"] == probe.Q89
    assert fork_b1["input"][-1]["content"] == probe.INTERVENING
    # b's second turn carries b's own first turn, never a's.
    texts = json.dumps(fork_b2["input"])
    assert "ENC-3" in texts and "ENC-2" not in texts
    assert len(set(fake.keys)) == len(fake.keys)


class AnthropicFake:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post(self, url, payload, *, timeout_seconds):
        self.payloads.append(copy.deepcopy(dict(payload)))
        n = len(self.payloads)
        return {
            "id": f"msg_{n}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": f"summary {n}", "signature": f"SIG-{n}"},
                {"type": "text", "text": f"A{n}"},
            ],
            "stop_reason": "end_turn", "stop_sequence": None, "stop_details": None,
            "usage": {"input_tokens": 10, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 0,
                      "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
                      "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 80},
                      "service_tier": "standard", "inference_geo": "global"},
            "container": None,
        }


def test_anthropic_forks_replay_signed_thinking_and_stay_independent() -> None:
    fake = AnthropicFake()
    root = AnthropicNativeSession(model="claude-opus-5-5", instructions=probe.SYSTEM, effort="medium",
                                  api_key="k", transport=fake, pricer=lambda usage: 0.0)
    root.turn([probe.T1])
    a, b = probe.fork(root), probe.fork(root)
    a.turn([probe.Q89])
    b.turn([probe.INTERVENING])
    _, fork_a, fork_b = fake.payloads
    for payload in (fork_a, fork_b):
        assert payload["messages"][1]["content"][0]["signature"] == "SIG-1"
    assert "SIG-2" not in json.dumps(fork_b["messages"])
    assert root.turn_index == 1 and a.turn_index == 2 and b.turn_index == 2


sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_retained_providers as providers  # noqa: E402  (shared fakes)


def test_gemini_forks_replay_signed_parts_and_stay_independent() -> None:
    fake = providers.GeminiFake()
    model = "gemini/gemini-3.8-flash"
    root = probe.factory_for(model)(
        model=model, instructions=probe.SYSTEM, effort="medium", service_tier=probe.session_tier(model),
        compact_threshold=None, api_key="k", transport=fake, price_date="2026-09-25",
    )
    root.turn([probe.T1])
    a, b = probe.fork(root), probe.fork(root)
    a.turn([probe.Q89])
    b.turn([probe.INTERVENING])
    _, fork_a, fork_b = fake.payloads
    for payload in (fork_a, fork_b):
        assert payload["contents"][1]["parts"][1]["thoughtSignature"] == providers.SIGNATURE + "1"
    assert providers.SIGNATURE + "2" not in json.dumps(fork_b["contents"])
    assert root.turn_index == 1 and a.turn_index == 2 and b.turn_index == 2


def test_deepseek_forks_replay_reasoning_details_and_stay_independent(monkeypatch) -> None:
    monkeypatch.setattr(providers.on, "_MODELS_INDEX_CACHE", copy.deepcopy(providers.INDEX))
    fake = providers.OpenRouterFake()
    model = "openrouter/deepseek/deepseek-v4.1-flash"
    root = probe.factory_for(model)(
        model=model, instructions=probe.SYSTEM, effort="high", service_tier=probe.session_tier(model),
        compact_threshold=None, api_key="k", transport=fake, price_date=None,
    )
    root.turn([probe.T1])
    a, b = probe.fork(root), probe.fork(root)
    a.turn([probe.Q89])
    b.turn([probe.INTERVENING])
    _, fork_a, fork_b = fake.payloads
    for payload in (fork_a, fork_b):
        assistant = [m for m in payload["messages"] if m["role"] == "assistant"]
        assert assistant[0]["reasoning_details"][0]["data"] == providers.ENCRYPTED + "1"
        assert payload["tools"] == [providers.INERT_TOOL]
    assert providers.ENCRYPTED + "2" not in json.dumps(fork_b["messages"])


class _Turn:
    def __init__(self, text: str) -> None:
        self.text, self.usage, self.cost_usd = text, {}, 0.0
        self.reasoning_summaries = ("I pick 4729 and keep it.",)


class _Root:
    def turn(self, texts):
        return _Turn(str(4729 % 97))


class _BrokenFork:
    def turn(self, texts):
        raise RuntimeError("HTTP 500 from upstream")


class _Control:
    def call_with_meta(self, messages):
        return "UNKNOWN", {"usage": {}}


def test_a_failed_branch_records_its_exception_and_is_not_an_abstention(monkeypatch) -> None:
    monkeypatch.setattr(probe, "factory_for", lambda model: (lambda **kw: _Root()))
    monkeypatch.setattr(probe, "fork", lambda session: _BrokenFork())
    monkeypatch.setattr(probe, "make_adapter", lambda **kw: _Control())
    record = probe.run_trial("gpt-5.6-luna", "medium", "escape", 0, "2026-09-30")
    assert record["branch_errors"]["r1"] == "RuntimeError: HTTP 500 from upstream"
    assert "r2" in record["branch_errors"]
    verdict = record["verdict"]
    assert verdict["summary_n"] == [4729]
    assert not verdict["retained_complete"] and not verdict["retained_abstained"]
    assert not verdict["matches_summary_n"]


def test_openrouter_controls_carry_the_reported_bill() -> None:
    resp = {"choices": [{"message": {"content": "15"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 5, "cost": 0.00042}}
    _, meta = probe._normalize_with_cost(resp)
    assert meta["reported_cost_usd"] == 0.00042
    model = "openrouter/deepseek/deepseek-v4.1-flash"
    assert probe.control_cost(model, meta["usage"], "2026-09-30", meta["reported_cost_usd"]) == 0.00042
