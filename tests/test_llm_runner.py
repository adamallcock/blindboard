"""Prove the violation accounting would catch rule-breaking if it happened
(the zero-violations pilot result is only meaningful if these paths work).

Uses a scripted fake adapter — no network."""

from __future__ import annotations

import json

import pytest

from blindboard.client import APIAdapter, LossyMultiTurnStateError
from blindboard.env import EnvConfig
from blindboard.llm_runner import run_llm_episode
from blindboard.tiers import TIERS


class FakeAdapter:
    """Plays scripted replies, then falls back to a legal round-robin bot."""

    supports_lossless_multi_turn = True

    def __init__(self, scripted: list[str], config: EnvConfig) -> None:
        self.scripted = list(scripted)
        self.board = config.board_obj()
        self.k = config.chord_size
        self.cursor = 0
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        self.model = "fake"
        self.effort = "none"

    def _legal_reply(self, last_user: str) -> str:
        if "final guess" in last_user or "GUESS" in last_user.upper():
            letters = list(self.board.symbols)
            pairs = ", ".join(
                f'"{a}": "{letters[i]}"' for i, a in enumerate(self.board.addresses)
            )
            return f'{{"guess": {{{pairs}}}}}'
        addrs = [
            self.board.addresses[(self.cursor + i) % self.board.n] for i in range(self.k)
        ]
        self.cursor += self.k
        return '{"presses": ' + str(addrs).replace("'", '"') + "}"

    def call_with_meta(self, messages) -> tuple[str, dict]:
        self.calls += 1
        reply = self.scripted.pop(0) if self.scripted else self._legal_reply(messages[-1].text)
        return reply, {"reasoning": ["fake trace"], "usage": {"input_tokens": 1, "output_tokens": 1}}

    def __call__(self, messages) -> str:
        return self.call_with_meta(messages)[0]


CFG = TIERS["t0_frozen"]
A = CFG.board_obj().addresses


def test_four_presses_is_rejected_not_truncated() -> None:
    """The parser must never silently keep the first K presses."""
    four = json.dumps({"presses": [A[0], A[1], A[2], A[3]]})
    ok = json.dumps({"presses": [A[0], A[1], A[2]]})
    adapter = FakeAdapter([four, ok], CFG)
    result = run_llm_episode(adapter, CFG, seed=0)
    assert result["total_retries"] == 1  # the 4-press reply consumed the retry
    assert result["violations"] == []  # corrected on retry, no forfeit
    # First record's presses are the corrected 3, not a truncation of the 4.
    assert result["records"][0]["presses"] == [A[0], A[1], A[2]]


def test_double_illegal_forfeits_turn() -> None:
    bad = json.dumps({"presses": [A[0], A[1]]})  # too few, twice
    adapter = FakeAdapter([bad, bad], CFG)
    result = run_llm_episode(adapter, CFG, seed=0)
    assert result["total_retries"] == 1
    assert len(result["violations"]) == 1
    assert result["records"][0]["presses"] == []  # turn forfeited, no info
    assert result["records"][0].get("skipped")


def test_duplicate_press_via_case_alias_is_caught() -> None:
    """'l_ring_top' normalizes to 'L-ring-top': same key twice must be illegal."""
    dup = json.dumps({"presses": ["L-ring-top", "l_ring_top", A[0]]})
    ok = json.dumps({"presses": [A[0], A[1], A[2]]})
    adapter = FakeAdapter([dup, ok], CFG)
    result = run_llm_episode(adapter, CFG, seed=0)
    assert result["total_retries"] == 1


def test_unparseable_guess_phase_scores_zero() -> None:
    adapter = FakeAdapter([], CFG)
    # Legal presses all game, then garbage at guess phase (twice).
    adapter.scripted = []
    legal = FakeAdapter([], CFG)
    replies = [legal._legal_reply("press") for _ in range(CFG.horizon)]
    replies += ["I am lost", "still lost"]
    adapter = FakeAdapter(replies, CFG)
    result = run_llm_episode(adapter, CFG, seed=0)
    assert result["correct"] == 0
    assert result["meta"].get("forced_zero") is True


def test_reasoning_traces_saved(tmp_path) -> None:
    adapter = FakeAdapter([], CFG)
    run_llm_episode(adapter, CFG, seed=0, transcript_dir=tmp_path)

    saved = json.loads((tmp_path / "seed0000.json").read_text())
    assistant = [m for m in saved["transcript"] if m["role"] == "assistant"]
    assert assistant and all(m.get("reasoning") == ["fake trace"] for m in assistant)
    assert all("usage" in m for m in assistant)


def test_text_only_provider_is_rejected_before_first_billed_call() -> None:
    adapter = APIAdapter(model="gpt-5.6-sol", api_key="test-only")

    with pytest.raises(LossyMultiTurnStateError, match="full-transcript episode"):
        run_llm_episode(adapter, CFG, seed=0)

    assert adapter.calls == 0
    assert adapter.usage_totals == {}
