"""The legacy visible-transcript condition is opt-in twice over.

A text-only adapter can replay only visible text, dropping the model's
earlier reasoning. The runner refuses it unless the caller asks for the
legacy condition AND built the adapter lossy on purpose; the episode is then
stamped so it can never pass for a retained-reasoning run. No network."""

from __future__ import annotations

import json

import pytest

from blindboard.client import APIAdapter, LossyMultiTurnStateError
from blindboard.llm_runner import LEGACY_VISIBLE_HARNESS, run_llm_episode
from blindboard.tiers import TIERS

CFG = TIERS["t0_frozen"]


class TextOnlyAdapter:
    """A legal round-robin bot with the text adapters' flags."""

    supports_lossless_multi_turn = False

    def __init__(self, *, allow_lossy_multi_turn: bool) -> None:
        self.allow_lossy_multi_turn = allow_lossy_multi_turn
        self.board = CFG.board_obj()
        self.cursor = 0
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        self.model = "fake-text"
        self.effort = "none"

    def call_with_meta(self, messages) -> tuple[str, dict]:
        self.calls += 1
        last = messages[-1].text.lower()
        if "guess" in last and "final" in last:
            reply = json.dumps({"guess": dict(zip(self.board.addresses, self.board.symbols))})
        else:
            k = CFG.chord_size
            reply = json.dumps(
                {"presses": [self.board.addresses[(self.cursor + i) % self.board.n] for i in range(k)]}
            )
            self.cursor += k
        return reply, {"reasoning": [], "usage": {"input_tokens": 1, "output_tokens": 1}}


class LosslessAdapter(TextOnlyAdapter):
    """Declares lossless replay and has no lossy opt-in at all."""

    supports_lossless_multi_turn = True

    def __init__(self) -> None:
        super().__init__(allow_lossy_multi_turn=False)
        del self.allow_lossy_multi_turn


def test_text_adapter_is_refused_by_default_even_when_built_lossy() -> None:
    adapter = TextOnlyAdapter(allow_lossy_multi_turn=True)
    with pytest.raises(LossyMultiTurnStateError, match="full-transcript episode"):
        run_llm_episode(adapter, CFG, seed=0)
    assert adapter.calls == 0


def test_legacy_flag_alone_does_not_unlock_a_strict_adapter() -> None:
    fake = TextOnlyAdapter(allow_lossy_multi_turn=False)
    real = APIAdapter(model="gpt-5.6-sol", api_key="test-only")
    for adapter in (fake, real):
        with pytest.raises(LossyMultiTurnStateError, match="allow_lossy_multi_turn=True"):
            run_llm_episode(adapter, CFG, seed=0, legacy_visible_transcript=True)
    assert fake.calls == 0 and real.calls == 0


def test_a_lossless_adapter_cannot_be_relabelled_legacy() -> None:
    adapter = LosslessAdapter()
    with pytest.raises(LossyMultiTurnStateError):
        run_llm_episode(adapter, CFG, seed=0, legacy_visible_transcript=True)
    assert adapter.calls == 0


def test_double_opt_in_runs_and_stamps_the_episode(tmp_path) -> None:
    adapter = TextOnlyAdapter(allow_lossy_multi_turn=True)
    result = run_llm_episode(
        adapter, CFG, seed=0, transcript_dir=tmp_path, legacy_visible_transcript=True
    )
    assert adapter.calls == result["api_calls"] > 1
    assert result["harness"] == LEGACY_VISIBLE_HARNESS == "visible-transcript (legacy opt-in)"
    saved = json.loads((tmp_path / "seed0000.json").read_text())
    assert saved["result"]["harness"] == LEGACY_VISIBLE_HARNESS


def test_default_runs_carry_no_legacy_stamp() -> None:
    result = run_llm_episode(LosslessAdapter(), CFG, seed=0)
    assert result.get("harness") != LEGACY_VISIBLE_HARNESS
