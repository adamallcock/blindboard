"""KS-1 regression: the rules text must tell the truth about the schedule.

Red-team finding 2026-07-20: bb-r1's announced-mode text promised rotations
'eventually stop' on six constant-schedule tiers where they never do. This
gate renders every registered tier's system prompt and checks the schedule
statement against the actual rotation_schedule."""

from __future__ import annotations

from blindboard.prompts import PROMPT_VERSION, build_system_prompt
from blindboard.tiers import TIERS


def _never_stops(config) -> bool:
    sizes = [config.rotation_size(t) for t in range(1, config.horizon + 1)]
    return bool(sizes) and sizes[-1] > 0


def test_prompt_version_is_current() -> None:
    assert PROMPT_VERSION == "bb-r2"


def test_announced_rules_match_actual_schedule() -> None:
    for name, config in TIERS.items():
        text = build_system_prompt(config)
        if config.rotation_mode not in ("announced", "announced_full"):
            assert "eventually stop" not in text, name
            continue
        if _never_stops(config):
            assert "never stop" in text, f"{name}: constant schedule must say so"
            assert "eventually stop" not in text, f"{name}: false decay promise"
        elif any(config.rotation_schedule):
            assert "eventually stop" in text, f"{name}: decaying schedule keeps bb-r1 text"
            assert "never stop" not in text, name


def test_storm_and_observer_texts_state_their_own_dynamics() -> None:
    text = build_system_prompt(TIERS["t8_storm"])
    assert "This never stops" in text
    text = build_system_prompt(TIERS["t3_observer"])
    assert "eventually stop" not in text
