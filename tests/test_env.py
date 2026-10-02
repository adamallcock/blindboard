"""Environment invariants: determinism, rotation semantics, rule
enforcement, scoring."""

from __future__ import annotations

import random

import pytest

from blindboard.agents import RandomAgent
from blindboard.boards import BOARDS
from blindboard.env import Action, EnvConfig, Episode
from blindboard.runner import run_scripted_episode
from blindboard.tiers import TIERS


def drive(config: EnvConfig, seed: int, agent_seed: int = 0) -> Episode:
    ep = Episode(config, seed)
    agent = RandomAgent(config, agent_seed)
    while not ep.done:
        action = agent.act(ep.observation())
        assert ep.validate(action) is None
        ep.apply(action)
    return ep


def test_determinism() -> None:
    config = TIERS["t2_announced"]
    a = drive(config, seed=7)
    b = drive(config, seed=7)
    assert a.initial_layout == b.initial_layout
    assert a.records == b.records
    assert a.result == b.result


def test_layout_is_permutation() -> None:
    for tier in TIERS.values():
        ep = Episode(tier, seed=3)
        assert sorted(ep.layout) == sorted(ep.board.symbols)


def test_rotation_is_derangement_and_preserves_letters() -> None:
    config = TIERS["t2_announced"]
    for seed in range(30):
        ep = Episode(config, seed)
        agent = RandomAgent(config, seed)
        while not ep.done:
            before = list(ep.layout)
            action = agent.act(ep.observation())
            ep.apply(action)
            if ep.done:
                break
            rotation = ep.records[-1]["rotation"]
            positions = rotation["positions"]
            after = ep.layout
            for p in range(ep.board.n):
                if p in positions:
                    assert after[p] != before[p], "rotated key kept its letter"
                else:
                    assert after[p] == before[p], "unrotated key moved"
            assert sorted(after[p] for p in positions) == sorted(
                before[p] for p in positions
            ), "rotation changed the letter-set"


def test_observer_rotates_exactly_pressed_keys() -> None:
    config = TIERS["t3_observer"]
    ep = Episode(config, seed=1)
    agent = RandomAgent(config, seed=1)
    action = agent.act(ep.observation())
    pressed = sorted(ep.board.index_of(a) for a in action.presses)
    ep.apply(action)
    assert ep.records[-1]["rotation"]["positions"] == pressed


def test_rotation_schedule_decays_to_zero() -> None:
    config = TIERS["t2_announced"]
    ep = drive(config, seed=5)
    for record in ep.records:
        turn = record["turn"]
        expected_m = config.rotation_size(turn)
        got = len(record["rotation"]["positions"])
        assert got == (expected_m if expected_m >= 2 else 0)


def test_validation_rules() -> None:
    config = TIERS["t0_frozen"]
    ep = Episode(config, seed=0)
    addrs = ep.board.addresses
    assert ep.validate(Action(presses=[addrs[0], addrs[1]])) is not None  # too few
    assert ep.validate(Action(presses=[addrs[0], addrs[0], addrs[1]])) is not None  # dup
    assert ep.validate(Action(presses=[addrs[0], addrs[1], "L-thumb-top"])) is not None
    assert ep.validate(Action(presses=list(addrs[:3]))) is None
    assert ep.validate(Action(presses=list(addrs[:3]), locks={addrs[5]: "Z"})) is not None
    assert ep.validate(Action(presses=list(addrs[:3]), locks={addrs[5]: "A"})) is None
    partial_guess = {a: "A" for a in addrs[:5]}
    assert ep.validate(Action(guess=partial_guess)) is not None


def test_lock_evaluated_at_lock_time_and_immutable() -> None:
    config = TIERS["t0_frozen"]
    ep = Episode(config, seed=0)
    addr = ep.board.addresses[0]
    truth = ep.layout[0]
    wrong = next(x for x in ep.board.symbols if x != truth)
    ep.apply(Action(presses=list(ep.board.addresses[:3]), locks={addr: truth}))
    # Second lock on the same key is ignored, not an error.
    ep.apply(Action(presses=list(ep.board.addresses[3:6]), locks={addr: wrong}))
    assert ep.locks[0] == truth
    assert ep.lock_eval[0] is True


def test_guess_scoring_and_forced_phase() -> None:
    config = TIERS["t0_frozen"]
    ep = Episode(config, seed=2)
    agent = RandomAgent(config, seed=2)
    for _ in range(config.horizon):
        ep.apply(agent.act(ep.observation()))
    assert ep.phase == "guess"
    err = ep.validate(Action(presses=list(ep.board.addresses[:3])))
    assert err is not None  # presses rejected in guess phase
    perfect = dict(zip(ep.board.addresses, ep.layout))
    assert ep.validate(Action(guess=perfect)) is None
    ep.apply(Action(guess=perfect))
    assert ep.result is not None
    assert ep.result["correct"] == ep.board.n
    assert ep.result["forced_guess"] is True


def test_near_qwerty_swap_count() -> None:
    board = BOARDS["qwerty26"]
    rng = random.Random(0)
    layout = board.near_qwerty_layout(rng, swaps=6)
    diffs = sum(1 for a, b in zip(layout, board.qwerty_layout) if a != b)
    assert diffs == 12
    assert sorted(layout) == sorted(board.symbols)


def test_skip_turn_consumes_turn() -> None:
    config = TIERS["t0_frozen"]
    ep = Episode(config, seed=0)
    ep.skip_turn("test violation")
    assert ep.turn == 2
    assert ep.records[-1]["skipped"] == "test violation"
    assert len(ep.violations) == 1


@pytest.mark.parametrize("tier_name", list(TIERS))
def test_all_tiers_run_end_to_end(tier_name: str) -> None:
    config = TIERS[tier_name]
    agent = RandomAgent(config, 0)
    result = run_scripted_episode(agent, config, seed=0)
    assert 0 <= result["board_accuracy"] <= 1
    assert result["guess_turn"] == config.horizon + 1
