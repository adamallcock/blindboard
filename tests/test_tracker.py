"""Tracker soundness: whatever the tracker deduces must match the true
layout, at every step, across modes and seeds. Also: the reference agent
must actually solve the easy tiers."""

from __future__ import annotations

import pytest

from blindboard.agents import AssumedOrderAgent, ReferenceAgent
from blindboard.env import Episode
from blindboard.runner import run_scripted_episode
from blindboard.tiers import TIERS

SOUND_MODES = ["t0_frozen", "t1_bookkeeping", "t2_announced", "t3_observer", "t8_storm"]


@pytest.mark.parametrize("tier_name", SOUND_MODES)
@pytest.mark.parametrize("seed", range(8))
def test_tracker_soundness_stepwise(tier_name: str, seed: int) -> None:
    config = TIERS[tier_name]
    ep = Episode(config, seed)
    agent = ReferenceAgent(config, seed, samples=25)
    while not ep.done:
        obs = ep.observation()
        action = agent.act(obs)
        # After ingesting, every deduced letter must be true RIGHT NOW.
        for pos, letter in agent.tracker.deduced().items():
            assert ep.layout[pos] == letter, (
                f"{tier_name} seed {seed} turn {obs.turn}: tracker deduced "
                f"{letter} at {ep.board.addresses[pos]} but truth is {ep.layout[pos]}"
            )
        # Truth is always among the candidates (the soundness invariant).
        for pos in range(ep.board.n):
            assert ep.layout[pos] in agent.tracker.cand[pos]
        assert ep.validate(action) is None
        ep.apply(action)
    assert agent.tracker_resets == 0


def test_reference_solves_frozen_board() -> None:
    config = TIERS["t0_frozen"]
    solved = 0
    for seed in range(10):
        agent = ReferenceAgent(config, seed)
        result = run_scripted_episode(agent, config, seed)
        solved += result["board_accuracy"] == 1.0
        assert result["lock_wrong"] == 0, "reference locks must never be wrong"
    assert solved >= 9, f"reference should solve nearly all frozen boards, got {solved}/10"


def test_reference_beats_baselines_on_headline_tier() -> None:
    config = TIERS["t2_announced"]
    ref_acc, naive_acc = [], []
    for seed in range(10):
        ref = run_scripted_episode(ReferenceAgent(config, seed, samples=25), config, seed)
        naive = run_scripted_episode(AssumedOrderAgent(config, seed), config, seed)
        ref_acc.append(ref["board_accuracy"])
        naive_acc.append(naive["board_accuracy"])
    assert sum(ref_acc) > sum(naive_acc), (sum(ref_acc), sum(naive_acc))


def test_announced_full_is_lossless_bookkeeping() -> None:
    """In announced_full mode, once something is deduced it stays deduced
    (knowledge is only relabeled, never destroyed)."""
    config = TIERS["t1_bookkeeping"]
    for seed in range(5):
        ep = Episode(config, seed)
        agent = ReferenceAgent(config, seed, samples=25)
        deduced_counts = []
        while not ep.done:
            action = agent.act(ep.observation())
            deduced_counts.append(len(agent.tracker.deduced()))
            ep.apply(action)
        assert all(
            b >= a for a, b in zip(deduced_counts, deduced_counts[1:])
        ), f"seed {seed}: deduction count regressed in lossless mode: {deduced_counts}"
