"""Planner (solver v3) invariants: legality, determinism, lock precision,
frozen-board mastery, stepwise soundness of its internal beliefs, and exact
reference fallback (certificate preservation)."""

from __future__ import annotations

import pytest

from blindboard.agents import AGENTS, ReferenceAgent
from blindboard.env import Episode
from blindboard.planner import PlannerAgent, loss_floor, select_policy
from blindboard.runner import run_scripted_episode
from blindboard.tiers import TIERS

# Tiers where the planner runs its own policy (observer/storm/perpetual)
# plus one fallback tier, all with recoverable rotation info.
PLANNED_TIERS = ["t3_observer", "t6_observer26", "t8_storm", "t8_storm_hard", "t7_perpetual"]


def test_planner_registered() -> None:
    assert AGENTS["planner"] is PlannerAgent


@pytest.mark.parametrize("tier_name", PLANNED_TIERS)
@pytest.mark.parametrize("seed", range(6))
def test_planner_stepwise_soundness_and_legality(tier_name: str, seed: int) -> None:
    """Mirror of test_tracker_soundness_stepwise, driven by the planner.

    The planner's internal beliefs are its reference engine's tracker (its
    stop rule and resolver chords are pure functions of that tracker), so
    stepwise soundness means: everything deduced matches the true layout,
    and the truth stays inside every candidate set, at every turn."""
    config = TIERS[tier_name]
    ep = Episode(config, seed)
    agent = PlannerAgent(config, seed, samples=25)
    while not ep.done:
        obs = ep.observation()
        action = agent.act(obs)
        for pos, letter in agent.tracker.deduced().items():
            assert ep.layout[pos] == letter, (
                f"{tier_name} seed {seed} turn {obs.turn}: planner deduced "
                f"{letter} at {ep.board.addresses[pos]} but truth is {ep.layout[pos]}"
            )
        for pos in range(ep.board.n):
            assert ep.layout[pos] in agent.tracker.cand[pos]
        if agent.belief is not None and agent.belief.active:
            # The weighted posterior's soundness invariant: the true current
            # layout is always inside its support, with positive weight.
            assert agent.belief.support.get(tuple(ep.layout), 0.0) > 0.0, (
                f"{tier_name} seed {seed} turn {obs.turn}: truth fell out of "
                "the weighted belief support"
            )
        assert ep.validate(action) is None, "planner must be legal at every turn"
        ep.apply(action)
    assert agent.tracker_resets == 0
    assert ep.result is not None and ep.result["lock_wrong"] == 0, (
        "planner locks must keep perfect precision"
    )


@pytest.mark.parametrize("tier_name", ["t3_observer", "t8_storm_hard", "t7_perpetual"])
def test_planner_determinism(tier_name: str) -> None:
    config = TIERS[tier_name]
    a = run_scripted_episode(PlannerAgent(config, 3), config, 3)
    b = run_scripted_episode(PlannerAgent(config, 3), config, 3)
    assert a["guess"] == b["guess"]
    assert a["board_accuracy"] == b["board_accuracy"]
    assert [r["presses"] for r in a["records"]] == [r["presses"] for r in b["records"]]
    assert a["locks"] == b["locks"]


def test_planner_solves_frozen_board_10_of_10() -> None:
    config = TIERS["t0_frozen"]
    for seed in range(10):
        result = run_scripted_episode(PlannerAgent(config, seed), config, seed)
        assert result["board_accuracy"] == 1.0, f"seed {seed} not solved"
        assert result["lock_wrong"] == 0


@pytest.mark.parametrize("tier_name", PLANNED_TIERS)
def test_planner_lock_precision(tier_name: str) -> None:
    config = TIERS[tier_name]
    for seed in range(4):
        result = run_scripted_episode(PlannerAgent(config, seed), config, seed)
        assert result["lock_wrong"] == 0


@pytest.mark.parametrize(
    "tier_name", ["t2_announced", "t1_bookkeeping", "t9_dual", "t10_relabel"]
)
def test_planner_fallback_is_byte_identical_to_reference(tier_name: str) -> None:
    """Fallback configs must reproduce reference v2 play exactly, so no
    frozen certificate can regress. (t10_relabel: big-board perpetual and
    relabel tiers are deliberate fallbacks — reference is at ceiling.)"""
    config = TIERS[tier_name]
    for seed in (0, 11):
        rp = run_scripted_episode(PlannerAgent(config, seed), config, seed)
        rr = run_scripted_episode(ReferenceAgent(config, seed), config, seed)
        assert [r["presses"] for r in rp["records"]] == [
            r["presses"] for r in rr["records"]
        ]
        assert rp["guess"] == rr["guess"]
        assert rp["board_accuracy"] == rr["board_accuracy"]
        assert rp["locks"] == rr["locks"]


def test_planner_policy_selection() -> None:
    assert select_policy(TIERS["t3_observer"]) == "observer"
    assert select_policy(TIERS["t6_observer26"]) == "observer"
    assert select_policy(TIERS["t8_storm"]) == "storm"
    assert select_policy(TIERS["t8_storm_hard"]) == "storm"
    assert select_policy(TIERS["t7_perpetual"]) == "perpetual"
    # At-ceiling / solved-at-1.000 configs stay on exact reference play.
    for name in (
        "t0_frozen", "t1_bookkeeping", "t2_announced", "t4_unannounced",
        "t5_qwerty", "t6_gauntlet", "t7_blitz", "t9_numrow", "t9_macbook",
        "t9_dual", "t10_relabel", "t11_maelstrom",
    ):
        assert select_policy(TIERS[name]) == "fallback", name


def test_planner_stop_rule_scores_at_the_blob_floor() -> None:
    """When the planner stops early in observer mode, the remaining damage
    must be exactly the freshest-blob floor: everything outside the last
    rotation set is guessed correctly."""
    config = TIERS["t3_observer"]
    stopped = 0
    for seed in range(12):
        result = run_scripted_episode(PlannerAgent(config, seed), config, seed)
        if result["guess_turn"] <= config.horizon:
            stopped += 1
            records = [r for r in result["records"] if r["rotation"]["positions"]]
            last = set(records[-1]["rotation"]["positions"])
            board = config.board_obj()
            wrong = [
                addr
                for addr, letter in result["final_layout"].items()
                if result["guess"].get(addr) != letter
            ]
            assert all(board.index_of(a) in last for a in wrong), (
                f"seed {seed}: early guess lost keys outside the final blob"
            )
            assert len(wrong) <= len(last), "cannot lose more than the blob"
    assert stopped > 0, "stop rule should fire on some observer episodes"


def test_loss_floor_values() -> None:
    assert loss_floor(2) == 0.0
    assert loss_floor(3) == pytest.approx(1.5)
    assert loss_floor(4) == pytest.approx(8 / 3)
    assert loss_floor(5) == pytest.approx(15 / 4)
