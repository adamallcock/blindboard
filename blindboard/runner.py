"""Episode runners (scripted agents) and post-hoc diagnostics.

Diagnostics replay an episode's observation stream through the sound
tracker to compute what was DEDUCIBLE from the information the agent
actually received — separating information-acquisition failures from
deduction/memory failures:
- deducible_at_guess: keys uniquely determined by received info (lower
  bound; tracker is sound but incomplete)
- deduced_correct_in_guess: of those, how many the agent guessed right
  (the gap is pure reasoning/memory loss, not missing information)
- known_presses / zero_info_chords: presses spent on already-determined
  keys (forgetting or poor planning).
Only meaningful for modes with recoverable rotation info (none, announced,
announced_full, observer)."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

from blindboard.agents import ReferenceAgent, ScriptedAgent
from blindboard.dual import DualEpisode, make_episode
from blindboard.env import Action, EnvConfig, Episode, Observation
from blindboard.tracker import BeliefTracker, Contradiction

DIAGNOSABLE_MODES = ("none", "announced", "announced_full", "observer", "storm")


def _physical_obs(sub: Episode, obs: Observation) -> Observation:
    """Rebuild an observation in canonical-physical addressing from the
    records, so relabel-blind scripted agents stay sound. (Relabeling is
    pure translation load; scripted baselines and reference bypass it —
    the reference ceiling of a relabel tier equals its base tier.)"""
    board = sub.board
    if not sub.records:
        return obs
    r = sub.records[-1]
    rotation = r["rotation"]
    notice = None
    mapping = None
    if rotation["mode"] in ("announced", "observer", "storm") and rotation["positions"]:
        notice = sorted(board.addresses[p] for p in rotation["positions"])
    if rotation["mode"] == "announced_full" and rotation["positions"]:
        mapping = sorted(
            (board.addresses[int(src)], board.addresses[dst])
            for src, dst in rotation["permutation"].items()
        )
    return dc_replace(
        obs,
        last_presses=[board.addresses[p] for p in r["press_positions"]],
        last_results=list(r["results"]),
        rotation_notice=notice,
        rotation_mapping=mapping,
    )


def _translate_action(sub: Episode, action: Action, board) -> Action:
    """Physical-intent addresses -> current addressing."""

    def current(addr: str) -> str:
        return sub.addr_of_phys(board.index_of(addr))

    return Action(
        presses=[current(a) for a in action.presses],
        locks={current(a): v for a, v in action.locks.items()},
        guess=None if action.guess is None else {current(a): v for a, v in action.guess.items()},
    )


def run_scripted_episode(
    agent: ScriptedAgent,
    config: EnvConfig,
    seed: int,
    privileged_rotation: bool = False,
) -> dict[str, Any]:
    """Run one episode with a scripted agent. Scripted agents must be legal:
    validation errors raise (they indicate bugs, not gameplay)."""
    ep = make_episode(config, seed)
    board = ep.board
    agents: dict[str, ScriptedAgent] = {"A": agent}
    if config.dual:
        agents["B"] = type(agent)(config, seed + 999_331)
    while not ep.done:
        sub: Episode = ep.active if isinstance(ep, DualEpisode) else ep
        current_agent = agents[ep.active_label if isinstance(ep, DualEpisode) else "A"]
        obs = ep.observation()
        if config.relabel_every > 0:
            obs = _physical_obs(sub, obs)
        if isinstance(current_agent, ReferenceAgent):
            priv = None
            if privileged_rotation and sub.records:
                rotation = sub.records[-1]["rotation"]
                priv = [board.addresses[p] for p in rotation.get("positions", [])]
            action: Action = current_agent.act(obs, privileged_rotation=priv)
        else:
            action = current_agent.act(obs)
        if config.relabel_every > 0:
            action = _translate_action(sub, action, board)
        err = ep.validate(action)
        if err:
            raise RuntimeError(f"scripted agent produced illegal action: {err}")
        ep.apply(action)
    result = dict(ep.result or {})
    result["records"] = ep.records
    result.update(diagnose_any(ep))
    if isinstance(agent, ReferenceAgent):
        result["tracker_resets"] = agent.tracker_resets
    return result


def diagnose_any(ep: Episode | DualEpisode) -> dict[str, Any]:
    if isinstance(ep, DualEpisode):
        parts = [diagnose(ep.a).get("diagnostics"), diagnose(ep.b).get("diagnostics")]
        if any(p is None or "error" in p for p in parts):
            return {"diagnostics": None}
        merged = {k: parts[0][k] + parts[1][k] for k in parts[0]}
        return {"diagnostics": merged}
    return diagnose(ep)


def diagnose(ep: Episode) -> dict[str, Any]:
    """Replay the observation stream through a fresh sound tracker."""
    if ep.config.rotation_mode not in DIAGNOSABLE_MODES:
        return {"diagnostics": None}
    board = ep.board
    tracker = BeliefTracker(board)
    known_presses = 0
    zero_info_chords = 0
    total_presses = 0
    try:
        for record in ep.records:
            positions = record["press_positions"]
            if positions:
                deduced = tracker.deduced()
                already = sum(1 for p in positions if p in deduced)
                known_presses += already
                total_presses += len(positions)
                if already == len(positions):
                    zero_info_chords += 1
                tracker.observe_chord(positions, record["results"])
            rotation = record["rotation"]
            if rotation["mode"] == "announced_full":
                mapping = {int(k): v for k, v in rotation["permutation"].items()}
                tracker.apply_rotation_mapping(mapping)
            elif rotation["mode"] in ("announced", "observer", "storm"):
                tracker.apply_rotation(rotation["positions"])
    except Contradiction:  # pragma: no cover - would indicate an env bug
        return {"diagnostics": {"error": "contradiction during replay"}}
    deduced = tracker.deduced()
    guess = (ep.result or {}).get("guess", {})
    # Guess keys use the addressing CURRENT at guess time; positions are
    # physical — translate via addr_of_phys or relabel tiers mis-score.
    deduced_correct = sum(
        1
        for pos, letter in deduced.items()
        if guess.get(ep.addr_of_phys(pos)) == ep.layout[pos]
    )
    truth_matches = sum(
        1 for pos, letter in deduced.items() if ep.layout[pos] == letter
    )
    return {
        "diagnostics": {
            "deducible_at_guess": len(deduced),
            "deduced_correct_in_guess": deduced_correct,
            "tracker_truth_matches": truth_matches,  # == deducible if env+tracker sound
            "known_presses": known_presses,
            "zero_info_chords": zero_info_chords,
            "total_presses": total_presses,
        }
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {}
    accs = [r["board_accuracy"] for r in results]
    mean = sum(accs) / n
    var = sum((a - mean) ** 2 for a in accs) / max(n - 1, 1)
    solved = sum(1 for r in results if r["board_accuracy"] == 1.0)
    return {
        "episodes": n,
        "board_accuracy_mean": round(mean, 4),
        "board_accuracy_sem": round((var / n) ** 0.5, 4),
        "solve_rate": round(solved / n, 4),
        "guess_turn_mean": round(sum(r["guess_turn"] for r in results) / n, 2),
        "lock_score_mean": round(sum(r["lock_score"] for r in results) / n, 2),
        "lock_wrong_total": sum(r["lock_wrong"] for r in results),
        "violations_total": sum(len(r["violations"]) for r in results),
    }
