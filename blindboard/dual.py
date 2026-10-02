"""Two independent hidden boards played alternately — the interference
axis. Same address space, same symbol set, two hidden layouts: a sound
player keeps two trackers; a model must keep two near-identical fact-sets
from bleeding into each other.

Timeline: A1 B1 A2 B2 ... then board A's guess phase, then board B's.
Score = combined correct / 2n. Presents the Episode interface the runners
already use (observation/validate/apply/skip_turn/force_zero_guess/...)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from blindboard.env import Action, EnvConfig, Episode, Observation


class DualEpisode:
    def __init__(self, config: EnvConfig, seed: int) -> None:
        base = replace(config, dual=False)
        self.config = config
        self.seed = seed
        self.a = Episode(base, seed * 2)
        self.b = Episode(base, seed * 2 + 1)
        self.board = self.a.board

    # ------------------------------------------------------------ routing

    @property
    def active(self) -> Episode:
        if self.a.phase == "press" and self.a.turn <= self.b.turn:
            return self.a
        if self.b.phase == "press":
            return self.b
        return self.a if not self.a.done else self.b

    @property
    def active_label(self) -> str:
        return "A" if self.active is self.a else "B"

    @property
    def done(self) -> bool:
        return self.a.done and self.b.done

    @property
    def phase(self) -> str:
        return self.active.phase

    @property
    def turn(self) -> int:
        return self.active.turn

    @property
    def layout(self) -> list[str]:
        return self.active.layout

    @property
    def violations(self) -> list[dict[str, Any]]:
        return self.a.violations + self.b.violations

    @property
    def records(self) -> list[dict[str, Any]]:
        recs = [{**r, "dual_board": "A"} for r in self.a.records]
        recs += [{**r, "dual_board": "B"} for r in self.b.records]
        return recs

    # --------------------------------------------------------------- api

    def observation(self) -> Observation:
        sub = self.active.observation()
        label = self.active_label
        notes = [
            f"=== BOARD {label} === (two independent boards; this turn and these "
            f"results/notices are for board {label} only)"
        ] + list(sub.notes)
        return Observation(
            turn=sub.turn,
            horizon=sub.horizon,
            phase=sub.phase,
            last_presses=sub.last_presses,
            last_results=sub.last_results,
            rotation_notice=sub.rotation_notice,
            rotation_mapping=sub.rotation_mapping,
            relabel_notice=sub.relabel_notice,
            notes=notes,
        )

    def validate(self, action: Action) -> str | None:
        return self.active.validate(action)

    def apply(self, action: Action, meta: dict[str, Any] | None = None) -> None:
        self.active.apply(action, meta)

    def skip_turn(self, reason: str) -> None:
        self.active.skip_turn(reason)

    def force_zero_guess(self, reason: str) -> None:
        self.active.force_zero_guess(reason)

    def phys_of_addr(self, address: str) -> int:
        return self.active.phys_of_addr(address)

    def addr_of_phys(self, pos: int) -> str:
        return self.active.addr_of_phys(pos)

    @property
    def result(self) -> dict[str, Any] | None:
        if not self.done:
            return None
        ra, rb = self.a.result or {}, self.b.result or {}
        n = self.board.n
        return {
            "seed": self.seed,
            "config": self.config.to_dict(),
            "dual": True,
            "guess_turn": max(ra["guess_turn"], rb["guess_turn"]),
            "forced_guess": ra["forced_guess"] or rb["forced_guess"],
            "board_n": 2 * n,
            "correct": ra["correct"] + rb["correct"],
            "board_accuracy": (ra["correct"] + rb["correct"]) / (2 * n),
            "per_board": {"A": ra["correct"], "B": rb["correct"]},
            "locks": {"A": ra["locks"], "B": rb["locks"]},
            "lock_correct": ra["lock_correct"] + rb["lock_correct"],
            "lock_wrong": ra["lock_wrong"] + rb["lock_wrong"],
            "lock_score": ra["lock_score"] + rb["lock_score"],
            "violations": self.violations,
            "guess": {"A": ra["guess"], "B": rb["guess"]},
            "final_layout": {"A": ra["final_layout"], "B": rb["final_layout"]},
            "meta": {"A": ra.get("meta", {}), "B": rb.get("meta", {})},
        }


def make_episode(config: EnvConfig, seed: int) -> Episode | DualEpisode:
    return DualEpisode(config, seed) if config.dual else Episode(config, seed)
