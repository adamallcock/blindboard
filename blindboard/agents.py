"""Scripted agents: baselines and the reference solver.

- RandomAgent: random legal chords, random final guess. The floor.
- AssumedOrderAgent: a plausibly-naive player — presses keys round-robin,
  misreads the alphabetical results as if they were in press order, ignores
  rotations, guesses its last-seen letters. Measures how far shallow
  memorization gets without deduction.
- ReferenceAgent: sound belief tracker + myopic expected-information chord
  selection (scored over sampled consistent layouts). Locks newly deduced
  keys, guesses when solved. Not information-theoretically optimal, but a
  strong reference for score normalization ("reference-normalized regret").
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter

from blindboard.boards import Board
from blindboard.env import Action, EnvConfig, Observation
from blindboard.tracker import BeliefTracker, Contradiction


class ScriptedAgent:
    def __init__(self, config: EnvConfig, seed: int) -> None:
        self.config = config
        self.board: Board = config.board_obj()
        self.rng = random.Random(f"agent:{type(self).__name__}:{seed}")

    def act(self, obs: Observation) -> Action:  # pragma: no cover - interface
        raise NotImplementedError


class RandomAgent(ScriptedAgent):
    def act(self, obs: Observation) -> Action:
        if obs.phase == "guess":
            letters = list(self.board.symbols)
            self.rng.shuffle(letters)
            return Action(guess=dict(zip(self.board.addresses, letters)))
        return Action(presses=self.rng.sample(self.board.addresses, self.config.chord_size))


class AssumedOrderAgent(ScriptedAgent):
    def __init__(self, config: EnvConfig, seed: int) -> None:
        super().__init__(config, seed)
        self.believed: dict[str, str] = {}
        self.cursor = 0

    def act(self, obs: Observation) -> Action:
        if obs.last_presses and obs.last_results:
            # The misreading: attribute sorted results positionally.
            for addr, letter in zip(obs.last_presses, obs.last_results):
                self.believed[addr] = letter
        if obs.phase == "guess":
            return Action(guess=self._guess())
        k = self.config.chord_size
        addrs = self.board.addresses
        presses = [addrs[(self.cursor + i) % len(addrs)] for i in range(k)]
        self.cursor = (self.cursor + k) % len(addrs)
        return Action(presses=presses)

    def _guess(self) -> dict[str, str]:
        guess: dict[str, str] = {}
        used: set[str] = set()
        for addr in self.board.addresses:
            letter = self.believed.get(addr)
            if letter is not None and letter not in used:
                guess[addr] = letter
                used.add(letter)
        leftovers = [x for x in self.board.symbols if x not in used]
        self.rng.shuffle(leftovers)
        for addr in self.board.addresses:
            if addr not in guess:
                guess[addr] = leftovers.pop()
        return guess


class ReferenceAgent(ScriptedAgent):
    """Tracker + myopic expected-information chord choice.

    `privileged_rotation=True` lets calibration runs feed it rotation sets
    even in unannounced modes (an optimistic upper bound for those tiers).
    """

    def __init__(
        self,
        config: EnvConfig,
        seed: int,
        samples: int = 40,
        max_chords_scored: int = 3000,
        observer_anchor_penalty: float = 1.0,
        use_locks: bool = True,
    ) -> None:
        super().__init__(config, seed)
        self.tracker = BeliefTracker(self.board)
        self.samples = samples
        self.max_chords_scored = max_chords_scored
        self.observer_anchor_penalty = observer_anchor_penalty
        self.use_locks = use_locks
        self.locked: set[str] = set()
        self.tracker_resets = 0

    # -- belief updates from the observation stream ----------------------

    def ingest(self, obs: Observation, privileged_rotation: list[str] | None = None) -> None:
        try:
            if obs.last_presses and obs.last_results:
                positions = [self.board.index_of(a) for a in obs.last_presses]
                self.tracker.observe_chord(positions, list(obs.last_results))
            rotated = obs.rotation_notice
            if privileged_rotation is not None:
                rotated = privileged_rotation
            if obs.rotation_mapping:
                mapping = {
                    self.board.index_of(src): self.board.index_of(dst)
                    for src, dst in obs.rotation_mapping
                }
                self.tracker.apply_rotation_mapping(mapping)
            elif rotated:
                self.tracker.apply_rotation([self.board.index_of(a) for a in rotated])
        except Contradiction:
            # Only reachable when beliefs go stale (unannounced modes without
            # privileged info). Reset to ignorance rather than crash.
            self.tracker = BeliefTracker(self.board)
            self.tracker_resets += 1

    def act(self, obs: Observation, privileged_rotation: list[str] | None = None) -> Action:
        self.ingest(obs, privileged_rotation)
        locks = self._new_locks() if self.use_locks else {}
        if obs.phase == "guess":
            return Action(guess=self._best_guess())
        if self.tracker.solved() and obs.turn >= self.config.min_guess_turn:
            return Action(guess=self._best_guess())
        presses = self._choose_chord()
        return Action(presses=presses, locks=locks)

    # -- internals --------------------------------------------------------

    def _new_locks(self) -> dict[str, str]:
        locks = {}
        for pos, letter in self.tracker.deduced().items():
            addr = self.board.addresses[pos]
            if addr not in self.locked:
                locks[addr] = letter
                self.locked.add(addr)
        return locks

    def _best_guess(self) -> dict[str, str]:
        # Exact posterior when the consistent-layout space enumerates:
        # per-position marginal argmax maximizes expected keys-correct
        # (reference v2). Falls back to sampled voting on huge spaces.
        layouts = self.tracker.enumerate_layouts(cap=20000)
        if layouts is None:
            layouts = self.tracker.sample_layouts(self.rng, max(self.samples, 60))
        votes: list[Counter[str]] = [Counter() for _ in range(self.board.n)]
        for layout in layouts:
            for p, letter in enumerate(layout):
                votes[p][letter] += 1
        guess: dict[str, str] = {}
        for p, addr in enumerate(self.board.addresses):
            if votes[p]:
                guess[addr] = votes[p].most_common(1)[0][0]
            else:  # no layout found; fall back to any candidate
                guess[addr] = sorted(self.tracker.cand[p])[0]
        return guess

    def _choose_chord(self) -> list[str]:
        k = self.config.chord_size
        n = self.board.n
        layouts = self.tracker.sample_layouts(self.rng, self.samples)
        if len(layouts) < 2:
            unknown = [p for p in range(n) if len(self.tracker.cand[p]) > 1]
            pool = unknown + [p for p in range(n) if p not in unknown]
            return [self.board.addresses[p] for p in pool[:k]]
        total = math.comb(n, k)
        if total <= self.max_chords_scored:
            chords = list(itertools.combinations(range(n), k))
        else:
            chords = [
                tuple(sorted(self.rng.sample(range(n), k)))
                for _ in range(self.max_chords_scored)
            ]
        deduced = set(self.tracker.deduced())
        best_score = float("inf")
        best: list[tuple[int, ...]] = []
        for chord in chords:
            buckets: Counter[tuple[str, ...]] = Counter()
            for layout in layouts:
                buckets[tuple(sorted(layout[p] for p in chord))] += 1
            m = len(layouts)
            score = sum((c / m) * math.log2(c) for c in buckets.values())
            if self.config.rotation_mode in ("observer", "storm"):
                score += self.observer_anchor_penalty * sum(
                    1 for p in chord if p in deduced
                )
            if score < best_score - 1e-9:
                best_score, best = score, [chord]
            elif abs(score - best_score) <= 1e-9:
                best.append(chord)
        chosen = self.rng.choice(best)
        return [self.board.addresses[p] for p in chosen]


from blindboard.planner import PlannerAgent  # noqa: E402  (bottom import: planner subclasses/wraps ReferenceAgent)

AGENTS = {
    "random": RandomAgent,
    "assumed_order": AssumedOrderAgent,
    "reference": ReferenceAgent,
    "planner": PlannerAgent,
}
