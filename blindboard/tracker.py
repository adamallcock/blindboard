"""Sound (but deliberately incomplete) belief tracker over the hidden layout.

Belief state, always about the CURRENT layout:
- cand[p]: set of letters position p could hold (truth always contained —
  the soundness invariant, enforced by tests)
- groups: (positions, letters) constraints, |positions| == |letters|,
  meaning "exactly these letters occupy exactly these positions".

Chords create groups (results are unordered, so a chord pins a letter-set
to a position-set, not an assignment). Rotations are derangements within a
set S: every position in S changes letter, the letter-set on S is
preserved. Announced rotations degrade knowledge to unions over S and drop
groups that partially overlap S (a sound weakening; exact propagation of
unordered chords through partial rotations is intentionally out of scope).

This tracker doubles as: the reference agent's brain, and the diagnostics
engine that replays any agent's observation stream to compute what was
deducible from the information it actually received.
"""

from __future__ import annotations

import math
import random
from blindboard.boards import Board


class Contradiction(Exception):
    pass


class BeliefTracker:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.n = board.n
        self.symbols = list(board.symbols)
        self.cand: list[set[str]] = [set(self.symbols) for _ in range(self.n)]
        self.groups: list[tuple[frozenset[int], frozenset[str]]] = []

    # ------------------------------------------------------------- updates

    def observe_chord(self, positions: list[int], letters: list[str]) -> None:
        """Unordered chord result: letter-set `letters` occupies `positions`."""
        if not positions:
            return
        self._add_group(frozenset(positions), frozenset(letters))
        self.propagate()

    def apply_rotation(self, rotated: list[int]) -> None:
        """Announced derangement within `rotated` (permutation unknown)."""
        s = frozenset(rotated)
        if len(s) < 2:
            return
        # The letter-set on S is preserved; record it if fully known.
        if all(len(self.cand[p]) == 1 for p in s):
            letters = frozenset(next(iter(self.cand[p])) for p in s)
            self._add_group(s, letters)
        # Groups survive only if untouched or wholly containing S.
        self.groups = [
            (pset, lset)
            for pset, lset in self.groups
            if not (pset & s) or s <= pset
        ]
        old = [set(c) for c in self.cand]
        for p in s:
            union: set[str] = set()
            for q in s:
                if q != p:
                    union |= old[q]
            self.cand[p] = union
        self.propagate()

    def apply_rotation_mapping(self, mapping: dict[int, int]) -> None:
        """Fully announced rotation: letter at src moved to dst. Lossless."""
        if not mapping:
            return
        old = [set(c) for c in self.cand]
        for src, dst in mapping.items():
            self.cand[dst] = old[src]
        remap = {src: dst for src, dst in mapping.items()}
        self.groups = [
            (frozenset(remap.get(p, p) for p in pset), lset)
            for pset, lset in self.groups
        ]
        self.propagate()

    # --------------------------------------------------------- propagation

    def _add_group(self, pset: frozenset[int], lset: frozenset[str]) -> None:
        if len(pset) != len(lset):
            raise Contradiction(f"group size mismatch: {pset} vs {lset}")
        if (pset, lset) not in self.groups:
            self.groups.append((pset, lset))

    def propagate(self) -> None:
        changed = True
        while changed:
            changed = False
            for pset, lset in self.groups:
                for p in range(self.n):
                    before = len(self.cand[p])
                    if p in pset:
                        self.cand[p] &= lset
                    else:
                        self.cand[p] -= lset
                    if len(self.cand[p]) != before:
                        changed = True
                    if not self.cand[p]:
                        raise Contradiction(f"no candidates left for position {p}")
            # Singleton elimination.
            for p in range(self.n):
                if len(self.cand[p]) == 1:
                    letter = next(iter(self.cand[p]))
                    for q in range(self.n):
                        if q != p and letter in self.cand[q]:
                            self.cand[q].discard(letter)
                            changed = True
                            if not self.cand[q]:
                                raise Contradiction(f"no candidates left for position {q}")
            # Hidden singles.
            for letter in self.symbols:
                spots = [p for p in range(self.n) if letter in self.cand[p]]
                if not spots:
                    raise Contradiction(f"letter {letter} has no possible position")
                if len(spots) == 1 and len(self.cand[spots[0]]) > 1:
                    self.cand[spots[0]] = {letter}
                    changed = True
        # Drop groups that have become fully implied by singletons.
        self.groups = [
            (pset, lset)
            for pset, lset in self.groups
            if not all(len(self.cand[p]) == 1 for p in pset)
        ]

    # ------------------------------------------------------------- queries

    def deduced(self) -> dict[int, str]:
        return {p: next(iter(c)) for p, c in enumerate(self.cand) if len(c) == 1}

    def solved(self) -> bool:
        return all(len(c) == 1 for c in self.cand)

    def entropy_upper_bound(self) -> float:
        return sum(math.log2(len(c)) for c in self.cand)

    def enumerate_layouts(self, cap: int = 20000) -> list[list[str]] | None:
        """ALL layouts consistent with cand + groups, or None if more than
        `cap` exist (bail early). Exact posterior support — used for
        endgame guessing, where the pruned space is small."""
        order = sorted(range(self.n), key=lambda p: len(self.cand[p]))
        out: list[list[str]] = []
        layout: dict[int, str] = {}
        used: set[str] = set()

        def walk(i: int) -> bool:
            if len(out) > cap:
                return False
            if i == len(order):
                full = [layout[p] for p in range(self.n)]
                if self._check_groups(full):
                    out.append(full)
                return True
            p = order[i]
            for x in sorted(self.cand[p]):
                if x in used:
                    continue
                layout[p] = x
                used.add(x)
                ok = walk(i + 1)
                used.discard(x)
                del layout[p]
                if not ok:
                    return False
            return True

        complete = walk(0)
        return out if complete and len(out) <= cap else None

    def sample_layouts(self, rng: random.Random, k: int, tries_per: int = 200) -> list[list[str]]:
        """Sample up to k layouts consistent with cand + groups (randomized
        backtracking; not guaranteed uniform, good enough for chord scoring)."""
        samples: list[list[str]] = []
        order = sorted(range(self.n), key=lambda p: len(self.cand[p]))
        for _ in range(k):
            layout = self._sample_one(rng, order, tries_per)
            if layout is not None:
                samples.append(layout)
        return samples

    def _sample_one(
        self, rng: random.Random, order: list[int], tries: int
    ) -> list[str] | None:
        for _ in range(tries):
            layout: dict[int, str] = {}
            used: set[str] = set()
            if self._backtrack(rng, order, 0, layout, used, budget=[2000]):
                full = [layout[p] for p in range(self.n)]
                if self._check_groups(full):
                    return full
        return None

    def _backtrack(
        self,
        rng: random.Random,
        order: list[int],
        i: int,
        layout: dict[int, str],
        used: set[str],
        budget: list[int],
    ) -> bool:
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        if i == len(order):
            return True
        p = order[i]
        # Sorted before shuffle: set iteration order is hash-seed-dependent,
        # and shuffling a hash-ordered list makes sampled layouts (and every
        # reference score built on them) vary across interpreter runs.
        options = sorted(x for x in self.cand[p] if x not in used)
        rng.shuffle(options)
        for x in options:
            layout[p] = x
            used.add(x)
            if self._backtrack(rng, order, i + 1, layout, used, budget):
                return True
            used.discard(x)
            del layout[p]
        return False

    def _check_groups(self, layout: list[str]) -> bool:
        return all(
            frozenset(layout[p] for p in pset) == lset for pset, lset in self.groups
        )
