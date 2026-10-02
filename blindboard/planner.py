"""Solver v3 ("planner"): the reference engine plus genuine multi-turn play.

Where the reference (v2) actually loses points
----------------------------------------------
Loss decomposition over the sub-1.0 tiers shows that essentially ALL of the
reference's wrong keys sit inside the LAST rotation set of the episode.
That loss has a hard information-theoretic floor: for a uniform derangement
over an announced set S, the post-rotation marginal of any position in S is
capped at 1/(|S|-1) (p receives the letter of a uniformly-chosen other
member of S, and a letter occupies at most one member), so any guess loses
at least

    loss_floor(s) = s - s/(s-1)        (s = |S| >= 3; a deranged pair is a
                                        swap, s = 2 costs nothing)

keys on that set: 1.5 for s=3, 8/3 for s=4, 15/4 for s=5.  No player can
beat this, whatever the strategy — which also means per-tier ceilings of
(n - floor)/n: .875 for t3_observer, .8974 for t6_observer26/t8_storm,
.9423 for t7_perpetual/t10_relabel, .8558 for t8_storm_hard, .9615 for
t9_numrow and .9681 for t9_macbook/t11_maelstrom.  The reference already
measures AT (or, by seed luck, above) that ceiling everywhere except
t3_observer, t7_perpetual and t8_storm_hard.

What the planner adds
---------------------
The planner plays byte-identically to the reference EXCEPT for two
deviations that target exactly the remaining gap:

1. OPTIMAL STOPPING (observer/storm).  Since every press turn scrambles the
   pressed keys (plus storm extras), every future stopping point pays the
   loss_floor of its own freshest blob — so the planner guesses at the
   FIRST provably-floor-tight moment instead of pressing to the horizon and
   frequently ending on a flatter (worse-than-sharp) final chord like the
   reference does.  Two triggers:
   (a) SHARP STATE (tracker-only): everything resolved except the freshest
       rotation set, whose members hold (s-1)-candidate sets with distinct
       missing letters.  The consistent space is then exactly the uniform
       set of s-derangements, and the exact-posterior guess realises
       loss_floor(s) — no continuation can expect better.  With decaying
       storm extras the threshold compares against the minimum future blob
       floor, so the planner plays through the extra-churn phase and stops
       in the pure-observer phase.
   (b) WEIGHTED DAMAGE (storm): an exact weighted posterior (WeightedBelief
       below) tracks the true distribution over current layouts once the
       consistent space is enumerable; its damage — expected keys lost by
       guessing now — is exact, so the planner stops whenever damage dips
       to the floor (plus, for never-decaying extras a la storm_hard,
       where the floor itself is unreachable and steady-state horizon
       damage sits ~1.2 keys above it, a dev-tuned residual margin).

2. RESOLVER CLOSERS (perpetual announced, and K=3 observer as a safety
   net).  In the last press turns the planner forces deterministic chords
   {candidate-disjoint probes + deduced companions}.  The unordered result
   then attributes every letter uniquely, so plain tracker propagation pins
   each probe; the penultimate rotation set gets resolved retroactively by
   the final chord's result, and the final random rotation lands on a
   fully-known board, costing exactly its floor.

3. EXACT WEIGHTED GUESSING (storm): at the final guess the weighted
   posterior's per-position argmax replaces the reference's uniform vote
   over tracker-consistent layouts, recovering the derangement path-weight
   and joint no-fixed-point information the tracker's sound-but-lossy
   propagation discards.

Everything else — exploration chords, lock policy (tracker-deduced keys
only, perfect precision), guessing, contradiction handling — is delegated
to an internal ReferenceAgent constructed with the same (config, seed), so
its RNG stream, and hence fallback play, is byte-identical to the
registered reference.  Configs where the reference is already at ceiling
run in pure fallback mode: certificates cannot regress there.

The frozen BeliefTracker is used unchanged and read-only; determinism per
(config, seed) holds because every planner decision is a pure function of
the tracker state and the shared seeded RNG streams.
"""

from __future__ import annotations

import random
from itertools import permutations

from blindboard.boards import Board
from blindboard.env import Action, EnvConfig, Observation

# Stop-rule float tolerance: the sharp-state comparison is exact rationals
# in float form; the epsilon only absorbs representation error.
STOP_DELTA = 1e-6

# Perpetual announced: force resolver chords for the last N press turns.
CLOSER_TURNS = 2

# Observer with chord_size 3: a sharp 3-blob is a 2-cycle, so one probe
# resolves it wholly and a resolver pipeline is stable; use it as a closer
# when the natural stop state has not appeared late in the episode.
OBSERVER3_CLOSER_TURNS = 4

# Weighted stopping margin for perpetual-extras storm (storm_hard): the
# floor is unreachable there (the rolling extra key keeps ~1+ keys of
# residue alive), so stop once the EXACT damage is within this margin of
# the floor — comfortably below the ~1.2-above-floor steady-state horizon
# damage.  Swept on held-out dev seeds (1000-1059), not on the certificate
# seeds: 0.75 beat {0.25, 0.5, 1.0, 1.25} there.
STORM_RESIDUAL_MARGIN = 0.75

# Trust the weighted belief only after this many chord filters since
# seeding (the uniform seed weights need observations to sharpen), and only
# while its support is small enough that exact marginals are cheap (a large
# support means high damage anyway — no stopping decision is lost).
BELIEF_TRUST_FILTERS = 2
BELIEF_TRUST_SUPPORT = 25000


def loss_floor(s: int) -> float:
    """Minimum expected keys lost on the freshest announced uniform
    derangement of s positions — unavoidable by any strategy."""
    if s <= 2:
        return 0.0
    return s - s / (s - 1)


_PATTERN_CACHE: dict[int, list[tuple[int, ...]]] = {}


def derangement_patterns(s: int) -> list[tuple[int, ...]]:
    """All derangements of range(s), as tuples q with q[i] != i meaning
    'the letter at slot i moves to slot q[i]' (the env's convention:
    layout[dst] = old[src])."""
    if s not in _PATTERN_CACHE:
        _PATTERN_CACHE[s] = [
            q for q in permutations(range(s)) if all(q[i] != i for i in range(s))
        ]
    return _PATTERN_CACHE[s]


def bounded_enumerate(
    tracker, cap: int, step_budget: int = 250_000
) -> list[tuple[str, ...]] | None:
    """All layouts consistent with the tracker's cand + groups, or None if
    more than `cap` exist OR the walk exceeds `step_budget` nodes.

    Equivalent result set to tracker.enumerate_layouts(cap) but safe on
    SPARSE spaces: group constraints are enforced during the walk (not only
    at leaves), and the step budget hard-bounds worst-case time — the
    tracker's own enumerator can churn exponentially on mid-game storm
    states whose candidate product is huge but whose consistent set is
    small.  Read-only on the tracker; deterministic."""
    n = tracker.n
    order = sorted(range(n), key=lambda p: (len(tracker.cand[p]), p))
    # Per-position intersection of every containing group's letter set:
    # a partial assignment violating it can never complete a valid group
    # (equal sizes force the assigned letters to equal lset exactly), so
    # membership pruning at assignment time is equivalent to the tracker's
    # leaf-time _check_groups.
    allowed: list[set[str]] = [set(tracker.cand[p]) for p in range(n)]
    for pset, lset in tracker.groups:
        for p in pset:
            allowed[p] &= lset
    domains: list[list[str]] = [sorted(allowed[p]) for p in range(n)]
    out: list[tuple[str, ...]] = []
    layout: dict[int, str] = {}
    used: set[str] = set()
    budget = [step_budget]

    def walk(i: int) -> bool:
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        if len(out) > cap:
            return False
        if i == n:
            out.append(tuple(layout[p] for p in range(n)))
            return True
        p = order[i]
        for x in domains[p]:
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


class WeightedBelief:
    """Exact-forward-filtered weighted posterior over CURRENT layouts.

    Seeded from the frozen tracker's full consistent enumeration (a sound
    superset of the true posterior support) with uniform weights; from then
    on every transition is exact: chord results filter the support, and an
    announced rotation of set S mixes each layout over all |S|-derangements
    with equal probability (the env draws its derangements uniformly).
    This recovers exactly the information the tracker's sound-but-lossy
    union/group propagation discards — joint no-fixed-point structure and
    path multiplicities — which is what distinguishes storm endgames.

    Soundness invariant (tested): while `active`, the true current layout
    is in `support`.  The uniform seeding makes WEIGHTS approximate at
    first; `filters` counts chord filters since seeding, and callers treat
    the belief as trustworthy only after a couple of them."""

    def __init__(self, board: Board, seed_cap: int = 3000, expand_budget: int = 1_000_000) -> None:
        self.board = board
        self.n = board.n
        self.seed_cap = seed_cap
        self.expand_budget = expand_budget
        self.active = False
        self.filters = 0
        self.support: dict[tuple[str, ...], float] = {}

    def try_seed(self, tracker) -> bool:
        if self.active:
            return True
        layouts = bounded_enumerate(tracker, cap=self.seed_cap)
        if layouts is None or not layouts:
            return False
        w = 1.0 / len(layouts)
        self.support = {l: w for l in layouts}
        self.active = True
        self.filters = 0
        return True

    def deactivate(self) -> None:
        self.active = False
        self.support = {}

    def observe_chord(self, positions: list[int], letters: list[str]) -> None:
        if not self.active or not positions:
            return
        target = sorted(letters)
        new = {
            layout: w
            for layout, w in self.support.items()
            if sorted(layout[p] for p in positions) == target
        }
        if not new:  # would mean the seed missed the truth: never, but safe
            self.deactivate()
            return
        total = sum(new.values())
        self.support = {layout: w / total for layout, w in new.items()}
        self.filters += 1

    def apply_rotation(self, rotated: list[int]) -> None:
        if not self.active:
            return
        s = len(rotated)
        if s < 2:
            return
        pos = sorted(rotated)
        patterns = derangement_patterns(s)
        if len(self.support) * len(patterns) > self.expand_budget:
            self.deactivate()
            return
        share = 1.0 / len(patterns)
        new: dict[tuple[str, ...], float] = {}
        for layout, w in self.support.items():
            ws = w * share
            for q in patterns:
                moved = list(layout)
                for i in range(s):
                    moved[pos[q[i]]] = layout[pos[i]]
                key = tuple(moved)
                new[key] = new.get(key, 0.0) + ws
        self.support = new

    def apply_rotation_mapping(self, mapping: dict[int, int]) -> None:
        if not self.active or not mapping:
            return
        new: dict[tuple[str, ...], float] = {}
        for layout, w in self.support.items():
            moved = list(layout)
            for src, dst in mapping.items():
                moved[dst] = layout[src]
            key = tuple(moved)
            new[key] = new.get(key, 0.0) + w
        self.support = new

    def marginals(self) -> list[dict[str, float]]:
        margs: list[dict[str, float]] = [{} for _ in range(self.n)]
        for layout, w in self.support.items():
            for p, letter in enumerate(layout):
                margs[p][letter] = margs[p].get(letter, 0.0) + w
        return margs

    def damage(self) -> float:
        """Exact expected keys lost by the optimal guess right now."""
        return float(self.n) - sum(
            max(m.values()) for m in self.marginals() if m
        )

    def best_guess(self) -> dict[int, str]:
        """Per-position argmax of the exact marginals (maximises expected
        keys-correct). Deterministic tie-break: higher weight, then letter."""
        guess: dict[int, str] = {}
        for p, m in enumerate(self.marginals()):
            guess[p] = sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return guess


def select_policy(config: EnvConfig) -> str:
    """Pick the planner policy for a config ("fallback" = exact reference).

    Conservative on purpose: configs whose reference certificate already
    sits at the information-theoretic ceiling (decaying announced schedules
    that solve to 1.000; observer/storm with chord_size >= 4 at the
    K-derangement floor; perpetual churn on the 39/47-key boards and the
    relabel tiers, measured at/above (n - 1.5)/n) keep byte-identical
    reference behaviour so no certificate can regress.  The planning
    policies own the three configs with a real remaining gap: observer,
    storm, and single-board 26-key perpetual announced churn.
    """
    mode = config.rotation_mode
    if config.dual:
        return "fallback"
    if mode == "observer":
        return "observer"
    if mode == "storm":
        return "storm"
    if mode == "announced":
        if (
            config.rotation_size(config.horizon) >= 3
            and config.relabel_every == 0
            and config.board_obj().n <= 26
        ):
            return "perpetual"
        return "fallback"
    return "fallback"


class PlannerAgent:
    """Solver v3. Registry-compatible: PlannerAgent(config, seed, **kwargs)."""

    def __init__(
        self,
        config: EnvConfig,
        seed: int,
        samples: int = 40,
        max_chords_scored: int = 3000,
        observer_anchor_penalty: float = 1.0,
        use_locks: bool = True,
    ) -> None:
        # Lazy import: blindboard.agents imports this module at its bottom
        # to register the planner; a top-level import here would cycle.
        from blindboard.agents import ReferenceAgent

        self.config = config
        self.board: Board = config.board_obj()
        self.seed = seed
        self.rng = random.Random(f"agent:PlannerAgent:{seed}")
        self._ref = ReferenceAgent(
            config,
            seed,
            samples=samples,
            max_chords_scored=max_chords_scored,
            observer_anchor_penalty=observer_anchor_penalty,
            use_locks=use_locks,
        )
        self.policy = select_policy(config)
        self.use_locks = use_locks
        self._last_rotation: list[int] = []
        # Storm only: exact weighted posterior for endgame guessing and
        # non-sharp stopping (storm_hard endgames are never sharp: the
        # rolling extra key keeps residual uncertainty alive).
        self.belief: WeightedBelief | None = (
            WeightedBelief(self.board) if self.policy == "storm" else None
        )
        # Stop margin above the theoretical floor for weighted stopping:
        # only meaningful when extras never decay to zero (storm_hard), where
        # the achievable steady-state damage sits well above the floor.
        self._stop_margin = (
            STORM_RESIDUAL_MARGIN if config.rotation_size(config.horizon) >= 1 else 0.0
        ) if self.policy == "storm" else 0.0

    # Expose the reference's diagnostics so runners/tests can read them.
    @property
    def tracker(self):
        return self._ref.tracker

    @property
    def locked(self) -> set[str]:
        return self._ref.locked

    @property
    def tracker_resets(self) -> int:
        return self._ref.tracker_resets

    # ------------------------------------------------------------------ act

    def act(self, obs: Observation, privileged_rotation: list[str] | None = None) -> Action:
        if self.policy == "fallback":
            return self._ref.act(obs, privileged_rotation)

        resets_before = self._ref.tracker_resets
        self._ref.ingest(obs, privileged_rotation)
        rotated = obs.rotation_notice if privileged_rotation is None else privileged_rotation
        self._last_rotation = [self.board.index_of(a) for a in rotated] if rotated else []
        self._update_belief(obs, resets_before)
        locks = self._ref._new_locks() if self.use_locks else {}
        if obs.phase == "guess":
            return Action(guess=self._final_guess())
        if obs.turn >= self.config.min_guess_turn:
            if self._ref.tracker.solved():
                return Action(guess=self._final_guess())
            if self.policy in ("observer", "storm") and self._should_stop(obs):
                return Action(guess=self._final_guess())
        return Action(presses=self._choose_chord(obs), locks=locks)

    # ------------------------------------------------------ weighted belief

    def _update_belief(self, obs: Observation, resets_before: int) -> None:
        """Mirror the tracker's update sequence into the exact weighted
        posterior (chord filter, then announced rotation mixing)."""
        b = self.belief
        if b is None:
            return
        if self._ref.tracker_resets != resets_before:
            b.deactivate()  # tracker went stale and reset; belief is stale too
            return
        if b.active:
            if obs.last_presses and obs.last_results:
                positions = [self.board.index_of(a) for a in obs.last_presses]
                b.observe_chord(positions, list(obs.last_results))
            if obs.rotation_mapping:
                mapping = {
                    self.board.index_of(src): self.board.index_of(dst)
                    for src, dst in obs.rotation_mapping
                }
                b.apply_rotation_mapping(mapping)
            elif self._last_rotation:
                b.apply_rotation(self._last_rotation)
        # Seed only when the NEXT rotation's blob is small enough that the
        # derangement expansion stays cheap (storm early-game extras can push
        # blob sizes to 6-7, whose pattern counts explode).
        upcoming = self.config.chord_size + self.config.rotation_size(obs.turn)
        if obs.phase == "press" and upcoming <= 5:
            b.try_seed(self._ref.tracker)

    def _belief_trusted(self) -> bool:
        b = self.belief
        return (
            b is not None
            and b.active
            and bool(b.support)
            and b.filters >= BELIEF_TRUST_FILTERS
            and len(b.support) <= BELIEF_TRUST_SUPPORT
        )

    def _final_guess(self) -> dict[str, str]:
        if self._belief_trusted():
            by_pos = self.belief.best_guess()  # type: ignore[union-attr]
            return {self.board.addresses[p]: by_pos[p] for p in range(self.board.n)}
        return self._ref._best_guess()

    # -------------------------------------------------------------- stopping

    def _future_min_floor(self, obs: Observation) -> float:
        """Best loss_floor achievable at any future stopping point."""
        k = self.config.chord_size
        if self.config.rotation_mode == "observer":
            return loss_floor(k)
        # storm: blob = pressed U extras(turn); guessing at press turn g
        # means the freshest blob came from press turn g-1.
        floors = [
            loss_floor(k + self.config.rotation_size(u))
            for u in range(obs.turn, self.config.horizon + 1)
        ]
        return min(floors)

    def _sharp_state(self) -> bool:
        """True iff the belief state is exactly 'everything resolved except
        the freshest rotation set, whose pre-rotation assignment was fully
        known'.  Then the tracker-consistent space is exactly the uniform
        set of |S|-derangements, the exact-posterior guess realises
        loss_floor(|S|) in expectation, and no continuation can beat it."""
        last = self._last_rotation
        s = len(last)
        if s < 3:
            return False
        cand = self._ref.tracker.cand
        non_single = sorted(p for p in range(self.board.n) if len(cand[p]) > 1)
        if non_single != sorted(last):
            return False
        if any(len(cand[p]) != s - 1 for p in last):
            return False
        union: set[str] = set()
        for p in last:
            union |= cand[p]
        if len(union) != s:
            return False
        missing = [next(iter(union - cand[p])) for p in last]
        return len(set(missing)) == s

    def _should_stop(self, obs: Observation) -> bool:
        """Guess now iff the expected loss of guessing is provably no worse
        than the floor of ANY future stopping point (plus, for storm with
        never-decaying extras, a dev-tuned residual margin — the floor is
        unreachable there and steady-state damage sits far above it)."""
        if not self._last_rotation:
            return False
        floor_now = self._future_min_floor(obs) + STOP_DELTA
        if self._sharp_state() and loss_floor(len(self._last_rotation)) <= floor_now:
            return True
        if self._belief_trusted():
            return self.belief.damage() <= floor_now + self._stop_margin  # type: ignore[union-attr]
        return False

    # ---------------------------------------------------------------- chords

    def _choose_chord(self, obs: Observation) -> list[str]:
        k = self.config.chord_size
        turns_left = self.config.horizon - obs.turn
        force = False
        if self.policy == "perpetual":
            force = turns_left < CLOSER_TURNS
        elif self.policy == "observer" and k == 3:
            force = turns_left < OBSERVER3_CLOSER_TURNS
        if force:
            unresolved = [
                p for p in range(self.board.n) if len(self._ref.tracker.cand[p]) > 1
            ]
            chord = self._resolver_chord(unresolved, k)
            if chord is not None:
                return chord
        return self._ref._choose_chord()

    def _resolver_chord(self, unresolved: list[int], k: int) -> list[str] | None:
        """{pairwise candidate-disjoint probes + deduced companions}.

        Companions' letters are already excluded from every candidate set
        (singleton elimination), so with disjoint probes each result letter
        is attributable to exactly one pressed key: tracker propagation pins
        every probe, and every pressed key's pre-press letter is known
        (companions now, probes retroactively) — the freshest blob stays
        sharp.  Deterministic: sorted orders, no RNG."""
        tracker = self._ref.tracker
        if not unresolved:
            return None
        last = set(self._last_rotation)
        # Oldest structures first (drain the pipeline from the back), then
        # tightest candidate sets, then position index.
        order = sorted(unresolved, key=lambda p: (p in last, len(tracker.cand[p]), p))
        probes: list[int] = []
        union: set[str] = set()
        for p in order:
            if len(probes) == k:
                break
            cand = tracker.cand[p]
            if union & cand:
                continue
            probes.append(p)
            union |= cand
        if not probes:
            return None
        chord = list(probes)
        companions = [p for p in sorted(tracker.deduced()) if p not in chord]
        chord += companions[: k - len(chord)]
        if len(chord) < k:
            # Not enough deduced keys: top up with further unresolved
            # positions (candidate overlap allowed; the group constraint is
            # still sound information), then arbitrary distinct keys.
            chord += [p for p in order if p not in chord][: k - len(chord)]
        if len(chord) < k:
            chord += [p for p in range(self.board.n) if p not in chord][: k - len(chord)]
        if len(chord) != k:
            return None
        return [self.board.addresses[p] for p in sorted(chord)]
