"""The hidden-keyboard episode: turn loop, rotation modes, scoring.

Timeline within press-turn t (1-based):
  1. agent receives observation (results + rotation notice from turn t-1)
  2. agent submits Action (exactly K distinct presses; optional locks;
     or a full-board guess, which ends the episode)
  3. locks are evaluated silently against the CURRENT layout
  4. chord result = letters at pressed positions (current layout),
     reported alphabetically (unordered) in the NEXT observation
  5. rotation is applied (mode-specific); announced modes report WHICH
     positions rotated in the next observation, never the permutation
     (announced_full reports the permutation too)

After the final press turn the agent must submit a guess (guess phase).
Guesses and locks are always evaluated against the layout at submission
time. Rotations use derangements: every rotated position changes letter,
and the SET of letters on the rotated positions is preserved.

Episode seeding (K2 hardening, 2026-07-28)
------------------------------------------
Every random draw in an episode -- the hidden layout, the rotor subset, the
rotated sets, the derangements, the relabel choices -- comes from one RNG
keyed by (salt, seed). The generator is public and `random.Random` is a
published PRNG, so a *public* seed pool is fully replayable: an attacker who
can name the seed can precompute the whole trajectory and guess perfectly
(see `tests/test_seed_replay_attack.py`). `EnvConfig.salt` closes that path
for official scoring: the salt is high-entropy, never published alongside
the seed list, and the seed alone is then useless. An empty salt reproduces
the legacy key byte-for-byte, so every stored artifact replays unchanged.
"""

from __future__ import annotations

import hashlib
import random
import secrets
from dataclasses import dataclass, field, replace
from typing import Any

from blindboard.boards import BOARDS, Board

ROTATION_MODES = (
    "none",
    "announced",
    "announced_full",
    "observer",
    "unannounced",
    "rotor",
    "storm",  # one announced derangement over (pressed keys ∪ random sample)
)

# Domain separator for the published salt commitment. Changing it changes
# every commitment, so it is versioned rather than edited.
SALT_COMMITMENT_DOMAIN = "blindboard-salt-commitment:v1"

# Default salt entropy in bytes (secrets.token_hex -> 64 hex chars).
SALT_BYTES = 32


def generate_salt(nbytes: int = SALT_BYTES) -> str:
    """A fresh high-entropy episode salt for an official evaluation.

    Hex only, so it can never contain the ':' separator used by the seed
    key and can be pasted through shells, JSON and env vars unescaped.
    """
    if nbytes < 16:
        raise ValueError("official salts need at least 16 bytes of entropy")
    return secrets.token_hex(nbytes)


def salt_commitment(salt: str) -> str:
    """Publishable digest of a salt: proves after the fact which salt a run
    used, without revealing it while the salt is still live."""
    return hashlib.sha256(f"{SALT_COMMITMENT_DOMAIN}:{salt}".encode()).hexdigest()


def episode_seed_key(salt: str, seed: int) -> str:
    """The string handed to `random.Random` for an episode.

    An empty salt yields the pre-hardening key EXACTLY, so every artifact
    recorded before 2026-07-28 replays byte-for-byte. A non-empty salt
    prefixes the seed, which reshuffles every draw in the episode.
    """
    return f"blindboard:{seed}" if not salt else f"blindboard:{salt}:{seed}"


def with_salt(config: EnvConfig, salt: str) -> EnvConfig:
    """Official-eval helper: the same tier, bound to a secret salt."""
    return replace(config, salt=salt)


@dataclass
class EnvConfig:
    board: str = "fingers12"
    chord_size: int = 3
    horizon: int = 18  # number of press turns; guess phase follows
    rotation_mode: str = "announced"
    # rotated-set size per press turn (index t-1); missing entries = 0.
    rotation_schedule: tuple[int, ...] = ()
    layout_kind: str = "random"  # "random" | "near_qwerty"
    near_qwerty_swaps: int = 6
    rotor_size: int = 4  # rotor mode: fixed derangement over a fixed subset
    min_guess_turn: int = 1
    lock_bonus: int = 1
    lock_penalty: int = -3
    # Every k turns the ADDRESS SPACE is relabeled (two signature-matched
    # finger columns swap addresses, announced). 0 = off. Pure translation
    # load: the hidden layout and the information content are untouched.
    relabel_every: int = 0
    # Two independent boards played alternately (interference axis).
    dual: bool = False
    # SECRET per-episode salt (K2). "" = the public/legacy path: the episode
    # is fully replayable from its seed, which is fine for calibration,
    # pilots and anything already published, and fatal for official scoring.
    # Official runs bind a `generate_salt()` value that is withheld until the
    # seed list rotates. Never serialized: `to_dict` emits only its
    # commitment, so a run artifact can be tied to a salt after reveal
    # without leaking the live one.
    salt: str = ""

    def board_obj(self) -> Board:
        return BOARDS[self.board]

    def rotation_size(self, turn: int) -> int:
        idx = turn - 1
        if idx < len(self.rotation_schedule):
            return self.rotation_schedule[idx]
        return 0

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["rotation_schedule"] = list(self.rotation_schedule)
        salt = d.pop("salt", "")
        if salt:
            d["salt_commitment"] = salt_commitment(salt)
        return d


@dataclass
class Action:
    presses: list[str] = field(default_factory=list)
    locks: dict[str, str] = field(default_factory=dict)
    guess: dict[str, str] | None = None


@dataclass
class Observation:
    turn: int  # press turn number; horizon+1 = guess phase
    horizon: int
    phase: str  # "press" | "guess" | "done"
    last_presses: list[str] | None = None
    last_results: list[str] | None = None  # in canonical symbol order
    rotation_notice: list[str] | None = None  # addresses rotated after last turn
    rotation_mapping: list[tuple[str, str]] | None = None  # announced_full only
    relabel_notice: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "horizon": self.horizon,
            "phase": self.phase,
            "last_presses": self.last_presses,
            "last_results": self.last_results,
            "rotation_notice": self.rotation_notice,
            "rotation_mapping": self.rotation_mapping,
            "relabel_notice": self.relabel_notice,
            "notes": list(self.notes),
        }


def _derangement(rng: random.Random, items: list[int]) -> dict[int, int]:
    """Uniform random derangement mapping old position -> new position."""
    if len(items) < 2:
        return {}
    while True:
        shuffled = items[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(items, shuffled)):
            return dict(zip(items, shuffled))


class Episode:
    def __init__(self, config: EnvConfig, seed: int, salt: str | None = None) -> None:
        if config.rotation_mode not in ROTATION_MODES:
            raise ValueError(f"unknown rotation mode {config.rotation_mode}")
        salt = config.salt if salt is None else salt
        if ":" in salt:
            # Keeps (salt, seed) -> key injective; without it, a salt ending
            # in ":<digits>" could impersonate another episode's stream.
            raise ValueError("episode salt must not contain ':'")
        self.config = config
        self.seed = seed
        self.salt = salt
        self.board = config.board_obj()
        self.rng = random.Random(episode_seed_key(salt, seed))
        if config.layout_kind == "near_qwerty":
            self.layout = self.board.near_qwerty_layout(self.rng, config.near_qwerty_swaps)
        else:
            self.layout = self.board.random_layout(self.rng)
        self.initial_layout = list(self.layout)
        self.turn = 1
        self.done = False
        self.phase = "press"
        self.locks: dict[int, str] = {}  # position -> letter (immutable once set)
        self.lock_eval: dict[int, bool] = {}
        self.records: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self._pending_obs = Observation(turn=1, horizon=config.horizon, phase="press")
        self._pending_notes: list[str] = []
        if config.rotation_mode == "rotor":
            subset = sorted(self.rng.sample(range(self.board.n), config.rotor_size))
            self._rotor = _derangement(self.rng, subset)
        else:
            self._rotor = {}
        # Address relabeling: addr slot i currently refers to physical key
        # addr_to_phys[i]. Identity unless relabel_every > 0 fires.
        self.addr_to_phys: list[int] = list(range(self.board.n))
        self.result: dict[str, Any] | None = None

    # ------------------------------------------------------- addressing

    def phys_of_addr(self, address: str) -> int:
        return self.addr_to_phys[self.board.index_of(address)]

    def addr_of_phys(self, pos: int) -> str:
        return self.board.addresses[self.addr_to_phys.index(pos)]

    # ---------------------------------------------------------------- obs

    def observation(self) -> Observation:
        return self._pending_obs

    # ------------------------------------------------------------ validate

    def validate(self, action: Action) -> str | None:
        """Return an error message if the action is illegal, else None."""
        board = self.board
        cfg = self.config
        if action.guess is not None:
            if self.phase == "press" and self.turn < cfg.min_guess_turn:
                return f"You may not guess before turn {cfg.min_guess_turn}."
            missing = [a for a in board.addresses if a not in action.guess]
            if missing:
                return (
                    f"Your guess must cover every key. Missing {len(missing)} addresses, "
                    f"e.g. {missing[:3]}."
                )
            bad = [a for a in action.guess if a not in board.addresses]
            if bad:
                return f"Unknown addresses in guess: {bad[:3]}."
            bad_letters = sorted({v for v in action.guess.values() if v not in board.symbols})
            if bad_letters:
                return f"Guess letters must be in {board.symbols[0]}..{board.symbols[-1]}; got {bad_letters[:3]}."
            return None
        if self.phase == "guess":
            return "The game is over: you must submit your final guess now (a 'guess' object covering every key)."
        if len(action.presses) != cfg.chord_size:
            return f"You must press exactly {cfg.chord_size} keys this turn (you sent {len(action.presses)})."
        bad = [a for a in action.presses if a not in board.addresses]
        if bad:
            return f"Unknown key addresses: {bad}. Valid addresses were listed in the rules."
        if len(set(action.presses)) != len(action.presses):
            return "You may not press the same key twice in one turn."
        bad = [a for a in action.locks if a not in board.addresses]
        if bad:
            return f"Unknown addresses in locks: {bad}."
        bad_letters = sorted({v for v in action.locks.values() if v not in board.symbols})
        if bad_letters:
            return f"Lock letters must be in {board.symbols[0]}..{board.symbols[-1]}; got {bad_letters}."
        return None

    # --------------------------------------------------------------- apply

    def apply(self, action: Action, meta: dict[str, Any] | None = None) -> None:
        """Apply a validated action. Call validate() first."""
        assert not self.done
        if action.guess is not None:
            self._apply_guess(action.guess, meta)
            return
        board = self.board
        notes: list[str] = []
        accepted_locks: dict[str, str] = {}
        for addr, letter in action.locks.items():
            pos = self.phys_of_addr(addr)
            if pos in self.locks:
                notes.append(f"Lock on {addr} ignored: that key is already locked.")
                continue
            self.locks[pos] = letter
            self.lock_eval[pos] = self.layout[pos] == letter
            accepted_locks[addr] = letter
        press_positions = [self.phys_of_addr(a) for a in action.presses]
        results = board.sort_symbols([self.layout[p] for p in press_positions])
        rotation = self._rotate(press_positions)
        relabel = self._maybe_relabel()
        # Rotation/relabel notices are read NEXT turn, so both are rendered
        # in the post-relabel addressing.
        if rotation.get("positions") and self.config.rotation_mode in (
            "announced",
            "announced_full",
            "observer",
            "storm",
        ):
            rotation["addresses"] = sorted(self.addr_of_phys(p) for p in rotation["positions"])
            if self.config.rotation_mode == "announced_full":
                rotation["mapping"] = sorted(
                    (self.addr_of_phys(int(src)), self.addr_of_phys(dst))
                    for src, dst in rotation["permutation"].items()
                )
        self.records.append(
            {
                "turn": self.turn,
                "presses": list(action.presses),
                "press_positions": press_positions,
                "results": results,
                "locks_accepted": accepted_locks,
                "rotation": rotation,
                "relabel": relabel,
                "meta": meta or {},
            }
        )
        next_turn = self.turn + 1
        phase = "press" if next_turn <= self.config.horizon else "guess"
        self.turn = next_turn
        self.phase = phase
        self._pending_obs = Observation(
            turn=next_turn,
            horizon=self.config.horizon,
            phase=phase,
            last_presses=list(action.presses),
            last_results=results,
            rotation_notice=rotation.get("addresses"),
            rotation_mapping=rotation.get("mapping"),
            relabel_notice=relabel["notice"] if relabel else None,
            notes=notes,
        )

    def skip_turn(self, reason: str) -> None:
        """Consume the turn with no presses (repeated illegal actions)."""
        self.violations.append({"turn": self.turn, "reason": reason})
        rotation = self._rotate([])
        relabel = self._maybe_relabel()
        if rotation.get("positions") and self.config.rotation_mode in (
            "announced",
            "announced_full",
            "observer",
            "storm",
        ):
            rotation["addresses"] = sorted(self.addr_of_phys(p) for p in rotation["positions"])
            if self.config.rotation_mode == "announced_full":
                rotation["mapping"] = sorted(
                    (self.addr_of_phys(int(src)), self.addr_of_phys(dst))
                    for src, dst in rotation["permutation"].items()
                )
        self.records.append(
            {
                "turn": self.turn,
                "presses": [],
                "press_positions": [],
                "results": [],
                "locks_accepted": {},
                "rotation": rotation,
                "relabel": relabel,
                "skipped": reason,
                "meta": {},
            }
        )
        next_turn = self.turn + 1
        phase = "press" if next_turn <= self.config.horizon else "guess"
        self.turn = next_turn
        self.phase = phase
        self._pending_obs = Observation(
            turn=next_turn,
            horizon=self.config.horizon,
            phase=phase,
            last_presses=[],
            last_results=[],
            rotation_notice=rotation.get("addresses"),
            rotation_mapping=rotation.get("mapping"),
            relabel_notice=relabel["notice"] if relabel else None,
            notes=[f"Turn {next_turn - 1} was forfeited: {reason}"],
        )

    def force_zero_guess(self, reason: str) -> None:
        """Model failed to produce any valid guess: score zero."""
        self.violations.append({"turn": self.turn, "reason": f"guess failed: {reason}"})
        self._apply_guess({}, {"forced_zero": True}, allow_incomplete=True)

    # ------------------------------------------------------------ internals

    def _apply_guess(
        self, guess: dict[str, str], meta: dict[str, Any] | None, allow_incomplete: bool = False
    ) -> None:
        board = self.board
        correct = sum(
            1
            for addr in board.addresses
            if guess.get(addr) == self.layout[self.phys_of_addr(addr)]
        )
        lock_correct = sum(1 for ok in self.lock_eval.values() if ok)
        lock_wrong = len(self.lock_eval) - lock_correct
        self.done = True
        self.phase = "done"
        self.result = {
            "seed": self.seed,
            "config": self.config.to_dict(),
            "guess_turn": self.turn,
            "forced_guess": self.turn > self.config.horizon,
            "board_n": board.n,
            "correct": correct,
            "board_accuracy": correct / board.n,
            "locks": {self.addr_of_phys(p): v for p, v in self.locks.items()},
            "lock_correct": lock_correct,
            "lock_wrong": lock_wrong,
            "lock_score": lock_correct * self.config.lock_bonus + lock_wrong * self.config.lock_penalty,
            "violations": list(self.violations),
            "guess": dict(guess),
            "final_layout": {self.addr_of_phys(p): v for p, v in enumerate(self.layout)},
            "meta": meta or {},
        }
        self._pending_obs = Observation(turn=self.turn, horizon=self.config.horizon, phase="done")

    def _rotate(self, press_positions: list[int]) -> dict[str, Any]:
        cfg = self.config
        mode = cfg.rotation_mode
        board = self.board
        mapping: dict[int, int] = {}
        if mode == "none":
            return {"mode": mode, "positions": []}
        if mode == "observer":
            mapping = _derangement(self.rng, sorted(press_positions))
        elif mode == "rotor":
            mapping = dict(self._rotor)
        elif mode == "storm":
            m = cfg.rotation_size(self.turn)
            extra = [p for p in range(board.n) if p not in press_positions]
            sample = self.rng.sample(extra, min(m, len(extra))) if m > 0 else []
            union = sorted(set(press_positions) | set(sample))
            if len(union) >= 2:
                mapping = _derangement(self.rng, union)
        else:  # announced / announced_full / unannounced
            m = cfg.rotation_size(self.turn)
            if m >= 2:
                positions = sorted(self.rng.sample(range(board.n), m))
                mapping = _derangement(self.rng, positions)
        if mapping:
            old = list(self.layout)
            for src, dst in mapping.items():
                self.layout[dst] = old[src]
        return {
            "mode": mode,
            "positions": sorted(mapping.keys()),
            "permutation": {str(k): v for k, v in mapping.items()},
        }

    def _maybe_relabel(self) -> dict[str, Any] | None:
        k = self.config.relabel_every
        if k <= 0 or self.turn % k != 0 or self.turn >= self.config.horizon:
            return None
        groups = self.board.column_groups
        by_signature: dict[tuple[str, ...], list[int]] = {}
        for gi, (_, _, signature) in enumerate(groups):
            by_signature.setdefault(signature, []).append(gi)
        eligible = [gis for gis in by_signature.values() if len(gis) >= 2]
        if not eligible:
            return None
        ga, gb = self.rng.sample(self.rng.choice(sorted(eligible, key=len, reverse=True)), 2)
        label_a, idx_a, _ = groups[ga]
        label_b, idx_b, _ = groups[gb]
        for ia, ib in zip(idx_a, idx_b):
            self.addr_to_phys[ia], self.addr_to_phys[ib] = (
                self.addr_to_phys[ib],
                self.addr_to_phys[ia],
            )
        notice = (
            f"RELABEL NOTICE: the {label_a} finger column and the {label_b} finger column "
            f"have swapped addresses (every row). From now on, addresses starting {label_a}- "
            f"refer to the physical keys previously addressed {label_b}-, and vice versa. "
            f"Any rotation notice in this message already uses the NEW addressing."
        )
        return {"a": label_a, "b": label_b, "notice": notice}
