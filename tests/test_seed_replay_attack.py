"""K2 (red-team 2026-07-27): the deployed evaluation was seed-replayable.

THE ATTACK
----------
Blindboard is procedurally generated, and the release policy leaned on that
for contamination resistance. Procedural generation defeats *answer*
memorization only if the answer cannot be recomputed. It could be. Every
draw in an episode came from `random.Random("blindboard:<seed>")` -- a
published PRNG, a published generator, and a seed pool of enumerable size.
So:

  1. Press two fixed chords on `t3_observer` and read back the two
     unordered letter sets. That pair is a fingerprint.
  2. Offline, compute the fingerprint of every seed in the candidate pool.
     Across seeds 0..59 (calibration) and across a 50-seed official-style
     pool, all fingerprints are UNIQUE -- two chords name the episode.
  3. Replay the identified seed locally, read the layout straight out of
     the simulator, and submit it. Score: 1.000, above the .875 analytic
     ceiling for honest play.

Holding the seed list back does not fix this: a fingerprint identifies the
episode, and a leaked, guessed, or brute-forced seed hands over the whole
trajectory.

THE FIX (implemented; these tests gate it)
------------------------------------------
`EnvConfig.salt` / `Episode(..., salt=...)` key the episode RNG as
`blindboard:<salt>:<seed>` (`blindboard.env.episode_seed_key`). Official
evaluation binds a high-entropy salt (`blindboard.env.generate_salt`) that
is never published with the seed list -- only its commitment is. Without
the salt, a precomputed fingerprint map is worthless and the replay attack
falls back to chance.

The legacy (empty-salt) tests below are kept deliberately: they assert the
vulnerability that existed, so the hardened assertions can never be
satisfied vacuously and the regression cannot silently return.
"""

from __future__ import annotations

import glob
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from blindboard.dual import make_episode
from blindboard.env import (
    Action,
    Episode,
    EnvConfig,
    episode_seed_key,
    generate_salt,
    salt_commitment,
    with_salt,
)
from blindboard.tiers import TIERS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_release_seeds as mrs  # noqa: E402


def _in_git_checkout() -> bool:
    try:
        proc = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
                              capture_output=True, text=True, check=False)
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "true"

# The two chords from the confirmed finding. Both are legal t3_observer
# actions (fingers12 board, chord size 3); the second re-presses two keys
# the first one scrambled, which is what makes the pair so discriminating.
CHORD_A = ["L-pinky-top", "L-pinky-home", "L-pinky-bottom"]
CHORD_B = ["L-pinky-top", "L-pinky-home", "L-ring-top"]

CALIBRATION_SEEDS = list(range(60))

# Fixed salts, not freshly generated ones: these tests must be
# deterministic. Real official salts come from generate_salt().
PROBE_SALTS = [
    "9f" * 32,
    "a3c1" * 16,
    "0123456789abcdef" * 4,
]

Fingerprint = tuple[tuple[str, ...], ...]


def _official_style_pool(n: int = 50) -> list[int]:
    """A pool shaped like a real release cut: n distinct 9-digit seeds.

    `make_release_seeds.make_seeds` uses `secrets`, so it cannot be
    reproduced in a test; this mirrors its shape deterministically.
    """
    rng = random.Random("official-style-pool")
    seeds: set[int] = set()
    while len(seeds) < n:
        seeds.add(rng.randrange(10**9))
    return sorted(seeds)


def _probe(seed: int, salt: str = "") -> tuple[Fingerprint, Episode]:
    """Press the two chords; return the fingerprint and the live episode."""
    ep = Episode(TIERS["t3_observer"], seed, salt=salt)
    results: list[tuple[str, ...]] = []
    for chord in (CHORD_A, CHORD_B):
        action = Action(presses=chord)
        assert ep.validate(action) is None
        ep.apply(action)
        results.append(tuple(ep.records[-1]["results"]))
    return tuple(results), ep


def _fingerprint_map(seeds: list[int], salt: str = "") -> dict[Fingerprint, list[int]]:
    table: dict[Fingerprint, list[int]] = {}
    for seed in seeds:
        table.setdefault(_probe(seed, salt)[0], []).append(seed)
    return table


def _replayed_guess(seed: int, salt: str = "") -> dict[str, str]:
    """What the attacker submits: the layout of a locally replayed episode,
    read out of the simulator after the same two chords."""
    _, sim = _probe(seed, salt)
    return {addr: sim.layout[sim.phys_of_addr(addr)] for addr in sim.board.addresses}


def _run_attack(
    seeds: list[int],
    table: dict[Fingerprint, list[int]],
    salt: str = "",
    replay_salt: str = "",
) -> dict:
    """Fingerprint each live episode, look it up, guess the replayed layout.

    `salt` is what the environment runs under; `replay_salt` is what the
    attacker replays under. They differ for everyone but the salt holder --
    that gap is the whole defense.
    """
    identified = correct = total = 0
    for seed in seeds:
        fingerprint, ep = _probe(seed, salt)
        candidates = table.get(fingerprint, [])
        if candidates == [seed]:
            identified += 1
        # An attacker with no match still has to answer; unmatched episodes
        # fall back to the first pool entry, i.e. an uninformed permutation.
        guess = _replayed_guess(candidates[0] if candidates else seeds[0], replay_salt)
        assert ep.validate(Action(guess=guess)) is None
        ep.apply(Action(guess=guess))
        correct += ep.result["correct"]
        total += ep.board.n
    return {
        "identified": identified,
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }


# --------------------------------------------------------------------------
# The vulnerability, as it existed on the unsalted path. These assertions
# must keep passing: they are the witness that the hardened assertions below
# are testing something real.
# --------------------------------------------------------------------------


def test_legacy_two_chords_uniquely_fingerprint_the_calibration_pool() -> None:
    """Two chords name the episode: 60 seeds -> 60 distinct fingerprints."""
    table = _fingerprint_map(CALIBRATION_SEEDS)
    collisions = {fp: seeds for fp, seeds in table.items() if len(seeds) > 1}
    assert not collisions, f"expected the documented hole, got collisions: {collisions}"
    assert len(table) == len(CALIBRATION_SEEDS)


def test_legacy_two_chords_uniquely_fingerprint_an_official_style_pool() -> None:
    """The hole is not an artifact of small integer seeds: a 50-seed pool of
    9-digit release-shaped seeds is just as identifiable."""
    pool = _official_style_pool()
    table = _fingerprint_map(pool)
    assert len(table) == len(pool)


def test_legacy_seed_replay_attack_scores_perfectly() -> None:
    """THE KILL SHOT. Identify by fingerprint, replay the public PRNG, submit
    the simulator's layout: 720/720 keys, accuracy 1.000, against a .875
    analytic ceiling for honest play. Fix: bind a secret salt (see below)."""
    table = _fingerprint_map(CALIBRATION_SEEDS)
    outcome = _run_attack(CALIBRATION_SEEDS, table)
    assert outcome["identified"] == len(CALIBRATION_SEEDS)
    assert outcome["correct"] == outcome["total"] == 720
    assert outcome["accuracy"] == 1.0
    assert outcome["accuracy"] > 0.875, "the attack must exceed the analytic ceiling"


# --------------------------------------------------------------------------
# The vulnerability, closed on the hardened path.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("salt", PROBE_SALTS)
def test_salt_breaks_the_precomputed_fingerprint_map(salt: str) -> None:
    """The attacker's offline map is built from public seeds on the public
    PRNG. Under a withheld salt, not one of its 60 entries identifies the
    episode it is looking at."""
    public_table = _fingerprint_map(CALIBRATION_SEEDS)
    for seed in CALIBRATION_SEEDS:
        fingerprint, _ = _probe(seed, salt)
        assert public_table.get(fingerprint) != [seed]


@pytest.mark.parametrize("salt", PROBE_SALTS)
def test_salt_reduces_the_replay_attack_to_chance(salt: str) -> None:
    """End-to-end: the same attack against salted episodes scores at chance
    (1/12 per key on a 12-key board), not 1.000."""
    public_table = _fingerprint_map(CALIBRATION_SEEDS)
    outcome = _run_attack(CALIBRATION_SEEDS, public_table, salt=salt)
    assert outcome["identified"] == 0
    assert outcome["accuracy"] < 0.25, outcome
    assert outcome["accuracy"] < 0.875, "must sit far below the analytic ceiling"


@pytest.mark.parametrize("salt", PROBE_SALTS)
def test_official_style_pool_is_not_identifiable_under_salt(salt: str) -> None:
    """Finding (b) closed: enumerating the whole official seed list no longer
    buys the attacker the episode."""
    pool = _official_style_pool()
    public_table = _fingerprint_map(pool)
    for seed in pool:
        fingerprint, _ = _probe(seed, salt)
        assert public_table.get(fingerprint) != [seed]


def test_the_defense_is_the_secret_not_the_obscurity() -> None:
    """Salting does not reduce what a chord leaks in-episode: the salted pool
    is still internally unique. What it removes is the OFFLINE precomputation
    -- the map can only be built by whoever holds the salt. Recorded so a
    future maintainer does not mistake the salt for an information-hiding
    measure and 'simplify' it away."""
    salt = PROBE_SALTS[0]
    salted_table = _fingerprint_map(CALIBRATION_SEEDS, salt=salt)
    assert len(salted_table) == len(CALIBRATION_SEEDS)
    holder = _run_attack(CALIBRATION_SEEDS, salted_table, salt=salt, replay_salt=salt)
    assert holder["identified"] == len(CALIBRATION_SEEDS)
    assert holder["accuracy"] == 1.0, "the salt holder can still replay, by design"


# --------------------------------------------------------------------------
# The salt must not disturb anything that already exists.
# --------------------------------------------------------------------------


def test_empty_salt_reproduces_the_legacy_rng_key() -> None:
    for seed in (0, 7, 999_331):
        assert episode_seed_key("", seed) == f"blindboard:{seed}"
        assert episode_seed_key("deadbeef", seed) == f"blindboard:deadbeef:{seed}"
    # And the key is actually what drives the layout.
    for tier_name in ("t2_announced", "t3_observer", "t11_maelstrom"):
        config = TIERS[tier_name]
        ep = Episode(config, seed=7)
        expected = config.board_obj().random_layout(random.Random("blindboard:7"))
        assert ep.initial_layout == expected, tier_name


def test_stored_artifacts_replay_byte_identically() -> None:
    """Every stored t11_maelstrom episode must replay to the same trajectory
    under the salted engine with the default empty salt: same press
    positions, same chord results, same rotations (sets, permutations and
    announced addresses), same relabels, same final layout."""
    checked = 0
    for run_dir in sorted(glob.glob(str(ROOT / "results/**/*t11_maelstrom*"), recursive=True)):
        for path in sorted(Path(run_dir).glob("seed*.json")):
            stored = json.loads(path.read_text())
            ep = Episode(TIERS["t11_maelstrom"], stored["result"]["seed"])
            for record in stored["records"]:
                if "skipped" in record:
                    ep.skip_turn(record["skipped"])
                else:
                    ep.apply(
                        Action(
                            presses=record["presses"],
                            locks=record.get("locks_accepted") or {},
                        )
                    )
                fresh = ep.records[-1]
                for key in ("turn", "press_positions", "results", "rotation", "relabel"):
                    assert fresh.get(key) == record.get(key), f"{path.name} turn {record['turn']} {key}"
            final = {ep.addr_of_phys(p): letter for p, letter in enumerate(ep.layout)}
            assert final == stored["result"]["final_layout"], path.name
            checked += 1
    assert checked >= 25, f"expected the stored maelstrom runs to be present, saw {checked}"


def test_default_config_is_unsalted_and_serializes_unchanged() -> None:
    """The salt is opt-in, and an unsalted config's serialized form is
    exactly what pre-hardening artifacts carry (no new keys)."""
    assert EnvConfig().salt == ""
    artifacts = sorted(glob.glob(str(ROOT / "results/llm/*t11_maelstrom*/seed*.json")))
    assert artifacts, "expected stored maelstrom artifacts to compare against"
    stored = json.loads(Path(artifacts[0]).read_text())
    assert TIERS["t11_maelstrom"].to_dict() == stored["result"]["config"]


def test_salt_never_appears_in_serialized_results() -> None:
    """A run artifact must be publishable while its salt is still live: it
    carries the commitment, never the salt."""
    salt = PROBE_SALTS[0]
    config = with_salt(TIERS["t0_frozen"], salt)
    ep = Episode(config, seed=3)
    ep.force_zero_guess("test")
    blob = json.dumps(ep.result, sort_keys=True)
    assert salt not in blob
    assert "salt" not in ep.result["config"]
    assert ep.result["config"]["salt_commitment"] == salt_commitment(salt)
    # ... and everything else in the config is untouched by the salt.
    assert {k: v for k, v in ep.result["config"].items() if k != "salt_commitment"} == TIERS[
        "t0_frozen"
    ].to_dict()


def test_distinct_salts_give_distinct_episodes() -> None:
    config = TIERS["t2_announced"]
    layouts = {
        salt: tuple(Episode(config, seed=11, salt=salt).initial_layout)
        for salt in ["", *PROBE_SALTS]
    }
    assert len(set(layouts.values())) == len(layouts)
    # Same salt, same episode: salting stays deterministic and reproducible.
    a = Episode(config, seed=11, salt=PROBE_SALTS[0]).initial_layout
    b = Episode(with_salt(config, PROBE_SALTS[0]), seed=11).initial_layout
    assert a == b


def test_explicit_salt_argument_overrides_the_config() -> None:
    config = with_salt(TIERS["t0_frozen"], PROBE_SALTS[0])
    assert Episode(config, seed=1).salt == PROBE_SALTS[0]
    assert Episode(config, seed=1, salt="").salt == ""
    assert Episode(config, seed=1, salt="").initial_layout == Episode(
        TIERS["t0_frozen"], seed=1
    ).initial_layout


def test_salt_containing_the_separator_is_rejected() -> None:
    """Otherwise (salt, seed) -> key is not injective: a salt ending in
    ':<digits>' could impersonate another episode's stream."""
    with pytest.raises(ValueError, match="salt"):
        Episode(TIERS["t0_frozen"], seed=1, salt="abc:1")
    with pytest.raises(ValueError, match="salt"):
        Episode(with_salt(TIERS["t0_frozen"], "abc:1"), seed=1)


def test_dual_episodes_inherit_the_config_salt() -> None:
    """The interference tier builds two sub-episodes internally; both must be
    salted or half the official board stays replayable."""
    salt = PROBE_SALTS[0]
    dual = make_episode(with_salt(TIERS["t9_dual"], salt), seed=5)
    assert dual.a.salt == salt and dual.b.salt == salt
    plain = make_episode(TIERS["t9_dual"], seed=5)
    assert dual.a.initial_layout != plain.a.initial_layout
    assert dual.b.initial_layout != plain.b.initial_layout


# --------------------------------------------------------------------------
# Release tooling: the salt has to survive the cut without being published.
# --------------------------------------------------------------------------


def test_generate_salt_is_high_entropy_hex() -> None:
    salts = {generate_salt() for _ in range(16)}
    assert len(salts) == 16
    for salt in salts:
        assert len(salt) == 64 and ":" not in salt
        int(salt, 16)  # hex only, so it is shell/JSON/env safe
    assert len(generate_salt(16)) == 32
    with pytest.raises(ValueError):
        generate_salt(8)  # too little entropy for an official cut


def test_salt_commitment_is_stable_and_hides_the_salt() -> None:
    salt = generate_salt()
    commitment = salt_commitment(salt)
    assert commitment == salt_commitment(salt)
    assert commitment != salt_commitment(generate_salt())
    assert salt not in commitment and len(commitment) == 64


def test_seedlist_commitment_binds_both_the_salt_and_the_seeds() -> None:
    """The published digest must pin the pair. Swapping either half after a
    score is reported has to be detectable at reveal time."""
    seeds = {"t3_observer": [1, 2, 3], "t7_blitz": [9, 8]}
    salt = "a" * 64
    base = mrs.seedlist_commitment(salt, seeds)
    assert base == mrs.seedlist_commitment(salt, {"t7_blitz": [9, 8], "t3_observer": [1, 2, 3]})
    assert base != mrs.seedlist_commitment("b" * 64, seeds)
    assert base != mrs.seedlist_commitment(salt, {"t3_observer": [1, 2, 4], "t7_blitz": [9, 8]})


@pytest.mark.skipif(not _in_git_checkout(), reason="needs a git checkout: git decides what is ignored")
def test_release_cut_refuses_the_burned_v1_and_untracked_paths() -> None:
    assert "v1" in mrs.BURNED_VERSIONS
    assert mrs.SEED_FORMAT == 2
    # The K5 gate: a seed path git would track must abort the cut.
    mrs.assert_gitignored(mrs.PRIVATE_DIR / "official_seeds_v9.json")
    with pytest.raises(SystemExit):
        mrs.assert_gitignored(ROOT / "README.md")


def test_release_cut_fails_closed_where_git_cannot_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a git checkout nothing can show a seed path is ignored, so the
    cut refuses to write it rather than warning and carrying on."""
    monkeypatch.setattr(mrs, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        mrs.assert_gitignored(tmp_path / "private" / "official_seeds_v9.json")
