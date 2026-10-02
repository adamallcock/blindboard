# Blindboard release policy

Blindboard is procedurally generated: every episode is a fresh board layout
and rotation schedule keyed by (tier, salt, seed). That property is what a
held-out seed list buys — it turns "don't leak the eval set" from an
unenforceable request into something verifiable after the fact.

**Procedural generation alone does not defeat answer memorization.** It
removes the fixed answer key; it does not remove the answer. The generator
is public, `random.Random` is a published PRNG, and a seed pool is
enumerable, so anyone who can name the seed can recompute the episode
exactly. The salt below is what turns "no fixed answer key" into "no
recoverable answer". See [The replay attack](#the-replay-attack-k2).

## Public / private split

**Public (this repo, always):**
- Environment (`blindboard/`): board definitions, rotation modes, scoring,
  the turn loop.
- Tier definitions (`blindboard/tiers.py`) and their parameters.
- Reference solver (`blindboard/agents.py`) and the naive/random floors.
- Diagnostics (deducible-at-guess replay, known-press/zero-info tracking).
- Tests, pilot run artifacts, and calibration tables (`results/calibration`,
  `docs/pilot-*.md`).

**Private until rotation — official-eval seed lists AND their salt.**
`private/official_seeds_<version>.json` (built by
`scripts/make_release_seeds.py`) holds the seeds a reported score must be
run against to count as official, plus the secret salt those episodes run
under. Only digests are published (`private/SEED_HASHES.md`) at release
time: commit-reveal. The file itself is revealed on the *next* rotation, so
anyone can hash it, match it against the published digest, and confirm a
past score was computed on the seeds — and under the salt — that were live
then, not cherry-picked after the fact.

**Private until release — canary tier variants.**
The engine already implements prior-intrusion (near-QWERTY layouts,
`layout_kind="near_qwerty"`) and rotor (`rotation_mode="rotor"`) mechanics,
but neither ships as a published, run tier — `tiers.py` has no `rotor`
entry, and `t5_qwerty_prior` is built but excluded from the calibrated
headline set. At release, specific hidden parameters (swap count, rotor
subset and size, which base tier they're layered on) get chosen and
instantiated as new tiers whose exact configuration was never public. A
model trained against the visible mechanics rather than the general task
has no way to have prepared for these.

## The replay attack (K2)

Found by the 2026-07-27 red-team review, confirmed by the maintainer, fixed
2026-07-28. It is written up here rather than filed away because the false
belief that produced it — "procedural generation makes contamination a
non-issue" — is the natural one to fall back into.

**The attack.** On `t3_observer` (12 keys, chord size 3), press two fixed
chords and read back the two unordered letter sets:

```
turn 1:  L-pinky-top, L-pinky-home, L-pinky-bottom
turn 2:  L-pinky-top, L-pinky-home, L-ring-top
```

The second chord re-presses two keys the first one scrambled, which makes
the pair extremely discriminating. That pair of results is a fingerprint,
and the fingerprint is *unique* across all 60 calibration seeds and all 50
official-v1 seeds — two chords name the episode. From the name, everything
else follows: `random.Random("blindboard:<seed>")` is a published PRNG, so
the attacker replays the episode locally, reads the layout out of their own
simulator, and submits it. Measured score: **240/240 = 1.000**, against the
`.875` analytic ceiling for honest play on that tier.

**Why procedural generation does not stop it.** Procedural generation
removes the fixed answer *key*; it does not remove the answer. When the
generator is public, the PRNG is public, and the candidate pool is
enumerable, the answer is *recomputable* — and a recomputable answer is
memorizable in the only sense that matters, since the model (or its
harness) can derive it at eval time rather than recall it. Holding the seed
list back does not save this either: the fingerprint identifies the episode
without being told the seed, and a leaked or brute-forced seed hands over
the entire trajectory.

**The fix: a secret per-episode salt.** `EnvConfig.salt` (default `""`)
enters the episode RNG key:

```
random.Random(episode_seed_key(salt, seed))
    salt == ""  ->  "blindboard:<seed>"          # legacy, byte-identical
    salt != ""  ->  "blindboard:<salt>:<seed>"   # official
```

The empty default reproduces the pre-hardening key exactly, so every stored
artifact and every published calibration number replays unchanged (gated by
`tests/test_seed_replay_attack.py::test_stored_artifacts_replay_byte_identically`).
Official evaluation binds a 32-byte `blindboard.env.generate_salt()` value
that is never published with the seed list. Without it, a precomputed
fingerprint map matches nothing and the attack drops to chance (~`.083` per
key on a 12-key board).

Two properties worth stating explicitly, because they are easy to
misremember:

- The salt is **not** an information-hiding measure. A salted pool is still
  internally unique — the chords leak exactly as much as before. What the
  salt removes is *offline precomputation*: the map can only be built by
  whoever holds the salt.
- The salt is **never serialized**. `EnvConfig.to_dict()` emits only
  `salt_commitment`, so a run artifact stays publishable while its salt is
  still live, and after reveal anyone can verify which salt it ran under.

**Rule: official scoring runs behind an unrevealed salt.** A score is not
official if it was produced with `salt == ""`, with a salt that had already
been published, or under a salt whose commitment was not published *before*
the run. Everything else — calibration, pilots, robustness runs, canaries,
anything in `results/` — stays unsalted and fully replayable on purpose:
that reproducibility is the point of those artifacts.

## Rotation procedure

1. Cut a new version: `scripts/make_release_seeds.py --version vN`. Writes
   `private/official_seeds_vN.json` (seeds **and** a fresh salt; the file
   is gitignored and the script refuses to write a path git would track)
   and appends three digests to `private/SEED_HASHES.md`: the file's
   SHA-256, the seed-list commitment
   `sha256("blindboard-seedlist:v2" + salt + canonical_json(seeds))`, and
   the salt commitment.
2. Publish the digests (`SEED_HASHES.md` is public metadata; the file it
   describes is not). Official scores from here on are run against vN's
   seeds, bound to vN's salt via `blindboard.env.with_salt(tier, salt)`.
3. On the *next* rotation (vN+1's cut), reveal vN's file — seeds and salt
   together. The commitment binds both halves at once, so neither can be
   swapped after a score was reported; revealing it is a proof, not a trust
   request.
4. Only the current version's seeds and salt stay secret. Every prior
   version is fully public once superseded — layouts, rotations, everything
   needed to replay it.

Canary tiers rotate on their own, slower clock: new hidden parameters
whenever a release needs a fresh detection surface, independent of
seed-file rotation.

## Published commitments

The digests of every official seed list, published when the list is cut.
A list's file, its seeds and salt together, is revealed at the next
rotation. Anyone can then hash the file and recompute both commitments.

| Version | Status | Cut | Tiers × seeds |
|---|---|---|---|
| v2 | live: official scores run on it | 2026-10-02 | 9 × 50 |
| v1 | burned; never used for scoring (see the incident below) | 2026-07 | — |

**v2** (format 2) covers `t2_announced`, `t3_observer`, `t6_gauntlet`,
`t7_blitz`, `t7_perpetual`, `t9_numrow`, `t9_macbook`, `t10_relabel`, and
`t11_maelstrom`:

- File, `official_seeds_v2.json`, SHA-256:
  `389f24a17f82231db272d17c44e1d356d9b3e06b1a47c9a64bdf14942735ed60`
- Seed-list commitment,
  `sha256("blindboard-seedlist:v2" + salt + canonical_json(seeds))`:
  `a11bce72462f40a9c71cae1a13cc3e12c7574ed474ad4a581d89afdcc24c75d0`
- Salt commitment, `blindboard.env.salt_commitment(salt)`:
  `11355f71350c73f25a0c9237759d4d4e0548d1a1612c83e46ce24351c6170273`

## Threat model

- **Memorization contamination** (a model has seen this exact board or
  answer before): procedural generation handles the *recall* half — there
  is no fixed answer key, so a leaked transcript contaminates one (tier,
  seed) sample, not the tier. It does nothing about the *recompute* half:
  with a public generator and PRNG, the answer for any named seed can be
  derived at eval time. That half is handled by the official salt (see
  [The replay attack](#the-replay-attack-k2)), which is the only reason a
  held-out seed list is worth anything.
- **Generator-training** (a model, or its training pipeline, was tuned
  against Blindboard's visible generator, reference solver, or published
  tiers specifically, rather than the general task): detectable, not
  preventable. The repo, generator, and reference solver are public by
  design, so this can't be blocked directly. What's detectable is anomalous
  performance on canary tiers (unseen parameterizations a model could only
  do well on by having modeled the generator) plus anomalies in the
  effort/token-vs-accuracy curve set during calibration — a model that gets
  suspiciously cheap or short specifically on Blindboard-shaped tasks.
- **Mechanics overfitting** (a solver tuned to the specific announced /
  observer / unannounced / rotation-schedule shapes that are public, rather
  than the underlying state-tracking-under-interference skill): same
  detector as generator-training. Canary tiers exercise structurally
  related but never-published mechanics (prior-intrusion, rotor), so a
  solver that only learned the public shapes shows a gap between headline
  and canary performance that a solver of the general skill would not.

## INCIDENT (2026-07-20, red-team K5): v1 seed list committed — burned

`private/official_seeds_v1.json` was tracked in git from commit 04ca5fc.
Anyone with repo history holds the seeds, so the v1 commit-reveal protects
nothing: **v1 must never be used for an official score.** Remediation: cut
v2 through a clean process (file never tracked; .gitignore now blocks it;
publish digest only) and add a release gate that fails if any
official-seeds blob is reachable via `git rev-list --objects --all`.

K2 makes v1 burned twice over: it is format 1, so it carries no salt, and
every one of its 50 seeds is fingerprintable in two chords. Held-out seeds
without a salt were never sufficient. `make_release_seeds.py` now refuses
`--version v1` outright and emits format 2 only.
