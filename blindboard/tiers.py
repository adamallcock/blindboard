"""Named difficulty tiers. Each mechanism has an off-switch so failures can
be attributed: t0 isolates deduction+memory, t1 adds pure bookkeeping,
t2 is the headline (constraint decay), t3 couples probing to disturbance,
t4 removes the announcements, t5 scales to the full QWERTY board."""

from __future__ import annotations

from blindboard.env import EnvConfig

TIERS: dict[str, EnvConfig] = {
    # Frozen board: pure unordered-chord deduction under memory load.
    "t0_frozen": EnvConfig(
        rotation_mode="none",
        horizon=12,
    ),
    # Rotations fully announced (src -> dst): lossless if you bookkeep.
    "t1_bookkeeping": EnvConfig(
        rotation_mode="announced_full",
        horizon=16,
        rotation_schedule=(3, 3, 3, 3, 3, 3, 3, 3),
    ),
    # Headline tier: announced rotation sets, decaying to zero.
    "t2_announced": EnvConfig(
        rotation_mode="announced",
        horizon=18,
        rotation_schedule=(3, 3, 3, 3, 2, 2, 2, 2),
    ),
    # Observer effect: pressing scrambles what you pressed.
    "t3_observer": EnvConfig(
        rotation_mode="observer",
        horizon=20,
    ),
    # Silent rotations: probabilistic belief decay (expert tier).
    "t4_unannounced": EnvConfig(
        rotation_mode="unannounced",
        horizon=20,
        rotation_schedule=(2,) * 10,
    ),
    # Full QWERTY board, random layout.
    "t5_qwerty": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=24,
        rotation_mode="announced",
        rotation_schedule=(4, 4, 4, 3, 3, 2, 2, 2),
    ),
    # Prior-intrusion probe: layout is QWERTY with 6 hidden swaps.
    "t5_qwerty_prior": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=24,
        rotation_mode="announced",
        rotation_schedule=(4, 4, 4, 3, 3, 2, 2, 2),
        layout_kind="near_qwerty",
        near_qwerty_swaps=6,
    ),
    # Frontier gauntlet: 26 keys, sustained rotations through turn 16,
    # 10 clean endgame turns. Hard but solvable with strong play
    # (reference-verified; see calibration).
    "t6_gauntlet": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=26,
        rotation_mode="announced",
        rotation_schedule=(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2),
    ),
    # Frontier observer: probing disturbs a 26-key board, no other motion.
    "t6_observer26": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=26,
        rotation_mode="observer",
    ),
    # Perpetual churn: rotations NEVER stop — no clean endgame to grind.
    # Solve during the storm or not at all.
    "t7_perpetual": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=24,
        rotation_mode="announced",
        rotation_schedule=(3,) * 24,
    ),
    # Blitz: rotations through turn 12, horizon 16 — reference barely fits,
    # so slow constraint play costs accuracy directly.
    "t7_blitz": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=16,
        rotation_mode="announced",
        rotation_schedule=(4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2),
    ),
    # Storm: observer + churn composed — every chord scrambles itself plus
    # extra sampled keys (one announced derangement). Extra churn decays to
    # zero by turn 12; the self-scramble never stops. Reference ceiling .892.
    # (Constant extra=3 forever gives reference .208 — luck-bound, rejected.)
    "t8_storm": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=24,
        rotation_mode="storm",
        rotation_schedule=(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0),
    ),
    # Storm hard: one extra key churns forever on top of the self-scramble.
    # Analytic ceiling .855769 (scripts/build_paper_assets.py CEILINGS); the
    # planner certificate is .8282 ± .0066 over 60 seeds
    # (results/calibration/calibration-20260718-215957.json), i.e. .0276
    # BELOW the ceiling — heuristic headroom, not seed noise. The deepest
    # tier where strong play still works.
    # (The ".799 reference ceiling" this comment used to quote predated
    # solver v3; the 2026-07-20 red-team review flagged it as a trap for
    # future maintainers. Do not quote .799 anywhere.)
    "t8_storm_hard": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=24,
        rotation_mode="storm",
        rotation_schedule=(1,) * 24,
    ),
    # --- Scale axis: bigger boards under perpetual churn ---
    "t9_numrow": EnvConfig(
        board="qwerty39",
        chord_size=4,
        horizon=30,
        rotation_mode="announced",
        rotation_schedule=(3,) * 30,
    ),
    "t9_macbook": EnvConfig(
        board="macbook47",
        chord_size=5,
        horizon=36,
        rotation_mode="announced",
        rotation_schedule=(3,) * 36,
    ),
    # --- Interference axis: two boards, alternating turns ---
    "t9_dual": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=20,  # per board; 40 press turns total
        rotation_mode="announced",
        rotation_schedule=(2,) * 20,
        dual=True,
    ),
    # --- Translation axis: perpetual + address relabeling every 6 turns.
    # Relabeling is invisible to the reference (pure model load), so the
    # reference ceiling equals the base tier's (t7_perpetual: .936).
    "t10_relabel": EnvConfig(
        board="qwerty26",
        chord_size=4,
        horizon=24,
        rotation_mode="announced",
        rotation_schedule=(3,) * 24,
        relabel_every=6,
    ),
    # --- The stack: 47 keys + perpetual churn + relabeling ---
    "t11_maelstrom": EnvConfig(
        board="macbook47",
        chord_size=5,
        horizon=36,
        rotation_mode="announced",
        rotation_schedule=(3,) * 36,
        relabel_every=6,
    ),
}


def get_tier(name: str) -> EnvConfig:
    if name not in TIERS:
        raise KeyError(f"unknown tier {name!r}; known: {sorted(TIERS)}")
    return TIERS[name]
