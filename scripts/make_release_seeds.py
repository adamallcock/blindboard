#!/usr/bin/env python3
"""Generate an official-eval seed list for a Blindboard release.

Official seeds are held out (written under private/, not published) so a
reported score can't be produced by a solver tuned against publicly known
seeds. The output file's SHA-256 is printed and logged to
private/SEED_HASHES.md so the hash can be published at release time and the
seed file itself revealed on the next rotation (commit-reveal): anyone can
then re-run the eval and confirm it matches the hash that was live when a
score was reported.

Format 2 (K2, 2026-07-28) additionally binds a SECRET SALT to the list.
Holding out the seeds is not enough on its own: the generator is public and
`random.Random` is a published PRNG, so anyone who learns or fingerprints a
seed replays the whole episode and answers perfectly. The salt goes into the
episode RNG key (`blindboard:<salt>:<seed>`), so a seed is worthless without
it. Published at release time: the file digest and the *commitment*
sha256(domain || salt || seeds). Revealed at the next rotation: the salt and
the seeds, at which point anyone can recompute the commitment and verify
that the score reported earlier was run against exactly this list.

Usage:
  python scripts/make_release_seeds.py --version v2
  python scripts/make_release_seeds.py --version v2 --n 100
  python scripts/make_release_seeds.py --version v2 --tiers t2_announced t3_observer
  python scripts/make_release_seeds.py --version v3 --force  # overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blindboard.env import generate_salt, salt_commitment  # noqa: E402
from blindboard.tiers import TIERS  # noqa: E402

DEFAULT_TIERS = [
    "t2_announced",
    "t3_observer",
    "t6_gauntlet",
    "t7_blitz",
    "t7_perpetual",
    "t9_numrow",
    "t9_macbook",
    "t10_relabel",
    "t11_maelstrom",
]

PRIVATE_DIR = ROOT / "private"

SEED_FORMAT = 2
# Domain separator for the seed-list commitment. Versioned, never edited.
COMMITMENT_DOMAIN = "blindboard-seedlist:v2"
COMMITMENT_SCHEME = 'sha256("blindboard-seedlist:v2" + salt + canonical_json(seeds))'

# v1's seed file was committed to git (red-team K5, 2026-07-20). It is
# burned for official scoring and must never be regenerated under that name.
BURNED_VERSIONS = {"v1"}


def make_seeds(n: int) -> list[int]:
    """n cryptographically random seeds, deduped to exactly n, sorted."""
    seeds: set[int] = set()
    while len(seeds) < n:
        seeds.add(secrets.randbelow(10**9))
    return sorted(seeds)


def canonical_seeds(seeds: dict[str, list[int]]) -> str:
    """Stable serialization of the seed map, so the commitment does not
    depend on dict order or pretty-printing."""
    return json.dumps(seeds, sort_keys=True, separators=(",", ":"))


def seedlist_commitment(salt: str, seeds: dict[str, list[int]]) -> str:
    """The publishable commitment: sha256 over the salt AND the seeds.

    Publishing this binds both halves at once. Revealing the salt and seeds
    later reproduces it exactly, so neither can be swapped after a score is
    reported.
    """
    blob = f"{COMMITMENT_DOMAIN}:{salt}:{canonical_seeds(seeds)}".encode()
    return hashlib.sha256(blob).hexdigest()


def assert_gitignored(path: Path) -> None:
    """Refuse to write a seed file that git would track (K5 remediation).
    Fails closed: if git cannot say the path is ignored (no git, or not a
    checkout), nothing is written."""
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        print(f"error: could not run 'git check-ignore' ({exc}); refusing to write {path}", file=sys.stderr)
        sys.exit(1)
    if proc.returncode == 0:
        return
    if proc.returncode == 1:
        print(
            f"error: {path} is NOT gitignored. v1 was burned exactly this way "
            "(red-team K5). Add it to .gitignore before cutting a release.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(
        f"error: 'git check-ignore' failed (rc={proc.returncode}), so {path} cannot be shown "
        "to be ignored; cut releases from a git checkout",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v2", help="release version tag, e.g. v2")
    parser.add_argument("--n", type=int, default=50, help="seeds per tier")
    parser.add_argument("--tiers", nargs="*", default=DEFAULT_TIERS, choices=sorted(TIERS))
    parser.add_argument("--force", action="store_true", help="overwrite an existing version file")
    args = parser.parse_args()

    if args.version in BURNED_VERSIONS:
        print(
            f"error: {args.version} is burned (its seed file was committed to git history); "
            "cut the next version instead",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = PRIVATE_DIR / f"official_seeds_{args.version}.json"
    hashes_path = PRIVATE_DIR / "SEED_HASHES.md"

    if out_path.exists() and not args.force:
        print(f"error: {out_path} already exists; pass --force to overwrite", file=sys.stderr)
        sys.exit(1)

    assert_gitignored(out_path)

    created = date.today().isoformat()
    salt = generate_salt()
    seeds = {tier: make_seeds(args.n) for tier in args.tiers}
    commitment = seedlist_commitment(salt, seeds)
    payload = {
        "benchmark": "blindboard",
        "format": SEED_FORMAT,
        "version": args.version,
        "created": created,
        # SECRET. Never publish this file or this field until the version is
        # superseded; official scores are only meaningful while it is held.
        "salt": salt,
        "salt_commitment": salt_commitment(salt),
        "commitment": commitment,
        "commitment_scheme": COMMITMENT_SCHEME,
        "seeds": seeds,
    }

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, sort_keys=True, indent=1), encoding="utf-8")

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()

    with hashes_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"- official_seeds_{args.version}.json (format {SEED_FORMAT}) "
            f"sha256: {digest} | seedlist commitment: {commitment} "
            f"| salt commitment: {payload['salt_commitment']} (created {created})\n"
        )

    print(f"wrote {out_path} ({len(args.tiers)} tiers x {args.n} seeds)")
    print(f"sha256:              {digest}")
    print(f"seedlist commitment: {commitment}")
    print(f"salt commitment:     {payload['salt_commitment']}")
    print(
        "\nPublish the three digests above. Do NOT publish the file: it holds\n"
        "the salt, and a published seed without a withheld salt is replayable.\n"
        "Official runs bind the salt via blindboard.env.with_salt(tier, salt)."
    )


if __name__ == "__main__":
    main()
