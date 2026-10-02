#!/usr/bin/env python3
"""Rebuild every table and number in the Blindboard paper from this release.

    python3 rebuild.py

1. Checks every file listed in SHA256SUMS.
2. Runs scripts/build_paper_assets.py --skip-gates into a temporary
   directory, regenerating every table, number, and CSV from results/.
3. Compares each regenerated file byte for byte with paper/tables/.

Python 3.11 or later, standard library only, no network. readiness_gates.tex
is not regenerated here: it records a run of the test suite, which runs on
its own (python3 -m pytest -q). Exits nonzero on any mismatch.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "paper" / "tables"
NOT_REBUILT = {"readiness_gates.tex"}


def check_sums() -> tuple[int, list[str]]:
    problems: list[str] = []
    lines = (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"changed: {rel}")
    return len(lines), problems


def main() -> int:
    count, problems = check_sums()
    if problems:
        print(f"SHA256SUMS: {len(problems)} of {count} files do not match:")
        for line in problems[:20]:
            print(f"  {line}")
        return 1
    print(f"SHA256SUMS: all {count} files match.")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        proc = subprocess.run(
            [sys.executable, "scripts/build_paper_assets.py", "--skip-gates", "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print("The asset build failed:")
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            return 1
        shipped = sorted(p.name for p in TABLES.iterdir() if p.is_file() and p.name not in NOT_REBUILT)
        rebuilt = sorted(p.name for p in out.iterdir() if p.is_file())
        differing = [
            name for name in shipped
            if not (out / name).is_file() or (out / name).read_bytes() != (TABLES / name).read_bytes()
        ]
        unexpected = sorted(set(rebuilt) - set(shipped))
    if differing or unexpected:
        for name in differing:
            print(f"differs from the shipped copy: paper/tables/{name}")
        for name in unexpected:
            print(f"rebuilt but not shipped: {name}")
        return 1
    print(f"REBUILD OK: all {len(shipped)} generated tables, number files, and CSVs match paper/tables/ byte for byte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
