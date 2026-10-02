#!/usr/bin/env python3
"""Aggregate all LLM runs into a per-(model, tier) pilot table.

Merges multiple run directories (e.g. seed 0-4 and 5-19 extensions) by
reading per-episode transcript files, so cells report combined n."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.client import PRICES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/llm")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    # Cell key includes effort: medium and max runs of the same (model, tier)
    # share seed numbers and must never dedup against each other.
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    costs: dict[tuple[str, str, str], float] = defaultdict(float)
    for run_dir in sorted(Path(args.results).iterdir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = (summary["model"], summary["tier"], summary.get("effort", "medium"))
        if summary.get("cost_usd_list_price"):
            costs[key] += summary["cost_usd_list_price"]
        for ep_path in sorted(run_dir.glob("seed*.json")):
            episode = json.loads(ep_path.read_text(encoding="utf-8"))
            r = dict(episode["result"])
            r["diagnostics"] = episode["result"].get("diagnostics")
            cells[key].append(r)

    rows = []
    for (model, tier, effort), eps in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        # Dedup by seed (later runs win), then aggregate.
        by_seed = {e["seed"]: e for e in eps}
        eps = sorted(by_seed.values(), key=lambda e: e["seed"])
        n = len(eps)
        accs = [e["board_accuracy"] for e in eps]
        mean = sum(accs) / n
        sem = (sum((a - mean) ** 2 for a in accs) / max(n - 1, 1) / n) ** 0.5
        diag = [e.get("diagnostics") or {} for e in eps]
        ded = [d.get("deducible_at_guess") for d in diag if d.get("deducible_at_guess") is not None]
        dedc = [d.get("deduced_correct_in_guess") for d in diag if d.get("deduced_correct_in_guess") is not None]
        collapses = sum(1 for a in accs if a <= 0.5)
        rows.append(
            {
                "tier": tier,
                "model": model,
                "effort": effort,
                "n": n,
                "acc_mean": round(mean, 3),
                "acc_sem": round(sem, 3),
                "solve_rate": round(sum(1 for a in accs if a == 1.0) / n, 2),
                "collapse_rate_le_half": round(collapses / n, 2),
                "deducible_mean": round(sum(ded) / len(ded), 2) if ded else None,
                "deduction_gap": round((sum(ded) - sum(dedc)) / len(ded), 2) if ded and dedc else None,
                "lock_wrong_total": sum(e["lock_wrong"] for e in eps),
                "lock_score_mean": round(sum(e["lock_score"] for e in eps) / n, 2),
                "violations": sum(len(e["violations"]) for e in eps),
                "cost_usd": round(costs[(model, tier, effort)], 3),
            }
        )
    header = list(rows[0].keys()) if rows else []
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in header) + " |")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nsaved {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
