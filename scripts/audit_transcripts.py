#!/usr/bin/env python3
"""Audit saved transcripts: re-parse every assistant reply and look for
anything the harness could have silently mangled — oversized chords,
duplicate presses, multiple JSON objects per reply (last-object-wins
risk), guess+presses in one action, non-action JSON.

Run over results/llm/*/seed*.json; exits nonzero if any reply that was
accepted at runtime looks illegal on re-parse."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindboard.boards import BOARDS
from blindboard.protocol import _balanced_spans, parse_action
from blindboard.tiers import TIERS


def main() -> None:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/llm")
    stats: Counter[str] = Counter()
    problems: list[str] = []
    for run_dir in sorted(results_dir.iterdir()):
        if not (run_dir / "summary.json").exists():
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        config = TIERS[summary["tier"]]
        board = config.board_obj()
        for ep_path in sorted(run_dir.glob("seed*.json")):
            data = json.loads(ep_path.read_text())
            transcript = data.get("transcript", [])
            for i, m in enumerate(transcript):
                if m["role"] != "assistant":
                    continue
                # A reply followed by an INVALID ACTION correction was
                # rejected at runtime — the harness handled it; auditing it
                # as "accepted" would be a false positive. Count separately.
                nxt = transcript[i + 1] if i + 1 < len(transcript) else None
                if nxt and nxt["role"] == "user" and nxt["text"].startswith("INVALID ACTION:"):
                    stats["runtime_rejected_replies"] += 1
                    continue
                text = m["text"]
                stats["assistant_replies"] += 1
                objects = _balanced_spans(text, "{", "}")
                if len(objects) > 1:
                    stats["multi_object_replies"] += 1
                action, err = parse_action(text, board)
                if err is not None:
                    stats["unparseable_or_illegal_addr"] += 1
                    continue
                raw = None
                for obj in reversed(objects):
                    try:
                        raw = json.loads(obj)
                        break
                    except json.JSONDecodeError:
                        continue
                if isinstance(raw, dict):
                    if raw.get("guess") is not None and raw.get("presses"):
                        stats["guess_plus_presses"] += 1
                    extra = set(raw) - {"presses", "locks", "guess"}
                    if extra:
                        stats["extra_top_level_keys"] += 1
                if action.guess is None:
                    n = len(action.presses)
                    stats[f"chord_size_{n}"] += 1
                    if n != config.chord_size:
                        problems.append(f"{ep_path.name}: chord of {n} in accepted reply")
                    if len(set(action.presses)) != n:
                        problems.append(f"{ep_path.name}: duplicate press in accepted reply")
                else:
                    stats["guesses"] += 1
                    if set(action.guess) != set(board.addresses):
                        problems.append(f"{ep_path.name}: guess not covering board")
    print(json.dumps(dict(sorted(stats.items())), indent=1))
    if problems:
        print("\nPROBLEMS:")
        print("\n".join(problems))
        sys.exit(1)
    print("\nNo accepted reply violates the rules on re-parse.")


if __name__ == "__main__":
    main()
