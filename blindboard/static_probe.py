"""Tier 0 static probe: finger-movement typing on a REAL QWERTY keyboard.

Single-turn tasks that isolate the addressing/translation skill (and its
inverse) with no hidden state at all:
- encode_word:   "type CRANE" -> finger tokens
- decode:        finger tokens -> "what word is this?"
- encode_answer: trivia question -> answer expressed ONLY as finger tokens

The model must know the QWERTY layout and touch-typing finger assignments
on its own — the prompt defines the token format but never the letter map.
(The single format example intentionally leaks one letter, D; accepted.)

Scoring: tokens are canonicalized (case/underscores tolerated). encode
tasks score letter-level accuracy against the canonical map plus exact-
word rate; encode_answer decodes the model's tokens and checks membership
in the accepted answer set (unknown tokens decode to '?')."""

from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from blindboard.boards import QWERTY_ADDR_TO_LETTER, QWERTY_LETTER_TO_ADDR
from blindboard.client import APIAdapter, Message
from blindboard.data import TRIVIA, WORDS
from blindboard.protocol import extract_json

SYSTEM_PROMPT = """You describe touch-typing finger movements on a standard QWERTY keyboard (letters only, ignore shift).

Every letter key is struck by a specific finger in standard touch typing. Describe a letter as a token: hand-finger-row, where hand is L or R, finger is pinky/ring/middle/index, and row is top/home/bottom. The two index fingers also each cover one extra column toward the center of the keyboard; for letters in those two center columns, append "-in" to the token (e.g. "L-index-top-in").

Example token: "L-middle-home" is the letter D.

Answer with ONLY the JSON asked for, no other text."""

ENCODE_WORD_PROMPT = (
    'Spell the word "{word}" as finger-movement tokens, one token per letter, in typing order.\n'
    'Reply with ONLY a JSON array of tokens, e.g. ["L-middle-home", "..."].'
)

DECODE_PROMPT = (
    "These finger movements type an English word (tokens in typing order):\n{tokens}\n"
    'What word is it? Reply with ONLY a JSON object: {{"word": "..."}}.'
)

ENCODE_ANSWER_PROMPT = (
    "{question}\n"
    "Answer with a single English word — but do NOT write the word. Express your answer "
    "ONLY as finger-movement tokens, one token per letter, in typing order.\n"
    'Reply with ONLY a JSON array of tokens.'
)


def canonical_token(raw: str) -> str | None:
    token = raw.strip().lower().replace("_", "-").replace(" ", "-")
    parts = [p for p in token.split("-") if p]
    if not 3 <= len(parts) <= 4:
        return None
    hand = {"l": "L", "left": "L", "r": "R", "right": "R"}.get(parts[0])
    finger = parts[1] if parts[1] in ("pinky", "ring", "middle", "index") else None
    row = parts[2] if parts[2] in ("top", "home", "bottom") else None
    inner = len(parts) == 4
    if inner and parts[3] not in ("in", "inner"):
        return None
    if hand is None or finger is None or row is None:
        return None
    return f"{hand}-{finger}-{row}" + ("-in" if inner else "")


def encode(word: str) -> list[str]:
    return [QWERTY_LETTER_TO_ADDR[ch] for ch in word.upper()]


def decode(tokens: list[str]) -> str:
    letters = []
    for raw in tokens:
        canon = canonical_token(raw) if isinstance(raw, str) else None
        letters.append(QWERTY_ADDR_TO_LETTER.get(canon or "", "?"))
    return "".join(letters)


def build_items(task: str, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"static:{task}:{seed}")
    items: list[dict[str, Any]] = []
    if task == "encode_word":
        for word in rng.sample(WORDS, min(n, len(WORDS))):
            items.append({"task": task, "word": word, "prompt": ENCODE_WORD_PROMPT.format(word=word)})
    elif task == "decode":
        for word in rng.sample(WORDS, min(n, len(WORDS))):
            tokens = json.dumps(encode(word))
            items.append({"task": task, "word": word, "prompt": DECODE_PROMPT.format(tokens=tokens)})
    elif task == "encode_answer":
        for question, answers in rng.sample(TRIVIA, min(n, len(TRIVIA))):
            items.append(
                {
                    "task": task,
                    "question": question,
                    "answers": answers,
                    "prompt": ENCODE_ANSWER_PROMPT.format(question=question),
                }
            )
    else:
        raise ValueError(f"unknown task {task!r}")
    return items


def score_item(item: dict[str, Any], reply: str) -> dict[str, Any]:
    data = extract_json(reply)
    out: dict[str, Any] = {"format_ok": False, "letter_acc": 0.0, "exact": False}
    if item["task"] == "decode":
        if isinstance(data, dict) and isinstance(data.get("word"), str):
            got = data["word"].strip().upper()
            out["format_ok"] = True
            out["got"] = got
            out["exact"] = got == item["word"]
            truth = item["word"]
            hits = sum(1 for a, b in zip(got, truth) if a == b)
            out["letter_acc"] = hits / len(truth)
        return out
    # encode tasks: expect a JSON array (be lenient: accept {"tokens": [...]}).
    tokens = data if isinstance(data, list) else None
    if tokens is None and isinstance(data, dict) and isinstance(data.get("tokens"), list):
        tokens = data["tokens"]
    if tokens is None:
        return out
    out["format_ok"] = True
    decoded = decode(tokens)
    out["decoded"] = decoded
    if item["task"] == "encode_word":
        truth = item["word"]
        hits = sum(1 for a, b in zip(decoded, truth) if a == b)
        out["letter_acc"] = hits / len(truth) if truth else 0.0
        out["exact"] = decoded == truth
        out["per_letter"] = [
            {"letter": t, "token": tok if isinstance(tok, str) else str(tok), "ok": d == t}
            for t, d, tok in zip(truth, decoded, tokens)
        ]
    else:  # encode_answer
        out["exact"] = decoded in item["answers"]
        canonical = item["answers"][0]
        hits = sum(1 for a, b in zip(decoded, canonical) if a == b)
        out["letter_acc"] = hits / len(canonical)
    return out


def run_static_probe(
    adapter: APIAdapter,
    tasks: list[str],
    n_per_task: int,
    seed: int,
    out_dir: Path | None = None,
    parallel: int = 16,
) -> dict[str, Any]:
    items = [item for task in tasks for item in build_items(task, n_per_task, seed)]

    def run_one(item: dict[str, Any]) -> dict[str, Any]:
        messages = [Message("system", SYSTEM_PROMPT), Message("user", item["prompt"])]
        reply, meta = adapter.call_with_meta(messages)
        row = {**{k: v for k, v in item.items() if k != "prompt"}, "reply": reply}
        row["reasoning"] = meta.get("reasoning", [])
        row.update(score_item(item, reply))
        return row

    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            all_rows = list(pool.map(run_one, items))
    else:
        all_rows = [run_one(item) for item in items]
    summary: dict[str, Any] = {"model": adapter.model, "effort": adapter.effort, "seed": seed}
    for task in tasks:
        rows = [r for r in all_rows if r["task"] == task]
        summary[task] = {
            "n": len(rows),
            "format_ok": round(sum(r["format_ok"] for r in rows) / len(rows), 3),
            "exact": round(sum(r["exact"] for r in rows) / len(rows), 3),
            "letter_acc": round(sum(r["letter_acc"] for r in rows) / len(rows), 3),
        }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rows.json").write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return {"summary": summary, "rows": all_rows}
