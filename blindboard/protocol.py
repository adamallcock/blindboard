"""Parse model replies into Actions, tolerantly but auditably.

Accepts: bare JSON, fenced ```json blocks, JSON embedded in prose (last
balanced object wins). Address strings are normalized case-insensitively
with '_'/' ' treated as '-'. Anything unparseable returns an error string
that is sent back verbatim for the single retry."""

from __future__ import annotations

import json
from typing import Any

from blindboard.boards import Board
from blindboard.env import Action


def _balanced_spans(text: str, open_ch: str, close_ch: str) -> list[str]:
    spans = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch:
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start : i + 1])
    return spans


def extract_json(text: str) -> Any | None:
    """Best JSON value in the text: the whole reply if it parses, else the
    last balanced object, else the last balanced array (so arrays nested in
    an action object never shadow the object itself)."""
    stripped = text.strip()
    candidates = [stripped] if stripped.startswith(("{", "[")) else []
    candidates.extend(reversed(_balanced_spans(text, "{", "}")))
    candidates.extend(reversed(_balanced_spans(text, "[", "]")))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_address(raw: Any, board: Board, lookup: dict[str, str]) -> str | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower().replace("_", "-").replace(" ", "-")
    return lookup.get(key)


def _address_lookup(board: Board) -> dict[str, str]:
    return {addr.lower(): addr for addr in board.addresses}


def parse_action(text: str, board: Board) -> tuple[Action | None, str | None]:
    """Returns (action, None) or (None, error)."""
    data = extract_json(text)
    if data is None or not isinstance(data, dict):
        return None, "Could not find a JSON object in your reply. Reply with ONLY the action JSON."
    lookup = _address_lookup(board)
    action = Action()
    if data.get("guess") is not None:
        guess_raw = data["guess"]
        if not isinstance(guess_raw, dict):
            return None, "'guess' must be an object mapping every key address to a letter."
        guess: dict[str, str] = {}
        for k, v in guess_raw.items():
            addr = _normalize_address(k, board, lookup)
            if addr is None:
                return None, f"Unknown key address in guess: {k!r}."
            if not isinstance(v, str):
                return None, f"Guess value for {k!r} must be a letter string."
            guess[addr] = v.strip().upper()
        action.guess = guess
        return action, None
    presses_raw = data.get("presses")
    if presses_raw is None:
        return None, "Your JSON must contain either 'presses' (a list of key addresses) or 'guess'."
    if not isinstance(presses_raw, list):
        return None, "'presses' must be a list of key address strings."
    presses: list[str] = []
    for item in presses_raw:
        addr = _normalize_address(item, board, lookup)
        if addr is None:
            return None, f"Unknown key address in presses: {item!r}. Use addresses from the rules list."
        presses.append(addr)
    action.presses = presses
    locks_raw = data.get("locks") or {}
    if not isinstance(locks_raw, dict):
        return None, "'locks' must be an object mapping key addresses to letters."
    for k, v in locks_raw.items():
        addr = _normalize_address(k, board, lookup)
        if addr is None:
            return None, f"Unknown key address in locks: {k!r}."
        if not isinstance(v, str):
            return None, f"Lock value for {k!r} must be a letter string."
        action.locks[addr] = v.strip().upper()
    return action, None
