"""QWERTY finger map correctness, encode/decode roundtrip, reply parsing."""

from __future__ import annotations

from blindboard.boards import BOARDS, QWERTY_LETTER_TO_ADDR
from blindboard.protocol import parse_action
from blindboard.static_probe import canonical_token, decode, encode, score_item

BOARD12 = BOARDS["fingers12"]

CANONICAL_SPOT_CHECKS = {
    "Q": "L-pinky-top",
    "A": "L-pinky-home",
    "Z": "L-pinky-bottom",
    "W": "L-ring-top",
    "E": "L-middle-top",
    "R": "L-index-top",
    "T": "L-index-top-in",
    "G": "L-index-home-in",
    "B": "L-index-bottom-in",
    "Y": "R-index-top-in",
    "H": "R-index-home-in",
    "N": "R-index-bottom-in",
    "U": "R-index-top",
    "J": "R-index-home",
    "M": "R-index-bottom",
    "I": "R-middle-top",
    "K": "R-middle-home",
    "O": "R-ring-top",
    "L": "R-ring-home",
    "P": "R-pinky-top",
}


def test_qwerty_finger_map() -> None:
    assert len(QWERTY_LETTER_TO_ADDR) == 26
    for letter, addr in CANONICAL_SPOT_CHECKS.items():
        assert QWERTY_LETTER_TO_ADDR[letter] == addr, letter


def test_encode_decode_roundtrip_all_letters() -> None:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    assert decode(encode(alphabet)) == alphabet


def test_token_normalization() -> None:
    assert canonical_token("l_index_top_in") == "L-index-top-in"
    assert canonical_token(" Right-Middle-Home ") == "R-middle-home"
    assert canonical_token("left index top inner") == "L-index-top-in"
    assert canonical_token("L-thumb-top") is None
    assert canonical_token("nonsense") is None


def test_score_encode_word() -> None:
    item = {"task": "encode_word", "word": "CAB"}
    reply = '["L-middle-bottom", "L-pinky-home", "L-index-bottom-in"]'
    out = score_item(item, reply)
    assert out["format_ok"] and out["exact"] and out["letter_acc"] == 1.0
    wrong = '["L-middle-bottom", "L-pinky-home", "R-index-bottom"]'  # B -> M
    out = score_item(item, wrong)
    assert not out["exact"] and abs(out["letter_acc"] - 2 / 3) < 1e-9


def test_score_encode_answer_accepts_any_listed_answer() -> None:
    item = {"task": "encode_answer", "answers": ["PUPPY", "PUP"]}
    reply = str(encode("PUP")).replace("'", '"')
    assert score_item(item, reply)["exact"]


def test_parse_action_variants() -> None:
    text = 'Sure! Here is my move:\n```json\n{"presses": ["l-ring-top", "L_index_home", "L-pinky-bottom"]}\n```'
    action, err = parse_action(text, BOARD12)
    assert err is None
    assert action.presses == ["L-ring-top", "L-index-home", "L-pinky-bottom"]

    action, err = parse_action('{"guess": {"L-ring-top": "a"}}', BOARD12)
    assert err is None and action.guess == {"L-ring-top": "A"}

    action, err = parse_action("I think I'll press the ring finger", BOARD12)
    assert action is None and err is not None

    action, err = parse_action('{"presses": ["L-thumb-top", "L-ring-top", "L-pinky-top"]}', BOARD12)
    assert action is None and "Unknown key address" in (err or "")

    # Reasoning preamble with a stray brace in a string, then the real JSON.
    tricky = 'note: "{" is fine. {"presses": ["L-ring-top", "L-ring-home", "L-ring-bottom"], "locks": {"L-middle-top": "c"}}'
    action, err = parse_action(tricky, BOARD12)
    assert err is None
    assert action.locks == {"L-middle-top": "C"}
