"""Physical boards and finger addressing.

A board is a fixed set of physical positions, each with a canonical
finger-address string (`L-ring-top`, `R-index-home-in`, ...). The hidden
layout assigns one letter per position. Addresses are the ONLY way the
agent can refer to positions; results come back as bare letters.

Boards:
- fingers12: left hand only, 4 fingers x 3 rows. Every finger owns exactly
  one column, so (finger, row) is unambiguous. Symbols A..L.
- qwerty26: full 3-row letter block with authentic touch-typing columns,
  including the index-finger inner-reach columns (`-in`). Symbols A..Z.
  Supports `near_qwerty` layouts (QWERTY with k random swaps) for the
  prior-intrusion tier.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

FINGERS = ["pinky", "ring", "middle", "index"]
ROWS = ["num", "top", "home", "bottom"]

# Fixed, published symbol order for reporting chord results (alphabetical
# for letter-only boards; symbols/digits precede letters on big boards).
SYMBOL_ORDER = "`-=[]\\;',./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The full MacBook main block, letters + number row + punctuation, as
# touch-typing columns: (hand, finger, reach, [(row, char), ...]).
# reach "" = the finger's home column; "in" = index inner reach;
# "out"/"o1"/"o2"/"o3" = pinky outer reaches (1st/2nd/3rd column outward).
MACBOOK_COLUMNS: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    ("L", "pinky", "out", [("num", "`")]),
    ("L", "pinky", "", [("num", "1"), ("top", "Q"), ("home", "A"), ("bottom", "Z")]),
    ("L", "ring", "", [("num", "2"), ("top", "W"), ("home", "S"), ("bottom", "X")]),
    ("L", "middle", "", [("num", "3"), ("top", "E"), ("home", "D"), ("bottom", "C")]),
    ("L", "index", "", [("num", "4"), ("top", "R"), ("home", "F"), ("bottom", "V")]),
    ("L", "index", "in", [("num", "5"), ("top", "T"), ("home", "G"), ("bottom", "B")]),
    ("R", "index", "in", [("num", "6"), ("top", "Y"), ("home", "H"), ("bottom", "N")]),
    ("R", "index", "", [("num", "7"), ("top", "U"), ("home", "J"), ("bottom", "M")]),
    ("R", "middle", "", [("num", "8"), ("top", "I"), ("home", "K"), ("bottom", ",")]),
    ("R", "ring", "", [("num", "9"), ("top", "O"), ("home", "L"), ("bottom", ".")]),
    ("R", "pinky", "", [("num", "0"), ("top", "P"), ("home", ";"), ("bottom", "/")]),
    ("R", "pinky", "o1", [("num", "-"), ("top", "["), ("home", "'")]),
    ("R", "pinky", "o2", [("num", "="), ("top", "]")]),
    ("R", "pinky", "o3", [("top", "\\")]),
]

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUMROW = set("`1234567890-=")


def _addr(hand: str, finger: str, row: str, reach: str = "") -> str:
    return f"{hand}-{finger}-{row}" + (f"-{reach}" if reach else "")


@dataclass(frozen=True)
class Board:
    name: str
    addresses: tuple[str, ...]  # canonical order; position index = list index
    symbols: tuple[str, ...]  # symbols in play, in SYMBOL_ORDER
    qwerty_layout: tuple[str, ...] | None = None  # true legend per position
    # Finger columns for address relabeling: (label, addr indices, row signature)
    column_groups: tuple[tuple[str, tuple[int, ...], tuple[str, ...]], ...] = ()

    @property
    def n(self) -> int:
        return len(self.addresses)

    def index_of(self, address: str) -> int:
        try:
            return self.addresses.index(address)
        except ValueError:
            raise KeyError(address) from None

    def sort_symbols(self, symbols: list[str]) -> list[str]:
        return sorted(symbols, key=SYMBOL_ORDER.index)

    def random_layout(self, rng: random.Random) -> list[str]:
        layout = list(self.symbols)
        rng.shuffle(layout)
        return layout

    def near_qwerty_layout(self, rng: random.Random, swaps: int) -> list[str]:
        """QWERTY layout with `swaps` random transpositions (all displaced
        positions distinct, so exactly 2*swaps keys differ from QWERTY)."""
        if self.qwerty_layout is None:
            raise ValueError(f"board {self.name} has no qwerty layout")
        layout = list(self.qwerty_layout)
        positions = rng.sample(range(self.n), 2 * swaps)
        for i in range(swaps):
            a, b = positions[2 * i], positions[2 * i + 1]
            layout[a], layout[b] = layout[b], layout[a]
        return layout


def _fingers12() -> Board:
    rows3 = ["top", "home", "bottom"]
    addresses = [_addr("L", f, r) for f in FINGERS for r in rows3]
    groups = tuple(
        (f"L-{f}", tuple(i * 3 + j for j in range(3)), tuple(rows3))
        for i, f in enumerate(FINGERS)
    )
    return Board("fingers12", tuple(addresses), tuple("ABCDEFGHIJKL"), column_groups=groups)


def _from_macbook(name: str, allowed: set[str] | None) -> Board:
    """Build a board from the MacBook column spec, keeping only `allowed`
    characters (None = all). Column order is fixed, so qwerty26 built this
    way is byte-identical to the original letters-only board."""
    addresses: list[str] = []
    legend: list[str] = []
    groups: list[tuple[str, tuple[int, ...], tuple[str, ...]]] = []
    for hand, finger, reach, keys in MACBOOK_COLUMNS:
        indices: list[int] = []
        rows: list[str] = []
        for row, char in keys:
            if allowed is not None and char not in allowed:
                continue
            indices.append(len(addresses))
            rows.append(row)
            addresses.append(_addr(hand, finger, row, reach))
            legend.append(char)
        if indices:
            label = f"{hand}-{finger}" + (f"-{reach}" if reach else "")
            groups.append((label, tuple(indices), tuple(rows)))
    symbols = tuple(sorted(legend, key=SYMBOL_ORDER.index))
    return Board(name, tuple(addresses), symbols, tuple(legend), tuple(groups))


BOARDS: dict[str, Board] = {
    b.name: b
    for b in (
        _fingers12(),
        _from_macbook("qwerty26", LETTERS),
        _from_macbook("qwerty39", LETTERS | NUMROW),
        _from_macbook("macbook47", None),
    )
}

# Canonical QWERTY letter -> finger address, for the static probe.
_Q26 = BOARDS["qwerty26"]
QWERTY_LETTER_TO_ADDR: dict[str, str] = {
    letter: _Q26.addresses[i] for i, letter in enumerate(_Q26.qwerty_layout or ())
}
QWERTY_ADDR_TO_LETTER = {v: k for k, v in QWERTY_LETTER_TO_ADDR.items()}
