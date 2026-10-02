"""Boards (39/47), address relabeling, and dual-board interference axes."""

from __future__ import annotations

from dataclasses import replace

from blindboard.agents import RandomAgent, ReferenceAgent
from blindboard.boards import BOARDS
from blindboard.prompts import build_system_prompt
from blindboard.runner import run_scripted_episode
from blindboard.tiers import TIERS


def test_board_sizes_and_spot_checks() -> None:
    b39, b47 = BOARDS["qwerty39"], BOARDS["macbook47"]
    assert b39.n == 39 and b47.n == 47
    for board, char, addr in [
        (b47, "6", "R-index-num-in"),
        (b47, "`", "L-pinky-num-out"),
        (b47, "]", "R-pinky-top-o2"),
        (b47, "\\", "R-pinky-top-o3"),
        (b47, ",", "R-middle-bottom"),
        (b47, ";", "R-pinky-home"),
        (b47, "'", "R-pinky-home-o1"),
        (b39, "1", "L-pinky-num"),
        (b39, "=", "R-pinky-num-o2"),
    ]:
        assert board.qwerty_layout is not None
        assert board.addresses[board.qwerty_layout.index(char)] == addr, char


def test_qwerty26_unchanged_by_factory_refactor() -> None:
    b = BOARDS["qwerty26"]
    assert b.n == 26
    assert b.addresses[0] == "L-pinky-top"
    assert b.qwerty_layout is not None and b.qwerty_layout[0] == "Q"
    assert b.symbols == tuple(sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def test_symbol_sort_order() -> None:
    b47 = BOARDS["macbook47"]
    assert b47.sort_symbols(["A", "3", "-", "Z", "["]) == ["-", "[", "3", "A", "Z"]


def test_relabel_swaps_address_meaning() -> None:
    config = replace(TIERS["t0_frozen"], relabel_every=2, horizon=8)
    result = run_scripted_episode(RandomAgent(config, 0), config, seed=0)
    relabels = [r["relabel"] for r in result["records"] if r.get("relabel")]
    assert relabels, "relabel_every=2 must fire"
    # After a relabel, records still hold PHYSICAL positions; the layout
    # never moved (rotation none), so pressing all addresses at guess time
    # must still cover all physical keys exactly once (bijection intact).
    assert result["board_n"] == 12


def test_reference_solves_with_relabeling_translation() -> None:
    config = replace(TIERS["t2_announced"], relabel_every=3)
    for seed in range(5):
        result = run_scripted_episode(ReferenceAgent(config, seed, samples=25), config, seed)
        assert result["board_accuracy"] == 1.0, (
            f"seed {seed}: relabeling must be invisible to the translated reference"
        )
        assert result["lock_wrong"] == 0
        # Diagnostics must translate addressing too: a perfect solver
        # converts every deducible key even under relabeling.
        d = result["diagnostics"]
        assert d["deduced_correct_in_guess"] == d["deducible_at_guess"] == 12


def test_dual_end_to_end_and_scoring() -> None:
    config = TIERS["t9_dual"]
    a = run_scripted_episode(RandomAgent(config, 3), config, seed=3)
    b = run_scripted_episode(RandomAgent(config, 3), config, seed=3)
    assert a["board_n"] == 52
    assert a["correct"] == b["correct"], "dual episodes must be deterministic"
    assert set(a["per_board"]) == {"A", "B"}


def test_dual_reference_solves_frozen_boards() -> None:
    config = replace(
        TIERS["t9_dual"], rotation_mode="none", rotation_schedule=(), horizon=12
    )
    result = run_scripted_episode(ReferenceAgent(config, 1, samples=25), config, seed=1)
    assert result["board_accuracy"] == 1.0, result["per_board"]
    d = result["diagnostics"]
    assert d and d["deducible_at_guess"] == 52


def test_prompts_cover_new_rules() -> None:
    mac = build_system_prompt(TIERS["t11_maelstrom"])
    assert "backslash" in mac and "RELABEL" in mac.upper()
    assert "same fixed order" in mac
    dual = build_system_prompt(TIERS["t9_dual"])
    assert "TWO BOARDS" in dual
