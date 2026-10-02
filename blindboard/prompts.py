"""System prompt and observation formatting for LLM players.

Kept deliberately tight (~700 tokens): the pilot models are small and the
game is the load, not the rules. Every rule the validator enforces is
stated; the complete address list is enumerated (weak models cannot be
trusted to construct addresses)."""

from __future__ import annotations

from blindboard.boards import Board
from blindboard.env import EnvConfig, Observation

# Prompt instrument version, stamped into every episode result. Bumped to
# bb-r2 on 2026-07-20: red-team KS-1 found the announced-mode rules text
# promised rotations "eventually stop" on six constant-schedule tiers where
# they never do. The schedule-truth tail is now derived from the config;
# decaying-tier text is byte-identical to bb-r1 (those cells keep their
# instrument).
PROMPT_VERSION = "bb-r2"

ROTATION_RULES = {
    "none": "The keys never move: the arrangement is fixed for the whole game.",
    "announced": (
        "After your presses each turn, some keys may be ROTATED: the letters on the "
        "rotated keys are rearranged among those same keys, and every rotated key ends "
        "up with a different letter than it had (the set of letters on those keys is "
        "unchanged; only their places among those keys change). Next turn you are told "
        "exactly WHICH keys were rotated, but not how they were rearranged."
    ),
    "announced_full": (
        "After your presses each turn, some keys may be ROTATED: the letters on the "
        "rotated keys are rearranged among those same keys (every rotated key gets a "
        "different letter than it had). Next turn you are told exactly which keys "
        "rotated AND where each key's letter moved, e.g. 'L-ring-top -> L-index-home' "
        "means the letter that was on L-ring-top is now on L-index-home."
    ),
    "observer": (
        "Pressing disturbs the board: after each of your chords, the keys you just "
        "pressed are rotated among themselves — each of those keys ends up with a "
        "different one of those same letters. No other keys ever move. Your results "
        "always describe the keys as they were BEFORE that rotation."
    ),
    "unannounced": (
        "After your presses each turn, some keys may be silently ROTATED: the letters "
        "on the rotated keys are rearranged among those same keys (every rotated key "
        "gets a different letter than it had). You are NOT told which keys rotated, "
        "or whether any rotation happened at all."
    ),
    "rotor": (
        "A fixed secret mechanism rotates the same set of keys in the same way after "
        "every turn. You are not told which keys, but the mechanism never changes."
    ),
    "storm": (
        "After your presses each turn, ONE rotation hits a group of keys: the keys you "
        "just pressed PLUS some additional random keys are all rearranged among "
        "themselves — every key in the group ends up with a different letter than it "
        "had, and the set of letters on the group is unchanged. Your results always "
        "describe the keys as they were BEFORE that rotation. Next turn you are told "
        "exactly which keys were in the rotated group, but not how they were "
        "rearranged. This never stops."
    ),
}


BOARD_DESCS = {
    "fingers12": (
        "a one-handed keyboard with 12 keys: 4 fingers (pinky, ring, middle, index) x 3 rows (top, home, bottom)"
    ),
    "qwerty26": (
        "a standard QWERTY-shaped keyboard of 26 letter keys addressed by touch-typing "
        "finger and row; '-in' marks the index fingers' inner (center-column) reach"
    ),
    "qwerty39": (
        "a QWERTY keyboard's 26 letters plus the 13-key number row (39 keys), addressed by "
        "touch-typing finger and row (rows: num, top, home, bottom); '-in' marks index-finger "
        "inner reach; '-out'/'-o1'/'-o2' mark pinky outer-reach columns"
    ),
    "macbook47": (
        "the full main block of a MacBook keyboard: 47 keys (letters, number row, punctuation), "
        "addressed by touch-typing finger and row (rows: num, top, home, bottom); '-in' marks "
        "index-finger inner reach; '-out'/'-o1'/'-o2'/'-o3' mark pinky outer-reach columns"
    ),
}

RELABEL_RULES = (
    "ADDRESS RELABELING\n"
    "Every {k} turns the ADDRESSING ITSELF changes: two same-shaped finger columns swap "
    "addresses, and you are told which (e.g. 'the L-ring and R-index columns have swapped "
    "addresses'). The hidden symbols do NOT move when this happens — only what you must call "
    "each key changes. Every notice and every press/lock/guess you make uses the NEWEST "
    "addressing. Relabelings compose over the game: always track what each address currently "
    "means."
)

DUAL_RULES = (
    "TWO BOARDS\n"
    "You are playing TWO independent boards, A and B, with identical addressing but different "
    "hidden arrangements. Turns alternate between boards and every message is labeled with its "
    "board. Presses, results, rotation notices, locks and guesses apply ONLY to the labeled "
    "board. You will submit a separate final guess for each board. Keep the two boards' facts "
    "strictly separate — mixing them up is the main way to lose."
)


def schedule_tail(config: EnvConfig) -> str:
    """Truthful description of how rotations evolve, derived from the
    ACTUAL schedule (KS-1 fix). Only announced modes carry a schedule
    statement; other modes' texts already state their own dynamics."""
    if config.rotation_mode not in ("announced", "announced_full"):
        return ""
    sizes = [config.rotation_size(t) for t in range(1, config.horizon + 1)]
    if not any(sizes):
        return ""
    if sizes[-1] > 0:
        return (
            " Rotations continue for the ENTIRE game and never stop: "
            "there is no rotation-free endgame."
        )
    # bb-r1 wording, byte-identical, kept for decaying schedules (true there).
    return " Rotations become rarer as the game goes on and eventually stop."


def build_system_prompt(config: EnvConfig) -> str:
    board: Board = config.board_obj()
    k = config.chord_size
    letters = ", ".join(board.symbols)
    addresses = ", ".join(board.addresses)
    hand_desc = BOARD_DESCS[board.name]
    guess_when = (
        f"From turn {config.min_guess_turn} onward you may end the game at any time by submitting your full guess."
        if config.min_guess_turn > 1
        else "You may end the game at any time by submitting your full guess."
    )
    extra_sections = []
    if config.relabel_every > 0:
        extra_sections.append(RELABEL_RULES.format(k=config.relabel_every))
    if config.dual:
        extra_sections.append(DUAL_RULES)
    extra = ("\n\n" + "\n\n".join(extra_sections)) if extra_sections else ""
    sort_desc = (
        "an ALPHABETICALLY SORTED list"
        if all(s.isalpha() for s in board.symbols)
        else "a list sorted in the same fixed order as the symbol list above"
    )
    escape_note = (
        ' (note: the backslash symbol is written "\\\\" inside JSON strings)'
        if "\\" in board.symbols
        else ""
    )
    return f"""You are playing BLINDBOARD, a deduction game about a hidden keyboard.

THE BOARD
The board is {hand_desc}. Each key is addressed by finger and row. The complete list of valid key addresses:
{addresses}

The {board.n} symbols {letters} are assigned to the {board.n} keys in a hidden arrangement (each symbol on exactly one key). Your job: work out which symbol is on which key{escape_note}.

EACH TURN
- You press exactly {k} DIFFERENT keys (a chord) by giving their addresses.
- Next turn you are told which {k} symbols those keys held, as {sort_desc}. You are NOT told which symbol came from which key. Overlap your chords cleverly to pin symbols to keys.

ROTATIONS
{ROTATION_RULES[config.rotation_mode]}{schedule_tail(config)}

LOCKS (optional, side score)
On any press turn you may lock letters onto keys you are sure of: {config.lock_bonus:+d} point if correct at that moment, {config.lock_penalty:+d} if wrong. You get no feedback on locks during the game. Each key can be locked only once, ever.

FINAL GUESS
{guess_when} After turn {config.horizon} you MUST submit it. Your main score is how many of the {board.n} keys you guess correctly, judged against the arrangement at the moment you guess.{extra}

RESPONSE FORMAT
Reply with ONLY one JSON object, no other text.
To press: {{"presses": ["{board.addresses[0]}", "{board.addresses[4]}", "{board.addresses[8]}"]}}
To press and lock: {{"presses": [ ... {k} addresses ... ], "locks": {{"{board.addresses[1]}": "{board.symbols[0]}"}}}}
To guess (ends the game): {{"guess": {{"{board.addresses[0]}": "{board.symbols[0]}", ... one entry for every key ...}}}}"""


def format_observation(obs: Observation, config: EnvConfig) -> str:
    lines: list[str] = []
    notes = list(obs.notes)
    if notes and notes[0].startswith("=== BOARD"):
        lines.append(notes.pop(0))
    if obs.phase == "guess":
        lines.append(
            f"TURN {obs.turn}: press phase over ({config.horizon} turns used)."
        )
    else:
        lines.append(f"TURN {obs.turn} of {config.horizon}.")
    if obs.relabel_notice:
        lines.append(obs.relabel_notice)
    if obs.last_presses is not None:
        if obs.last_presses:
            lines.append("Last turn you pressed: " + ", ".join(obs.last_presses) + ".")
            lines.append(
                "Their symbols (canonical order, unordered): "
                + ", ".join(obs.last_results or [])
                + "."
            )
        else:
            lines.append("Last turn was forfeited: no presses, no results.")
    if obs.rotation_mapping:
        moves = ", ".join(f"{src} -> {dst}" for src, dst in obs.rotation_mapping)
        lines.append(f"Rotation notice (letter movements): {moves}.")
    elif obs.rotation_notice:
        lines.append(
            "Rotation notice: these keys were rotated after your presses: "
            + ", ".join(obs.rotation_notice)
            + "."
        )
    elif obs.last_presses is not None and config.rotation_mode in ("announced", "announced_full"):
        lines.append("Rotation notice: no keys were rotated.")
    lines.extend(notes)
    if obs.phase == "guess":
        lines.append(
            'You MUST now submit your final guess: {"guess": { ... one letter for every key ... }}.'
        )
    else:
        lines.append("Reply with your action JSON.")
    return "\n".join(lines)
