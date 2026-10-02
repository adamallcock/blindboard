# Vendored from benchkit@0.3.10 (sha256:5b2058d0a866f75c). Do not hand-edit; run 'benchkit sync' to update.
"""Shared contract for provider-native, lossless multi-turn sessions.

A *native session* holds one provider's own opaque reasoning state and
replays it verbatim on the next request, so a model carries its earlier
reasoning into its next turn: Anthropic signed thinking blocks, Gemini
thought signatures, OpenRouter ``reasoning_details``. (OpenAI's equivalent
is ``responses_stateful.ResponsesSession`` in ``typed_item_replay`` mode.)
The single-inference text adapters cannot do this: ``Message(role, text)``
has nowhere to carry that state, which is why they refuse assistant history.

Every provider session implements this contract:

- ``supports_lossless_multi_turn = True`` and a ``model`` attribute.
- ``turn(user_texts)`` appends the new user message(s) to the session's own
  native history, makes ONE provider request, stores the complete native
  assistant output exactly as returned (reasoning parts, signatures, text,
  in provider order), and returns a ``NativeTurn``.
- ``NativeTurn.usage`` uses the pricing classes of
  ``blindboard.pricing``: ``input_tokens`` is the TOTAL prompt;
  ``cache_read_tokens`` and ``cache_write_tokens`` are subsets of it;
  ``reasoning_tokens`` is inside ``output_tokens``. A class the provider did
  not report is omitted, never zero-filled.
- ``NativeTurn.cost_usd`` is the provider's standard-tier list cost of that
  one request on the day it ran (reporting rule, PLAYBOOK section 7), or
  ``None`` when published rates cannot price it. Never a guess.
- ``NativeTurn.public`` and ``NativeTurn.provenance`` must pass
  ``assert_public_safe``: opaque state never leaves the session.
- ``public["retention"]`` states what was replayed, so a reader can tell
  a retained-reasoning run from a visible-transcript one.
- A turn the provider ends incomplete raises ``IncompleteTurnError``
  (history unchanged) carrying its billed ``NativeTurn``.

Whether a given model actually USES replayed reasoning is an empirical
question per model and route (some chat templates strip earlier thinking).
Each provider module documents a live check; never claim retention for a
model that has not passed it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from blindboard.pricing import price_call

# Keys whose values are opaque provider state. Their presence anywhere in
# public metadata is a leak.
OPAQUE_KEYS = frozenset(
    {
        "signature",
        "thoughtSignature",
        "thought_signature",
        "encrypted_content",
        "reasoning_details",
        "redacted_thinking",
        "data",
    }
)


class RetentionUnsupportedError(RuntimeError):
    """The model or route cannot carry earlier reasoning into a new turn."""


class IncompleteTurnError(RuntimeError):
    """The provider ended the turn without a complete answer (for example at
    the output-token limit). History is unchanged. ``turn`` is the billed,
    public-safe record of the attempt (usually empty text), so its tokens
    and cost are never lost and a runner can score it as an empty reply."""

    def __init__(self, message: str, *, turn: NativeTurn) -> None:
        super().__init__(message)
        self.turn = turn


@dataclass(frozen=True)
class NativeTurn:
    text: str
    usage: Mapping[str, int]
    reasoning_summaries: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    public: Mapping[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None


class NativeSession(Protocol):
    supports_lossless_multi_turn: bool
    model: str

    def turn(self, user_texts: Sequence[str]) -> NativeTurn: ...


Pricer = Callable[[Mapping[str, int]], "float | None"]


def today_utc() -> str:
    return datetime.now(UTC).date().isoformat()


def registry_pricer(registry_key: str, *, on: str | None = None) -> Pricer:
    """Price one request's usage at the published standard-tier list for
    ``registry_key``, on ``on`` or (default) the UTC day the request runs."""

    def price(usage: Mapping[str, int]) -> float | None:
        return price_call(registry_key, dict(usage), on=on or today_utc())

    return price


def assert_public_safe(metadata: Any, *, max_string: int = 4096) -> None:
    """Raise if public metadata carries opaque state or an oversized blob."""

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in OPAQUE_KEYS:
                    raise ValueError(f"opaque provider state in public metadata at {path}.{key}")
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, str) and len(value) > max_string:
            raise ValueError(f"oversized string in public metadata at {path}")

    walk(metadata, "$")
