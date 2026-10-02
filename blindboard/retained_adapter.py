"""Retained-reasoning adapter for the Blindboard runner, any provider.

The legacy runner condition ("visible transcript") rebuilt every request
from plain text, so each call saw the game log and its own earlier moves
but none of its earlier reasoning. This adapter is the lossless condition.
It drives a provider-native session (the kit's shared contract,
``native_session.NativeSession``): the session keeps the provider's own
opaque reasoning state and replays it verbatim each turn, so the model
carries its earlier reasoning into its next move.

One adapter is one episode. The runner still owns the visible transcript
and sends all of it on every call; this bridge sends the session only the
new user input, and first checks that the runner's transcript still matches
what the session has seen and returned, message for message. Any mismatch
raises before a billed call, so the visible log and the native state can
never drift apart silently.

Each request is priced at the provider's published standard-tier list in
effect on the day it ran (cache reads, cache writes and long-context tiers
included; flex priced as standard). ``price_date`` pins that day for tests.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from blindboard.anthropic_native import AnthropicNativeSession
from blindboard.client import Message
from blindboard.gemini_native import GeminiNativeSession
from blindboard.native_session import (
    IncompleteTurnError,
    NativeSession,
    registry_pricer,
    today_utc,
)
from blindboard.openrouter_native import INERT_TOOL, OpenRouterNativeSession
from blindboard.responses_native import OpenAINativeSession

HARNESS = "retained-reasoning"


class TranscriptDivergenceError(RuntimeError):
    """The runner's visible transcript no longer matches the session state."""


SessionFactory = Callable[..., NativeSession]


def _openai(*, model: str, instructions: str, effort: str, service_tier: str | None,
            compact_threshold: int | None, api_key: str | None, transport: Any,
            price_date: str | None) -> NativeSession:
    return OpenAINativeSession(
        model=model,
        instructions=instructions,
        effort=effort,
        service_tier=service_tier,
        compact_threshold=compact_threshold,
        api_key=api_key,
        transport=transport,
        pricer=registry_pricer(model, on=price_date) if price_date else None,
    )


def _openai_only(compact_threshold: int | None, model: str) -> None:
    if compact_threshold:
        raise ValueError(f"server-side compaction is OpenAI-only; not available for {model!r}")


def _anthropic(*, model: str, instructions: str, effort: str, service_tier: str | None,
               compact_threshold: int | None, api_key: str | None, transport: Any,
               price_date: str | None) -> NativeSession:
    # No flex tier on Anthropic: requests run (and are priced) at standard.
    _openai_only(compact_threshold, model)
    slug = model.split("/", 1)[1] if model.startswith("anthropic/") else model
    return AnthropicNativeSession(
        model=slug,
        instructions=instructions,
        effort=effort,
        api_key=api_key,
        transport=transport,
        # None keeps the session's own registry pricer and its 1-hour-write guard.
        pricer=registry_pricer(f"anthropic/{slug}", on=price_date) if price_date else None,
    )


def _openrouter(*, model: str, instructions: str, effort: str, service_tier: str | None,
                compact_threshold: int | None, api_key: str | None, transport: Any,
                price_date: str | None) -> NativeSession:
    _openai_only(compact_threshold, model)
    slug = model.split("/", 1)[1]
    return OpenRouterNativeSession(
        model=slug,
        instructions=instructions,
        effort=effort,
        api_key=api_key,
        transport=transport,
        pricer=registry_pricer(model, on=price_date) if price_date else None,
        # DeepSeek keeps earlier-turn reasoning only when the request carries
        # tools (live check 2026-09-25); the inert tool is that request shape.
        # Every other model is refused unless it has its own verified verdict.
        tools=(INERT_TOOL,) if slug.startswith("deepseek/") else (),
    )


def _gemini(*, model: str, instructions: str, effort: str, service_tier: str | None,
            compact_threshold: int | None, api_key: str | None, transport: Any,
            price_date: str | None) -> NativeSession:
    _openai_only(compact_threshold, model)
    slug = model.split("/", 1)[1]
    return GeminiNativeSession(
        model=slug,
        instructions=instructions,
        effort=effort,
        service_tier=service_tier,
        api_key=api_key,
        transport=transport,
        pricer=registry_pricer(model, on=price_date) if price_date else None,
        # Flex first; after 3 flex refusals of one request, send it on standard.
        flex_fallback_after=3 if service_tier == "flex" else None,
    )


# Model-name prefix -> session factory. A provider joins by adding a line.
SESSION_FACTORIES: dict[str, SessionFactory] = {
    "gpt-": _openai,
    "claude-": _anthropic,
    "anthropic/": _anthropic,
    "openrouter/": _openrouter,
    "gemini/": _gemini,
}


def factory_for(model: str) -> SessionFactory:
    for prefix, factory in SESSION_FACTORIES.items():
        if model.startswith(prefix):
            return factory
    raise ValueError(
        f"no native retained-reasoning session for {model!r}; "
        f"supported prefixes: {sorted(SESSION_FACTORIES)}"
    )


class RetainedReasoningAdapter:
    """Same runner interface as the text adapters, backed by a native session."""

    supports_lossless_multi_turn = True

    def __init__(
        self,
        *,
        model: str,
        effort: str = "medium",
        service_tier: str | None = "flex",
        compact_threshold: int | None = None,
        api_key: str | None = None,
        transport: Any = None,
        price_date: str | None = None,
    ) -> None:
        self._factory = factory_for(model)  # refuse unsupported models up front
        self.model = model
        self.effort = effort
        self.service_tier = service_tier
        self.compact_threshold = compact_threshold
        self._api_key = api_key
        self._transport = transport
        self._price_date = price_date  # fixed ISO date (tests); None = day of each request
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        # Standard-tier list cost; None once any request is unpriceable.
        self._cost: float | None = 0.0
        self._lock = threading.Lock()
        self._session: NativeSession | None = None
        self._instructions: str | None = None
        # Everything the session has seen or produced, in order, as the
        # runner should present it back: (role, text) after the system prompt.
        self._seen: list[tuple[str, str]] = []
        # User input sent in a turn the provider left incomplete: the native
        # session never received it, so it goes out with the next input.
        self._undelivered: list[str] = []

    def __call__(self, messages: list[Message]) -> str:
        return self.call_with_meta(messages)[0]

    def _new_input(self, messages: list[Message]) -> list[Message]:
        """Validate the transcript against session state; return the unsent tail."""
        if not messages or messages[0].role != "system":
            raise TranscriptDivergenceError("transcript must open with the system prompt")
        if any(m.role == "system" for m in messages[1:]):
            raise TranscriptDivergenceError("only one system prompt is supported")
        instructions = messages[0].text
        if self._session is None:
            self._session = self._factory(
                model=self.model,
                instructions=instructions,
                effort=self.effort,
                service_tier=self.service_tier,
                compact_threshold=self.compact_threshold,
                api_key=self._api_key,
                transport=self._transport,
                price_date=self._price_date,
            )
            self._instructions = instructions
        elif instructions != self._instructions:
            raise TranscriptDivergenceError("system prompt changed mid-episode")

        rest = [(m.role, m.text) for m in messages[1:]]
        if rest[: len(self._seen)] != self._seen:
            raise TranscriptDivergenceError(
                "visible transcript diverged from the session's native state"
            )
        tail = messages[1 + len(self._seen):]
        if not tail:
            raise TranscriptDivergenceError("no new input since the last turn")
        if any(m.role != "user" for m in tail):
            raise TranscriptDivergenceError(
                "new input may contain user messages only; assistant turns come from the session"
            )
        return tail

    def call_with_meta(self, messages: list[Message]) -> tuple[str, dict[str, Any]]:
        with self._lock:
            tail = self._new_input(messages)
            assert self._session is not None
            texts = self._undelivered + [m.text for m in tail]
            incomplete: str | None = None
            try:
                turn = self._session.turn(texts)
                self._undelivered = []
            except IncompleteTurnError as exc:
                # No answer (e.g. the output-token limit). As in the visible
                # harness the runner gets an empty reply, so the corrective
                # retry or forfeit applies; the tokens are billed; the attempt
                # is not retained and its input goes out with the next turn.
                turn = exc.turn
                incomplete = str(turn.public.get("incomplete_reason") or "incomplete")
                self._undelivered = texts
            self._seen.extend((m.role, m.text) for m in tail)
            self._seen.append(("assistant", turn.text))

            usage = dict(turn.usage)
            self.calls += 1
            for key, value in usage.items():
                self.usage_totals[key] = self.usage_totals.get(key, 0) + int(value)
            if turn.cost_usd is None or self._cost is None:
                self._cost = None
            else:
                self._cost += turn.cost_usd

            meta: dict[str, Any] = {
                "reasoning": list(turn.reasoning_summaries),
                "usage": usage,
                "cost_usd_list": turn.cost_usd,
                "price_date": self._price_date or today_utc(),
                "stop_reason": turn.provenance.get("finish_reason"),
                "provenance": dict(turn.provenance),
                # Public, digest-only session facts (never the native state).
                "harness": {"name": HARNESS, **dict(turn.public)},
            }
            if incomplete:
                meta["stop_reason"] = incomplete
                meta["harness"]["incomplete"] = incomplete
            return turn.text, meta

    def cost_usd(self) -> float | None:
        """Standard-tier list cost of every request so far, each priced at the
        published rates in effect on the day it ran."""
        return self._cost
