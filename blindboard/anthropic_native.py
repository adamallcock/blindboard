# Vendored from benchkit@0.3.10 (sha256:9b2ca94dfbcced27). Do not hand-edit; run 'benchkit sync' to update.
"""Provider-native, lossless multi-turn sessions for the Anthropic Messages
API (``POST /v1/messages``), stdlib only.

``AnthropicNativeSession`` implements ``blindboard.native_session``: it
owns the conversation's native ``messages`` history and, on every turn,
resends every earlier assistant turn's complete content-block list exactly as
the API returned it -- ``thinking`` blocks with their ``signature``, any
``redacted_thinking`` blocks, ``text`` blocks, in provider order -- then
appends the new user message. The single-inference ``AnthropicAdapter``
cannot do this (its ``Message(role, text)`` has nowhere to carry a signed
thinking block), which is why it refuses assistant history.

Why replay everything, unmodified (Anthropic's thinking docs; sources and
page digests in docs/2026-09-25-anthropic-native-sessions.md):

- a thinking block's ``signature`` is the encrypted full reasoning and must
  go back unchanged; with ``display: "summarized"`` the readable
  ``thinking`` text is a summary written by a different model, so it is
  telemetry, never state;
- the API itself filters replayed thinking per model and bills input only
  for the blocks the model is shown, so the client never prunes;
- edited, reordered or partially dropped thinking blocks are rejected with a
  400, and on Claude Fable 5.1 / Opus 5.5 a block is valid only while the
  ``system`` prompt, ``tools`` and every earlier message are unchanged. The
  session is therefore append-only: ``system`` is frozen at construction and
  history is never edited.

What replay buys is model-dependent. Anthropic documents that Opus 4.5+ and
Sonnet 4.6+ keep every prior turn's thinking in context, while all Haiku
models through Haiku 4.5 keep only the current turn's (the API strips earlier
thinking). The live check in the doc above measured it per model;
``RETENTION_VERDICTS`` records the verdicts, and a model without a
"retained" verdict raises ``RetentionUnsupportedError`` at construction
unless the caller passes ``allow_visible_only=True`` (the run is then
labelled in ``public["retention"]["effective"]``).

Transport behavior (frozen):

- Thinking config mirrors ``anthropic_messages`` (checked by a test):
  adaptive models get ``{"type": "adaptive", "display": "summarized"}`` and
  ``output_config.effort`` in {low, medium, high, xhigh, max};
  claude-haiku-4-5 gets legacy ``{"type": "enabled", "budget_tokens": N}``.
  Any other effort string is rejected (fail closed). No temperature, no
  tools, no server-side fallback (a benchmark measures the named model).
- Top-level automatic prompt caching ``{"type": "ephemeral"}`` (5-minute
  TTL) on every request, as in the one-shot adapter: the breakpoint moves to
  the newest block, so each turn reads the replayed prefix from cache.
- Non-streaming raw HTTP. 408/409/429, every 5xx (529 overloaded included)
  and network failures are retried with bounded exponential backoff
  (honouring ``retry-after``); every other 4xx fails at once and is never
  retried.
- History advances only when a response has ``stop_reason == "end_turn"``
  and a replayable content list (non-empty; every ``thinking`` block
  signed; every ``redacted_thinking`` block carrying ``data``). Anything else
  (``max_tokens``, ``refusal``, ``model_context_window_exceeded``, ...)
  raises ``AnthropicIncompleteTurnError`` carrying a public-safe
  ``NativeTurn`` (stop reason, stop details, usage, cost) and leaves the
  history untouched, so the caller can retry the identical turn or end the
  episode.

Usage and cost: Anthropic's ``input_tokens`` EXCLUDES cached tokens (the
Messages reference defines the total prompt as the sum of the uncached,
cache-write and cache-read counts), so
``input_tokens`` = input + cache_read + cache_creation (the TOTAL prompt);
``cache_read_tokens`` = cache_read_input_tokens; ``cache_write_tokens`` =
cache_creation_input_tokens, with the per-TTL sub-counts kept as
``cache_write_5m_tokens`` / ``cache_write_1h_tokens`` (priced differently);
``output_tokens`` = output_tokens; ``reasoning_tokens`` =
``output_tokens_details.thinking_tokens`` (inside output). Unreported classes
are omitted, never zero-filled. ``cost_usd`` is the dated standard-tier list
cost, or None when the registry cannot price the request exactly: US-only
inference (a 1.1x multiplier the registry does not model), or, for the
default registry pricer (one cache-write rate, the 5-minute one), any 1-hour
write or a write without its TTL breakdown.
"""

from __future__ import annotations

import copy
import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from blindboard.native_session import (
    IncompleteTurnError,
    NativeTurn,
    Pricer,
    RetentionUnsupportedError,
    assert_public_safe,
    registry_pricer,
    today_utc,
)
from blindboard.client import build_provenance

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Thinking configuration, kept in lockstep with benchkit.client.anthropic_messages
# (tests/test_client_anthropic_native.py asserts equality) without importing
# it, so vendoring this module does not drag in the Gemini/OpenRouter clients.
ADAPTIVE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
LEGACY_MARKERS = ("haiku-4-5",)
LEGACY_BUDGETS = {"low": 8000, "medium": 16000, "high": 32000, "xhigh": 32000, "max": 32000}
ADAPTIVE_MAX_TOKENS = 128000
LEGACY_MAX_TOKENS = 64000

# Top-level automatic caching, default 5-minute TTL (same as the adapter).
CACHE_CONTROL = {"type": "ephemeral"}
# Transient by definition (the SDKs' retry set): request timeout, conflict,
# rate limit, and every 5xx (500 api_error, 504 timeout_error, 529
# overloaded_error, edge 52x). Every other 4xx is a request/account problem
# (400 invalid request -- also used for spend limits -- 401, 402 billing,
# 403, 404, 413) and is never retried.
RETRYABLE_4XX = frozenset({408, 409, 429})


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_4XX or status_code >= 500
ACCEPTED_STOP_REASON = "end_turn"
REPLAY_MODE = "thinking_blocks"
# inference_geo values billed at the standard rate. "us" carries a 1.1x
# multiplier on Claude 4.6+ that the pricing registry does not model.
STANDARD_PRICE_GEOS = frozenset({"global", "not_available"})

# Live retention verdicts (docs/2026-09-25-anthropic-native-sessions.md).
# "retained": turn 2's billed prompt grew by turn 1's thinking when the
# signed blocks were replayed, versus a text-only control of the same
# conversation. "not_retained": no growth -- the API strips prior-turn
# thinking for that model. Keys are the API ids requested and the ids the
# API reported back.
RETENTION_VERDICTS: dict[str, dict[str, str]] = {
    # turn-2 billed prompt, native replay minus text-only control, vs turn-1
    # thinking tokens (effort medium; haiku low = budget 8000).
    "claude-haiku-4-5": {
        "verdict": "not_retained", "verified_on": "2026-09-25",
        "evidence": "delta 0 with 4272 turn-1 thinking tokens (791 vs 791)",
    },
    "claude-haiku-4-5-20251001": {
        "verdict": "not_retained", "verified_on": "2026-09-25",
        "evidence": "dated id the alias resolved to; see claude-haiku-4-5",
    },
    "claude-sonnet-5": {
        "verdict": "retained", "verified_on": "2026-09-25",
        "evidence": "delta 1124 of 1124 turn-1 thinking tokens (1264 vs 140)",
    },
    "claude-opus-4-8": {
        "verdict": "retained", "verified_on": "2026-09-25",
        "evidence": "delta 965 of 965 turn-1 thinking tokens (1103 vs 138)",
    },
    "claude-opus-5-5": {
        "verdict": "retained", "verified_on": "2026-09-25",
        "evidence": "delta 616 of 614 turn-1 thinking tokens (756 vs 140)",
    },
}

_OPAQUE_RUN = re.compile(r"[A-Za-z0-9+/=_-]{64,}")
_MAX_ERROR_MESSAGE = 300
_MAX_STOP_EXPLANATION = 1000


class AnthropicHTTPError(RuntimeError):
    """Sanitized HTTP failure: status, Anthropic's error type, a short
    redacted message, the request id, and a digest of the body -- never the
    raw body."""

    def __init__(
        self,
        *,
        status_code: int,
        error_type: str | None = None,
        message: str | None = None,
        request_id: str | None = None,
        body_sha256: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        detail = f" {error_type}" if error_type else ""
        text = f": {message}" if message else ""
        request = f" (request_id={request_id})" if request_id else ""
        super().__init__(f"Anthropic HTTP {status_code}{detail}{text}{request}")
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.request_id = request_id
        self.body_sha256 = body_sha256
        self.retry_after_seconds = retry_after_seconds
        self.retryable = is_retryable_status(status_code)


class AnthropicTransportError(RuntimeError):
    """Network-level failure (timeout, reset, unparseable body); retryable."""

    retryable = True

    def __init__(self, kind: str) -> None:
        super().__init__(f"Anthropic transport failed ({kind})")
        self.kind = kind


class AnthropicResponseError(RuntimeError):
    """A 200 response that violates the Messages contract (error envelope,
    non-assistant role, malformed content). Not retried; history unchanged."""


class AnthropicIncompleteTurnError(IncompleteTurnError):
    """The turn did not end in a complete, replayable ``end_turn`` response;
    history is unchanged.

    ``turn`` carries the public-safe record of the attempt (empty text,
    usage, cost, ``stop_reason`` / ``stop_details`` / ``incomplete_reason``
    in ``turn.public``), so billed tokens are never lost to the caller."""


class AnthropicTransport(Protocol):
    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout_seconds: int
    ) -> dict[str, Any]: ...


def _redact(message: str, limit: int) -> str:
    return _OPAQUE_RUN.sub("<redacted>", message)[:limit]


def _retry_after_seconds(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


class AnthropicHTTPTransport:
    """Stdlib JSON POST with the house headers (``x-api-key``,
    ``anthropic-version``)."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set (use: secret run anthropic --env ANTHROPIC_API_KEY -- ...)"
            )
        self._api_key = api_key

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout_seconds: int
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read() or b""
            parsed: Any = None
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            error = parsed.get("error") if isinstance(parsed, Mapping) else None
            error = error if isinstance(error, Mapping) else {}
            request_id = parsed.get("request_id") if isinstance(parsed, Mapping) else None
            headers = getattr(exc, "headers", None)
            if not isinstance(request_id, str) and headers is not None:
                request_id = headers.get("request-id")
            message, error_type = error.get("message"), error.get("type")
            raise AnthropicHTTPError(
                status_code=exc.code,
                error_type=error_type if isinstance(error_type, str) else None,
                message=_redact(message, _MAX_ERROR_MESSAGE) if isinstance(message, str) else None,
                request_id=request_id if isinstance(request_id, str) else None,
                body_sha256=hashlib.sha256(raw).hexdigest(),
                retry_after_seconds=_retry_after_seconds(headers),
            ) from None
        except (OSError, http.client.HTTPException) as exc:
            raise AnthropicTransportError(type(exc).__name__) from None
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AnthropicTransportError("InvalidJSON") from None
        if not isinstance(parsed, dict):
            raise AnthropicTransportError("NonObjectJSON")
        return parsed


def is_legacy_thinking_model(model: str) -> bool:
    return any(marker in model for marker in LEGACY_MARKERS)


def thinking_request_config(
    model: str, effort: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """``(thinking, output_config)`` exactly as a session sends them.

    Adaptive models: summarized adaptive thinking plus ``output_config.effort``.
    claude-haiku-4-5: the legacy ``budget_tokens`` mapping and no
    output_config (effort is rejected there). Unknown efforts raise."""
    if is_legacy_thinking_model(model):
        if effort not in LEGACY_BUDGETS:
            raise ValueError(f"effort must be one of {sorted(LEGACY_BUDGETS)} for {model}")
        return {"type": "enabled", "budget_tokens": LEGACY_BUDGETS[effort]}, None
    if effort not in ADAPTIVE_EFFORTS:
        raise ValueError(f"effort must be one of {list(ADAPTIVE_EFFORTS)} for {model}")
    return {"type": "adaptive", "display": "summarized"}, {"effort": effort}


def anthropic_usage_classes(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Map a Messages ``usage`` object onto the pricing classes (omit, never
    zero-fill).

    input_tokens = input_tokens + cache_read_input_tokens +
    cache_creation_input_tokens (the TOTAL prompt; Anthropic's own
    ``input_tokens`` is only the uncached tail); cache_read_tokens and
    cache_write_tokens are its subsets; cache_write_5m_tokens /
    cache_write_1h_tokens are the per-TTL split of the writes;
    reasoning_tokens = output_tokens_details.thinking_tokens (inside
    output_tokens)."""
    usage = usage if isinstance(usage, Mapping) else {}

    def count(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    classes: dict[str, int] = {}
    uncached = count(usage.get("input_tokens"))
    reads = count(usage.get("cache_read_input_tokens"))
    writes = count(usage.get("cache_creation_input_tokens"))
    if uncached is not None:
        classes["input_tokens"] = uncached + (reads or 0) + (writes or 0)
    if reads is not None:
        classes["cache_read_tokens"] = reads
    if writes is not None:
        classes["cache_write_tokens"] = writes
    by_ttl = usage.get("cache_creation")
    if isinstance(by_ttl, Mapping):
        for src, dst in (
            ("ephemeral_5m_input_tokens", "cache_write_5m_tokens"),
            ("ephemeral_1h_input_tokens", "cache_write_1h_tokens"),
        ):
            value = count(by_ttl.get(src))
            if value is not None:
                classes[dst] = value
    output = count(usage.get("output_tokens"))
    if output is not None:
        classes["output_tokens"] = output
    details = usage.get("output_tokens_details")
    if isinstance(details, Mapping):
        thinking = count(details.get("thinking_tokens"))
        if thinking is not None:
            classes["reasoning_tokens"] = thinking
    return classes


def _usage_identity_holds(usage: Mapping[str, int]) -> bool | None:
    """The documented identities the mapping relies on: the TTL split sums
    to the writes, and thinking is inside output. None if unreported."""
    checks: list[bool] = []
    if "cache_write_tokens" in usage and (
        "cache_write_5m_tokens" in usage or "cache_write_1h_tokens" in usage
    ):
        split = usage.get("cache_write_5m_tokens", 0) + usage.get("cache_write_1h_tokens", 0)
        checks.append(split == usage["cache_write_tokens"])
    if "reasoning_tokens" in usage and "output_tokens" in usage:
        checks.append(usage["reasoning_tokens"] <= usage["output_tokens"])
    return all(checks) if checks else None


def retention_verdict(model: str) -> dict[str, str] | None:
    verdict = RETENTION_VERDICTS.get(model)
    return dict(verdict) if verdict is not None else None


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _block_counts(content: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "blocks": len(content),
        "thinking_blocks": 0,
        "signed_thinking_blocks": 0,
        "empty_thinking_blocks": 0,
        "redacted_thinking_blocks": 0,
        "text_blocks": 0,
        "other_blocks": 0,
    }
    for block in content:
        kind = block.get("type")
        if kind == "thinking":
            counts["thinking_blocks"] += 1
            if isinstance(block.get("signature"), str) and block["signature"]:
                counts["signed_thinking_blocks"] += 1
            if not block.get("thinking"):
                counts["empty_thinking_blocks"] += 1
        elif kind == "redacted_thinking":
            counts["redacted_thinking_blocks"] += 1
        elif kind == "text":
            counts["text_blocks"] += 1
        else:
            counts["other_blocks"] += 1
    return counts


def _unreplayable_reason(content: Sequence[Mapping[str, Any]]) -> str | None:
    """Why this content cannot be replayed verbatim next turn, or None."""
    if not content:
        return "empty_content"
    for block in content:
        kind = block.get("type")
        if kind == "thinking" and not (
            isinstance(block.get("signature"), str) and block["signature"]
        ):
            return "unsigned_thinking_block"
        if kind == "redacted_thinking" and not (
            isinstance(block.get("data"), str) and block["data"]
        ):
            return "redacted_thinking_without_data"
    return None


def _user_message(user_texts: Sequence[str]) -> dict[str, Any]:
    if isinstance(user_texts, (str, bytes)) or not isinstance(user_texts, Sequence):
        raise TypeError("user_texts must be a sequence of strings, not a single string")
    if not user_texts:
        raise ValueError("a turn needs at least one user text")
    for text in user_texts:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every user text must be a string with non-whitespace content")
    return {"role": "user", "content": [{"type": "text", "text": text} for text in user_texts]}


class AnthropicNativeSession:
    """One Claude conversation whose assistant turns replay natively.

    ``model`` is the bare API model id (no "anthropic/" prefix; the registry
    key is "anthropic/<model>"). ``instructions`` becomes the top-level
    ``system`` prompt on every request, byte-identical (None omits it). A
    session is single-threaded: concurrent ``turn`` calls raise; separate
    sessions are independent."""

    supports_lossless_multi_turn = True

    def __init__(
        self,
        *,
        model: str,
        instructions: str | None,
        effort: str,
        api_key: str | None = None,
        transport: AnthropicTransport | None = None,
        pricer: Pricer | None = None,
        request_timeout_seconds: int = 1800,
        max_retries: int = 5,
        retry_initial_seconds: float = 10.0,
        retry_backoff: float = 2.0,
        retry_max_seconds: float = 300.0,
        sleeper: Callable[[float], None] = time.sleep,
        max_tokens: int | None = None,
        allow_visible_only: bool = False,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if model.startswith("anthropic/"):
            raise ValueError("pass the bare API model id (no 'anthropic/' prefix)")
        if instructions is not None and (not isinstance(instructions, str) or not instructions.strip()):
            raise ValueError("instructions must be a nonempty string or None")
        if not isinstance(effort, str):
            raise TypeError("effort must be a string")
        if not isinstance(allow_visible_only, bool):
            raise TypeError("allow_visible_only must be bool")
        if not isinstance(request_timeout_seconds, int) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be a positive integer")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if retry_initial_seconds < 0 or retry_max_seconds < 0 or retry_backoff < 1:
            raise ValueError("retry delays must be nonnegative and retry_backoff >= 1")
        thinking, output_config = thinking_request_config(model, effort)
        legacy = is_legacy_thinking_model(model)
        if max_tokens is None:
            max_tokens = LEGACY_MAX_TOKENS if legacy else ADAPTIVE_MAX_TOKENS
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if legacy and max_tokens <= thinking["budget_tokens"]:
            raise ValueError("max_tokens must exceed the thinking budget_tokens")

        verdict = retention_verdict(model)
        retained = verdict is not None and verdict["verdict"] == "retained"
        if not retained and not allow_visible_only:
            if verdict is None:
                reason = "has no live retention verdict"
            else:
                reason = (
                    "does not retain prior-turn thinking (live check "
                    f"{verdict['verified_on']}: the API strips it)"
                )
            raise RetentionUnsupportedError(
                f"{model} {reason}; pass allow_visible_only=True to run it as a "
                "labelled visible-transcript condition"
            )
        if retained:
            effective = "thinking_retained"
        elif verdict is None:
            effective = "unverified"
        else:
            effective = "visible_only"

        self.model = model
        self.instructions = instructions
        self.effort = effort
        self.max_tokens = max_tokens
        self.allow_visible_only = allow_visible_only
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.retry_initial_seconds = float(retry_initial_seconds)
        self.retry_backoff = float(retry_backoff)
        self.retry_max_seconds = float(retry_max_seconds)
        self._thinking = thinking
        self._output_config = output_config
        self._retention_verdict = verdict
        self._retention_effective = effective
        self._sleep = sleeper
        self._registry_pricing = pricer is None
        self._pricer = pricer or registry_pricer(f"anthropic/{model}")
        self._transport = transport or AnthropicHTTPTransport(
            api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        self._messages: list[dict[str, Any]] = []
        self._turn_index = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"AnthropicNativeSession(model={self.model!r}, effort={self.effort!r}, "
            f"retention={self._retention_effective!r}, turn_index={self._turn_index})"
        )

    @property
    def turn_index(self) -> int:
        """Completed turns so far (the index the next turn will carry)."""
        return self._turn_index

    def native_messages(self) -> list[dict[str, Any]]:
        """PRIVATE opaque state: a deep copy of the native history (user
        messages as sent, assistant content exactly as returned, signatures
        included). For private checkpoints and audits; never publish it."""
        return copy.deepcopy(self._messages)

    def build_request(self, user_texts: Sequence[str]) -> dict[str, Any]:
        """The exact body the next ``turn`` would POST; no I/O, no mutation."""
        return self._payload(_user_message(user_texts))

    def turn(self, user_texts: Sequence[str]) -> NativeTurn:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("one AnthropicNativeSession cannot be driven concurrently")
        try:
            user_message = _user_message(user_texts)
            payload = self._payload(user_message)
            response, retried = self._post_with_retries(payload)
            return self._accept(response, user_message, payload, retried)
        finally:
            self._lock.release()

    def _payload(self, user_message: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": copy.deepcopy(self._messages) + [copy.deepcopy(dict(user_message))],
            "thinking": copy.deepcopy(self._thinking),
            "cache_control": dict(CACHE_CONTROL),
        }
        if self.instructions is not None:
            body["system"] = self.instructions
        if self._output_config is not None:
            body["output_config"] = copy.deepcopy(self._output_config)
        return body

    def _post_with_retries(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[int | str]]:
        retried: list[int | str] = []
        for attempt in range(self.max_retries + 1):
            try:
                response = self._transport.post(
                    ANTHROPIC_URL,
                    copy.deepcopy(dict(payload)),
                    timeout_seconds=self.request_timeout_seconds,
                )
                return response, retried
            except AnthropicHTTPError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                retried.append(exc.status_code)
                server_delay = exc.retry_after_seconds
            except AnthropicTransportError as exc:
                if attempt >= self.max_retries:
                    raise
                retried.append(exc.kind)
                server_delay = None
            delay = min(
                self.retry_max_seconds,
                self.retry_initial_seconds * self.retry_backoff**attempt,
            )
            if server_delay is not None:
                delay = max(delay, min(server_delay, self.retry_max_seconds))
            if delay:
                self._sleep(delay)
        raise AssertionError("unreachable retry loop")  # pragma: no cover

    def _cost(self, usage: Mapping[str, int], inference_geo: Any) -> tuple[float | None, str | None]:
        """(cost, reason it is None). Never a guess."""
        if "input_tokens" not in usage or "output_tokens" not in usage:
            return None, "usage_incomplete"
        if _usage_identity_holds(usage) is False:
            return None, "usage_identity_broken"
        if isinstance(inference_geo, str) and inference_geo not in STANDARD_PRICE_GEOS:
            return None, f"inference_geo_{inference_geo}_multiplier_not_in_registry"
        if self._registry_pricing and (
            usage.get("cache_write_1h_tokens")
            or (usage.get("cache_write_tokens") and "cache_write_5m_tokens" not in usage)
        ):
            return None, "cache_write_ttl_not_priceable_by_registry"
        cost = self._pricer(dict(usage))
        return cost, None if cost is not None else "no_published_rate_in_pricer"

    def _accept(
        self,
        response: Mapping[str, Any],
        user_message: Mapping[str, Any],
        payload: Mapping[str, Any],
        retried: list[int | str],
    ) -> NativeTurn:
        if not isinstance(response, Mapping):
            raise AnthropicResponseError("Messages response is not an object")
        if response.get("type") == "error":
            raise AnthropicResponseError("Messages returned an error envelope")
        if response.get("role") != "assistant":
            raise AnthropicResponseError("Messages response role is not 'assistant'")
        content = response.get("content")
        if not isinstance(content, list) or not all(
            isinstance(block, Mapping) and isinstance(block.get("type"), str)
            for block in content
        ):
            raise AnthropicResponseError("Messages content must be a list of typed blocks")

        stop_reason = response.get("stop_reason")
        unreplayable = _unreplayable_reason(content)
        complete = stop_reason == ACCEPTED_STOP_REASON and unreplayable is None
        text = "".join(
            block["text"]
            for block in content
            if block["type"] == "text" and isinstance(block.get("text"), str)
        )
        summaries = tuple(
            block["thinking"]
            for block in content
            if block["type"] == "thinking"
            and isinstance(block.get("thinking"), str)
            and block["thinking"]
        )
        raw_usage = response.get("usage")
        usage = anthropic_usage_classes(raw_usage)
        raw_usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        inference_geo = raw_usage.get("inference_geo")
        returned_tier = raw_usage.get("service_tier")
        cost, unpriced_reason = self._cost(usage, inference_geo)
        replayed = [block for message in self._messages if message["role"] == "assistant"
                    for block in message["content"]]
        replayed_counts = _block_counts(replayed)

        public: dict[str, Any] = {
            "provider": "anthropic",
            "api": "messages",
            "model": self.model,
            "turn_index": self._turn_index,
            "complete": complete,
            "stop_reason": stop_reason,
            "retention": {
                "replayed": REPLAY_MODE,
                "prior_turns": self._turn_index,
                "replayed_thinking_blocks": replayed_counts["thinking_blocks"],
                "replayed_redacted_thinking_blocks": replayed_counts["redacted_thinking_blocks"],
                "verdict": self._retention_verdict["verdict"] if self._retention_verdict else None,
                "verified_on": self._retention_verdict["verified_on"] if self._retention_verdict else None,
                "effective": self._retention_effective,
            },
            "effort": self.effort,
            "thinking_config": copy.deepcopy(self._thinking),
            "output_config": copy.deepcopy(self._output_config),
            "max_tokens": self.max_tokens,
            "cache_control": dict(CACHE_CONTROL),
            "output": _block_counts(content),
            "usage": dict(usage),
            "usage_identity_holds": _usage_identity_holds(usage),
            "cost_usd": cost,
            "priced_on": today_utc(),
            "service_tier": returned_tier if isinstance(returned_tier, str) else None,
            "inference_geo": inference_geo if isinstance(inference_geo, str) else None,
            "request_sha256": _sha256_json(payload),
            "http": {"attempts": len(retried) + 1, "retried": list(retried)},
        }
        if unpriced_reason is not None:
            public["cost_unpriced_reason"] = unpriced_reason
        stop_details = response.get("stop_details")
        if isinstance(stop_details, Mapping):
            explanation = stop_details.get("explanation")
            public["stop_details"] = {
                "type": stop_details.get("type"),
                "category": stop_details.get("category"),
                "explanation": _redact(explanation, _MAX_STOP_EXPLANATION)
                if isinstance(explanation, str)
                else None,
            }
        if not complete:
            public["incomplete_reason"] = (
                f"stop_reason_{stop_reason}" if stop_reason != ACCEPTED_STOP_REASON else unreplayable
            )
        provenance = build_provenance(
            response_id=response.get("id"),
            routed_model=response.get("model"),
            service_tier=returned_tier if isinstance(returned_tier, str) else None,
            finish_reason=stop_reason,
        )
        assert_public_safe(public)
        assert_public_safe(provenance)
        turn = NativeTurn(
            text=text if complete else "",
            usage=usage,
            reasoning_summaries=summaries if complete else (),
            provenance=provenance,
            public=public,
            cost_usd=cost,
        )
        if not complete:
            raise AnthropicIncompleteTurnError(
                f"Anthropic turn {self._turn_index} incomplete "
                f"({public['incomplete_reason']}); history unchanged",
                turn=turn,
            )
        self._messages.append(copy.deepcopy(dict(user_message)))
        self._messages.append({"role": "assistant", "content": copy.deepcopy(content)})
        self._turn_index += 1
        return turn


__all__ = [
    "ACCEPTED_STOP_REASON",
    "ADAPTIVE_EFFORTS",
    "ANTHROPIC_URL",
    "ANTHROPIC_VERSION",
    "CACHE_CONTROL",
    "LEGACY_BUDGETS",
    "RETENTION_VERDICTS",
    "RETRYABLE_4XX",
    "AnthropicHTTPError",
    "AnthropicHTTPTransport",
    "AnthropicIncompleteTurnError",
    "AnthropicNativeSession",
    "AnthropicResponseError",
    "AnthropicTransport",
    "AnthropicTransportError",
    "anthropic_usage_classes",
    "is_retryable_status",
    "retention_verdict",
    "thinking_request_config",
]
