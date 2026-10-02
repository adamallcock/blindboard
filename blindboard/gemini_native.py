# Vendored from benchkit@0.3.10 (sha256:fdd5cc7bc15ad682). Do not hand-edit; run 'benchkit sync' to update.
"""Provider-native, lossless multi-turn sessions for the Gemini API
(``generateContent``), stdlib only.

``GeminiNativeSession`` implements ``blindboard.native_session``: it
owns the conversation's native ``contents`` history and, on every turn,
resends every earlier model turn's complete ``content`` exactly as the API
returned it -- all parts, in provider order, including thought-summary parts
(``"thought": true``) and every ``thoughtSignature`` byte-for-byte -- then
appends the new user content. The single-inference ``GeminiAdapter`` cannot
do this (its ``Message(role, text)`` has nowhere to carry a signature),
which is why it refuses assistant history.

Why replay everything, unmodified (Google's generateContent docs; exact
quotes and snapshot dates in docs/2026-09-25-gemini-native-sessions.md):

- signatures are "encrypted representations of the model's internal thought
  process" and must be returned "exactly as they were received";
- "Return the entire response with all parts back to the model in
  subsequent turns. Don't concatenate parts with signatures together. Don't
  merge one part with a signature with another part without a signature.";
- without a function call, Gemini 3 puts the signature on the LAST part
  (when streaming, possibly an empty-text part), so no part may be dropped,
  merged or reordered. For text turns the API does not reject a missing signature --
  "omitting signatures will degrade the model's reasoning" -- so an
  accidental strip would be silent. That is why replay here is structural
  (deep copies of the returned ``content``), never re-derived from text.

What replay buys is model-dependent. Google documents thought
*preservation* (reasoning from earlier turns re-entering later prompts)
"beginning with Gemini 3.5 Flash"; "Earlier models do not use reasoning
context from previous turns in the same manner." Live check 2026-09-25,
gemini-3.8-flash (details in the doc): replaying the signature adds exactly
the earlier turns' thoughtsTokenCount to promptTokenCount (7,833 of 7,833;
122 of 122; 683 of 682), and the model computed with a number that existed
only in its turn-1 reasoning (2/2 correct; plain-text control 0/2). But
preservation is conditional: a user turn saying the model's private
reasoning was in context was served WITHOUT it (6/6). So every turn
records a free audit, ``public["retention"]["prompt_growth_tokens"]`` =
prompt - (previous prompt + previous output). It is >= 0 when earlier
reasoning was carried and clearly negative (``reasoning_drop_detected``)
when it was left out. ``RETENTION_VERDICTS`` records the live verdicts. A
model without a "retained" verdict raises ``RetentionUnsupportedError`` at
construction unless the caller passes ``allow_visible_only=True``. Replay
stays native either way; the run is then labelled
``public["retention"]["effective"]`` = "unverified" (no verdict) or
"visible_only" (a not-retained verdict), never "thinking_retained".

Transport behavior (frozen):

- ``effort`` is required. "default" OMITS ``thinkingLevel`` so the model
  runs at its documented vendor default, which differs by model (``high``
  for Gemini 3 Flash Preview and 3.1 Pro Preview, ``medium`` for 3.5 Flash
  and later Flash models), so never send a level and call it "default".
  "none" sends ``thinkingBudget: 0`` (the text adapter's mapping). Any
  other string is sent verbatim as ``thinkingLevel`` and the API validates
  it (3.8 Flash: low/medium/high; ``minimal`` is an error).
  ``public["thinking_config"]`` records exactly what was sent.
- ``service_tier`` defaults to "flex" (Google's discounted synchronous
  tier -- NOT the Batch API); ``None`` omits the field. The tier the API
  applied comes back in ``usageMetadata.serviceTier`` and is recorded.
- 408/429/5xx and network failures are retried with bounded exponential
  backoff (honouring a server retry delay); every other 4xx fails at once.
  History advances only when a response has exactly one candidate with a
  non-empty ``model`` content and ``finishReason == "STOP"``. Anything else
  raises ``GeminiIncompleteTurnError`` carrying a public-safe ``NativeTurn``
  (finish/block reason, usage, cost) and leaves the history untouched, so
  the caller can retry the identical turn or end the episode.

Usage and cost: ``usageMetadata`` maps onto the pricing classes as
``input_tokens`` = promptTokenCount (the TOTAL prompt, cached tokens
included); ``cache_read_tokens`` = cachedContentTokenCount;
``output_tokens`` = candidatesTokenCount + thoughtsTokenCount (Google bills
output "including thinking tokens", and reports thoughts OUTSIDE
candidatesTokenCount: totalTokenCount = prompt + thoughts + candidates);
``reasoning_tokens`` = thoughtsTokenCount. Unreported classes are omitted.
A response whose totalTokenCount breaks that identity, or that reports
tool-use prompt tokens (unpriced here), gets ``cost_usd=None``.
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
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from blindboard.gemini_client import GEMINI_URL
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

# Transient by definition: timeout, rate limit / resource exhaustion (the
# documented flex "at capacity" answers are 429 and 503), server errors.
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
ACCEPTED_FINISH_REASON = "STOP"
REPLAY_MODE = "thought_signatures"

# Live retention verdicts (docs/2026-09-25-gemini-native-sessions.md).
# "retained": the next prompt grew by the earlier turn's thoughtsTokenCount
# when its signature was replayed, versus a plain-text control of the same
# conversation, and the model used content that existed only in that
# reasoning. A model absent here has no live verdict (Google documents
# preservation only from Gemini 3.5 Flash on).
RETENTION_VERDICTS: dict[str, dict[str, str]] = {
    "gemini-3.8-flash": {
        "verdict": "retained",
        "verified_on": "2026-09-25",
        "evidence": (
            "signature-only replay minus text-only control = 7833 of 7833 turn-1 "
            "thought tokens (7902 vs 69); 122 of 122; 683 of 682; N mod 89 from "
            "turn-1-only reasoning 2/2 correct vs 0/2; a follow-up saying the "
            "model's private reasoning was in context was served without it (6/6)"
        ),
    },
}


def retention_verdict(model: str) -> dict[str, str] | None:
    verdict = RETENTION_VERDICTS.get(model)
    return dict(verdict) if verdict is not None else None

_SIGNATURE_KEY = "thoughtSignature"
_PRICED_CLASSES = ("input_tokens", "cache_read_tokens", "output_tokens")
# Re-tokenization jitter allowed before a prompt shrink counts as a drop
# (measured: re-entered thoughts = thoughtsTokenCount +0/+1).
_GROWTH_SLACK_TOKENS = 16
_OPAQUE_RUN = re.compile(r"[A-Za-z0-9+/=_-]{64,}")
_MAX_ERROR_MESSAGE = 300
_MAX_FINISH_MESSAGE = 1000


class GeminiHTTPError(RuntimeError):
    """Sanitized HTTP failure: status, Google's error status, a short
    redacted message, and a digest of the body -- never the raw body."""

    def __init__(
        self,
        *,
        status_code: int,
        error_status: str | None = None,
        message: str | None = None,
        body_sha256: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        detail = f" {error_status}" if error_status else ""
        text = f": {message}" if message else ""
        super().__init__(f"Gemini HTTP {status_code}{detail}{text}")
        self.status_code = status_code
        self.error_status = error_status
        self.message = message
        self.body_sha256 = body_sha256
        self.retry_after_seconds = retry_after_seconds
        self.retryable = status_code in RETRYABLE_HTTP


class GeminiTransportError(RuntimeError):
    """Network-level failure (timeout, reset, unparseable body); retryable."""

    retryable = True

    def __init__(self, kind: str) -> None:
        super().__init__(f"Gemini transport failed ({kind})")
        self.kind = kind


class GeminiResponseError(RuntimeError):
    """A 200 response that violates the generateContent contract (error
    envelope, several candidates, a non-model content). Not retried."""


class GeminiIncompleteTurnError(IncompleteTurnError):
    """The turn did not end in a complete STOP response; history is unchanged.

    ``turn`` carries the public-safe record of the attempt (empty text,
    usage, cost, finish/block reason in ``turn.public``)."""


class GeminiTransport(Protocol):
    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout_seconds: int
    ) -> dict[str, Any]: ...


def _redact(message: str, limit: int) -> str:
    return _OPAQUE_RUN.sub("<redacted>", message)[:limit]


def _retry_delay_seconds(headers: Any, error: Mapping[str, Any]) -> float | None:
    """Server-requested wait: a numeric Retry-After header, else the
    google.rpc.RetryInfo ``retryDelay`` ("12s") in the error details."""
    value = headers.get("Retry-After") if headers is not None else None
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    for detail in error.get("details") or []:
        delay = detail.get("retryDelay") if isinstance(detail, Mapping) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(0.0, float(delay[:-1]))
            except ValueError:
                continue
    return None


class GeminiHTTPTransport:
    """Stdlib JSON POST. The key travels in the ``x-goog-api-key`` header
    (Google's REST examples), never in the URL."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set (use: secret run gemini --env GEMINI_API_KEY -- ...)"
            )
        self._api_key = api_key

    def post(
        self, url: str, payload: Mapping[str, Any], *, timeout_seconds: int
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json", "x-goog-api-key": self._api_key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read() or b""
            error: Mapping[str, Any] = {}
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, Mapping) and isinstance(parsed.get("error"), Mapping):
                    error = parsed["error"]
            except (UnicodeDecodeError, json.JSONDecodeError):
                pass
            message = error.get("message")
            status = error.get("status")
            raise GeminiHTTPError(
                status_code=exc.code,
                error_status=status if isinstance(status, str) else None,
                message=_redact(message, _MAX_ERROR_MESSAGE) if isinstance(message, str) else None,
                body_sha256=hashlib.sha256(raw).hexdigest(),
                retry_after_seconds=_retry_delay_seconds(getattr(exc, "headers", None), error),
            ) from None
        except (OSError, http.client.HTTPException) as exc:
            # URLError, timeouts, resets, TLS errors, IncompleteRead mid-body.
            raise GeminiTransportError(type(exc).__name__) from None
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise GeminiTransportError("InvalidJSON") from None
        if not isinstance(parsed, dict):
            raise GeminiTransportError("NonObjectJSON")
        return parsed


def thinking_config(effort: str, *, include_thoughts: bool) -> dict[str, Any] | None:
    """The ``generationConfig.thinkingConfig`` a session sends, or None to
    omit it. "default" sends no level (true vendor default); "none" sends
    ``thinkingBudget: 0``; anything else is sent verbatim as the level."""
    if not isinstance(effort, str) or not effort.strip():
        raise ValueError("effort must be a nonempty string ('default' = vendor default)")
    config: dict[str, Any] = {}
    if effort == "none":
        config["thinkingBudget"] = 0
    elif effort != "default":
        config["thinkingLevel"] = effort
    if include_thoughts:
        config["includeThoughts"] = True
    return config or None


def gemini_usage_classes(usage_metadata: Mapping[str, Any] | None) -> dict[str, int]:
    """Map ``usageMetadata`` onto the pricing classes (omit, never zero-fill).

    input_tokens = promptTokenCount (total prompt; cached tokens inside);
    cache_read_tokens = cachedContentTokenCount; output_tokens =
    candidatesTokenCount + thoughtsTokenCount (billed output); reasoning_tokens
    = thoughtsTokenCount; tool_use_prompt_tokens = toolUsePromptTokenCount;
    total_tokens = totalTokenCount."""
    meta = usage_metadata if isinstance(usage_metadata, Mapping) else {}

    def count(key: str) -> int | None:
        value = meta.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    classes: dict[str, int] = {}
    prompt, cached = count("promptTokenCount"), count("cachedContentTokenCount")
    candidates, thoughts = count("candidatesTokenCount"), count("thoughtsTokenCount")
    tool_use, total = count("toolUsePromptTokenCount"), count("totalTokenCount")
    if prompt is not None:
        classes["input_tokens"] = prompt
    if cached is not None:
        classes["cache_read_tokens"] = cached
    if candidates is not None or thoughts is not None:
        classes["output_tokens"] = (candidates or 0) + (thoughts or 0)
    if thoughts is not None:
        classes["reasoning_tokens"] = thoughts
    if tool_use is not None:
        classes["tool_use_prompt_tokens"] = tool_use
    if total is not None:
        classes["total_tokens"] = total
    return classes


def _usage_consistent(usage: Mapping[str, int]) -> bool | None:
    """totalTokenCount == prompt + thoughts + candidates (+ tool-use prompt),
    the documented identity the output mapping relies on. None if the
    components needed to check it were not reported."""
    if "total_tokens" not in usage or "input_tokens" not in usage:
        return None
    parts = (
        usage["input_tokens"]
        + usage.get("output_tokens", 0)
        + usage.get("tool_use_prompt_tokens", 0)
    )
    return parts == usage["total_tokens"]


def _reasoning_audit(
    previous: Mapping[str, int] | None, usage: Mapping[str, int]
) -> dict[str, Any]:
    """Per-turn check that earlier reasoning is still in the prompt.

    With preservation, this prompt = previous prompt + previous output
    (answer + thoughts, re-entering at thoughtsTokenCount) + replayed
    summaries + new user text, so ``prompt_growth_tokens`` is >= 0. A clearly
    negative growth means the server left earlier reasoning out of this
    request (observed 2026-09-25 when the user turn said the model's private
    reasoning was in context). Growth >= 0 is NOT proof of preservation when
    the prior thoughts are smaller than the new user text plus summaries."""
    if previous is None:
        return {
            "prompt_growth_tokens": None,
            "prior_turn_thought_tokens": None,
            "reasoning_drop_detected": None,
        }
    prompt = usage.get("input_tokens")
    prev_prompt, prev_output = previous.get("input_tokens"), previous.get("output_tokens")
    growth = (
        prompt - prev_prompt - prev_output
        if prompt is not None and prev_prompt is not None and prev_output is not None
        else None
    )
    return {
        "prompt_growth_tokens": growth,
        "prior_turn_thought_tokens": previous.get("reasoning_tokens"),
        "reasoning_drop_detected": None if growth is None else growth < -_GROWTH_SLACK_TOKENS,
    }


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _part_counts(contents: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    parts = [part for content in contents for part in content.get("parts") or []]
    return {
        "parts": len(parts),
        "thought_parts": sum(1 for part in parts if part.get("thought") is True),
        "signatures": sum(1 for part in parts if _SIGNATURE_KEY in part),
    }


def _user_content(user_texts: Sequence[str]) -> dict[str, Any]:
    if isinstance(user_texts, (str, bytes)) or not isinstance(user_texts, Sequence):
        raise TypeError("user_texts must be a sequence of strings, not a single string")
    if not user_texts:
        raise ValueError("a turn needs at least one user text")
    for text in user_texts:
        if not isinstance(text, str) or not text:
            raise ValueError("every user text must be a nonempty string")
    return {"role": "user", "parts": [{"text": text} for text in user_texts]}


class GeminiNativeSession:
    """One Gemini conversation whose model turns replay natively.

    ``model`` is the bare API model id (no "gemini/" prefix; the registry key
    is "gemini/<model>"). ``instructions`` becomes ``systemInstruction`` on
    every request (None omits it). A session is single-threaded: concurrent
    ``turn`` calls raise; separate sessions are independent."""

    supports_lossless_multi_turn = True

    def __init__(
        self,
        *,
        model: str,
        instructions: str | None,
        effort: str,
        service_tier: str | None = "flex",
        api_key: str | None = None,
        transport: GeminiTransport | None = None,
        pricer: Pricer | None = None,
        include_thoughts: bool = True,
        allow_visible_only: bool = False,
        request_timeout_seconds: int = 1800,
        # Flex sheds load in multi-minute 503 streaks (observed ~6 min on
        # 2026-09-25): back off patiently, ~20 min worst case, then fail.
        max_retries: int = 8,
        retry_initial_seconds: float = 10.0,
        retry_backoff: float = 2.0,
        retry_max_seconds: float = 300.0,
        sleeper: Callable[[float], None] = time.sleep,
        # When flex keeps refusing (429/503) a request, send it on the
        # standard tier after this many flex attempts. None: never fall back.
        # Reported cost is the standard-tier list either way.
        flex_fallback_after: int | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if model.startswith(("gemini/", "models/")):
            raise ValueError("pass the bare API model id (no 'gemini/' or 'models/' prefix)")
        if instructions is not None and (not isinstance(instructions, str) or not instructions):
            raise ValueError("instructions must be a nonempty string or None")
        if service_tier is not None and (not isinstance(service_tier, str) or not service_tier):
            raise ValueError("service_tier must be a nonempty string or None")
        if not isinstance(include_thoughts, bool):
            raise TypeError("include_thoughts must be bool")
        if not isinstance(allow_visible_only, bool):
            raise TypeError("allow_visible_only must be bool")
        if not isinstance(request_timeout_seconds, int) or request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be a positive integer")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")
        if retry_initial_seconds < 0 or retry_max_seconds < 0 or retry_backoff < 1:
            raise ValueError("retry delays must be nonnegative and retry_backoff >= 1")
        if flex_fallback_after is not None and (
            not isinstance(flex_fallback_after, int)
            or isinstance(flex_fallback_after, bool)
            or flex_fallback_after < 1
        ):
            raise ValueError("flex_fallback_after must be a positive integer or None")
        thinking = thinking_config(effort, include_thoughts=include_thoughts)

        # Retention gate, before any credential or network work. Replay is
        # native for every model; the verdict decides whether a run may be
        # called "retained reasoning".
        verdict = retention_verdict(model)
        retained = verdict is not None and verdict["verdict"] == "retained"
        if not retained and not allow_visible_only:
            reason = (
                "has no live retention verdict"
                if verdict is None
                else f"did not retain earlier reasoning (live check {verdict['verified_on']})"
            )
            raise RetentionUnsupportedError(
                f"{model} {reason}; pass allow_visible_only=True to run it as a "
                "labelled unverified/visible-transcript condition"
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
        self.service_tier = service_tier
        self.include_thoughts = include_thoughts
        self.allow_visible_only = allow_visible_only
        self._thinking_config = thinking
        self._retention_verdict = verdict
        self._retention_effective = effective
        self.request_timeout_seconds = request_timeout_seconds
        self.flex_fallback_after = flex_fallback_after
        self.max_retries = max_retries
        self.retry_initial_seconds = float(retry_initial_seconds)
        self.retry_backoff = float(retry_backoff)
        self.retry_max_seconds = float(retry_max_seconds)
        self._sleep = sleeper
        self._pricer = pricer or registry_pricer(f"gemini/{model}")
        self._transport = transport or GeminiHTTPTransport(
            api_key or os.environ.get("GEMINI_API_KEY", "")
        )
        self._url = f"{GEMINI_URL}/{urllib.parse.quote(model, safe='')}:generateContent"
        self._contents: list[dict[str, Any]] = []
        self._last_usage: dict[str, int] | None = None
        self._turn_index = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"GeminiNativeSession(model={self.model!r}, effort={self.effort!r}, "
            f"service_tier={self.service_tier!r}, retention={self._retention_effective!r}, "
            f"turn_index={self._turn_index})"
        )

    @property
    def turn_index(self) -> int:
        """Completed turns so far (the index the next turn will carry)."""
        return self._turn_index

    def native_contents(self) -> list[dict[str, Any]]:
        """PRIVATE opaque state: a deep copy of the native history (user
        contents as sent, model contents exactly as returned, signatures
        included). For private checkpoints and audits; never publish it."""
        return copy.deepcopy(self._contents)

    def build_request(self, user_texts: Sequence[str]) -> dict[str, Any]:
        """The exact body the next ``turn`` would POST; no I/O, no mutation."""
        return self._payload(_user_content(user_texts))

    def turn(self, user_texts: Sequence[str]) -> NativeTurn:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("one GeminiNativeSession cannot be driven concurrently")
        try:
            user_content = _user_content(user_texts)
            payload = self._payload(user_content)
            response, retried = self._post_with_retries(payload)
            return self._accept(response, user_content, payload, retried)
        finally:
            self._lock.release()

    def _payload(self, user_content: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": copy.deepcopy(self._contents) + [copy.deepcopy(dict(user_content))]
        }
        if self.instructions is not None:
            body["systemInstruction"] = {"parts": [{"text": self.instructions}]}
        if self._thinking_config is not None:
            body["generationConfig"] = {"thinkingConfig": copy.deepcopy(self._thinking_config)}
        if self.service_tier:
            body["serviceTier"] = self.service_tier
        return body

    def _post_with_retries(
        self, payload: Mapping[str, Any]
    ) -> tuple[dict[str, Any], list[int | str]]:
        retried: list[int | str] = []
        payload = dict(payload)
        flex_refusals = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = self._transport.post(
                    self._url,
                    copy.deepcopy(payload),
                    timeout_seconds=self.request_timeout_seconds,
                )
                return response, retried
            except GeminiHTTPError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                retried.append(exc.status_code)
                server_delay = exc.retry_after_seconds
                if payload.get("serviceTier") == "flex" and exc.status_code in (429, 503):
                    flex_refusals += 1
                    if self.flex_fallback_after is not None and flex_refusals >= self.flex_fallback_after:
                        # Same request on the standard tier (serviceTier omitted).
                        payload.pop("serviceTier")
                        retried.append("fallback_standard")
                        continue
            except GeminiTransportError as exc:
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

    def _accept(
        self,
        response: Mapping[str, Any],
        user_content: Mapping[str, Any],
        payload: Mapping[str, Any],
        retried: list[int | str],
    ) -> NativeTurn:
        if not isinstance(response, Mapping):
            raise GeminiResponseError("generateContent response is not an object")
        if response.get("error"):
            raise GeminiResponseError("generateContent returned an error envelope")
        candidates = response.get("candidates") or []
        if not isinstance(candidates, list) or len(candidates) > 1:
            raise GeminiResponseError("expected at most one candidate")
        candidate = candidates[0] if candidates else {}
        if not isinstance(candidate, Mapping):
            raise GeminiResponseError("candidate is not an object")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        has_parts = isinstance(parts, list) and bool(parts)
        if has_parts:
            if content.get("role") != "model":
                raise GeminiResponseError("candidate content role is not 'model'")
            if not all(isinstance(part, Mapping) for part in parts):
                raise GeminiResponseError("candidate parts must be objects")
        finish_reason = candidate.get("finishReason")
        complete = has_parts and finish_reason == ACCEPTED_FINISH_REASON

        text = (
            "".join(
                part["text"]
                for part in parts
                if part.get("thought") is not True and isinstance(part.get("text"), str)
            )
            if has_parts
            else ""
        )
        summaries = (
            tuple(
                part["text"]
                for part in parts
                if part.get("thought") is True and isinstance(part.get("text"), str)
            )
            if has_parts
            else ()
        )
        usage_metadata = response.get("usageMetadata")
        usage = gemini_usage_classes(usage_metadata)
        consistent = _usage_consistent(usage)
        cost = None
        if (
            "input_tokens" in usage
            and "output_tokens" in usage
            and consistent is not False
            and not usage.get("tool_use_prompt_tokens")
        ):
            cost = self._pricer({key: usage[key] for key in _PRICED_CLASSES if key in usage})
        returned_tier = (
            usage_metadata.get("serviceTier") if isinstance(usage_metadata, Mapping) else None
        )
        feedback = response.get("promptFeedback")
        block_reason = feedback.get("blockReason") if isinstance(feedback, Mapping) else None
        finish_message = candidate.get("finishMessage")
        replayed = _part_counts([c for c in self._contents if c.get("role") == "model"])

        public: dict[str, Any] = {
            "provider": "gemini",
            "api": "generateContent",
            "model": self.model,
            "turn_index": self._turn_index,
            "complete": complete,
            "retention": {
                "replayed": REPLAY_MODE,
                "prior_turns": self._turn_index,
                "replayed_parts": replayed["parts"],
                "replayed_thought_parts": replayed["thought_parts"],
                "replayed_signatures": replayed["signatures"],
                "verdict": self._retention_verdict["verdict"] if self._retention_verdict else None,
                "verified_on": (
                    self._retention_verdict["verified_on"] if self._retention_verdict else None
                ),
                "effective": self._retention_effective,
                **_reasoning_audit(self._last_usage, usage),
            },
            "effort": self.effort,
            "thinking_config": copy.deepcopy(self._thinking_config),
            "service_tier": {"requested": self.service_tier, "returned": returned_tier},
            "finish_reason": finish_reason,
            "output": _part_counts([content]) if has_parts else {"parts": 0},
            "usage": dict(usage),
            "usage_identity_holds": consistent,
            "cost_usd": cost,
            "date_utc": today_utc(),
            "request_sha256": _sha256_json(payload),
            "http": {
                "attempts": sum(1 for r in retried if r != "fallback_standard") + 1,
                "retried": list(retried),
            },
        }
        if block_reason is not None:
            public["block_reason"] = block_reason
        if isinstance(finish_message, str) and finish_message:
            public["finish_message"] = _redact(finish_message, _MAX_FINISH_MESSAGE)
        provenance = build_provenance(
            response_id=response.get("responseId"),
            model_version=response.get("modelVersion"),
            service_tier=returned_tier,
            finish_reason=finish_reason,
        )
        assert_public_safe(public)
        assert_public_safe(provenance)
        turn = NativeTurn(
            text=text if complete else "",  # never score a partial answer
            usage=usage,
            reasoning_summaries=summaries,
            provenance=provenance,
            public=public,
            cost_usd=cost,
        )
        if not complete:
            reason = block_reason or finish_reason or "no candidate"
            raise GeminiIncompleteTurnError(
                f"Gemini turn {self._turn_index} incomplete ({reason}); history unchanged",
                turn=turn,
            )
        self._contents.append(copy.deepcopy(dict(user_content)))
        self._contents.append(copy.deepcopy(dict(content)))
        self._last_usage = dict(usage)
        self._turn_index += 1
        return turn


__all__ = [
    "ACCEPTED_FINISH_REASON",
    "RETENTION_VERDICTS",
    "RETRYABLE_HTTP",
    "GeminiHTTPError",
    "GeminiHTTPTransport",
    "GeminiIncompleteTurnError",
    "GeminiNativeSession",
    "GeminiResponseError",
    "GeminiTransport",
    "GeminiTransportError",
    "gemini_usage_classes",
    "retention_verdict",
    "thinking_config",
]
