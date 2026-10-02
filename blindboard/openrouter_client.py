# Vendored from benchkit@0.3.10 (sha256:a78487f445197027). Do not hand-edit; run 'benchkit sync' to update.
"""OpenRouter chat completions API adapter (stdlib only).

Ported 2026-07-19 from scripts/research/reliability_abstention_pilot.py
(call_openrouter_chat) in benchmark-copybench, into the kit, with the
interactive-runner interface (call_with_meta, usage_totals) of
blindboard.client.APIAdapter. URL, auth header, and the bare
vendor/slug model convention also cross-checked against
obviousbench/runners/openrouter_batches.py's OpenRouter conventions
(OPENROUTER_API_KEY env, Bearer auth, model given as
openrouter/<vendor>/<slug> with the "openrouter/" prefix stripped before
calling the API).

- Request shape copied verbatim: temperature 0, usage={"include": True}
  for token accounting, and OpenRouter's unified `reasoning` parameter
  (effort passes through unvalidated; effort == "none" sends
  {"enabled": False} instead). Multi-turn Message lists map onto standard
  OpenAI-style chat roles (system/user/assistant) unchanged — the source
  only ever sent a single user turn ({"role": "user", "content": prompt}).
  The adapter rejects assistant history by default because a text-only
  message cannot preserve structured `reasoning_details`.
- Response extraction copied verbatim: choices[0].message.content is the
  text; usage.prompt_tokens/completion_tokens map to input/output tokens;
  completion_tokens_details.reasoning_tokens is surfaced again as
  thinking_tokens (an informational breakdown already folded into
  output_tokens, matching the Anthropic/Gemini adapters' handling); and
  finish_reason is folded to "completed" for {"stop", "end_turn", None}
  exactly as the source's own `status` field is computed (renamed to
  stop_reason for cross-adapter meta parity).
- Reasoning text: the source only ever surfaced a reasoning_tokens count,
  never trace text. Since the request already asks for reasoning (via the
  `reasoning` param above), we additionally collect
  choices[0].message.reasoning when the provider returns it, purely on
  the response-parsing side, so meta["reasoning"] matches the other
  adapters' shape. No new request field.
- service_tier is accepted-and-ignored, like the Anthropic adapter's
  interface-parity parameter: OpenRouter's per-request routing has no
  discount tier analogous to Gemini's flex or Anthropic's batch.
- Provenance: OpenRouter is a broker, so the response's own id/provider/
  model fields (which host served the turn, on which slug) are retained in
  meta["provenance"] — see blindboard.client.build_provenance.
- Effort binding is FAIL-CLOSED: see OpenRouterAdapter's docstring.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from blindboard.pricing import list_price_usd
from blindboard.client import Message, build_provenance, text_conversation_state

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Effort-binding verification (lesson 2026-07-19, qwen3.6-27b: the unified
# reasoning.effort field is silently ignored by models whose route does not
# list reasoning_effort in supported_parameters — an "effort sweep" on such
# a model re-runs one configuration). Cache the models index once per
# process and record, per adapter, whether the requested effort can bind.
_SUPPORTED_PARAMS_CACHE: dict[str, list[str]] | None = None


def effort_binds(model: str) -> bool | None:
    """True/False when the models index answers; None when it cannot be
    consulted (endpoint unreachable, or the model absent from the index).

    None is NOT a pass: it means the claim "this run swept effort" is
    unverified, and the adapter treats it as a failure unless the caller
    explicitly opted out (see OpenRouterAdapter.require_effort_binding)."""
    global _SUPPORTED_PARAMS_CACHE
    if _SUPPORTED_PARAMS_CACHE is None:
        try:
            with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))["data"]
            _SUPPORTED_PARAMS_CACHE = {
                m["id"]: list(m.get("supported_parameters") or []) for m in data
            }
        except Exception:  # noqa: BLE001 - metadata only
            return None
    params = _SUPPORTED_PARAMS_CACHE.get(model)
    if params is None:
        return None
    return "reasoning_effort" in params


class EffortBindingError(RuntimeError):
    """A graded reasoning effort was requested on a route that cannot be
    shown to honour it. Raised at construction, before any billed call."""

DEFAULT_MAX_RETRIES = 6
DEFAULT_RETRY_TIMEOUT_SECONDS = 900.0


def is_retryable_openrouter_code(code: object) -> bool:
    """True for transient OpenRouter transport and overload status codes."""
    return isinstance(code, int) and not isinstance(code, bool) and (
        code in {408, 429} or 500 <= code < 600
    )


class OpenRouterResponseError(RuntimeError):
    """Structured OpenRouter error from either HTTP status or response body."""

    def __init__(self, *, code: int | None, detail: str, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        label = str(code) if code is not None else "unknown"
        super().__init__(f"OpenRouter error {label}: {detail[:300]}")


def openrouter_response_error(resp: dict[str, Any]) -> OpenRouterResponseError | None:
    """Classify an OpenRouter error envelope, including HTTP-200 envelopes."""
    error = resp.get("error")
    if not isinstance(error, dict):
        return None
    raw_code = error.get("code")
    code = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
    detail = str(error.get("message") or json.dumps(error, sort_keys=True))
    return OpenRouterResponseError(
        code=code,
        detail=detail,
        retryable=is_retryable_openrouter_code(code),
    )


def _retry_after_seconds(
    *,
    headers: Mapping[str, str] | None = None,
    response: dict[str, Any] | None = None,
) -> float | None:
    """Read Retry-After from HTTP headers or OpenRouter error metadata."""
    candidates: list[Mapping[str, Any]] = []
    if headers is not None:
        candidates.append(headers)
    if response is not None:
        error = response.get("error")
        metadata = error.get("metadata") if isinstance(error, dict) else None
        embedded_headers = metadata.get("headers") if isinstance(metadata, dict) else None
        if isinstance(embedded_headers, dict):
            candidates.append(embedded_headers)
    for candidate in candidates:
        value = next(
            (value for key, value in candidate.items() if str(key).casefold() == "retry-after"),
            None,
        )
        try:
            seconds = float(value) if value is not None else None
        except (TypeError, ValueError):
            continue
        if seconds is not None and seconds > 0:
            return seconds
    return None


def build_openrouter_body(*, model: str, messages: list[Message], effort: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.text} for m in messages],
        "temperature": 0,
        # Explicit cap: without it the routed host's default completion cap
        # applies, and thinking-heavy models burn it entirely inside the
        # reasoning channel -> finish_reason "length" with EMPTY content
        # (qwen3.6-35b-a3b t2 screen, 2026-07-19: 38 forfeited turns). The
        # port source ran short single-turn tasks where defaults sufficed;
        # OpenRouter clamps to each model's own max, so a high value is safe.
        "max_tokens": 64000,
        "usage": {"include": True},
    }
    if effort == "none":
        payload["reasoning"] = {"enabled": False}
    else:
        payload["reasoning"] = {"effort": effort}
    return payload


def normalize_openrouter_response(resp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(text, meta) in the shape call_with_meta returns: meta carries
    reasoning text (when the provider returns it), a usage dict, and the
    folded finish reason."""
    if error := openrouter_response_error(resp):
        raise error
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    reasoning_text = message.get("reasoning")
    summaries = [reasoning_text] if isinstance(reasoning_text, str) and reasoning_text else []
    usage = resp.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    meta_usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    details = usage.get("completion_tokens_details") or {}
    if "reasoning_tokens" in details:
        meta_usage["thinking_tokens"] = int(details["reasoning_tokens"] or 0)
    finish_reason = choice.get("finish_reason")
    meta: dict[str, Any] = {
        "reasoning": summaries,
        "usage": meta_usage,
        "stop_reason": "completed" if finish_reason in {"stop", "end_turn", None} else finish_reason,
        # OpenRouter is a broker: `provider` names the upstream host that
        # actually served the turn and `model` the slug it served, both of
        # which can differ from the request when routing or fallback kicks
        # in. finish_reason is kept RAW here (stop_reason above is folded),
        # so a dispute sees what the host reported.
        "provenance": build_provenance(
            response_id=resp.get("id"),
            provider=resp.get("provider"),
            routed_model=resp.get("model"),
            finish_reason=finish_reason,
        ),
    }
    return text, meta


class OpenRouterAdapter:
    """Single-inference text adapter for OpenRouter chat-
    completions models. `model` is the OpenRouter vendor/slug (e.g.
    "deepseek/deepseek-v4-pro") — no "openrouter/" prefix (make_adapter
    strips it). service_tier is accepted-and-ignored for interface parity.

    Effort binding is FAIL-CLOSED (require_effort_binding=True, default).
    Requesting a graded effort on a route the models index does not show
    honouring reasoning_effort raises EffortBindingError at construction,
    before a single billed call — because such a run is not an effort
    sweep, it is one configuration collected N times under N labels
    (qwen3.6-27b, 2026-07-19). A failed lookup raises for the same reason:
    an unverified binding claim is not a verified one.

    ESCAPE HATCH — require_effort_binding=False is for DELIBERATE
    single-mode collection, where the point is to characterize the route's
    one thinking mode rather than to compare efforts. It warns instead of
    raising and stamps effort_binding=False onto the adapter, so the run
    artifact records that the effort label is not a graded setting. Never
    use it to make an effort sweep run; use it to declare that the run is
    not one."""

    supports_lossless_multi_turn = False

    def __init__(
        self,
        *,
        model: str,
        effort: str = "medium",
        service_tier: str | None = None,  # interface parity; no discount tier
        timeout_seconds: int = 900,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_sleep_seconds: float = 5.0,
        retry_timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
        api_key: str | None = None,
        require_effort_binding: bool = True,
        allow_lossy_multi_turn: bool = False,
    ) -> None:
        self.model = model
        self.effort = effort
        self.service_tier = None
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.retry_timeout_seconds = retry_timeout_seconds
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        self._lock = threading.Lock()
        self.require_effort_binding = require_effort_binding
        self.allow_lossy_multi_turn = allow_lossy_multi_turn
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set (use: secret run openrouter --env OPENROUTER_API_KEY -- ...)"
            )
        binding = effort_binds(model)
        # effort == "none" asks for reasoning to be switched OFF
        # ({"enabled": False}), which is not a graded setting and needs no
        # reasoning_effort support: nothing to verify, nothing to fail.
        if effort not in ("none",) and binding is not True:
            why = (
                f"the OpenRouter models index does not list reasoning_effort in "
                f"{model}'s supported_parameters"
                if binding is False
                else f"{model}'s supported_parameters could not be read "
                "(models index unreachable, or model absent from it)"
            )
            if require_effort_binding:
                raise EffortBindingError(
                    f"refusing to collect: effort {effort!r} cannot be shown to bind "
                    f"on {model} — {why}. A graded effort that does not bind makes "
                    "an 'effort sweep' re-run one configuration under several "
                    "labels. Pass require_effort_binding=False to collect this "
                    "route deliberately as single-mode (recorded as "
                    "effort_binding=False)."
                )
            print(
                f"WARNING: proceeding with require_effort_binding=False — {why}; "
                f"requested effort {effort!r} will NOT bind (single thinking "
                "mode); recording effort_binding=False.",
                flush=True,
            )
            binding = False
        self.effort_binding = binding

    def __call__(self, messages: list[Message]) -> str:
        return self.call_with_meta(messages)[0]

    def call_with_meta(self, messages: list[Message]) -> tuple[str, dict[str, Any]]:
        conversation_state = text_conversation_state(
            messages,
            provider="OpenRouter Chat Completions",
            allow_lossy_multi_turn=self.allow_lossy_multi_turn,
        )
        body = build_openrouter_body(model=self.model, messages=messages, effort=self.effort)
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        resp: dict[str, Any] | None = None
        retry_deadline = time.monotonic() + self.retry_timeout_seconds
        for retry in range(self.max_retries + 1):
            request = urllib.request.Request(
                OPENROUTER_URL, data=data, headers=headers, method="POST"
            )
            try:
                remaining_seconds = retry_deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise last_exc or TimeoutError("OpenRouter retry timeout exhausted")
                with urllib.request.urlopen(
                    request,
                    timeout=min(self.timeout_seconds, remaining_seconds),
                ) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                body_error = openrouter_response_error(resp)
                if body_error is None:
                    break
                last_exc = body_error
                if not body_error.retryable or retry >= self.max_retries:
                    raise body_error
                delay = _retry_after_seconds(response=resp) or self.retry_sleep_seconds * (
                    2**retry
                )
                if time.monotonic() + delay > retry_deadline:
                    raise body_error
                time.sleep(delay)
                resp = None
            except OpenRouterResponseError:
                raise
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_exc = OpenRouterResponseError(
                    code=exc.code,
                    detail=detail,
                    retryable=is_retryable_openrouter_code(exc.code),
                )
                if not last_exc.retryable or retry >= self.max_retries:
                    raise last_exc from exc
                delay = _retry_after_seconds(headers=exc.headers) or self.retry_sleep_seconds * (
                    2**retry
                )
                if time.monotonic() + delay > retry_deadline:
                    raise last_exc from exc
                time.sleep(delay)
            except Exception as exc:
                last_exc = exc
                if retry >= self.max_retries:
                    raise
                delay = self.retry_sleep_seconds * (2**retry)
                if time.monotonic() + delay > retry_deadline:
                    raise
                time.sleep(delay)
        if resp is None:  # pragma: no cover
            raise last_exc or RuntimeError("retry loop exited without response")
        text, meta = normalize_openrouter_response(resp)
        meta["conversation_state"] = conversation_state
        with self._lock:
            self.calls += 1
            for key, value in meta["usage"].items():
                self.usage_totals[key] = self.usage_totals.get(key, 0) + int(value)
        return text, meta

    def cost_usd(self) -> float | None:
        """Undiscounted list price; registry keys OpenRouter models as
        'openrouter/<vendor>/<slug>'."""
        return list_price_usd(f"openrouter/{self.model}", self.usage_totals)
