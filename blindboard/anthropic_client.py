# Vendored from benchkit@0.3.10 (sha256:0c72b6220de01be6). Do not hand-edit; run 'benchkit sync' to update.
"""Anthropic Messages API adapter (stdlib only) + provider dispatch.

Ported 2026-07-18 from obviousbench/runners/anthropic_messages.py into the
kit, with the interactive-runner interface (call_with_meta, usage_totals)
of blindboard.client.APIAdapter. Parameter shapes verified against
the claude-api reference, 2026-07:

- Adaptive-thinking models (fable-5, opus-4.7+, sonnet-5): thinking
  {"type": "adaptive", "display": "summarized"} + output_config.effort in
  {low, medium, high, xhigh, max}. display must be requested explicitly —
  the default is "omitted" and thinking blocks come back with EMPTY text
  (copybench 2026-06-03 adaptive-thinking diagnosis). The summary is a
  separate renderer's text, logged for debugging, never fed back.
- Pre-adaptive models (haiku-4-5): legacy {"type": "enabled",
  "budget_tokens": N}; adaptive/effort are rejected there.
- No temperature in either mode (rejected alongside thinking).
- Anthropic has no flex tier; batch is its only collection discount and
  lives in the batch runner world, not this sync adapter.
- Provenance: the response's id, resolved `model` (an alias comes back as
  the dated snapshot that served it) and stop_reason are retained in
  meta["provenance"] — see blindboard.client.build_provenance.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from blindboard.gemini_client import GeminiAdapter
from blindboard.openrouter_client import OpenRouterAdapter
from blindboard.pricing import list_price_usd
from blindboard.client import (
    APIAdapter,
    Message,
    build_provenance,
    text_conversation_state,
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

ADAPTIVE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
LEGACY_MARKERS = ("haiku-4-5",)
LEGACY_BUDGETS = {"low": 8000, "medium": 16000, "high": 32000, "xhigh": 32000, "max": 32000}
# HTTP statuses where a retry cannot help (auth, malformed request).
FATAL_HTTP = {400, 401, 403, 404}


def build_anthropic_body(*, model: str, messages: list[Message], effort: str) -> dict[str, Any]:
    system_text = "\n\n".join(m.text for m in messages if m.role == "system")
    turns = [
        {"role": m.role, "content": m.text}
        for m in messages
        if m.role in ("user", "assistant")
    ]
    body: dict[str, Any] = {
        "model": model,
        "messages": turns,
        # Top-level auto-caching: caches the growing multi-turn prefix.
        # Cache discounts are collection-only; reported costs stay at list
        # prices on the full raw prompt (see normalize_anthropic_response).
        "cache_control": {"type": "ephemeral"},
    }
    if system_text:
        body["system"] = system_text
    if any(marker in model for marker in LEGACY_MARKERS):
        body["max_tokens"] = 64000
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": LEGACY_BUDGETS.get(effort, 16000),
        }
    else:
        body["max_tokens"] = 128000
        body["thinking"] = {"type": "adaptive", "display": "summarized"}
        if effort in ADAPTIVE_EFFORTS:
            body["output_config"] = {"effort": effort}
    return body


def normalize_anthropic_response(resp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(text, meta) in the shape call_with_meta returns: meta carries
    reasoning summaries, a usage dict, and the stop reason."""
    if resp.get("type") == "error":
        raise RuntimeError(f"Anthropic error: {json.dumps(resp)[:300]}")
    content = resp.get("content") or []
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    summaries = [
        b["thinking"]
        for b in content
        if b.get("type") == "thinking" and b.get("thinking")
    ]
    usage = resp.get("usage") or {}
    # List-price accounting: cache reads/writes are billed at a discount,
    # but the raw prompt = input + cache_read + cache_creation. Fold them
    # into input_tokens so list_price_usd prices the undiscounted prompt.
    input_tokens = (
        int(usage.get("input_tokens", 0) or 0)
        + int(usage.get("cache_read_input_tokens", 0) or 0)
        + int(usage.get("cache_creation_input_tokens", 0) or 0)
    )
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    meta_usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    details = usage.get("output_tokens_details") or {}
    if "thinking_tokens" in details:  # billed thinking, opus-4.5+ era
        meta_usage["thinking_tokens"] = int(details["thinking_tokens"] or 0)
    meta: dict[str, Any] = {
        "reasoning": summaries,
        "usage": meta_usage,
        "stop_reason": resp.get("stop_reason"),
        # `model` is the resolved build: a request for an alias comes back
        # naming the dated snapshot that actually answered. `id` is the
        # message handle. stop_reason is repeated here so the provenance
        # block reads standalone across providers.
        "provenance": build_provenance(
            response_id=resp.get("id"),
            routed_model=resp.get("model"),
            finish_reason=resp.get("stop_reason"),
        ),
    }
    return text, meta


class AnthropicAdapter:
    """Single-inference text adapter for claude-* models.

    Assistant history is rejected by default because ``Message`` cannot carry
    complete signed thinking/tool blocks. The lossy opt-in is legacy-only.
    """

    supports_lossless_multi_turn = False

    def __init__(
        self,
        *,
        model: str,
        effort: str = "medium",
        service_tier: str | None = None,  # interface parity; no flex tier
        timeout_seconds: int = 900,
        max_retries: int = 3,
        retry_sleep_seconds: float = 5.0,
        api_key: str | None = None,
        allow_lossy_multi_turn: bool = False,
    ) -> None:
        self.model = model
        self.effort = effort
        self.service_tier = None
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.allow_lossy_multi_turn = allow_lossy_multi_turn
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        self._lock = threading.Lock()
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set (use: secret run anthropic --env ANTHROPIC_API_KEY -- ...)"
            )

    def __call__(self, messages: list[Message]) -> str:
        return self.call_with_meta(messages)[0]

    def call_with_meta(self, messages: list[Message]) -> tuple[str, dict[str, Any]]:
        conversation_state = text_conversation_state(
            messages,
            provider="Anthropic Messages",
            allow_lossy_multi_turn=self.allow_lossy_multi_turn,
        )
        body = build_anthropic_body(model=self.model, messages=messages, effort=self.effort)
        data = json.dumps(body).encode("utf-8")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        last_exc: Exception | None = None
        resp: dict[str, Any] | None = None
        for retry in range(self.max_retries + 1):
            request = urllib.request.Request(
                ANTHROPIC_URL, data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"Anthropic HTTP {exc.code}: {detail[:300]}")
                if exc.code in FATAL_HTTP or retry >= self.max_retries:
                    raise last_exc from exc
                time.sleep(self.retry_sleep_seconds * (2**retry))
            except Exception as exc:
                last_exc = exc
                if retry >= self.max_retries:
                    raise
                time.sleep(self.retry_sleep_seconds * (2**retry))
        if resp is None:  # pragma: no cover
            raise last_exc or RuntimeError("retry loop exited without response")
        text, meta = normalize_anthropic_response(resp)
        meta["conversation_state"] = conversation_state
        with self._lock:
            self.calls += 1
            for key, value in meta["usage"].items():
                self.usage_totals[key] = self.usage_totals.get(key, 0) + int(value)
        return text, meta

    def cost_usd(self) -> float | None:
        """Undiscounted list price; registry keys Anthropic direct-API
        models as 'anthropic/<model>'."""
        cost = list_price_usd(f"anthropic/{self.model}", self.usage_totals)
        if cost is None:
            cost = list_price_usd(self.model, self.usage_totals)
        return cost


def make_adapter(
    *,
    model: str,
    effort: str = "medium",
    service_tier: str | None = "flex",
    require_effort_binding: bool = True,
    **kwargs: Any,
):
    """Provider dispatch by model name. Accepts bare model IDs,
    'anthropic/<id>', 'gemini/<id>', or 'openrouter/<vendor>/<slug>';
    everything else goes to the OpenAI Responses adapter (which ignores
    unknown models only at pricing).

    require_effort_binding is OpenRouter-specific (only a broker can route
    a graded effort to a host that ignores it) and is accepted-and-dropped
    for the other providers, so a runner can expose one flag without
    branching on provider. Default True = fail closed."""
    if model.startswith(("claude", "anthropic/")):
        slug = model.split("/", 1)[1] if "/" in model else model
        return AnthropicAdapter(
            model=slug, effort=effort, service_tier=service_tier, **kwargs
        )
    if model.startswith("gemini/"):
        return GeminiAdapter(
            model=model.split("/", 1)[1], effort=effort, service_tier=service_tier, **kwargs
        )
    if model.startswith("openrouter/"):
        return OpenRouterAdapter(
            model=model.split("/", 1)[1],
            effort=effort,
            service_tier=service_tier,
            require_effort_binding=require_effort_binding,
            **kwargs,
        )
    return APIAdapter(model=model, effort=effort, service_tier=service_tier, **kwargs)
