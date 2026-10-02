# Vendored from benchkit@0.3.10 (sha256:1b70e5f8a3ddea44). Do not hand-edit; run 'benchkit sync' to update.
"""Gemini generateContent API adapter (stdlib only).

Ported 2026-07-19 from scripts/research/reliability_abstention_pilot.py
(call_gemini_generate) in benchmark-copybench, into the kit, with the
interactive-runner interface (call_with_meta, usage_totals) of
blindboard.client.APIAdapter. Request/response shapes copied
verbatim from that source; only the parts needed for a multi-turn
Message-based adapter (see below) are new.

- Flex service tier: the source hardcodes "serviceTier": "flex" — Google's
  discounted collection tier, NOT a batch API. Here service_tier defaults
  to "flex" and is included verbatim when truthy; service_tier=None omits
  the field entirely (interface parity with the OpenAI Responses adapter's
  own "if service_tier: payload[...] = service_tier" pattern).
- Thinking control: the source sets generationConfig.thinkingConfig to
  {"thinkingBudget": 0} when effort == "none", else {"thinkingLevel":
  effort} — passing our effort string straight through with no mapping or
  validation. Copied exactly; Gemini-unsupported effort strings are the
  caller's problem, same as in the source.
- Message body construction: the source only ever sent a single-turn prompt
  ("contents": [{"parts": [{"text": prompt}]}], no "role", no system
  turn). Building a `contents` list from a Message list is new: system
  messages join into a top-level "systemInstruction", and user/assistant
  messages map to Gemini's "user"/"model" content roles. The adapter itself
  rejects assistant history by default because this text-only representation
  cannot preserve Gemini thought signatures.
- Response extraction: candidates[0].content.parts, "thought"-flagged
  parts excluded from the joined text — copied exactly. The source then
  discards those thought parts' text entirely; we instead collect it into
  meta["reasoning"] so this adapter's meta shape matches the Anthropic
  adapter's (reasoning/usage/stop_reason). That is response-parsing only,
  no new request field.
- Usage mapping: promptTokenCount -> input_tokens; candidatesTokenCount +
  thoughtsTokenCount -> output_tokens (the source already combines these
  two into one output figure); thoughtsTokenCount is also surfaced as
  thinking_tokens (an informational breakdown already folded into
  output_tokens, not double-counted), mirroring the Anthropic adapter's
  output_tokens_details.thinking_tokens handling.
- Provenance: modelVersion (the build actually served behind a preview
  alias) and responseId are retained in meta["provenance"] — see
  blindboard.client.build_provenance.
- Error handling: the source only raises via HTTPError. We additionally
  raise on a top-level {"error": {...}} body (Google's standard API error
  envelope) for parity with the Anthropic/OpenRouter adapters' normalize
  functions, which both guard against a non-HTTP error payload.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from blindboard.pricing import list_price_usd
from blindboard.client import Message, build_provenance, text_conversation_state

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# HTTP statuses where a retry cannot help (auth, malformed request).
FATAL_HTTP = {400, 401, 403, 404}


def build_gemini_body(
    *,
    model: str,
    messages: list[Message],
    effort: str,
    service_tier: str | None,
) -> dict[str, Any]:
    system_text = "\n\n".join(m.text for m in messages if m.role == "system")
    contents = [
        {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.text}]}
        for m in messages
        if m.role in ("user", "assistant")
    ]
    generation_config: dict[str, Any] = {}
    if effort == "none":
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}
    else:
        generation_config["thinkingConfig"] = {"thinkingLevel": effort}
    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if service_tier:
        body["serviceTier"] = service_tier
    return body


def normalize_gemini_response(resp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(text, meta) in the shape call_with_meta returns: meta carries
    reasoning (thought-flagged parts), a usage dict, and the finish
    reason."""
    if resp.get("error"):
        raise RuntimeError(f"Gemini error: {json.dumps(resp)[:300]}")
    cand = (resp.get("candidates") or [{}])[0]
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(pt.get("text", "") for pt in parts if not pt.get("thought"))
    summaries = [pt["text"] for pt in parts if pt.get("thought") and pt.get("text")]
    usage = resp.get("usageMetadata") or {}
    input_tokens = int(usage.get("promptTokenCount", 0) or 0)
    thoughts_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)
    output_tokens = int(usage.get("candidatesTokenCount", 0) or 0) + thoughts_tokens
    meta_usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if "thoughtsTokenCount" in usage:
        meta_usage["thinking_tokens"] = thoughts_tokens
    meta: dict[str, Any] = {
        "reasoning": summaries,
        "usage": meta_usage,
        "stop_reason": cand.get("finishReason"),
        # Gemini names the build it served in `modelVersion` (a preview
        # alias resolves to a dated build) and the call in `responseId`.
        # There is no broker in front of it, so no provider field.
        "provenance": build_provenance(
            response_id=resp.get("responseId"),
            model_version=resp.get("modelVersion"),
        ),
    }
    return text, meta


class GeminiAdapter:
    """Single-inference text adapter for gemini-* models (direct
    Gemini API — not OpenRouter). `model` arrives WITHOUT the "gemini/"
    prefix (make_adapter strips it). service_tier defaults to "flex"
    (Google's discounted tier, distinct from any batch API); None omits
    the field from the request. Assistant history is rejected by default."""

    supports_lossless_multi_turn = False

    def __init__(
        self,
        *,
        model: str,
        effort: str = "medium",
        service_tier: str | None = "flex",
        timeout_seconds: int = 900,
        max_retries: int = 3,
        retry_sleep_seconds: float = 5.0,
        api_key: str | None = None,
        allow_lossy_multi_turn: bool = False,
    ) -> None:
        self.model = model
        self.effort = effort
        self.service_tier = service_tier
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.allow_lossy_multi_turn = allow_lossy_multi_turn
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        self._lock = threading.Lock()
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set (use: secret run gemini --env GEMINI_API_KEY -- ...)"
            )

    def __call__(self, messages: list[Message]) -> str:
        return self.call_with_meta(messages)[0]

    def call_with_meta(self, messages: list[Message]) -> tuple[str, dict[str, Any]]:
        conversation_state = text_conversation_state(
            messages,
            provider="Gemini generateContent",
            allow_lossy_multi_turn=self.allow_lossy_multi_turn,
        )
        body = build_gemini_body(
            model=self.model,
            messages=messages,
            effort=self.effort,
            service_tier=self.service_tier,
        )
        data = json.dumps(body).encode("utf-8")
        slug = urllib.parse.quote(self.model, safe="")
        url = f"{GEMINI_URL}/{slug}:generateContent?key={self.api_key}"
        headers = {"content-type": "application/json"}
        last_exc: Exception | None = None
        resp: dict[str, Any] | None = None
        for retry in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"Gemini HTTP {exc.code}: {detail[:300]}")
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
        text, meta = normalize_gemini_response(resp)
        meta["conversation_state"] = conversation_state
        with self._lock:
            self.calls += 1
            for key, value in meta["usage"].items():
                self.usage_totals[key] = self.usage_totals.get(key, 0) + int(value)
        return text, meta

    def cost_usd(self) -> float | None:
        """Undiscounted list price; registry keys Gemini direct-API models
        as 'gemini/<model>'. Flex is a collection-only discount and is
        never reflected here (house rule: report list price)."""
        return list_price_usd(f"gemini/{self.model}", self.usage_totals)
