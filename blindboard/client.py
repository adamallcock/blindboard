# Vendored from benchkit@0.3.10 (sha256:5764a70faab35bd9). Do not hand-edit; run 'benchkit sync' to update.
"""OpenAI Responses API adapter (stdlib only), ported from the gamebench/
copybench pilot conventions: bounded retries, usage accumulation, flex
service tier for collection. Pricing follows the house rule (copybench
build_report_tables.py, 2026-07-11): published costs use undiscounted list
prices; flex/batch discounts reduce what we PAY but never what we REPORT.

Also home to the cross-provider PROVENANCE contract (build_provenance /
provenance_rollup), which every adapter in benchkit.client uses so a run
artifact can name the exact implementation that produced it."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from blindboard.pricing import PRICES

API_URL = "https://api.openai.com/v1/responses"

# --- Provenance ----------------------------------------------------------
# What a dispute needs in order to name the implementation that answered:
# WHICH response (response_id), on WHOSE hardware (provider), running
# WHICH build (routed_model / model_version), at WHICH billing tier
# (service_tier), and how the turn ENDED (finish_reason). Every adapter
# fills the subset its API actually returns; nothing is synthesized, so an
# absent field is absent rather than null.
PROVENANCE_FIELDS = (
    "response_id",
    "provider",
    "routed_model",
    "model_version",
    "service_tier",
    "finish_reason",
)
# Identity fields worth summarizing across a run. response_id is excluded
# on purpose (one distinct value per call — no summary meaning; it stays
# per-call in the transcript for dispute-level lookup), as is finish_reason
# (per-turn outcome, already carried by meta["stop_reason"]).
ROLLUP_FIELDS = ("provider", "routed_model", "model_version", "service_tier")


def build_provenance(**fields: Any) -> dict[str, Any]:
    """Provenance sub-dict for meta, keeping only what the API returned.

    None/"" values are dropped so a field the provider did not send never
    appears as a null claim about routing."""
    unknown = set(fields) - set(PROVENANCE_FIELDS)
    if unknown:  # pragma: no cover - programmer error
        raise ValueError(f"unknown provenance field(s): {sorted(unknown)}")
    return {k: v for k, v in fields.items() if v is not None and v != ""}


def provenance_rollup(metas: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Distinct routed hosts / model builds / tiers seen across a run.

    Accepts per-call meta dicts, or transcript entries that carry meta
    wholesale (both hold provenance under the same key). Returns sorted
    distinct values for every ROLLUP_FIELDS key at least one call
    reported; keys nobody reported are omitted. More than one value for a
    key means the run was NOT served by a single implementation."""
    seen: dict[str, set[str]] = {}
    for meta in metas:
        if not isinstance(meta, Mapping):
            continue
        prov = meta.get("provenance")
        if not isinstance(prov, Mapping):
            continue
        for field in ROLLUP_FIELDS:
            value = prov.get(field)
            if value is None or value == "":
                continue
            seen.setdefault(field, set()).add(str(value))
    return {key: sorted(values) for key, values in sorted(seen.items())}

_ROLE_MAP = {"system": "developer", "user": "user", "assistant": "assistant"}
_CONTENT_TYPE = {"developer": "input_text", "user": "input_text", "assistant": "output_text"}


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "text": self.text}


class LossyMultiTurnStateError(RuntimeError):
    """Text-only message history cannot preserve native reasoning state."""


def require_lossless_multi_turn(adapter: object, *, purpose: str) -> None:
    """Refuse an iterative run before its first billed provider call."""

    if getattr(adapter, "supports_lossless_multi_turn", False) is not True:
        raise LossyMultiTurnStateError(
            f"{purpose} requires a provider-native lossless multi-turn adapter; "
            f"{type(adapter).__name__} does not declare "
            "supports_lossless_multi_turn=True"
        )


def text_conversation_state(
    messages: list[Message],
    *,
    provider: str,
    allow_lossy_multi_turn: bool,
) -> dict[str, Any]:
    """Fail closed when the text-only adapter is used as a reasoning loop.

    All shared adapters intentionally expose only ``Message(role, text)``.
    That type cannot carry Responses output items, Anthropic thinking blocks,
    Gemini thought signatures, or OpenRouter reasoning details. An assistant
    message therefore proves that a caller is reconstructing model history.
    """

    has_assistant_history = any(message.role == "assistant" for message in messages)
    if has_assistant_history and not allow_lossy_multi_turn:
        raise LossyMultiTurnStateError(
            f"{provider} text-only adapter cannot safely continue assistant history: "
            "native reasoning/tool state would be discarded. Use a provider-native "
            "stateful adapter, or set allow_lossy_multi_turn=True only to reproduce "
            "an explicitly labelled legacy condition."
        )
    mode = "lossy_visible_transcript" if has_assistant_history else "single_inference"
    return {
        "mode": mode,
        "lossless": not has_assistant_history,
        "legacy_opt_in": has_assistant_history and allow_lossy_multi_turn,
    }


def messages_to_input(messages: list[Message]) -> list[dict[str, Any]]:
    items = []
    for m in messages:
        role = _ROLE_MAP[m.role]
        items.append({"role": role, "content": [{"type": _CONTENT_TYPE[role], "text": m.text}]})
    return items


def extract_response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    parts: list[str] = []
    for item in response.get("output", []):
        if isinstance(item, dict):
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    return "".join(parts)


def responses_usage_classes(usage: dict[str, Any] | None) -> dict[str, int]:
    """Map an OpenAI Responses ``usage`` object onto the pricing classes.

    ``input_tokens`` stays the TOTAL prompt; ``input_tokens_details``
    supplies its cache subsets (``cached_tokens`` -> cache_read_tokens,
    ``cache_write_tokens`` -> cache_write_tokens), and
    ``output_tokens_details.reasoning_tokens`` is kept for reporting (it is
    already inside ``output_tokens``). Absent fields are omitted, not zeroed,
    so an unreported class is never mistaken for a measured zero."""
    usage = usage or {}
    classes: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            classes[key] = value
    details = usage.get("input_tokens_details") or {}
    for src, dst in (("cached_tokens", "cache_read_tokens"), ("cache_write_tokens", "cache_write_tokens")):
        value = details.get(src)
        if isinstance(value, int) and not isinstance(value, bool):
            classes[dst] = value
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if isinstance(reasoning, int) and not isinstance(reasoning, bool):
        classes["reasoning_tokens"] = reasoning
    return classes


def extract_reasoning_summaries(response: dict[str, Any]) -> list[str]:
    """Reasoning summary texts (requested via reasoning.summary='auto').
    Kept per-call for post-hoc debugging of state-tracking failures."""
    summaries: list[str] = []
    for item in response.get("output", []):
        if isinstance(item, dict) and item.get("type") == "reasoning":
            for s in item.get("summary", []) or []:
                if isinstance(s, dict) and isinstance(s.get("text"), str):
                    summaries.append(s["text"])
    return summaries


def extract_provenance(response: dict[str, Any]) -> dict[str, Any]:
    """Routing provenance from a Responses payload.

    `id` is the stored response handle, `model` is the exact build OpenAI
    served (a dated snapshot when an alias was requested), `status` is the
    Responses-level outcome (completed / incomplete / failed), and
    `service_tier` is the tier ACTUALLY applied — a flex request can be
    served on default, and only the response says which."""
    return build_provenance(
        response_id=response.get("id"),
        routed_model=response.get("model"),
        finish_reason=response.get("status"),
        service_tier=response.get("service_tier"),
    )


def call_responses_api(
    *,
    api_key: str,
    model: str,
    messages: list[Message],
    effort: str,
    service_tier: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": messages_to_input(messages),
        "text": {"format": {"type": "text"}, "verbosity": "low"},
        "reasoning": {"effort": effort, "summary": "auto"},
        "tools": [],
        "store": False,
    }
    if service_tier:
        payload["service_tier"] = service_tier
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc


class APIAdapter:
    """Adapter(messages) -> text, with bounded retries and usage tracking."""

    supports_lossless_multi_turn = False

    def __init__(
        self,
        *,
        model: str,
        effort: str = "medium",
        service_tier: str | None = "flex",
        timeout_seconds: int = 600,
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
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.calls = 0
        self.usage_totals: dict[str, int] = {}
        self._lock = threading.Lock()  # adapters may be shared across workers
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set (use: secret run copybench-openai --env OPENAI_API_KEY -- ...)"
            )

    def __call__(self, messages: list[Message]) -> str:
        return self.call_with_meta(messages)[0]

    def call_with_meta(self, messages: list[Message]) -> tuple[str, dict[str, Any]]:
        """Returns (text, meta) where meta carries per-call reasoning
        summaries and usage — thread-safe (nothing shared is returned)."""
        conversation_state = text_conversation_state(
            messages,
            provider="OpenAI Responses",
            allow_lossy_multi_turn=self.allow_lossy_multi_turn,
        )
        last_exc: Exception | None = None
        response: dict[str, Any] | None = None
        for retry in range(self.max_retries + 1):
            try:
                response = call_responses_api(
                    api_key=self.api_key,
                    model=self.model,
                    messages=messages,
                    effort=self.effort,
                    service_tier=self.service_tier,
                    timeout_seconds=self.timeout_seconds,
                )
                break
            except Exception as exc:
                last_exc = exc
                if retry >= self.max_retries:
                    raise
                time.sleep(self.retry_sleep_seconds * (2**retry))
        if response is None:  # pragma: no cover
            raise last_exc or RuntimeError("retry loop exited without response")
        usage = {k: v for k, v in (response.get("usage") or {}).items() if isinstance(v, int)}
        with self._lock:
            self.calls += 1
            for key, value in usage.items():
                self.usage_totals[key] = self.usage_totals.get(key, 0) + value
        meta = {
            "reasoning": extract_reasoning_summaries(response),
            "usage": usage,
            "provenance": extract_provenance(response),
            "conversation_state": conversation_state,
        }
        return extract_response_text(response), meta

    def cost_usd(self) -> float | None:
        """Undiscounted list-price cost of all calls so far."""
        prices = PRICES.get(self.model)
        if prices is None:
            return None
        cin = self.usage_totals.get("input_tokens", 0) / 1e6 * prices[0]
        cout = self.usage_totals.get("output_tokens", 0) / 1e6 * prices[1]
        return cin + cout
