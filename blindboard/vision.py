# Vendored from benchkit@0.3.10 (sha256:588138cd57ab383a). Do not hand-edit; run 'benchkit sync' to update.
"""Vision-capable adapter for image channels (attaches a PNG to the last
user message). Kept separate from the base Responses adapter per the
separation contract: tracks that need pixels vendor this file; the shared
client stays untouched.

- Image detail defaults to the MODEL FAMILY MAXIMUM and fails closed on an
  unregistered family, so a run never silently under-resolves the image
  channel; an explicit image_detail= is the documented opt-out. See
  VisionAdapter's docstring.
- Provenance: the served build / actual tier are retained in
  meta["provenance"], and the resolved detail in meta["image_detail"]."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from blindboard.client import (
    API_URL,
    APIAdapter,
    Message,
    extract_provenance,
    extract_reasoning_summaries,
    extract_response_text,
    messages_to_input,
    text_conversation_state,
)

OPENAI_VISION_GUIDE = (
    "https://developers.openai.com/api/docs/guides/images-vision"
    "#choose-an-image-detail-level"
)

_HIGH_ONLY_PREFIXES = (
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.2-chat-latest",
    "gpt-5.2",
    "gpt-5.1-codex-mini",
    "gpt-5-codex-mini",
    "o4-mini",
)
_ORIGINAL_PREFIXES = ("gpt-5.6", "gpt-5.5", "gpt-5.4")


def openai_max_image_detail(model: str) -> str:
    """Highest documented Responses image detail for a known model family.

    Raises ValueError for an unregistered family: benchkit cannot know what
    that model's maximum is, and guessing would silently under-resolve the
    image channel (see VisionAdapter for the rationale and the opt-out)."""
    normalized = model.lower()
    if normalized.startswith(_HIGH_ONLY_PREFIXES):
        return "high"
    if normalized.startswith(("gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14")):
        return "high"
    if normalized.startswith(_ORIGINAL_PREFIXES):
        return "original"
    raise ValueError(
        f"No frozen maximum OpenAI image-detail policy for {model!r}. Either "
        f"pass an explicit image_detail= to opt out of max resolution "
        f"('high'/'low'/'original', 'auto' to let the API choose, or None to "
        f"send no detail at all), or verify {OPENAI_VISION_GUIDE} and register "
        f"the family in benchkit/client/vision.py. Register it in the KIT and "
        f"re-sync — never in a vendored copy, which is hash-checked."
    )


class VisionAdapter(APIAdapter):
    """APIAdapter that attaches a PNG to the last user message.

    OpenAI-only by construction: it posts to the Responses API_URL and there
    is no base_url override, so a non-OpenAI model name has no working path
    through this class regardless of the detail policy below.

    IMAGE DETAIL — fail-closed at MAX by default. ``image_detail="max"``
    (the default) resolves to the highest level the model's family
    documents, because on a vision benchmark a silently under-resolved
    image is a confound: a channel-fidelity failure gets attributed to the
    model when it was really a resolution artifact. An unregistered family
    therefore RAISES rather than falling back to the API default.

    OPT-OUT — pass any explicit ``image_detail=`` to bypass the lookup
    entirely: "high"/"low"/"original" to pin a level, "auto" to let the API
    choose, or None to omit the field (the pre-policy behaviour). Use this
    for a model whose maximum is not registered, or when the run
    deliberately characterizes a lower detail level. The resolved value is
    recorded on every call's meta as ``image_detail``, so a run artifact
    always states which level produced it."""

    def __init__(self, *, image_detail: str | None = "max", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        resolved = (
            openai_max_image_detail(self.model)
            if image_detail == "max"
            else image_detail
        )
        if resolved not in {None, "low", "high", "original", "auto"}:
            raise ValueError(f"unsupported OpenAI image detail: {resolved}")
        self.image_detail = resolved

    def call_with_image(
        self, messages: list[Message], image_b64: str | None
    ) -> tuple[str, dict[str, Any]]:
        """Like call_with_meta, but attaches a PNG (data URL) to the LAST
        user message's content when image_b64 is provided."""
        conversation_state = text_conversation_state(
            messages,
            provider="OpenAI Responses vision",
            allow_lossy_multi_turn=self.allow_lossy_multi_turn,
        )
        items = messages_to_input(messages)
        if image_b64 is not None:
            for item in reversed(items):
                if item["role"] == "user":
                    image_item = {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    }
                    if self.image_detail is not None:
                        image_item["detail"] = self.image_detail
                    item["content"].insert(0, image_item)
                    break
        payload: dict[str, Any] = {
            "model": self.model,
            "input": items,
            "text": {"format": {"type": "text"}, "verbosity": "low"},
            "reasoning": {"effort": self.effort, "summary": "auto"},
            "tools": [],
            "store": False,
        }
        if self.service_tier:
            payload["service_tier"] = self.service_tier
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_exc: Exception | None = None
        response: dict[str, Any] | None = None
        for retry in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as resp:
                    response = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                last_exc = exc
                if retry >= self.max_retries:
                    raise
                time.sleep(self.retry_sleep_seconds * (2**retry))
        if response is None:  # pragma: no cover
            raise last_exc or RuntimeError("retry loop exited without response")
        usage = {
            k: v for k, v in (response.get("usage") or {}).items() if isinstance(v, int)
        }
        with self._lock:
            self.calls += 1
            for key, value in usage.items():
                self.usage_totals[key] = self.usage_totals.get(key, 0) + value
        return extract_response_text(response), {
            "reasoning": extract_reasoning_summaries(response),
            "usage": usage,
            "provenance": extract_provenance(response),
            "image_detail": self.image_detail,
            "conversation_state": conversation_state,
        }
