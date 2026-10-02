"""Vendored Gemini + OpenRouter adapters: dispatch, pricing wiring.

The adapters' internals are tested at the benchkit source; this guards the
consumer-side rendering (import map, registry entries, runner interface).
"""

from __future__ import annotations

import pytest

import blindboard.openrouter_client as oc
from blindboard.anthropic_client import make_adapter
from blindboard.gemini_client import GeminiAdapter, normalize_gemini_response
from blindboard.openrouter_client import (
    EffortBindingError,
    OpenRouterAdapter,
    normalize_openrouter_response,
)
from blindboard.pricing import PRICES

# Pinned copy of the OpenRouter models index: deepseek grades
# reasoning_effort, qwen does not. Keeps every test offline and makes
# effort binding a property of the fixture, not of the network.
_INDEX = {
    "deepseek/deepseek-v4-pro": ["reasoning", "reasoning_effort"],
    "qwen/qwen3.6-27b": ["reasoning", "include_reasoning"],
}


@pytest.fixture(autouse=True)
def _pinned_models_index(monkeypatch):
    monkeypatch.setattr(oc, "_SUPPORTED_PARAMS_CACHE", dict(_INDEX))


def test_registry_has_gemini_and_openrouter_entries() -> None:
    assert PRICES["gemini/gemini-3.1-pro-preview"] == (2.0, 12.0)
    assert PRICES["gemini/gemini-3-flash-preview"] == (0.5, 3.0)
    assert PRICES["openrouter/deepseek/deepseek-v4-pro"] == (0.435, 0.87)


def test_dispatch_gemini_and_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    g = make_adapter(model="gemini/gemini-3.1-pro-preview", effort="high")
    assert isinstance(g, GeminiAdapter) and g.model == "gemini-3.1-pro-preview"
    assert g.service_tier == "flex"
    o = make_adapter(model="openrouter/deepseek/deepseek-v4-pro", effort="low")
    assert isinstance(o, OpenRouterAdapter) and o.model == "deepseek/deepseek-v4-pro"
    assert o.effort_binding is True


def test_dispatch_fails_closed_on_a_route_that_ignores_effort(monkeypatch) -> None:
    """qwen3.6-27b takes reasoning but not graded reasoning_effort: an
    effort sweep here would re-run one configuration under several
    labels, so the runner must refuse before spending anything."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(EffortBindingError, match="cannot be shown to bind"):
        make_adapter(model="openrouter/qwen/qwen3.6-27b", effort="low")


def test_dispatch_escape_hatch_stamps_effort_binding_false(monkeypatch) -> None:
    """Deliberate single-mode collection stays possible, but the run is
    stamped so no downstream table can read it as a graded effort."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    adapter = make_adapter(
        model="openrouter/qwen/qwen3.6-27b", effort="low", require_effort_binding=False
    )
    assert adapter.effort_binding is False


def test_provenance_survives_the_vendor_rendering() -> None:
    """The import-map rewrite must not drop the shared provenance helper."""
    _, gemini_meta = normalize_gemini_response(
        {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            "modelVersion": "gemini-3.1-pro-preview-05-14",
            "responseId": "resp-1",
        }
    )
    assert gemini_meta["provenance"] == {
        "response_id": "resp-1",
        "model_version": "gemini-3.1-pro-preview-05-14",
    }
    _, or_meta = normalize_openrouter_response(
        {
            "id": "gen-1",
            "provider": "Fireworks",
            "model": "qwen/qwen3.6-27b",
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    assert or_meta["provenance"]["provider"] == "Fireworks"
    assert or_meta["provenance"]["routed_model"] == "qwen/qwen3.6-27b"


def test_gemini_cost_uses_prefixed_registry_key() -> None:
    adapter = GeminiAdapter(model="gemini-3.1-pro-preview", api_key="k")
    adapter.usage_totals = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert adapter.cost_usd() == pytest.approx(2.0 + 12.0)


def test_openrouter_cost_uses_prefixed_registry_key() -> None:
    adapter = OpenRouterAdapter(model="deepseek/deepseek-v4-pro", api_key="k")
    adapter.usage_totals = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert adapter.cost_usd() == pytest.approx(0.435 + 0.87)


def test_openrouter_adapter_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterAdapter(model="deepseek/deepseek-v4-pro")


def test_gemini_adapter_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiAdapter(model="gemini-3.1-pro-preview")
