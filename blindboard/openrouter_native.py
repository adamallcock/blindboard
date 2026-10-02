# Vendored from benchkit@0.3.10 (sha256:5d90cd983df96754). Do not hand-edit; run 'benchkit sync' to update.
"""Provider-native, lossless multi-turn session for OpenRouter chat completions.

OpenRouter returns a model's reasoning as ``choices[0].message.reasoning_details``
(typed items: ``reasoning.text``, ``reasoning.summary``, ``reasoning.encrypted``)
and accepts that array back on an assistant message in a later request. The
text adapter (``openrouter_chat.OpenRouterAdapter``) cannot carry it --
``Message(role, text)`` has nowhere to put it -- so it refuses assistant
history. ``OpenRouterNativeSession`` implements the
``blindboard.native_session`` contract instead: it stores every assistant
message exactly as returned and replays its ``content`` plus the complete
``reasoning_details`` array, unmodified and in provider order, on every later
request.

Replaying is not the same as retaining. The served model sees the replayed
reasoning only if the route's chat template keeps it, and OpenRouter cannot
tell you whether it did. Live check 2026-09-25
(docs/2026-09-25-openrouter-native-sessions.md): DeepSeek V4 Pro, V4 Flash and
V4.1 Flash DISCARD replayed reasoning unless the request carries a ``tools``
parameter -- as DeepSeek documents -- and keep it token-for-token when it does.
``RETENTION_VERDICTS`` records the verdict per model and request shape. A
model/shape whose verdict is not "verified", or a model absent from the table,
is refused at construction (``RetentionUnsupportedError``) unless the caller
passes ``allow_visible_only=True``; the session then replays visible content
only, says so in ``public["retention"]``, and reports
``supports_lossless_multi_turn = False``.

Frozen request (every turn): ``model``; ``messages`` = system instructions,
then the stored transcript with each assistant message replayed as
``{"role", "content", "reasoning_details"}``, then the new user message(s);
``temperature`` (0 unless overridden; DeepSeek documents that thinking mode
ignores it); an explicit ``max_tokens`` (a missing cap once let host defaults
spend the whole budget on thinking and return empty replies);
``usage: {"include": true}`` (documented as deprecated and always-on now;
kept for parity); ``reasoning`` = ``{"effort": e}``, ``{"enabled": true}``
(effort None) or ``{"enabled": false}`` (effort "none"); ``provider`` with
``require_parameters: true`` always; ``session_id`` (OpenRouter's sticky
routing key); and, only when configured, ``tools`` with
``tool_choice: "auto"`` (``"none"`` measured as dropping the tools, and with
them the retention, on 2026-09-25).

Fail-closed rules: graded effort must bind (``reasoning_effort`` listed and,
when the models index lists ``reasoning.supported_efforts``, the requested
level among them); model variant suffixes (``:batch``, ``:floor``, ...) are
refused; a turn that generated reasoning tokens but returned no
``reasoning_details`` is refused in native mode (nothing to carry forward);
a returned tool call is refused (this session executes no tools); and a
runtime tripwire refuses a turn whose prompt grew by clearly less than the
previous turn's output, i.e. whose route discarded the replayed reasoning. A
refused or failed turn never advances the stored transcript.

Cost: OpenRouter bills the serving endpoint's published per-token price and
reports it as ``usage.cost`` (USD). With no flex tier requested, that IS the
standard-tier price of the route that served the call, including cache reads
and any time-of-day or promotional rate the endpoint published at that
moment. ``NativeTurn.cost_usd`` uses it unless it is absent, the request was
BYOK (``usage.cost`` is then OpenRouter's fee), or the response reports a
non-default ``service_tier``; then the injected pricer (default: the registry
entry ``openrouter/<slug>`` on the UTC day of the call) prices the usage, or
the cost is None. ``public["cost_basis"]`` records which.

Runtime dependencies are Python's standard library only.
"""

from __future__ import annotations

import copy
import http.client
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from blindboard.native_session import (
    IncompleteTurnError,
    NativeTurn,
    Pricer,
    RetentionUnsupportedError,
    assert_public_safe,
    registry_pricer,
)
from blindboard.openrouter_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_TIMEOUT_SECONDS,
    OPENROUTER_MODELS_URL,
    OPENROUTER_URL,
    EffortBindingError,
    OpenRouterResponseError,
    _retry_after_seconds,
    is_retryable_openrouter_code,
    openrouter_response_error,
)
from blindboard.client import build_provenance

DEFAULT_MAX_TOKENS = 64000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 1800.0
DEFAULT_TEMPERATURE = 0.0

VERIFIED = "verified"
NOT_RETAINED = "not_retained"
UNMEASURABLE = "unmeasurable"
_VERDICT_VALUES = frozenset({VERIFIED, NOT_RETAINED, UNMEASURABLE})

REPLAY_NATIVE = "reasoning_details"
REPLAY_VISIBLE_ONLY = "visible_content_only"

# The exact tool carried by the with-tools arm of the 2026-09-25 live check.
# It exists only to change the request shape: DeepSeek keeps earlier-turn
# reasoning in context only when the request carries ``tools``. It adds the
# template's tool preamble to every prompt (~245 tokens on V4.1 Flash), so
# adopting it is a prompt-version change for a benchmark.
INERT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "noop",
        "description": "Does nothing. Never call this tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}


@dataclass(frozen=True)
class RetentionVerdict:
    """Live verdict on whether replayed reasoning re-enters the prompt.

    One verdict per request shape, because DeepSeek's template keys
    retention on the presence of a ``tools`` parameter. "verified": the
    turn-2 prompt exceeded a same-host visible-only control by turn 1's
    reasoning tokens. "not_retained": it did not exceed it at all.
    "unmeasurable": no decisive measurement.
    """

    without_tools: str
    with_tools: str
    checked: str  # ISO date of the live check
    hosts: tuple[str, ...]  # OpenRouter provider slugs that served it
    evidence: str

    def __post_init__(self) -> None:
        for value in (self.without_tools, self.with_tools):
            if value not in _VERDICT_VALUES:
                raise ValueError(f"unknown retention verdict {value!r}")

    def for_shape(self, *, with_tools: bool) -> str:
        return self.with_tools if with_tools else self.without_tools


# Qwen routes are deferred (not measured): absent models refuse by default.
RETENTION_VERDICTS: dict[str, RetentionVerdict] = {
    "deepseek/deepseek-v4-pro": RetentionVerdict(
        without_tools=NOT_RETAINED,
        with_tools=VERIFIED,
        checked="2026-09-25",
        hosts=("baidu", "deepinfra", "gmicloud"),
        evidence=(
            "turn-2 prompt minus same-host visible-only control: without tools 0 "
            "(r1 2636 baidu, 5287 deepinfra; tool_choice none: 4871 gmicloud, "
            "9271 deepinfra); with tools 2281 for r1 2281 (baidu), 5626 for "
            "5623 (deepinfra), 1946 for 2022 (gmicloud, whose counts jitter "
            "~76 tokens on identical prompts)"
        ),
    ),
    "deepseek/deepseek-v4-flash": RetentionVerdict(
        without_tools=NOT_RETAINED,
        with_tools=VERIFIED,
        checked="2026-09-25",
        hosts=("baidu", "deepinfra", "gmicloud", "novita"),
        evidence=(
            "turn-2 prompt minus same-host visible-only control: without tools 0 "
            "(r1 7302 baidu, 2830 deepinfra, 2428 novita; tool_choice none: "
            "4394 baidu); with tools 3078 for r1 3078 (baidu; the model then "
            "quoted a code it chose only in turn-1 reasoning), 2024 for 2024 "
            "(gmicloud), 2048 for 2048 (novita), 2998 for completion 3001 "
            "(deepinfra, which reports reasoning low: 2117)"
        ),
    ),
    "deepseek/deepseek-v4.1-flash": RetentionVerdict(
        without_tools=NOT_RETAINED,
        with_tools=VERIFIED,
        checked="2026-09-25",
        hosts=("deepinfra", "fireworks", "gmicloud", "together", "wafer"),
        evidence=(
            "turn-2 prompt minus same-host visible-only control: without tools 0 "
            "(r1 1985 gmicloud, 1654 deepinfra, 923 fireworks; tool_choice "
            "none: 2109 wafer, 1937 deepinfra); with tools 1170 for r1 1171 "
            "(wafer), 1420 for 1420 (deepinfra), 3165 for 3165 (gmicloud), "
            "1219 for 1219 (together), 1682 for completion 1685 (fireworks, "
            "which reports reasoning low: 1132); turn 3 carried turn-1 plus "
            "turn-2 reasoning within 2 tokens"
        ),
    ),
}

# Runtime tripwire. In a retained replay, prompt(t) - prompt(t-1) -
# output(t-1) is the new user message plus template overhead (+17 to +52
# measured, +/-80 across hosts' accounting). When a route discards the
# replayed reasoning it falls by about the previous turn's reasoning. The
# check fires only when that shortfall is unambiguous, so it can miss a
# stripping route (tiny reasoning, very long new user input) but should not
# refuse a retaining one.
RETENTION_CHECK_MIN_REASONING = 256
# A turn whose host visibly discarded the replay is re-sent at most this many
# times, each time avoiding every host that failed the check so far.
MAX_REROUTES = 2
RETENTION_CHECK_FLOOR_TOKENS = 128
RETENTION_CHECK_FRACTION = 0.5

_TIER_SUFFIXES = frozenset({"flex", "priority", "fast", "batch"})

# Models index, cached once per process (tests pin it). Effort binding is
# decided from it, as in openrouter_chat.effort_binds, plus the per-model
# reasoning.supported_efforts list the index now publishes.
_MODELS_INDEX_CACHE: dict[str, Mapping[str, Any]] | None = None


def _models_index_entry(model: str) -> tuple[Mapping[str, Any] | None, str | None]:
    """(entry, None), or (None, why) when the index cannot answer."""
    global _MODELS_INDEX_CACHE
    if _MODELS_INDEX_CACHE is None:
        try:
            with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))["data"]
            _MODELS_INDEX_CACHE = {
                m["id"]: m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str)
            }
        except Exception:  # noqa: BLE001 - metadata only; absence fails closed
            return None, "the OpenRouter models index could not be read"
    entry = _MODELS_INDEX_CACHE.get(model)
    if entry is None:
        return None, f"{model} is absent from the OpenRouter models index"
    return entry, None


def graded_effort_binding(model: str, effort: str) -> tuple[bool | None, str]:
    """(binds, why) for a graded ``effort`` on ``model``.

    True only when the models index lists ``reasoning_effort`` in the
    model's supported_parameters and, if it publishes
    ``reasoning.supported_efforts``, the requested level is in that list
    (DeepSeek V4 lists ["xhigh", "high"]; another level would be mapped to
    a neighbour, so a sweep over it re-runs one configuration). None when
    the index cannot answer -- which is not a pass."""
    entry, why = _models_index_entry(model)
    if entry is None:
        return None, why or "models index unavailable"
    params = entry.get("supported_parameters") or []
    if "reasoning_effort" not in params:
        return False, f"{model}'s supported_parameters do not list reasoning_effort"
    reasoning = entry.get("reasoning")
    efforts = reasoning.get("supported_efforts") if isinstance(reasoning, Mapping) else None
    if isinstance(efforts, list) and effort not in efforts:
        return False, f"{model}'s reasoning.supported_efforts {efforts} do not include {effort!r}"
    return True, f"{model} lists reasoning_effort" + (
        f" and supported_efforts {efforts}" if isinstance(efforts, list) else ""
    )


def _with_rerouted(
    usage: Mapping[str, int], cost: float | None, rerouted: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, int], float | None]:
    """Add billed, discarded attempts to a turn's usage and cost."""
    total = dict(usage)
    for r in rerouted:
        for key, value in r["usage"].items():
            total[key] = total.get(key, 0) + int(value)
        cost = None if cost is None or r["cost_usd"] is None else cost + r["cost_usd"]
    return total, cost


def openrouter_usage_classes(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Map an OpenRouter chat ``usage`` object onto the pricing classes.

    ``prompt_tokens`` is the TOTAL prompt (OpenRouter documents
    ``prompt_tokens_details`` as its breakdown), so it maps to input_tokens
    and ``cached_tokens`` / ``cache_write_tokens`` map to the cache subsets.
    ``completion_tokens`` includes reasoning; ``reasoning_tokens`` is kept
    for reporting. A field the response did not carry is omitted; a
    reported zero is kept."""
    usage = usage if isinstance(usage, Mapping) else {}
    classes: dict[str, int] = {}

    def put(name: str, value: Any) -> None:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            classes[name] = value

    put("input_tokens", usage.get("prompt_tokens"))
    put("output_tokens", usage.get("completion_tokens"))
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, Mapping):
        put("cache_read_tokens", prompt_details.get("cached_tokens"))
        put("cache_write_tokens", prompt_details.get("cache_write_tokens"))
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, Mapping):
        put("reasoning_tokens", completion_details.get("reasoning_tokens"))
    return classes


def replay_message(stored: Mapping[str, Any], *, replay: str = REPLAY_NATIVE) -> dict[str, Any]:
    """The request form of a stored assistant message.

    Native replay sends ``content`` and the stored ``reasoning_details``
    list itself (deep-copied, never re-ordered or re-built), as OpenRouter's
    reasoning guide specifies for passing reasoning back. The duplicate
    ``reasoning`` string is not sent. Visible-only replay sends content only.
    """
    message: dict[str, Any] = {"role": "assistant", "content": copy.deepcopy(stored.get("content"))}
    if replay == REPLAY_NATIVE and "reasoning_details" in stored:
        message["reasoning_details"] = copy.deepcopy(stored["reasoning_details"])
    return message


def _readable_reasoning(message: Mapping[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    for item in message.get("reasoning_details") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") == "reasoning.text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
        elif item.get("type") == "reasoning.summary" and isinstance(item.get("summary"), str):
            texts.append(item["summary"])
    if not texts and isinstance(message.get("reasoning"), str) and message["reasoning"]:
        texts.append(message["reasoning"])
    return tuple(t for t in texts if t)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def _tool_name(tool: Mapping[str, Any]) -> str | None:
    function = tool.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    return name if isinstance(name, str) else None


class OpenRouterTransportError(RuntimeError):
    """Network-level failure with no HTTP status: always retryable."""

    retryable = True


class UnsupportedToolCallError(IncompleteTurnError):
    """The model returned a tool call; this session executes no tools.
    History is unchanged; ``turn`` is the billed attempt, with empty text."""


class ConcurrentTurnError(RuntimeError):
    """Two threads called turn() on one session at once."""


class OpenRouterTransport(Protocol):
    def __call__(self, payload: Mapping[str, Any], *, timeout_seconds: float) -> Mapping[str, Any]: ...


class OpenRouterHTTPTransport:
    """Stdlib POST to the chat-completions endpoint.

    HTTP errors raise ``OpenRouterResponseError`` (code, retryable, and any
    Retry-After as ``retry_after_seconds``); network failures raise
    ``OpenRouterTransportError``. A 200 body is returned parsed, including an
    in-body error envelope, which the session classifies."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set (use: secret run openrouter --env OPENROUTER_API_KEY -- ...)"
            )
        self._api_key = api_key

    def __repr__(self) -> str:
        return "OpenRouterHTTPTransport()"

    def __call__(self, payload: Mapping[str, Any], *, timeout_seconds: float) -> Mapping[str, Any]:
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = OpenRouterResponseError(
                code=exc.code, detail=detail, retryable=is_retryable_openrouter_code(exc.code)
            )
            error.retry_after_seconds = _retry_after_seconds(headers=exc.headers)  # type: ignore[attr-defined]
            raise error from None
        except (OSError, http.client.HTTPException) as exc:  # URLError, timeouts, resets
            raise OpenRouterTransportError(
                f"OpenRouter transport failed ({type(exc).__name__})"
            ) from None
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OpenRouterTransportError("OpenRouter returned a non-JSON body") from None
        if not isinstance(parsed, dict):
            raise OpenRouterTransportError("OpenRouter returned non-object JSON")
        return parsed


def _response_error(resp: Mapping[str, Any]) -> OpenRouterResponseError | None:
    """Classify an HTTP-200 body that is not a usable completion."""
    if error := openrouter_response_error(dict(resp)):
        return error
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return OpenRouterResponseError(code=None, detail="response carries no choices", retryable=True)
    choice = choices[0]
    if isinstance(choice.get("error"), Mapping):
        error = openrouter_response_error({"error": dict(choice["error"])})
        if error is not None:
            return error
    if choice.get("finish_reason") == "error":
        return OpenRouterResponseError(code=None, detail="choice finished with error", retryable=True)
    if not isinstance(choice.get("message"), Mapping):
        return OpenRouterResponseError(code=None, detail="choice carries no message", retryable=True)
    return None


class OpenRouterNativeSession:
    """One lossless OpenRouter chat session (see module docstring).

    ``model`` is the bare OpenRouter slug (e.g. ``deepseek/deepseek-v4-pro``).
    ``effort`` is a graded level (must bind), ``None`` for the model's
    default reasoning (``{"enabled": true}``, no graded claim), or ``"none"``
    to disable reasoning. ``tools`` switches the request shape to the
    with-tools arm (see ``INERT_TOOL``); the session executes no tools.
    ``provider_routing`` is sent as OpenRouter's ``provider`` object;
    ``require_parameters`` is forced true and cannot be switched off.
    """

    def __init__(
        self,
        *,
        model: str,
        instructions: str | None,
        effort: str | None,
        api_key: str | None = None,
        transport: OpenRouterTransport | None = None,
        pricer: Pricer | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        provider_routing: Mapping[str, Any] | None = None,
        tools: Sequence[Mapping[str, Any]] = (),
        allow_visible_only: bool = False,
        require_effort_binding: bool = True,
        temperature: float | None = DEFAULT_TEMPERATURE,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_sleep_seconds: float = 5.0,
        retry_timeout_seconds: float = DEFAULT_RETRY_TIMEOUT_SECONDS,
        session_id: str | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty OpenRouter slug")
        if model.startswith("openrouter/"):
            raise ValueError("pass the bare OpenRouter slug (vendor/model), without 'openrouter/'")
        if ":" in model:
            raise ValueError(
                f"{model!r}: variant suffixes (:batch, :floor, :nitro, :free, ...) change the "
                "served endpoint pool or tier; retention verdicts cover base slugs only"
            )
        if instructions is not None and not isinstance(instructions, str):
            raise TypeError("instructions must be a string or None")
        if effort is not None and (not isinstance(effort, str) or not effort):
            raise ValueError("effort must be a nonempty string or None")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if temperature is not None and (
            not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
        ):
            raise TypeError("temperature must be a number or None")
        for name, value in (
            ("request_timeout_seconds", request_timeout_seconds),
            ("retry_timeout_seconds", retry_timeout_seconds),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a nonnegative integer")

        self.model = model
        self.instructions = instructions
        self.effort = effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.retry_timeout_seconds = float(retry_timeout_seconds)
        self.session_id = session_id or uuid.uuid4().hex
        self._sleeper = sleeper
        self._tools = self._validated_tools(tools)
        self._routing = self._validated_routing(provider_routing)

        # Retention gate: before any credential or network work.
        record = RETENTION_VERDICTS.get(model)
        with_tools = bool(self._tools)
        self.request_shape = "with_tools" if with_tools else "without_tools"
        self.retention_verdict = record.for_shape(with_tools=with_tools) if record else "absent"
        self.retention_checked = record.checked if record else None
        if self.retention_verdict == VERIFIED:
            self.replay = REPLAY_NATIVE
        elif allow_visible_only:
            self.replay = REPLAY_VISIBLE_ONLY
        else:
            hint = ""
            if record is not None and not with_tools and record.with_tools == VERIFIED:
                hint = (
                    f" The with-tools request shape IS verified for {model} (checked "
                    f"{record.checked}): pass tools=(INERT_TOOL,) to carry reasoning."
                )
            raise RetentionUnsupportedError(
                f"{model}: replayed reasoning is not shown to reach the model for requests "
                f"{'with' if with_tools else 'without'} tools (verdict "
                f"{self.retention_verdict!r}"
                + (f", checked {record.checked}" if record else ", never measured")
                + "). A multi-turn run would silently be a visible-transcript run. Pass "
                "allow_visible_only=True to collect that condition deliberately (it is "
                "labelled visible-only in public metadata)." + hint
            )
        # Lossless only when earlier reasoning actually rides along.
        self.supports_lossless_multi_turn = self.replay == REPLAY_NATIVE

        if transport is None:
            transport = OpenRouterHTTPTransport(api_key or os.environ.get("OPENROUTER_API_KEY", ""))
        self._transport = transport
        self._pricer = pricer if pricer is not None else registry_pricer(f"openrouter/{model}")
        self._pricer_basis = "injected_pricer" if pricer is not None else "registry_pricer"

        self._reasoning_param, self.effort_binding, self.effort_binding_note = self._effort_param(
            model, effort, require_effort_binding
        )

        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []
        self._turn_index = 0
        self._last_usage: dict[str, int] | None = None
        self._last_host: str | None = None
        # Hosts that discarded the replay in this session: never routed again.
        self._avoided_hosts: list[str] = []

    # -- configuration -------------------------------------------------------

    @staticmethod
    def _validated_tools(tools: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
        if isinstance(tools, (str, bytes, Mapping)) or not isinstance(tools, Sequence):
            raise TypeError("tools must be a sequence of tool definitions")
        out = tuple(copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping))
        if len(out) != len(tools):
            raise TypeError("every tool definition must be a mapping")
        try:
            json.dumps(out)
        except (TypeError, ValueError) as exc:
            raise ValueError("tool definitions must be JSON-serializable") from exc
        return out

    @staticmethod
    def _validated_routing(provider_routing: Mapping[str, Any] | None) -> dict[str, Any]:
        if provider_routing is not None and not isinstance(provider_routing, Mapping):
            raise TypeError("provider_routing must be a mapping or None")
        routing = copy.deepcopy(dict(provider_routing or {}))
        if routing.get("require_parameters", True) is not True:
            raise ValueError(
                "provider_routing.require_parameters must stay true: without it a host that "
                "cannot honour the reasoning parameters can serve the turn and ignore them"
            )
        routing["require_parameters"] = True
        for key in ("order", "only"):
            for slug in routing.get(key) or ():
                if isinstance(slug, str) and "/" in slug and slug.rsplit("/", 1)[1] in _TIER_SUFFIXES:
                    raise ValueError(
                        f"provider_routing.{key} names tier endpoint {slug!r}; tier endpoints "
                        "bill non-standard rates"
                    )
        try:
            json.dumps(routing)
        except (TypeError, ValueError) as exc:
            raise ValueError("provider_routing must be JSON-serializable") from exc
        assert_public_safe(routing)  # it is echoed into public metadata every turn
        return routing

    @staticmethod
    def _effort_param(
        model: str, effort: str | None, require_effort_binding: bool
    ) -> tuple[dict[str, Any], bool | str, str]:
        if effort is None:
            return {"enabled": True}, "model_default", "no graded effort requested"
        if effort == "none":
            return {"enabled": False}, "disabled", "reasoning disabled"
        binds, why = graded_effort_binding(model, effort)
        if binds is True:
            return {"effort": effort}, True, why
        if require_effort_binding:
            raise EffortBindingError(
                f"refusing to collect: effort {effort!r} cannot be shown to bind on {model} -- "
                f"{why}. A graded effort that does not bind makes an 'effort sweep' re-run one "
                "configuration under several labels. Pass effort=None for the model's default "
                "reasoning, or require_effort_binding=False to collect deliberately as "
                "single-mode (recorded as effort_binding=False)."
            )
        print(
            f"WARNING: require_effort_binding=False -- {why}; requested effort {effort!r} "
            "will NOT bind; recording effort_binding=False.",
            flush=True,
        )
        return {"effort": effort}, False, why

    # -- state ---------------------------------------------------------------

    @property
    def turn_index(self) -> int:
        """Index the next turn will carry (0 before the first turn)."""
        return self._turn_index

    @property
    def native_history(self) -> list[dict[str, Any]]:
        """PRIVATE transcript: every assistant message exactly as returned,
        reasoning_details included. Never write it into public artifacts."""
        return copy.deepcopy(self._history)

    def __repr__(self) -> str:
        return (
            f"OpenRouterNativeSession(model={self.model!r}, replay={self.replay!r}, "
            f"request_shape={self.request_shape!r}, turn_index={self._turn_index})"
        )

    # -- requests ------------------------------------------------------------

    def _new_user_messages(self, user_texts: Sequence[str]) -> list[dict[str, Any]]:
        if isinstance(user_texts, (str, bytes)) or not isinstance(user_texts, Sequence):
            raise TypeError("user_texts must be a sequence of strings, not a bare string")
        if not user_texts:
            raise ValueError("a turn needs at least one user message")
        messages = []
        for text in user_texts:
            if not isinstance(text, str) or not text:
                raise ValueError("every user message must be a nonempty string")
            messages.append({"role": "user", "content": text})
        return messages

    def _request_messages(self, new_user: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if self.instructions is not None:
            messages.append({"role": "system", "content": self.instructions})
        for stored in self._history:
            if stored.get("role") == "assistant":
                messages.append(replay_message(stored, replay=self.replay))
            else:
                messages.append(copy.deepcopy(stored))
        messages.extend(copy.deepcopy(new_user))
        return messages

    def build_turn_payload(self, user_texts: Sequence[str]) -> dict[str, Any]:
        """The exact next request body, without I/O or mutation."""
        return self._payload(self._request_messages(self._new_user_messages(user_texts)))

    def _payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
            "reasoning": copy.deepcopy(self._reasoning_param),
            "provider": self._routing_with_avoided(),
            "session_id": self.session_id,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self._tools:
            payload["tools"] = copy.deepcopy(list(self._tools))
            payload["tool_choice"] = "auto"
        return payload

    def _routing_with_avoided(self) -> dict[str, Any]:
        routing = copy.deepcopy(self._routing)
        if self._avoided_hosts:
            ignore = list(routing.get("ignore") or [])
            routing["ignore"] = ignore + [h for h in self._avoided_hosts if h not in ignore]
        return routing

    def _send(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """POST with bounded retries. Every attempt gets the full request
        timeout (a max-effort thinking turn can legitimately run long); the
        retry window only decides whether another attempt may start."""
        deadline = time.monotonic() + self.retry_timeout_seconds
        for retry in range(self.max_retries + 1):
            attempts = retry + 1
            delay_hint: float | None = None
            try:
                resp = self._transport(
                    copy.deepcopy(payload), timeout_seconds=self.request_timeout_seconds
                )
                if not isinstance(resp, Mapping):
                    raise OpenRouterTransportError("transport returned a non-object response")
                error: Exception | None = _response_error(resp)
                if error is None:
                    return copy.deepcopy(dict(resp)), attempts
                delay_hint = _retry_after_seconds(response=dict(resp))
            except OpenRouterResponseError as exc:
                error = exc
                delay_hint = getattr(exc, "retry_after_seconds", None)
            except OpenRouterTransportError as exc:
                error = exc
            if not getattr(error, "retryable", False) or retry >= self.max_retries:
                raise error
            delay = delay_hint or self.retry_sleep_seconds * (2**retry)
            if time.monotonic() + delay > deadline:
                raise error
            self._sleeper(delay)
        raise AssertionError("unreachable retry loop")  # pragma: no cover

    # -- the turn --------------------------------------------------------------

    def turn(self, user_texts: Sequence[str]) -> NativeTurn:
        if not self._lock.acquire(blocking=False):
            raise ConcurrentTurnError("one OpenRouterNativeSession cannot run two turns at once")
        try:
            return self._turn(user_texts)
        finally:
            self._lock.release()

    def _turn(self, user_texts: Sequence[str]) -> NativeTurn:
        new_user = self._new_user_messages(user_texts)
        messages = self._request_messages(new_user)
        rerouted: list[dict[str, Any]] = []
        while True:
            payload = self._payload(messages)
            resp, attempts = self._send(payload)

            choice = resp["choices"][0]
            message = copy.deepcopy(dict(choice["message"]))
            if message.get("role", "assistant") != "assistant":
                raise OpenRouterResponseError(
                    code=None, detail=f"choice message has role {message.get('role')!r}",
                    retryable=False,
                )
            details = message.get("reasoning_details")
            if details is not None and (
                not isinstance(details, list) or not all(isinstance(d, Mapping) for d in details)
            ):
                raise OpenRouterResponseError(
                    code=None, detail="reasoning_details is not a list of objects", retryable=False
                )

            usage = openrouter_usage_classes(resp.get("usage"))
            raw_usage = resp.get("usage") if isinstance(resp.get("usage"), Mapping) else {}
            service_tier = resp.get("service_tier")
            cost, cost_basis, cost_note = self._cost(raw_usage, usage, service_tier)
            host = resp.get("provider") if isinstance(resp.get("provider"), str) else None
            if message.get("tool_calls"):
                raise self._tool_call_error(resp, usage, cost, rerouted)
            if self.replay == REPLAY_NATIVE and usage.get("reasoning_tokens", 0) > 0 and not details:
                raise RetentionUnsupportedError(
                    f"{self.model} generated {usage['reasoning_tokens']} reasoning tokens but "
                    "returned no reasoning_details: that reasoning cannot be carried into the "
                    "next turn (turn not recorded)"
                )
            check = self._retention_check(usage)
            if check["status"] != "inconsistent":
                break
            if host is None or host in self._avoided_hosts or len(rerouted) >= MAX_REROUTES:
                raise RetentionUnsupportedError(
                    f"{self.model}: the prompt grew by {check['prompt_growth']} tokens after a "
                    f"turn that output {check['prior_output_tokens']} tokens "
                    f"({check['prior_reasoning_tokens']} reasoning); the route ({host!r}) "
                    "discarded the replayed reasoning (turn not recorded)"
                )
            # This host dropped the replay: the answer was produced without the
            # model's earlier reasoning. Bill it, avoid the host, send again.
            rerouted.append({"host": host, "usage": dict(usage), "cost_usd": cost,
                             "runtime_check": check})
            self._avoided_hosts.append(host)

        final_usage = dict(usage)
        usage, cost = _with_rerouted(usage, cost, rerouted)
        finish_reason = choice.get("finish_reason")
        replayed_turns = sum(1 for m in messages if m.get("role") == "assistant")
        replayed_items = sum(
            len(m.get("reasoning_details") or []) for m in messages if m.get("role") == "assistant"
        )
        public: dict[str, Any] = {
            "turn_index": self._turn_index,
            "session_id": self.session_id,
            "model_requested": self.model,
            "retention": {
                "replay": self.replay,
                "request_shape": self.request_shape,
                "verdict": self.retention_verdict,
                "verdict_checked": self.retention_checked,
                "replayed_assistant_turns": replayed_turns,
                "replayed_reasoning_items": replayed_items,
                "runtime_check": check,
            },
            "effort": {
                "requested": self.effort,
                "reasoning_param": copy.deepcopy(self._reasoning_param),
                "binding": self.effort_binding,
            },
            "routing_sent": copy.deepcopy(self._routing),
            "max_tokens_sent": self.max_tokens,
            "tools_sent": [_tool_name(t) for t in self._tools] or None,
            "finish_reason": finish_reason,
            "native_finish_reason": choice.get("native_finish_reason"),
            "cost_basis": cost_basis,
            "http_attempts": attempts,
        }
        if rerouted:
            # Usage and cost above include these billed, discarded attempts.
            public["rerouted"] = [
                {"host": r["host"], "usage": r["usage"], "cost_usd": r["cost_usd"],
                 "prompt_growth": r["runtime_check"].get("prompt_growth")}
                for r in rerouted
            ]
        if self._avoided_hosts:
            public["avoided_hosts"] = list(self._avoided_hosts)
        if cost_note:
            public["cost_basis_note"] = cost_note
        provenance = build_provenance(
            response_id=resp.get("id"),
            provider=resp.get("provider"),
            routed_model=resp.get("model"),
            service_tier=service_tier,
            finish_reason=finish_reason,
        )
        assert_public_safe(public)
        assert_public_safe(provenance)

        # Commit only now: nothing above may leave a half-advanced session.
        self._history.extend(copy.deepcopy(new_user))
        self._history.append(message)
        self._last_usage = final_usage  # the accepted attempt's prompt, for the next check
        self._last_host = resp.get("provider") if isinstance(resp.get("provider"), str) else None
        self._turn_index += 1
        return NativeTurn(
            text=_message_text(message.get("content")),
            usage=dict(usage),
            reasoning_summaries=_readable_reasoning(message),
            provenance=provenance,
            public=public,
            cost_usd=cost,
        )

    def _tool_call_error(
        self,
        resp: Mapping[str, Any],
        usage: Mapping[str, int],
        cost: float | None,
        rerouted: list[dict[str, Any]],
    ) -> UnsupportedToolCallError:
        """A tool call as a billed, unretained empty turn: the runner can score
        it like any reply that breaks the protocol."""
        usage, cost = _with_rerouted(usage, cost, rerouted)
        public: dict[str, Any] = {
            "turn_index": self._turn_index,
            "incomplete_reason": "tool_call",
            "retention": {"replay": self.replay, "request_shape": self.request_shape,
                          "this_turn_retained": False},
            "tools_sent": [_tool_name(t) for t in self._tools] or None,
        }
        if rerouted:
            public["rerouted"] = [{"host": r["host"], "cost_usd": r["cost_usd"]} for r in rerouted]
        provenance = build_provenance(
            response_id=resp.get("id"),
            provider=resp.get("provider"),
            routed_model=resp.get("model"),
            service_tier=resp.get("service_tier"),
            finish_reason="tool_calls",
        )
        assert_public_safe(public)
        assert_public_safe(provenance)
        turn = NativeTurn(text="", usage=dict(usage), provenance=provenance, public=public,
                          cost_usd=cost)
        return UnsupportedToolCallError(
            f"{self.model} returned a tool call; this session executes no tools "
            "(turn billed, not recorded)",
            turn=turn,
        )

    def _retention_check(self, usage: Mapping[str, int]) -> dict[str, Any]:
        if self.replay != REPLAY_NATIVE:
            return {"status": "not_applicable"}
        prev = self._last_usage
        if prev is None:
            return {"status": "first_turn"}
        needed = ("input_tokens", "output_tokens", "reasoning_tokens")
        if any(k not in prev for k in needed) or "input_tokens" not in usage:
            return {"status": "unmeasured"}
        prior_output = prev["output_tokens"]
        prior_reasoning = min(prev["reasoning_tokens"], prior_output)
        growth = usage["input_tokens"] - prev["input_tokens"]
        excess = growth - prior_output
        check: dict[str, Any] = {
            "prompt_growth": growth,
            "prior_output_tokens": prior_output,
            "prior_reasoning_tokens": prior_reasoning,
            "growth_minus_prior_output": excess,
            "prior_host": self._last_host,
        }
        if prior_reasoning < RETENTION_CHECK_MIN_REASONING:
            check["status"] = "not_decisive"
            return check
        floor = -max(RETENTION_CHECK_FRACTION * prior_reasoning, RETENTION_CHECK_FLOOR_TOKENS)
        check["status"] = "consistent" if excess >= floor else "inconsistent"
        return check

    def _cost(
        self, raw_usage: Mapping[str, Any], usage: Mapping[str, int], service_tier: Any
    ) -> tuple[float | None, str, str | None]:
        reported = raw_usage.get("cost")
        reasons: list[str] = []
        if not (
            isinstance(reported, (int, float))
            and not isinstance(reported, bool)
            and math.isfinite(reported)
            and reported >= 0
        ):
            reasons.append("usage.cost absent")
        if raw_usage.get("is_byok") is True:
            reasons.append("BYOK request: usage.cost is OpenRouter's fee, not the list price")
        if service_tier not in (None, "default"):
            reasons.append(f"served on the {service_tier!r} tier, not the standard tier")
        if not reasons:
            return float(reported), "openrouter_usage_cost", None
        price: float | None = None
        if self._pricer is not None:
            try:
                price = self._pricer(dict(usage))
            except ValueError:  # inconsistent cache classes: unpriceable, never guessed
                price = None
        basis = self._pricer_basis if price is not None else "unpriced"
        return price, basis, "; ".join(reasons)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "INERT_TOOL",
    "NOT_RETAINED",
    "REPLAY_NATIVE",
    "REPLAY_VISIBLE_ONLY",
    "RETENTION_VERDICTS",
    "UNMEASURABLE",
    "VERIFIED",
    "ConcurrentTurnError",
    "OpenRouterHTTPTransport",
    "OpenRouterNativeSession",
    "OpenRouterTransport",
    "OpenRouterTransportError",
    "RetentionVerdict",
    "UnsupportedToolCallError",
    "graded_effort_binding",
    "openrouter_usage_classes",
    "replay_message",
]
