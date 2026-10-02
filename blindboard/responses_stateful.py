# Vendored from benchkit@0.3.10 (sha256:e246b20b56c8cad8). Do not hand-edit; run 'benchkit sync' to update.
"""Provider-native OpenAI Responses continuation and compaction.

This module is deliberately separate from :mod:`blindboard.client`.
The historical ``APIAdapter`` remains a single-inference, text-only adapter;
this session preserves native typed output when a benchmark explicitly opts in
to a stateful protocol.

Runtime dependencies are Python's standard library only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from blindboard.pricing import list_price_usd
from blindboard.client import (
    API_URL,
    extract_reasoning_summaries,
    extract_response_text,
)
from blindboard.vision import openai_max_image_detail

_ACTIVE_STATUSES = frozenset({"queued", "in_progress"})
_RETRYABLE_RESPONSE_CODES = frozenset(
    {
        "internal_server_error",
        "rate_limit_exceeded",
        "server_error",
        "server_is_overloaded",
        "capacity_exceeded",
    }
)
_REASONING_CONTEXTS = frozenset({"auto", "current_turn", "all_turns"})
_REASONING_SUMMARIES = frozenset({"auto", "concise", "detailed"})
_IMAGE_DETAILS = frozenset({"low", "high", "original", "auto"})
_FORBIDDEN_NEW_INPUT_TYPES = frozenset(
    {
        "reasoning",
        "compaction",
        "function_call",
        "custom_tool_call",
        "computer_call",
        "web_search_call",
        "file_search_call",
        "code_interpreter_call",
    }
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "encrypted_content",
        "output",
        "output_items",
        "response_output",
        "typed_replay_state",
        "native_state",
        "checkpoint",
        "checkpoint_path",
        "request_payload",
    }
)
_CHECKPOINT_VERSION = "benchkit-openai-responses-session-v1"


class ContinuationMode(str, Enum):
    STATELESS_ONE_SHOT = "stateless_one_shot"
    PREVIOUS_RESPONSE_ID = "previous_response_id"
    TYPED_ITEM_REPLAY = "typed_item_replay"


class StorePolicy(str, Enum):
    AUTO = "auto"
    STORE = "store"
    DO_NOT_STORE = "do_not_store"


@dataclass(frozen=True)
class CompactionPolicy:
    enabled: bool
    compact_threshold: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("compaction enabled must be bool")
        if self.enabled:
            if (
                not isinstance(self.compact_threshold, int)
                or isinstance(self.compact_threshold, bool)
                or self.compact_threshold <= 0
            ):
                raise ValueError(
                    "enabled compaction requires a positive explicit compact_threshold"
                )
        elif self.compact_threshold is not None:
            raise ValueError("disabled compaction cannot carry a compact_threshold")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded HTTP retries, provider transports, and background polling."""

    max_post_retries: int = 3
    max_poll_retries: int = 3
    max_provider_attempts: int = 2
    initial_backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    poll_interval_seconds: float = 2.0
    max_poll_seconds: float = 600.0
    retryable_status_codes: tuple[int, ...] = (
        408,
        409,
        429,
        500,
        502,
        503,
        504,
        520,
        522,
        524,
    )
    # Provider error codes that refuse a request at random rather than for
    # anything in it. The identical payload is re-sent under a fresh
    # idempotency key, at most max_resends times per request. Off by default.
    # Live 2026-09-25: gpt-5.6-sol refused a request with HTTP 400
    # invalid_prompt and the verbatim re-POST completed.
    resend_error_codes: tuple[str, ...] = ()
    max_resends: int = 2

    def __post_init__(self) -> None:
        try:
            retryable_status_codes = tuple(self.retryable_status_codes)
        except TypeError as exc:
            raise TypeError("retryable_status_codes must be iterable") from exc
        object.__setattr__(self, "retryable_status_codes", retryable_status_codes)
        resend_error_codes = tuple(self.resend_error_codes)
        if any(not isinstance(code, str) or not code for code in resend_error_codes):
            raise ValueError("resend_error_codes must contain error-code strings")
        object.__setattr__(self, "resend_error_codes", resend_error_codes)
        if not isinstance(self.max_resends, int) or isinstance(self.max_resends, bool) or self.max_resends < 0:
            raise ValueError("max_resends must be a nonnegative integer")
        for name in ("max_post_retries", "max_poll_retries"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            not isinstance(self.max_provider_attempts, int)
            or isinstance(self.max_provider_attempts, bool)
            or self.max_provider_attempts <= 0
        ):
            raise ValueError("max_provider_attempts must be a positive integer")
        for name in (
            "initial_backoff_seconds",
            "poll_interval_seconds",
            "max_poll_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if (
            not isinstance(self.backoff_multiplier, (int, float))
            or isinstance(self.backoff_multiplier, bool)
            or self.backoff_multiplier < 1
        ):
            raise ValueError("backoff_multiplier must be at least 1")
        if not self.retryable_status_codes or any(
            not isinstance(code, int) or isinstance(code, bool) or code < 100
            for code in self.retryable_status_codes
        ):
            raise ValueError("retryable_status_codes must contain HTTP status integers")


@dataclass(frozen=True)
class ResponsesSessionConfig:
    model: str
    continuation_mode: ContinuationMode
    compaction_policy: CompactionPolicy
    instructions: str | None = None
    effort: str = "medium"
    reasoning_summary: str | None = "auto"
    reasoning_context: str | None = "auto"
    service_tier: str | None = "flex"
    background: bool = False
    max_output_tokens: int | None = None
    store_policy: StorePolicy = StorePolicy.AUTO
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    tools: tuple[Mapping[str, Any], ...] = ()
    text_verbosity: str = "low"
    request_timeout_seconds: int = 60

    def __post_init__(self) -> None:
        try:
            mode = ContinuationMode(self.continuation_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported continuation mode") from exc
        try:
            store_policy = StorePolicy(self.store_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported store policy") from exc
        object.__setattr__(self, "continuation_mode", mode)
        object.__setattr__(self, "store_policy", store_policy)

        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(self.compaction_policy, CompactionPolicy):
            raise TypeError("compaction_policy must be CompactionPolicy")
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise TypeError("instructions must be a string or None")
        if not isinstance(self.effort, str) or not self.effort:
            raise ValueError("effort must be a nonempty string")
        if (
            self.reasoning_summary is not None
            and self.reasoning_summary not in _REASONING_SUMMARIES
        ):
            raise ValueError("unsupported reasoning summary")
        if (
            self.reasoning_context is not None
            and self.reasoning_context not in _REASONING_CONTEXTS
        ):
            raise ValueError("unsupported reasoning context")
        if not isinstance(self.background, bool):
            raise TypeError("background must be bool")
        if self.service_tier is not None and (
            not isinstance(self.service_tier, str) or not self.service_tier
        ):
            raise ValueError("service_tier must be a nonempty string or None")
        if (
            self.max_output_tokens is not None
            and (
                not isinstance(self.max_output_tokens, int)
                or isinstance(self.max_output_tokens, bool)
                or self.max_output_tokens <= 0
            )
        ):
            raise ValueError("max_output_tokens must be a positive integer")
        if not isinstance(self.retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be RetryPolicy")
        if not isinstance(self.request_timeout_seconds, int) or self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be a positive integer")
        if not isinstance(self.text_verbosity, str) or not self.text_verbosity:
            raise ValueError("text_verbosity must be a nonempty string")
        tools = tuple(copy.deepcopy(tuple(self.tools)))
        if any(not isinstance(tool, Mapping) for tool in tools):
            raise TypeError("tools must contain provider objects")
        try:
            _canonical_json(tools)
        except (TypeError, ValueError) as exc:
            raise ValueError("tools must be JSON-serializable provider objects") from exc
        object.__setattr__(self, "tools", tools)

        effective = self.effective_store
        if mode is ContinuationMode.PREVIOUS_RESPONSE_ID and not effective:
            raise ValueError("previous_response_id continuation requires store=true")
        if mode is ContinuationMode.TYPED_ITEM_REPLAY and effective:
            raise ValueError("typed_item_replay continuation requires store=false")

    @property
    def effective_store(self) -> bool:
        if self.store_policy is StorePolicy.STORE:
            return True
        if self.store_policy is StorePolicy.DO_NOT_STORE:
            return False
        return self.continuation_mode is ContinuationMode.PREVIOUS_RESPONSE_ID


@dataclass(frozen=True, repr=False)
class ResponsesSessionState:
    logical_session_id: str
    next_turn_index: int
    completed_head_response_id: str | None
    last_completed_response_id: str | None
    state_digests: Mapping[str, str]
    _typed_replay_state: tuple[dict[str, Any], ...] = field(
        default=(), repr=False, compare=False
    )

    @property
    def typed_replay_state(self) -> list[dict[str, Any]]:
        """Explicit access to private opaque replay state; never artifact metadata."""

        return copy.deepcopy(list(self._typed_replay_state))

    def __repr__(self) -> str:
        return (
            "ResponsesSessionState("
            f"logical_session_id={self.logical_session_id!r}, "
            f"next_turn_index={self.next_turn_index}, "
            f"completed_head_response_id={self.completed_head_response_id!r}, "
            f"last_completed_response_id={self.last_completed_response_id!r}, "
            f"state_digests={dict(self.state_digests)!r}, "
            f"typed_replay_item_count={len(self._typed_replay_state)})"
        )


@dataclass(frozen=True)
class ResponsesTurnResult:
    text: str
    metadata: Mapping[str, Any]


class ResponsesTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        idempotency_key: str | None,
        timeout_seconds: int,
    ) -> dict[str, Any]: ...


class ResponsesHTTPError(RuntimeError):
    """Sanitized HTTP failure: raw provider bodies are never retained."""

    def __init__(
        self,
        *,
        method: str,
        status_code: int,
        error_code: str | None,
        body_sha256: str,
        request_id: str | None = None,
    ) -> None:
        code = f", code={error_code}" if error_code else ""
        request = f", request_id={request_id}" if request_id else ""
        super().__init__(
            f"OpenAI Responses {method} HTTP {status_code}{code}{request}; "
            f"body_sha256={body_sha256}"
        )
        self.method = method
        self.status_code = status_code
        self.error_code = error_code
        self.body_sha256 = body_sha256
        self.request_id = request_id
        self.retryable = False
        self.resume_same_request = False


class ResponsesTransportError(RuntimeError):
    retryable = True
    resume_same_request = True

    def __init__(self, *, method: str, exception_type: str) -> None:
        super().__init__(f"OpenAI Responses {method} transport failed ({exception_type})")
        self.method = method


class ResponsesTerminalError(RuntimeError):
    def __init__(
        self,
        *,
        response_id: str | None,
        status: str | None,
        error_code: str | None,
        retryable: bool,
        usage: Mapping[str, Any] | None = None,
        routed_model: str | None = None,
        service_tier: str | None = None,
    ) -> None:
        super().__init__(
            f"OpenAI response {response_id or '<missing-id>'} ended as "
            f"{status or '<missing-status>'}"
            + (f" (code={error_code})" if error_code else "")
        )
        self.response_id = response_id
        self.status = status
        self.error_code = error_code
        self.retryable = retryable
        self.resume_same_request = False
        # Billed usage of the failed response, when it reported any.
        self.usage = dict(usage) if isinstance(usage, Mapping) else None
        self.routed_model = routed_model
        self.service_tier = service_tier


class ResponsesPollTimeout(TimeoutError):
    retryable = True
    resume_same_request = True

    def __init__(self, *, response_id: str, status: str | None) -> None:
        super().__init__(
            f"OpenAI response {response_id} remains {status or '<unknown>'} "
            "after the local polling window"
        )
        self.response_id = response_id


class ConcurrentSessionMutationError(RuntimeError):
    pass


class PendingTurnError(RuntimeError):
    pass


class ResponsesCheckpointError(RuntimeError):
    pass


class ResponsesHTTPTransport:
    """Small stdlib JSON transport; safe errors never echo response bodies."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._api_key = api_key

    def request(
        self,
        method: str,
        url: str,
        *,
        payload: Mapping[str, Any] | None,
        idempotency_key: str | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        request = urllib.request.Request(
            url,
            data=None
            if payload is None
            else json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = response.read()
                try:
                    parsed = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ResponsesTransportError(
                        method=method, exception_type="InvalidJSON"
                    ) from None
        except urllib.error.HTTPError as exc:
            body = exc.read()
            error_code: str | None = None
            try:
                error = json.loads(body.decode("utf-8")).get("error")
                if isinstance(error, Mapping) and isinstance(error.get("code"), str):
                    error_code = error["code"]
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
            headers_obj = getattr(exc, "headers", None)
            request_id = headers_obj.get("x-request-id") if headers_obj else None
            raise ResponsesHTTPError(
                method=method,
                status_code=exc.code,
                error_code=error_code,
                body_sha256=hashlib.sha256(body).hexdigest(),
                request_id=request_id,
            ) from None
        except (TimeoutError, urllib.error.URLError) as exc:
            raise ResponsesTransportError(
                method=method, exception_type=type(exc).__name__
            ) from None
        if not isinstance(parsed, dict):
            raise ResponsesTransportError(method=method, exception_type="NonObjectJSON")
        return parsed


@dataclass(repr=False)
class _PendingTurn:
    turn_index: int
    request_key: str
    request_fingerprint: str
    parent_response_id: str | None
    input_items: list[dict[str, Any]]
    payload: dict[str, Any]
    provider_attempt: int
    idempotency_key: str
    response_id: str | None = None
    total_http_attempts: int = 0
    post_http_attempts: int = 0
    poll_http_attempts: int = 0
    resume_behavior: list[str] = field(default_factory=list)
    completed_response: dict[str, Any] | None = None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _idempotency_key(request_key: str, fingerprint: str, provider_attempt: int) -> str:
    material = f"{request_key}\0{fingerprint}\0{provider_attempt}".encode()
    return f"benchkit-{hashlib.sha256(material).hexdigest()}"


def _safe_error_code(response: Mapping[str, Any]) -> str | None:
    error = response.get("error") or response.get("incomplete_details")
    if not isinstance(error, Mapping):
        return None
    code = error.get("code") or error.get("reason")
    return code if isinstance(code, str) else None


def _item_type_counts(output: Sequence[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in output:
        item_type = item.get("type") if isinstance(item, Mapping) else None
        counts[item_type if isinstance(item_type, str) else "unknown"] += 1
    return dict(sorted(counts.items()))


def assert_public_metadata_safe(metadata: Mapping[str, Any]) -> None:
    """Recursively reject native Responses state from publishable metadata."""

    if not isinstance(metadata, Mapping):
        raise TypeError("public metadata must be a mapping")

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            item_type = value.get("type")
            if item_type in {"reasoning", "compaction"}:
                raise ValueError(f"public metadata contains native item at {'.'.join(path)}")
            for raw_key, nested in value.items():
                key = str(raw_key).casefold()
                if key in _FORBIDDEN_PUBLIC_KEYS:
                    raise ValueError(
                        f"public metadata contains forbidden key at "
                        f"{'.'.join((*path, str(raw_key)))}"
                    )
                walk(nested, (*path, str(raw_key)))
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, (*path, str(index)))

    walk(metadata, ())


def _normalize_new_input_items(
    items: Sequence[Mapping[str, Any]], *, model: str
) -> list[dict[str, Any]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or not items:
        raise ValueError("turn input must be a nonempty sequence of typed items")
    normalized: list[dict[str, Any]] = []
    for index, raw_item in enumerate(items):
        if not isinstance(raw_item, Mapping):
            raise TypeError(f"turn input item {index} must be an object")
        item = copy.deepcopy(dict(raw_item))
        item_type = item.get("type")
        role = item.get("role")
        if item_type in _FORBIDDEN_NEW_INPUT_TYPES or role == "assistant":
            raise ValueError("new turn input cannot contain prior native output items")
        if role is not None:
            if role != "user":
                raise ValueError("stable developer/system instructions must be top-level")
            content = item.get("content")
            if isinstance(content, str):
                if not content:
                    raise ValueError("user message content cannot be empty")
            elif isinstance(content, list) and content:
                for part_index, part in enumerate(content):
                    if not isinstance(part, Mapping):
                        raise TypeError(
                            f"user content part {part_index} must be an object"
                        )
                    part = copy.deepcopy(dict(part))
                    part_type = part.get("type")
                    if part_type == "input_text":
                        if not isinstance(part.get("text"), str) or not part["text"]:
                            raise ValueError("input_text requires nonempty text")
                    elif part_type == "input_image":
                        if not isinstance(part.get("image_url"), str) or not part["image_url"]:
                            raise ValueError("input_image requires image_url")
                        detail = part.get("detail", "max")
                        if detail == "max":
                            part["detail"] = openai_max_image_detail(model)
                        elif detail is None:
                            part.pop("detail", None)
                        elif detail not in _IMAGE_DETAILS:
                            raise ValueError("unsupported OpenAI image detail")
                    elif part_type != "input_file":
                        raise ValueError("user content must use typed input parts")
                    content[part_index] = part
            else:
                raise ValueError("user message content must be nonempty text or typed parts")
        elif not (isinstance(item_type, str) and item_type.endswith("_call_output")):
            raise ValueError("turn input must be a user message or native tool-result item")
        normalized.append(item)
    try:
        _canonical_json(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("turn input must be JSON-serializable") from exc
    return normalized


def _validate_replay_state(items: Sequence[Mapping[str, Any]]) -> None:
    for item in items:
        if item.get("type") != "reasoning":
            continue
        encrypted = item.get("encrypted_content")
        if not isinstance(encrypted, str) or not encrypted:
            raise ValueError(
                "typed replay cannot continue from a reasoning summary alone; "
                "encrypted_content is required"
            )


class ResponsesSession:
    """One independently locked provider-native Responses session.

    ``turn`` always creates a new scientific turn. ``resume_pending_turn``
    resumes an already prepared POST or polls its already-created background
    response; the two operations are intentionally distinct.
    """

    supports_lossless_multi_turn = True

    def __init__(
        self,
        config: ResponsesSessionConfig,
        *,
        transport: ResponsesTransport | None = None,
        api_key: str | None = None,
        logical_session_id: str | None = None,
        checkpoint_directory: Path | None = None,
        sleeper: Any = time.sleep,
        monotonic: Any = time.monotonic,
        _load_checkpoint: bool = False,
    ) -> None:
        if not isinstance(config, ResponsesSessionConfig):
            raise TypeError("config must be ResponsesSessionConfig")
        self.config = config
        self.logical_session_id = logical_session_id or str(uuid.uuid4())
        if not isinstance(self.logical_session_id, str) or not self.logical_session_id:
            raise ValueError("logical_session_id must be a nonempty string")
        self._transport = transport or ResponsesHTTPTransport(
            api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._mutation_lock = threading.Lock()
        self._next_turn_index = 0
        self._completed_head_response_id: str | None = None
        self._last_completed_response_id: str | None = None
        self._replay_items: list[dict[str, Any]] = []
        self._state_digests: dict[str, str] = {
            "raw_state_sha256": _sha256_json([]),
        }
        self._pending: _PendingTurn | None = None
        self._checkpoint_path: Path | None = None
        if checkpoint_directory is not None:
            directory = Path(checkpoint_directory)
            try:
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(directory, 0o700)
            except OSError:
                raise ResponsesCheckpointError(
                    "private checkpoint directory preparation failed"
                ) from None
            filename = hashlib.sha256(self.logical_session_id.encode("utf-8")).hexdigest()
            self._checkpoint_path = directory / f"responses-session-{filename[:24]}.json"
        if _load_checkpoint:
            self._load_checkpoint()

    def __repr__(self) -> str:
        return (
            "ResponsesSession("
            f"model={self.config.model!r}, "
            f"continuation_mode={self.config.continuation_mode.value!r}, "
            f"logical_session_id={self.logical_session_id!r}, "
            f"next_turn_index={self._next_turn_index}, "
            f"has_pending_turn={self._pending is not None})"
        )

    @classmethod
    def from_checkpoint(
        cls,
        config: ResponsesSessionConfig,
        *,
        logical_session_id: str,
        checkpoint_directory: Path,
        transport: ResponsesTransport | None = None,
        api_key: str | None = None,
        sleeper: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> ResponsesSession:
        return cls(
            config,
            logical_session_id=logical_session_id,
            checkpoint_directory=checkpoint_directory,
            transport=transport,
            api_key=api_key,
            sleeper=sleeper,
            monotonic=monotonic,
            _load_checkpoint=True,
        )

    @property
    def state(self) -> ResponsesSessionState:
        return ResponsesSessionState(
            logical_session_id=self.logical_session_id,
            next_turn_index=self._next_turn_index,
            completed_head_response_id=self._completed_head_response_id,
            last_completed_response_id=self._last_completed_response_id,
            state_digests=copy.deepcopy(self._state_digests),
            _typed_replay_state=tuple(copy.deepcopy(self._replay_items)),
        )

    def build_turn_payload(
        self, input_items: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Validate and return the next exact request body without mutation/I/O."""

        if self._pending is not None:
            raise PendingTurnError(
                "session has a pending turn; call resume_pending_turn instead"
            )
        if (
            self.config.continuation_mode is ContinuationMode.STATELESS_ONE_SHOT
            and self._next_turn_index != 0
        ):
            raise ValueError("stateless_one_shot sessions permit exactly one turn")
        normalized = _normalize_new_input_items(input_items, model=self.config.model)
        return self._build_payload(normalized)

    def turn(
        self,
        input_items: Sequence[Mapping[str, Any]],
        *,
        request_key: str | None = None,
    ) -> ResponsesTurnResult:
        if not self._mutation_lock.acquire(blocking=False):
            raise ConcurrentSessionMutationError(
                "one ResponsesSession cannot mutate concurrently"
            )
        try:
            if self._pending is not None:
                raise PendingTurnError(
                    "session has a pending turn; call resume_pending_turn instead"
                )
            payload = self.build_turn_payload(input_items)
            normalized = _normalize_new_input_items(input_items, model=self.config.model)
            logical_key = request_key or (
                f"{self.logical_session_id}:turn:{self._next_turn_index}"
            )
            if not isinstance(logical_key, str) or not logical_key:
                raise ValueError("request_key must be a nonempty string")
            fingerprint = _sha256_json(payload)
            parent = (
                self._completed_head_response_id
                if self.config.continuation_mode
                is ContinuationMode.PREVIOUS_RESPONSE_ID
                else None
            )
            pending = _PendingTurn(
                turn_index=self._next_turn_index,
                request_key=logical_key,
                request_fingerprint=fingerprint,
                parent_response_id=parent,
                input_items=normalized,
                payload=payload,
                provider_attempt=0,
                idempotency_key=_idempotency_key(logical_key, fingerprint, 0),
            )
            self._pending = pending
            self._persist_checkpoint()
            return self._drive_pending()
        finally:
            self._mutation_lock.release()

    def resume_pending_turn(self) -> ResponsesTurnResult:
        if not self._mutation_lock.acquire(blocking=False):
            raise ConcurrentSessionMutationError(
                "one ResponsesSession cannot mutate concurrently"
            )
        try:
            if self._pending is None:
                raise PendingTurnError("session has no pending turn to resume")
            behavior = (
                "resume_same_response_id"
                if self._pending.response_id
                else "resume_same_request"
            )
            self._pending.resume_behavior.append(behavior)
            self._persist_checkpoint()
            return self._drive_pending()
        finally:
            self._mutation_lock.release()

    def _build_payload(self, new_input: list[dict[str, Any]]) -> dict[str, Any]:
        mode = self.config.continuation_mode
        if mode is ContinuationMode.TYPED_ITEM_REPLAY:
            _validate_replay_state(self._replay_items)
            request_input = copy.deepcopy(self._replay_items) + copy.deepcopy(new_input)
        else:
            request_input = copy.deepcopy(new_input)
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": request_input,
            "text": {
                "format": {"type": "text"},
                "verbosity": self.config.text_verbosity,
            },
            "reasoning": {"effort": self.config.effort},
            "tools": copy.deepcopy(list(self.config.tools)),
            "store": self.config.effective_store,
            "background": self.config.background,
        }
        if self.config.instructions is not None:
            payload["instructions"] = self.config.instructions
        if self.config.reasoning_summary is not None:
            payload["reasoning"]["summary"] = self.config.reasoning_summary
        if self.config.reasoning_context is not None:
            payload["reasoning"]["context"] = self.config.reasoning_context
        if self.config.service_tier is not None:
            payload["service_tier"] = self.config.service_tier
        if self.config.max_output_tokens is not None:
            payload["max_output_tokens"] = self.config.max_output_tokens
        if self.config.compaction_policy.enabled:
            payload["context_management"] = [
                {
                    "type": "compaction",
                    "compact_threshold": self.config.compaction_policy.compact_threshold,
                }
            ]
        if mode is ContinuationMode.PREVIOUS_RESPONSE_ID:
            if self._replay_items:
                raise RuntimeError("previous_response_id cannot combine with typed replay")
            if self._completed_head_response_id is not None:
                payload["previous_response_id"] = self._completed_head_response_id
        elif "previous_response_id" in payload:
            raise RuntimeError("continuation modes cannot be combined")
        return payload

    def _drive_pending(self) -> ResponsesTurnResult:
        while True:
            pending = self._require_pending()
            if pending.completed_response is not None:
                response = copy.deepcopy(pending.completed_response)
            elif pending.response_id is None:
                response = self._post_with_retries()
                response_id = response.get("id")
                if isinstance(response_id, str) and response_id:
                    pending.response_id = response_id
                pending.completed_response = copy.deepcopy(response)
                self._persist_checkpoint()  # journal ID/output before any poll/accept
            else:
                response = self._poll_with_retries(pending.response_id)
                pending.completed_response = copy.deepcopy(response)
                self._persist_checkpoint()

            status = response.get("status")
            if status == "completed":
                return self._accept_completed(response)
            if status in _ACTIVE_STATUSES:
                pending.completed_response = None
                self._persist_checkpoint()
                return self._poll_until_terminal(initial_status=status)

            code = _safe_error_code(response)
            retryable = code in _RETRYABLE_RESPONSE_CODES
            error = ResponsesTerminalError(
                response_id=response.get("id")
                if isinstance(response.get("id"), str)
                else None,
                status=status if isinstance(status, str) else None,
                error_code=code,
                retryable=retryable,
                usage=response.get("usage") if isinstance(response.get("usage"), Mapping) else None,
                routed_model=response.get("model") if isinstance(response.get("model"), str) else None,
                service_tier=response.get("service_tier")
                if isinstance(response.get("service_tier"), str)
                else None,
            )
            if retryable and pending.provider_attempt + 1 < self.config.retry_policy.max_provider_attempts:
                pending.provider_attempt += 1
                pending.idempotency_key = _idempotency_key(
                    pending.request_key,
                    pending.request_fingerprint,
                    pending.provider_attempt,
                )
                pending.response_id = None
                pending.completed_response = None
                pending.resume_behavior.append(
                    "new_provider_transport_after_terminal_failure"
                )
                self._persist_checkpoint()
                continue
            self._pending = None
            self._persist_checkpoint()
            raise error

    def _post_with_retries(self) -> dict[str, Any]:
        pending = self._require_pending()
        policy = self.config.retry_policy
        retry = 0
        resends = 0
        while True:
            try:
                pending.total_http_attempts += 1
                pending.post_http_attempts += 1
                response = self._transport.request(
                    "POST",
                    API_URL,
                    payload=copy.deepcopy(pending.payload),
                    idempotency_key=pending.idempotency_key,
                    timeout_seconds=self.config.request_timeout_seconds,
                )
                return response
            except (ResponsesHTTPError, ResponsesTransportError) as exc:
                if (
                    isinstance(exc, ResponsesHTTPError)
                    and exc.error_code in policy.resend_error_codes
                    and resends < policy.max_resends
                ):
                    # The refused request created nothing: send the same
                    # payload again under a fresh key.
                    resends += 1
                    pending.idempotency_key = (
                        _idempotency_key(
                            pending.request_key,
                            pending.request_fingerprint,
                            pending.provider_attempt,
                        )
                        + f"-resend{resends}"
                    )
                    pending.resume_behavior.append(f"resend_after_{exc.error_code}")
                    self._persist_checkpoint()
                    self._backoff(0)
                    continue
                retryable = self._is_retryable_http_error(exc)
                if not retryable:
                    self._pending = None
                    self._persist_checkpoint()
                    raise
                if retry >= policy.max_post_retries:
                    pending.resume_behavior.append("awaiting_resume_same_request")
                    self._persist_checkpoint()
                    raise
                pending.resume_behavior.append("retry_same_post")
                self._persist_checkpoint()
                self._backoff(retry)
                retry += 1

    def _poll_with_retries(self, response_id: str) -> dict[str, Any]:
        pending = self._require_pending()
        policy = self.config.retry_policy
        for retry in range(policy.max_poll_retries + 1):
            try:
                pending.total_http_attempts += 1
                pending.poll_http_attempts += 1
                return self._transport.request(
                    "GET",
                    f"{API_URL}/{response_id}",
                    payload=None,
                    idempotency_key=None,
                    timeout_seconds=self.config.request_timeout_seconds,
                )
            except (ResponsesHTTPError, ResponsesTransportError) as exc:
                retryable = self._is_retryable_http_error(exc)
                if not retryable:
                    self._pending = None
                    self._persist_checkpoint()
                    raise
                if retry >= policy.max_poll_retries:
                    pending.resume_behavior.append("awaiting_resume_same_response_id")
                    self._persist_checkpoint()
                    raise
                pending.resume_behavior.append("retry_same_response_id")
                self._persist_checkpoint()
                self._backoff(retry)
        raise AssertionError("unreachable poll retry loop")

    def _poll_until_terminal(self, *, initial_status: str | None) -> ResponsesTurnResult:
        pending = self._require_pending()
        if pending.response_id is None:
            raise RuntimeError("polling requires a response ID")
        deadline = self._monotonic() + self.config.retry_policy.max_poll_seconds
        last_status = initial_status
        while True:
            if self._monotonic() >= deadline:
                pending.resume_behavior.append("poll_window_expired")
                self._persist_checkpoint()
                raise ResponsesPollTimeout(
                    response_id=pending.response_id, status=last_status
                )
            interval = self.config.retry_policy.poll_interval_seconds
            if interval:
                self._sleeper(interval)
            response = self._poll_with_retries(pending.response_id)
            status = response.get("status")
            last_status = status if isinstance(status, str) else None
            pending.completed_response = copy.deepcopy(response)
            self._persist_checkpoint()
            if status in _ACTIVE_STATUSES:
                pending.completed_response = None
                self._persist_checkpoint()
                continue
            return self._drive_pending()

    def _accept_completed(self, response: Mapping[str, Any]) -> ResponsesTurnResult:
        pending = self._require_pending()
        if response.get("status") != "completed":
            raise ValueError("only a completed response can advance session state")
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise ValueError("completed response is missing an ID")
        output = response.get("output")
        if not isinstance(output, list):
            raise TypeError("completed response output must be a list")
        exact_output = copy.deepcopy(output)
        output_digest = _sha256_json(exact_output)

        next_replay = copy.deepcopy(self._replay_items)
        if self.config.continuation_mode is ContinuationMode.TYPED_ITEM_REPLAY:
            next_replay.extend(copy.deepcopy(pending.input_items))
            next_replay.extend(exact_output)
            raw_state_digest = _sha256_json(next_replay)
        elif self.config.continuation_mode is ContinuationMode.PREVIOUS_RESPONSE_ID:
            raw_state_digest = _sha256_json({"completed_head_response_id": response_id})
        else:
            raw_state_digest = _sha256_json(
                {"response_id": response_id, "output_sha256": output_digest}
            )

        next_head = (
            response_id
            if self.config.continuation_mode
            is ContinuationMode.PREVIOUS_RESPONSE_ID
            else None
        )
        next_digests = {
            "raw_state_sha256": raw_state_digest,
            "request_sha256": pending.request_fingerprint,
            "output_sha256": output_digest,
        }
        metadata = self._public_metadata(
            response=response,
            pending=pending,
            raw_state_digest=raw_state_digest,
            output_digest=output_digest,
        )
        assert_public_metadata_safe(metadata)

        # Build and durably replace the accepted state before mutating memory.
        checkpoint_payload = self._checkpoint_payload(
            next_turn_index=self._next_turn_index + 1,
            completed_head_response_id=next_head,
            last_completed_response_id=response_id,
            replay_items=next_replay,
            state_digests=next_digests,
            pending=None,
        )
        self._write_checkpoint_payload(checkpoint_payload)
        self._next_turn_index += 1
        self._completed_head_response_id = next_head
        self._last_completed_response_id = response_id
        self._replay_items = next_replay
        self._state_digests = next_digests
        self._pending = None
        return ResponsesTurnResult(
            text=extract_response_text(dict(response)),
            metadata=metadata,
        )

    def _public_metadata(
        self,
        *,
        response: Mapping[str, Any],
        pending: _PendingTurn,
        raw_state_digest: str,
        output_digest: str,
    ) -> dict[str, Any]:
        output = response.get("output") or []
        counts = _item_type_counts(output if isinstance(output, list) else [])
        usage_raw = response.get("usage")
        usage = copy.deepcopy(dict(usage_raw)) if isinstance(usage_raw, Mapping) else {}
        usage_for_cost = {
            key: value
            for key, value in usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        reasoning = response.get("reasoning")
        returned_context = (
            reasoning.get("context") if isinstance(reasoning, Mapping) else None
        )
        metadata: dict[str, Any] = {
            "logical_session_id": self.logical_session_id,
            "turn_index": pending.turn_index,
            "continuation": {
                "requested_mode": self.config.continuation_mode.value,
                "effective_mode": self.config.continuation_mode.value,
                "lossless": True,
            },
            "compaction": {
                "enabled": self.config.compaction_policy.enabled,
                "compact_threshold": self.config.compaction_policy.compact_threshold,
                "present": counts.get("compaction", 0) > 0,
                "item_count": counts.get("compaction", 0),
            },
            "request_fingerprint": pending.request_fingerprint,
            "parent_response_id": pending.parent_response_id,
            "completed_response_id": response.get("id"),
            "poll_response_id": pending.response_id if self.config.background else None,
            "store": self.config.effective_store,
            "store_policy": {
                "requested": self.config.store_policy.value,
                "effective": self.config.effective_store,
            },
            "reasoning_context": {
                "requested": self.config.reasoning_context,
                "returned": returned_context,
            },
            "state_digest": raw_state_digest,
            "output_digest": output_digest,
            "output_item_type_counts": counts,
            "transport": {
                "provider_requests": pending.provider_attempt + 1,
                "http_attempts": pending.total_http_attempts,
                "post_attempts": pending.post_http_attempts,
                "poll_attempts": pending.poll_http_attempts,
                "resume_behavior": list(pending.resume_behavior),
            },
            "routed_model": response.get("model"),
            "service_tier": response.get("service_tier"),
            "terminal_status": response.get("status"),
            "finish_reason": response.get("status"),
            "usage": usage,
            "paid_cost_usd": None,
            "list_equivalent_cost_usd": list_price_usd(
                self.config.model, usage_for_cost
            ),
            "reasoning_summaries": extract_reasoning_summaries(dict(response)),
        }
        return metadata

    def _is_retryable_http_error(
        self, exc: ResponsesHTTPError | ResponsesTransportError
    ) -> bool:
        if isinstance(exc, ResponsesTransportError):
            return True
        retryable = exc.status_code in self.config.retry_policy.retryable_status_codes
        exc.retryable = retryable
        exc.resume_same_request = retryable
        return retryable

    def _backoff(self, retry_index: int) -> None:
        seconds = self.config.retry_policy.initial_backoff_seconds * (
            self.config.retry_policy.backoff_multiplier**retry_index
        )
        if seconds:
            self._sleeper(seconds)

    def _require_pending(self) -> _PendingTurn:
        if self._pending is None:
            raise PendingTurnError("session has no pending turn")
        return self._pending

    def _config_digest(self) -> str:
        retry = self.config.retry_policy
        return _sha256_json(
            {
                "model": self.config.model,
                "continuation_mode": self.config.continuation_mode.value,
                "compaction": {
                    "enabled": self.config.compaction_policy.enabled,
                    "compact_threshold": self.config.compaction_policy.compact_threshold,
                },
                "instructions": self.config.instructions,
                "effort": self.config.effort,
                "reasoning_summary": self.config.reasoning_summary,
                "reasoning_context": self.config.reasoning_context,
                "service_tier": self.config.service_tier,
                "background": self.config.background,
                "max_output_tokens": self.config.max_output_tokens,
                "store_policy": self.config.store_policy.value,
                "tools": self.config.tools,
                "text_verbosity": self.config.text_verbosity,
                "request_timeout_seconds": self.config.request_timeout_seconds,
                "retry_policy": {
                    "max_post_retries": retry.max_post_retries,
                    "max_poll_retries": retry.max_poll_retries,
                    "max_provider_attempts": retry.max_provider_attempts,
                    "initial_backoff_seconds": retry.initial_backoff_seconds,
                    "backoff_multiplier": retry.backoff_multiplier,
                    "poll_interval_seconds": retry.poll_interval_seconds,
                    "max_poll_seconds": retry.max_poll_seconds,
                    "retryable_status_codes": retry.retryable_status_codes,
                },
            }
        )

    def _pending_payload(self, pending: _PendingTurn | None) -> dict[str, Any] | None:
        if pending is None:
            return None
        return {
            "turn_index": pending.turn_index,
            "request_key": pending.request_key,
            "request_fingerprint": pending.request_fingerprint,
            "parent_response_id": pending.parent_response_id,
            "input_items": pending.input_items,
            "payload": pending.payload,
            "provider_attempt": pending.provider_attempt,
            "idempotency_key": pending.idempotency_key,
            "response_id": pending.response_id,
            "total_http_attempts": pending.total_http_attempts,
            "post_http_attempts": pending.post_http_attempts,
            "poll_http_attempts": pending.poll_http_attempts,
            "resume_behavior": pending.resume_behavior,
            "completed_response": pending.completed_response,
        }

    def _checkpoint_payload(
        self,
        *,
        next_turn_index: int | None = None,
        completed_head_response_id: str | None | object = ...,
        last_completed_response_id: str | None | object = ...,
        replay_items: list[dict[str, Any]] | None = None,
        state_digests: Mapping[str, str] | None = None,
        pending: _PendingTurn | None | object = ...,
    ) -> dict[str, Any]:
        return {
            "version": _CHECKPOINT_VERSION,
            "config_sha256": self._config_digest(),
            "logical_session_id": self.logical_session_id,
            "next_turn_index": self._next_turn_index
            if next_turn_index is None
            else next_turn_index,
            "completed_head_response_id": self._completed_head_response_id
            if completed_head_response_id is ...
            else completed_head_response_id,
            "last_completed_response_id": self._last_completed_response_id
            if last_completed_response_id is ...
            else last_completed_response_id,
            "replay_items": self._replay_items if replay_items is None else replay_items,
            "state_digests": self._state_digests
            if state_digests is None
            else dict(state_digests),
            "pending": self._pending_payload(self._pending)
            if pending is ...
            else self._pending_payload(pending),
        }

    def _persist_checkpoint(self) -> None:
        self._write_checkpoint_payload(self._checkpoint_payload())

    def _write_checkpoint_payload(self, payload: Mapping[str, Any]) -> None:
        if self._checkpoint_path is None:
            return
        path = self._checkpoint_path
        temp_path: str | None = None
        try:
            sealed = copy.deepcopy(dict(payload))
            if "checkpoint_sha256" in sealed:
                raise ValueError("checkpoint payload is already sealed")
            sealed["checkpoint_sha256"] = _sha256_json(sealed)
            fd, temp_path = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(sealed, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            temp_path = None
            os.chmod(path, 0o600)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                # Directory fsync is unavailable on some supported platforms;
                # the file itself was fsynced and atomically replaced above.
                pass
        except (OSError, TypeError, ValueError):
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            raise ResponsesCheckpointError("private checkpoint write failed") from None

    def _load_checkpoint(self) -> None:
        if self._checkpoint_path is None:
            raise ResponsesCheckpointError("checkpoint directory is required for recovery")
        try:
            payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raise ResponsesCheckpointError("private checkpoint load failed") from None
        if not isinstance(payload, Mapping):
            raise ResponsesCheckpointError("private checkpoint is not an object")
        payload = dict(payload)
        recorded_checkpoint_digest = payload.pop("checkpoint_sha256", None)
        if (
            not isinstance(recorded_checkpoint_digest, str)
            or recorded_checkpoint_digest != _sha256_json(payload)
        ):
            raise ResponsesCheckpointError("private checkpoint integrity drift")
        if payload.get("version") != _CHECKPOINT_VERSION:
            raise ResponsesCheckpointError("private checkpoint version drift")
        if payload.get("config_sha256") != self._config_digest():
            raise ResponsesCheckpointError("private checkpoint config drift")
        if payload.get("logical_session_id") != self.logical_session_id:
            raise ResponsesCheckpointError("private checkpoint session drift")
        replay = payload.get("replay_items")
        digests = payload.get("state_digests")
        if not isinstance(replay, list) or not all(isinstance(item, dict) for item in replay):
            raise ResponsesCheckpointError("private checkpoint replay state is invalid")
        if not isinstance(digests, Mapping) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in digests.items()
        ):
            raise ResponsesCheckpointError("private checkpoint digests are invalid")
        self._next_turn_index = int(payload.get("next_turn_index", -1))
        if self._next_turn_index < 0:
            raise ResponsesCheckpointError("private checkpoint turn index is invalid")
        self._completed_head_response_id = payload.get("completed_head_response_id")
        self._last_completed_response_id = payload.get("last_completed_response_id")
        self._replay_items = copy.deepcopy(replay)
        self._state_digests = dict(digests)
        if self.config.continuation_mode is ContinuationMode.TYPED_ITEM_REPLAY:
            expected_state_digest = _sha256_json(self._replay_items)
            if self._completed_head_response_id is not None:
                raise ResponsesCheckpointError("private checkpoint continuation drift")
        elif self.config.continuation_mode is ContinuationMode.PREVIOUS_RESPONSE_ID:
            if self._replay_items:
                raise ResponsesCheckpointError("private checkpoint continuation drift")
            expected_state_digest = (
                _sha256_json(
                    {"completed_head_response_id": self._completed_head_response_id}
                )
                if self._completed_head_response_id is not None
                else _sha256_json([])
            )
        else:
            if self._replay_items or self._completed_head_response_id is not None:
                raise ResponsesCheckpointError("private checkpoint continuation drift")
            output_digest = self._state_digests.get("output_sha256")
            expected_state_digest = (
                _sha256_json(
                    {
                        "response_id": self._last_completed_response_id,
                        "output_sha256": output_digest,
                    }
                )
                if self._last_completed_response_id is not None
                and output_digest is not None
                else _sha256_json([])
            )
        if self._state_digests.get("raw_state_sha256") != expected_state_digest:
            raise ResponsesCheckpointError("private checkpoint state digest drift")
        pending = payload.get("pending")
        if pending is None:
            self._pending = None
            return
        if not isinstance(pending, Mapping):
            raise ResponsesCheckpointError("private checkpoint pending turn is invalid")
        try:
            recovered = _PendingTurn(
                turn_index=int(pending["turn_index"]),
                request_key=str(pending["request_key"]),
                request_fingerprint=str(pending["request_fingerprint"]),
                parent_response_id=pending.get("parent_response_id"),
                input_items=copy.deepcopy(pending["input_items"]),
                payload=copy.deepcopy(pending["payload"]),
                provider_attempt=int(pending["provider_attempt"]),
                idempotency_key=str(pending["idempotency_key"]),
                response_id=pending.get("response_id"),
                total_http_attempts=int(pending.get("total_http_attempts", 0)),
                post_http_attempts=int(pending.get("post_http_attempts", 0)),
                poll_http_attempts=int(pending.get("poll_http_attempts", 0)),
                resume_behavior=list(pending.get("resume_behavior") or []),
                completed_response=copy.deepcopy(pending.get("completed_response")),
            )
        except (KeyError, TypeError, ValueError):
            raise ResponsesCheckpointError("private checkpoint pending turn is invalid") from None
        if _sha256_json(recovered.payload) != recovered.request_fingerprint:
            raise ResponsesCheckpointError("private checkpoint request fingerprint drift")
        expected_key = _idempotency_key(
            recovered.request_key,
            recovered.request_fingerprint,
            recovered.provider_attempt,
        )
        if recovered.idempotency_key != expected_key:
            raise ResponsesCheckpointError("private checkpoint idempotency drift")
        self._pending = recovered


__all__ = [
    "CompactionPolicy",
    "ConcurrentSessionMutationError",
    "ContinuationMode",
    "PendingTurnError",
    "ResponsesCheckpointError",
    "ResponsesHTTPError",
    "ResponsesHTTPTransport",
    "ResponsesPollTimeout",
    "ResponsesSession",
    "ResponsesSessionConfig",
    "ResponsesSessionState",
    "ResponsesTerminalError",
    "ResponsesTransportError",
    "ResponsesTurnResult",
    "RetryPolicy",
    "StorePolicy",
    "assert_public_metadata_safe",
]
