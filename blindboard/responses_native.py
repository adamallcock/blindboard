# Vendored from benchkit@0.3.10 (sha256:ea9a8efa919de5d5). Do not hand-edit; run 'benchkit sync' to update.
"""OpenAI Responses under the shared native-session contract.

A thin wrapper giving ``responses_stateful.ResponsesSession`` the same
``turn(user_texts) -> NativeTurn`` interface as the Anthropic, Gemini and
OpenRouter native sessions, so a benchmark runner drives every provider
one way. The frozen OpenAI condition: ``typed_item_replay`` (``store=false``;
every native output item, encrypted reasoning included, replayed before new
input), ``reasoning.context`` pinned to ``all_turns``, compaction off unless
an explicit threshold is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from blindboard.native_session import (
    IncompleteTurnError,
    NativeTurn,
    Pricer,
    assert_public_safe,
    registry_pricer,
)
from blindboard.client import build_provenance, responses_usage_classes
from blindboard.responses_stateful import (
    CompactionPolicy,
    ContinuationMode,
    ResponsesSession,
    ResponsesSessionConfig,
    ResponsesTerminalError,
    RetryPolicy,
    StorePolicy,
)


class OpenAINativeSession:
    """One episode's lossless OpenAI Responses session."""

    supports_lossless_multi_turn = True

    def __init__(
        self,
        *,
        model: str,
        instructions: str,
        effort: str = "medium",
        service_tier: str | None = "flex",
        reasoning_context: str = "all_turns",
        compact_threshold: int | None = None,
        request_timeout_seconds: int = 1800,
        api_key: str | None = None,
        transport: Any = None,
        pricer: Pricer | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self._pricer = pricer or registry_pricer(model)
        compaction = (
            CompactionPolicy(enabled=True, compact_threshold=compact_threshold)
            if compact_threshold
            else CompactionPolicy(enabled=False)
        )
        self._session = ResponsesSession(
            ResponsesSessionConfig(
                model=model,
                continuation_mode=ContinuationMode.TYPED_ITEM_REPLAY,
                compaction_policy=compaction,
                instructions=instructions,
                effort=effort,
                reasoning_summary="auto",
                reasoning_context=reasoning_context,
                service_tier=service_tier,
                store_policy=StorePolicy.DO_NOT_STORE,
                # Flex capacity 429s arrive in bursts under wide parallelism:
                # back off patiently (2s doubling, ~8.5 min worst case).
                # invalid_prompt refusals are random (a verbatim re-POST of
                # a refused request completed), so the same payload is
                # re-sent; the model never sees the refusal.
                retry_policy=RetryPolicy(
                    max_post_retries=8,
                    initial_backoff_seconds=2.0,
                    resend_error_codes=("invalid_prompt",),
                    max_resends=3,
                ),
                request_timeout_seconds=request_timeout_seconds,
            ),
            transport=transport,
            api_key=api_key,
        )

    def turn(self, user_texts: Sequence[str]) -> NativeTurn:
        if not user_texts:
            raise ValueError("a turn needs at least one user message")
        try:
            result = self._session.turn([{"role": "user", "content": text} for text in user_texts])
        except ResponsesTerminalError as exc:
            if exc.status != "incomplete":
                raise
            raise self._incomplete(exc) from exc
        md = result.metadata
        usage = responses_usage_classes(md.get("usage"))
        public: dict[str, Any] = {
            "turn_index": md.get("turn_index"),
            "retention": {
                "replayed": "typed_output_items",
                "reasoning_context": md.get("reasoning_context"),
                "continuation": md.get("continuation"),
            },
            "compaction": md.get("compaction"),
            "output_item_type_counts": md.get("output_item_type_counts"),
            "state_digest": md.get("state_digest"),
            # Attempt counts and retry/resend history for this turn.
            "transport": md.get("transport"),
        }
        provenance = build_provenance(
            response_id=md.get("completed_response_id"),
            routed_model=md.get("routed_model"),
            service_tier=md.get("service_tier"),
            finish_reason=md.get("finish_reason"),
        )
        assert_public_safe(public)
        assert_public_safe(provenance)
        return NativeTurn(
            text=result.text,
            usage=usage,
            reasoning_summaries=tuple(md.get("reasoning_summaries") or ()),
            provenance=provenance,
            public=public,
            cost_usd=self._pricer(usage),
        )

    def _incomplete(self, exc: ResponsesTerminalError) -> IncompleteTurnError:
        """An incomplete response (e.g. max_output_tokens) as a billed empty
        turn. The session did not advance: its output is not retained."""
        usage = responses_usage_classes(exc.usage)
        public: dict[str, Any] = {
            "incomplete_reason": exc.error_code or "incomplete",
            "retention": {"replayed": "typed_output_items", "this_turn_retained": False},
        }
        provenance = build_provenance(
            response_id=exc.response_id,
            routed_model=exc.routed_model,
            service_tier=exc.service_tier,
            finish_reason="incomplete",
        )
        assert_public_safe(public)
        assert_public_safe(provenance)
        turn = NativeTurn(
            text="",
            usage=usage,
            provenance=provenance,
            public=public,
            cost_usd=self._pricer(usage) if usage else None,
        )
        return IncompleteTurnError(
            f"OpenAI turn incomplete ({public['incomplete_reason']}); history unchanged",
            turn=turn,
        )
