"""The fast-searcher runtime.

Async-native: the loop awaits the ``AsyncGenerationFn`` seam for model turns
and the ``AsyncRetrievalClient`` seam for tools, fans parallel tool calls out
with ``asyncio.gather``, and pushes bulk token counting -- message-history
baselines and the round truncation pass -- off the event loop. The per-call
payload budget passes inside each tool task count inline: they are bounded by
one call's payload (single-digit milliseconds at the largest observed
payloads), where a thread dispatch per tool call would cost comparable
latency. The sync entry points in ``agent_harness.sync_api`` wrap these
coroutines with adapted seams.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel

import agent_harness.config as harness_config
from agent_harness.bridge_timing import emit as emit_bridge_timing
from agent_harness.config import (
    AGENTIC_SEARCH_DEFAULT_K,
    MAX_PARALLEL_TOOL_CALLS,
    METADATA_REPRESENTATIVE_VALUES_PER_FIELD,
    SEARCHER_PROMPT_TOKEN_LIMIT,
    SEARCHER_PRUNE_REMINDER_TOKENS,
    STRICT_TOPK,
    TURN_TOOL_PAYLOAD_TOKEN_BUDGET,
    HarnessTuning,
    MediaContentInput,
    searcher_agent_config,
    searcher_max_rounds,
)
from agent_harness.errors import AGENT_ERROR_KIND, error_kind
from agent_harness.llm import (
    AsyncGenerationFn,
    TokenUsage,
    completion_reasoning_tokens,
    extend_responses_api_trace,
    force_ranking,
    generation_failed,
    parse_ranking,
    require_generation_fn,
    response_message_to_dict,
    responses_api_trace_payload,
)
from agent_harness.metadata_guard import (
    build_metadata_registry,
    validate_metadata_filter_args,
    zero_result_filtered_search_count,
)
from agent_harness.prompts import (
    force_submit_message,
    over_budget_message,
    round_notice_message,
)
from agent_harness.retrieval import AsyncRetrievalClient
from agent_harness.schemas import (
    FilterChunksArgs,
    GetChunksArgs,
    GrepArgs,
    OverviewSearchArgs,
    PruneContextArgs,
    RankedChunkList,
    ReadDocumentArgs,
    SearchCorpusArgs,
)
from agent_harness.search import (
    ChunkIndex,
    ToolOutcome,
    execute_filter_chunks,
    execute_get_chunks,
    execute_grep,
    execute_inspect_metadata,
    execute_overview_search,
    execute_read_document,
    execute_search_corpus,
    redact_messages,
    truncate_round_payloads,
)
from agent_harness.searcher_prompts import fast_searcher_messages
from agent_harness.token_counter import ensure_rollout_token_counter
from agent_harness.tools.functions import resolve_async_retrieval_client
from agent_harness.tools.searcher_only import (
    filter_chunks_tool,
    grep_tool,
    search_corpus_tool,
    submit_ranking_tool,
)
from agent_harness.tools.shared import (
    get_chunks_tool,
    overview_search_tool,
    prune_context_tool,
    read_document_tool,
)

from .ranking import (
    finalize_chunks,
    normalize_top_k,
    ranking_trace_payload,
    ranking_unresolved,
    validate_ranked_chunk_ids,
)
from .shared import (
    agent_caused_payload_error,
    media_messages_for_payload,
    media_messages_for_tool_message,
    over_budget_round_missing_prune,
    parse_tool_args,
    tool_error,
    tool_message,
)
from .tool_trace import (
    _utc_now_iso,
    finish_tool_call_trace,
    jsonable,
    start_tool_call_trace,
    summarize_tool_call_iteration,
    summarize_tool_output,
    synthetic_tool_call_trace,
)


@dataclass(slots=True)
class FastAgenticSearchResult:
    """Structured fast-searcher result before rollout-record wrapping."""

    messages: list[dict[str, Any]]
    ranking: RankedChunkList | None
    chunks: list[dict[str, Any]]
    top_k: int | None
    strict_top_k: bool
    media_content: str
    additional_instructions: str | None
    queries_made: list[dict[str, Any]]
    initial_search_results: dict[str, Any]
    initial_metadata_facets: dict[str, Any]
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    max_input_tokens: int
    final_submit_input_tokens: int
    rounds_executed: int
    forced_ranking: bool
    ranking_unresolved: bool
    tool_call_iterations: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    openai_responses: list[dict[str, Any]]
    id_mapping: dict[str, Any]
    deleted_chunk_ids: list[dict[str, Any]]
    deleted_chunk_refs: list[dict[str, Any]]
    prompt_snapshot: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def retrieval(self) -> dict[str, Any]:
        return {
            "ranked_ids": [
                str(chunk["chunk_id"])
                for chunk in self.chunks
                if isinstance(chunk, Mapping) and chunk.get("chunk_id")
            ],
            "chunks": self.chunks,
            "ranking_strategy": self.ranking.ranking_strategy if self.ranking else None,
            "top_k": self.top_k,
            "strict_top_k": self.strict_top_k,
        }

    def to_record(self) -> dict[str, Any]:
        """Return the ``fast_agentic_search`` record payload."""
        result = {
            "ranking": self.ranking,
            "ranking_strategy": self.ranking.ranking_strategy if self.ranking else None,
            "chunks": self.chunks,
            "top_k": self.top_k,
            "strict_top_k": self.strict_top_k,
            "media_content": self.media_content,
            "token_counter_mode": harness_config.token_counter_mode(),
            "additional_instructions": self.additional_instructions,
            "queries_made": self.queries_made,
            "initial_search_results": self.initial_search_results,
            "initial_metadata_facets": self.initial_metadata_facets,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "agent_token_usage": {
                "searcher": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "reasoning_tokens": self.reasoning_tokens,
                    "total_tokens": self.total_tokens,
                    "max_input_tokens": self.max_input_tokens,
                    "final_submit_input_tokens": self.final_submit_input_tokens,
                }
            },
            "rounds_executed": self.rounds_executed,
            "forced_ranking": self.forced_ranking,
            "ranking_unresolved": self.ranking_unresolved,
            "tool_call_iterations": self.tool_call_iterations,
            "tool_trace": self.tool_trace,
            "openai_responses": responses_api_trace_payload(self.openai_responses),
            "id_mapping": self.id_mapping,
            "deleted_chunk_ids": self.deleted_chunk_ids,
            "deleted_chunk_refs": self.deleted_chunk_refs,
        }
        if self.prompt_snapshot is not None:
            result["monitoring"] = {"prompt_snapshot": self.prompt_snapshot}
        return result


@dataclass(slots=True)
class RolloutTotals:
    """Token accounting for one agent loop."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    reasoning_tokens: int = 0
    max_input_tokens: int = 0
    final_submit_input_tokens: int = 0

    def record_turn(self, response: Any) -> TokenUsage:
        turn = TokenUsage.of_response(response)
        self.usage = self.usage + turn
        self.reasoning_tokens += completion_reasoning_tokens(response)
        self.max_input_tokens = max(self.max_input_tokens, turn.input_tokens)
        return turn

    def record_forced(self, usage: TokenUsage, *, prompt_tokens: int) -> None:
        self.usage = self.usage + usage
        self.max_input_tokens = max(self.max_input_tokens, prompt_tokens)
        self.final_submit_input_tokens = prompt_tokens


@dataclass(frozen=True, slots=True)
class InitialSearchContext:
    """The bootstrap fetches: metadata facets and the seed search, with queries."""

    metadata_facets: dict[str, Any]
    metadata_query: dict[str, Any]
    search_results: dict[str, Any]
    search_query: dict[str, Any]
    trace_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SearcherToolRound:
    """Outcome of one round of searcher tool calls."""

    final: Any | None
    queries: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    pruned_context: bool


async def _estimate_messages_tokens_off_loop(messages: Sequence[Mapping[str, Any]]) -> int:
    """Token-count the transcript on a worker thread.

    Serializing and tokenizing a transcript that can reach 100k tokens is CPU
    work; on a shared event loop that is a stall, so counting never runs
    inline.
    """
    return await asyncio.to_thread(estimate_messages_tokens, messages)


async def fast_agentic_search(
    user_text: str,
    *,
    store_identifiers: Sequence[str],
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    generation_fn: AsyncGenerationFn | None = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Run the fast searcher and return the plain-dict record payload."""
    result = await run_fast_agentic_search(
        user_text,
        store_identifiers=store_identifiers,
        top_k=top_k,
        strict_top_k=strict_top_k,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
        additional_instructions=additional_instructions,
        include_prompt_snapshot=include_prompt_snapshot,
        media_content=media_content,
        generation_fn=generation_fn,
        tuning=tuning,
        as_of=as_of,
    )
    return result.to_record()


async def run_fast_agentic_search(
    user_text: str,
    *,
    store_identifiers: Sequence[str],
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    generation_fn: AsyncGenerationFn | None = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
) -> FastAgenticSearchResult:
    """Run the fast searcher and return structured loop state.

    ``as_of`` pins the runtime-context date instead of the UTC wall clock
    (see ``prompts._runtime_context``).
    """
    if tuning is not None:
        with harness_config.tuning_setting(tuning):
            return await run_fast_agentic_search(
                user_text,
                store_identifiers=store_identifiers,
                top_k=top_k,
                strict_top_k=strict_top_k,
                client=client,
                api_key=api_key,
                api_key_env=api_key_env,
                additional_instructions=additional_instructions,
                include_prompt_snapshot=include_prompt_snapshot,
                media_content=media_content,
                generation_fn=generation_fn,
                as_of=as_of,
            )
    if media_content is not None:
        with harness_config.media_content_setting(media_content):
            return await run_fast_agentic_search(
                user_text,
                store_identifiers=store_identifiers,
                top_k=top_k,
                strict_top_k=strict_top_k,
                client=client,
                api_key=api_key,
                api_key_env=api_key_env,
                additional_instructions=additional_instructions,
                include_prompt_snapshot=include_prompt_snapshot,
                generation_fn=generation_fn,
                as_of=as_of,
            )

    generate = require_generation_fn(generation_fn)
    # Every budget below measures through config.count_text_tokens: install the
    # policy tokenizer behind it before the first prompt is built. The first
    # install loads the tokenizer (possibly a hub download), so it runs off the
    # event loop; later calls are an O(1) resolved-model check.
    await asyncio.to_thread(ensure_rollout_token_counter, searcher_agent_config().get("model"))
    effective_top_k = normalize_top_k(top_k)
    if strict_top_k and effective_top_k is None:
        effective_top_k = AGENTIC_SEARCH_DEFAULT_K
    effective_strict_top_k = bool(strict_top_k and effective_top_k is not None)

    index = ChunkIndex()
    bootstrap = await _fetch_initial_context(
        user_text,
        index=index,
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    initial_metadata_facets = bootstrap.metadata_facets
    initial_search_results = bootstrap.search_results
    messages = fast_searcher_messages(
        user_text=user_text,
        initial_search_results=initial_search_results,
        initial_metadata_facets=initial_metadata_facets,
        top_k=effective_top_k,
        strict_top_k=effective_strict_top_k,
        additional_instructions=additional_instructions,
        as_of=as_of,
    )
    messages.extend(media_messages_for_payload(initial_search_results))
    prompt_snapshot = (
        _fast_searcher_prompt_snapshot(
            messages=messages,
            additional_instructions=additional_instructions,
        )
        if include_prompt_snapshot
        else None
    )
    queries_made: list[dict[str, Any]] = [
        {**bootstrap.metadata_query, "source": "initial_metadata_facets"},
        {**bootstrap.search_query, "source": "initial_original_query"},
    ]
    final_ranking: RankedChunkList | None = None
    totals = RolloutTotals()
    rounds_executed = 0
    prompt_tokens_estimate = 0
    tool_call_iterations: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = list(bootstrap.trace_events)
    openai_responses: list[dict[str, Any]] = []

    search_rounds = 0
    max_rounds = searcher_max_rounds()
    while search_rounds < max_rounds:
        rounds_executed += 1
        estimated_tokens = max(
            prompt_tokens_estimate, await _estimate_messages_tokens_off_loop(messages)
        )
        # The tools schema stays identical across every round -- over-budget
        # rounds included; a missing prune is recorded after generation instead
        # of swapping in a reduced tool list, so every round of one rollout
        # presents the same tool surface.
        tools = _searcher_tools(
            top_k=effective_top_k,
            strict_top_k=effective_strict_top_k,
        )
        if rounds_executed > 1:
            messages.append(round_notice_message(rounds_executed, max_rounds))
        over_budget = estimated_tokens >= SEARCHER_PRUNE_REMINDER_TOKENS
        if over_budget:
            messages.append(over_budget_message(estimated_tokens))
        search_rounds += 1

        response = await generate(
            messages,
            tools=tools,
            completion_config=searcher_agent_config(),
        )
        response_returned = time.perf_counter()
        extend_responses_api_trace(
            openai_responses,
            response,
            agent="fast_searcher",
            iteration=rounds_executed,
            phase="generation",
        )
        if generation_failed(response):
            break
        turn_usage = totals.record_turn(response)
        prompt_tokens_estimate = turn_usage.input_tokens

        assistant_message = response.choices[0].message if response.choices else None
        if assistant_message is None:
            break
        messages.append(response_message_to_dict(assistant_message))
        if not assistant_message.tool_calls:
            tool_call_iterations.append(
                summarize_tool_call_iteration(
                    agent="fast_searcher",
                    iteration=rounds_executed,
                    tool_calls=[],
                )
            )
            break
        if any(
            getattr(getattr(tool_call, "function", None), "name", "") == "submit_ranking"
            for tool_call in assistant_message.tool_calls
        ):
            totals.final_submit_input_tokens = turn_usage.input_tokens

        tool_call_iterations.append(
            summarize_tool_call_iteration(
                agent="fast_searcher",
                iteration=rounds_executed,
                tool_calls=assistant_message.tool_calls,
                over_budget_without_prune=over_budget_round_missing_prune(
                    over_budget,
                    assistant_message.tool_calls,
                    final_tool_name="submit_ranking",
                ),
            )
        )
        response_adapted = time.perf_counter()

        tool_round = await _handle_searcher_tool_calls(
            assistant_message.tool_calls,
            agent_iteration=rounds_executed,
            messages=messages,
            index=index,
            store_identifiers=store_identifiers,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            initial_metadata_facets=initial_metadata_facets,
            top_k=effective_top_k,
            strict_top_k=effective_strict_top_k,
            context_tokens_baseline=turn_usage.total_tokens,
        )
        tool_handler_finished = time.perf_counter()
        queries_made.extend(tool_round.queries)
        tool_trace.extend(tool_round.trace)
        if isinstance(tool_round.final, RankedChunkList):
            final_ranking = tool_round.final
        if tool_round.pruned_context:
            prompt_tokens_estimate = await _estimate_messages_tokens_off_loop(messages)
        if final_ranking is not None:
            emit_bridge_timing(
                "final_response_processing",
                forced=False,
                response_trace_and_adaptation_ms=round(
                    (response_adapted - response_returned) * 1000, 3
                ),
                ranking_handler_ms=round((tool_handler_finished - response_adapted) * 1000, 3),
                total_ms=round((tool_handler_finished - response_returned) * 1000, 3),
            )
            break

    forced_ranking = final_ranking is None
    if forced_ranking:
        messages.append(
            force_submit_message(
                top_k=effective_top_k,
                strict_top_k=effective_strict_top_k,
                round_index=rounds_executed,
                max_rounds=max_rounds,
            )
        )
        forced_submit_input_tokens = await _estimate_messages_tokens_off_loop(messages)
        forced = await force_ranking(
            messages,
            tools=_searcher_tools(
                top_k=effective_top_k,
                strict_top_k=effective_strict_top_k,
            ),
            completion_config=searcher_agent_config(),
            validate=lambda ranking: validate_ranked_chunk_ids(
                ranking,
                index,
                top_k=effective_top_k,
                strict_top_k=effective_strict_top_k,
            ),
            responses_trace=openai_responses,
            response_trace_metadata={
                "agent": "fast_searcher",
                "iteration": rounds_executed + 1,
            },
            generation_fn=generate,
            on_invalid_attempt=lambda attempt, validation_error: tool_trace.append(
                synthetic_tool_call_trace(
                    agent="fast_searcher",
                    iteration=rounds_executed + 1,
                    name="submit_ranking",
                    metadata={"forced": True, "attempt": attempt},
                    status="error",
                    error=validation_error,
                    error_kind=AGENT_ERROR_KIND,
                    attempt=attempt,
                )
            ),
        )
        totals.record_forced(forced.usage, prompt_tokens=forced_submit_input_tokens)
        final_ranking = forced.submission
        tool_trace.append(
            synthetic_tool_call_trace(
                agent="fast_searcher",
                iteration=rounds_executed + 1,
                name="submit_ranking",
                arguments=final_ranking,
                output={"ranking": ranking_trace_payload(final_ranking, index)},
                metadata={
                    "forced": True,
                    "input_tokens": forced.usage.input_tokens,
                    "output_tokens": forced.usage.output_tokens,
                },
                status="success" if final_ranking is not None else "error",
                error=None if final_ranking is not None else "forced ranking failed",
            )
        )

    result_build_started = time.perf_counter()
    chunks = finalize_chunks(
        index,
        final_ranking,
        top_k=effective_top_k,
        strict_top_k=effective_strict_top_k,
    )

    chunks_finished = time.perf_counter()
    result = FastAgenticSearchResult(
        messages=messages,
        ranking=final_ranking,
        chunks=chunks,
        top_k=effective_top_k,
        strict_top_k=effective_strict_top_k,
        media_content=harness_config.MEDIA_CONTENT,
        additional_instructions=additional_instructions,
        queries_made=queries_made,
        initial_search_results=initial_search_results,
        initial_metadata_facets=initial_metadata_facets,
        input_tokens=totals.usage.input_tokens,
        output_tokens=totals.usage.output_tokens,
        reasoning_tokens=totals.reasoning_tokens,
        max_input_tokens=totals.max_input_tokens,
        final_submit_input_tokens=totals.final_submit_input_tokens,
        rounds_executed=rounds_executed,
        forced_ranking=forced_ranking,
        ranking_unresolved=ranking_unresolved(index, final_ranking),
        tool_call_iterations=tool_call_iterations,
        tool_trace=tool_trace,
        openai_responses=openai_responses,
        id_mapping=index.refs.snapshot(),
        deleted_chunk_ids=[
            {"store_id": key[0], "file_id": key[1], "chunk_index": key[2]}
            for key in sorted(index.deleted_chunk_keys)
        ],
        deleted_chunk_refs=[
            {"chunk_id": index.refs.chunk_id_for_key(key)}
            for key in sorted(index.deleted_chunk_keys)
        ],
        prompt_snapshot=prompt_snapshot,
    )
    result_finished = time.perf_counter()
    emit_bridge_timing(
        "search_result_construction",
        forced=forced_ranking,
        chunk_count=len(chunks),
        finalize_chunks_ms=round((chunks_finished - result_build_started) * 1000, 3),
        snapshot_and_dataclass_ms=round((result_finished - chunks_finished) * 1000, 3),
        total_ms=round((result_finished - result_build_started) * 1000, 3),
    )
    return result


def _bootstrap_trace_event(
    name: str,
    outcome: ToolOutcome,
    *,
    started_at: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """One bootstrap fetch as a trace event, mirroring dispatched tool calls."""
    payload = outcome.payload if isinstance(outcome.payload, Mapping) else {}
    query = outcome.query if isinstance(outcome.query, Mapping) else {}
    error = payload.get("error") or query.get("error")
    return synthetic_tool_call_trace(
        agent="fast_searcher",
        iteration=0,
        name=name,
        arguments=outcome.query,
        output=outcome.payload,
        status="error" if error else "success",
        error=str(error) if error else None,
        error_kind=payload.get("error_kind") or query.get("error_kind"),
        call_id=f"fast-bootstrap-{name}",
        forced=False,
        started_at=started_at,
        duration_ms=duration_ms,
    )


async def _fetch_initial_context(
    user_text: str,
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> InitialSearchContext:
    """Fetch store metadata and query-specific seed results concurrently.

    ``execute_inspect_metadata`` independently overlaps its facet aggregation
    and chunk type-sampling calls, so this two-task join results in all three
    bootstrap provider requests being in flight together. Results are unpacked
    in the same stable metadata-then-search order used by the sequential path.
    """

    # Keyed by call name: the concurrent fetches complete in any order, but the
    # trace lists bootstrap events in the stable metadata-then-search order.
    bootstrap_trace: dict[str, dict[str, Any]] = {}

    async def timed_call(name: str, awaitable: Awaitable[ToolOutcome]) -> ToolOutcome:
        started_at = _utc_now_iso()
        started = time.perf_counter()
        try:
            outcome = await awaitable
        finally:
            emit_bridge_timing(
                "initial_bootstrap_call",
                operation=name,
                duration_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        bootstrap_trace[name] = _bootstrap_trace_event(
            name,
            outcome,
            started_at=started_at,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return outcome

    started = time.perf_counter()
    # Resolve the client and probe its stores resource before the concurrent
    # fetches start. This avoids racing any lazy client/resource setup and
    # guarantees all three requests use the same connection pool.
    try:
        resolved_client = resolve_async_retrieval_client(
            client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        _ = resolved_client.stores
    except Exception as exc:
        # Failure contract: metadata client setup fails first, then initial
        # search independently gets its own chance to resolve the cached
        # client and return its normal success/error payload.
        metadata_outcome = _initial_metadata_failure(
            exc,
            store_identifiers=store_identifiers,
        )
        # The injected client stays authoritative even when its resolution
        # failed once: falling back to the SDK here would silently send an
        # in-process deployment's rollout to the public API.
        search_outcome = await _fetch_initial_search_results(
            user_text,
            index=index,
            store_identifiers=store_identifiers,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        return InitialSearchContext(
            metadata_facets=metadata_outcome.payload,
            metadata_query=metadata_outcome.query,
            search_results=search_outcome.payload,
            search_query=search_outcome.query,
            trace_events=[
                _bootstrap_trace_event("inspect_metadata", metadata_outcome),
                _bootstrap_trace_event("search_corpus", search_outcome),
            ],
        )
    client_ready = time.perf_counter()
    metadata_outcome, search_outcome = await asyncio.gather(
        timed_call(
            "inspect_metadata",
            _fetch_initial_metadata_facets(
                store_identifiers=store_identifiers,
                client=resolved_client,
                api_key=api_key,
                api_key_env=api_key_env,
            ),
        ),
        timed_call(
            "search_corpus",
            _fetch_initial_search_results(
                user_text,
                index=index,
                store_identifiers=store_identifiers,
                client=resolved_client,
                api_key=api_key,
                api_key_env=api_key_env,
            ),
        ),
    )
    emit_bridge_timing(
        "initial_bootstrap_join",
        operation_count=3,
        client_setup_ms=round((client_ready - started) * 1000, 3),
        concurrent_calls_ms=round((time.perf_counter() - client_ready) * 1000, 3),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )
    return InitialSearchContext(
        metadata_facets=metadata_outcome.payload,
        metadata_query=metadata_outcome.query,
        search_results=search_outcome.payload,
        search_query=search_outcome.query,
        trace_events=[bootstrap_trace["inspect_metadata"], bootstrap_trace["search_corpus"]],
    )


async def _fetch_initial_metadata_facets(
    *,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    args = {"max_values_per_field": METADATA_REPRESENTATIVE_VALUES_PER_FIELD}
    try:
        outcome = await execute_inspect_metadata(
            args,
            store_identifiers=store_identifiers,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        return _initial_metadata_failure(exc, store_identifiers=store_identifiers)

    return ToolOutcome(
        {"type": "INITIAL_METADATA_FACETS", **outcome.payload},
        outcome.query,
    )


def _initial_metadata_failure(
    exc: Exception,
    *,
    store_identifiers: Sequence[str],
) -> ToolOutcome:
    store_ids = [str(store_id) for store_id in store_identifiers]
    payload = {
        "type": "INITIAL_METADATA_FACETS",
        "store_identifiers": store_ids,
        "rankable_fields": [],
        "field_types_sampled": False,
        "metadata_fields": {},
        "note": "Initial metadata inspection failed.",
        "error": str(exc),
    }
    metadata = {
        "tool": "inspect_metadata",
        "metadata_field_count": 0,
        "store_identifiers": store_ids,
        "error": str(exc),
        "error_kind": error_kind(exc),
    }
    return ToolOutcome(payload, metadata)


async def _fetch_initial_search_results(
    user_text: str,
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    args = {"query": user_text}
    try:
        outcome = await execute_search_corpus(
            args,
            index=index,
            store_identifiers=store_identifiers,
            top_k=harness_config.INITIAL_SEARCH_TOP_K,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        payload = {
            "type": "INITIAL_SEARCH_RESULTS",
            "query": user_text,
            "results": [],
            "error": str(exc),
        }
        metadata = {
            "tool": "search_corpus",
            "query": user_text,
            "k": 0,
            "new_chunks_added": 0,
            "error": str(exc),
            "error_kind": error_kind(exc),
        }
        return ToolOutcome(payload, metadata)

    payload = {
        "type": "INITIAL_SEARCH_RESULTS",
        "query": user_text,
        "results": outcome.payload.get("new_unseen_results") or [],
    }
    return ToolOutcome(payload, outcome.query)


def _searcher_tools(
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> list[dict[str, Any]]:
    # Deliberately not passing chunk_ids/document_ids here: nothing enforces
    # this enum (no strict/guided decoding on these tools), the IDs are
    # already visible to the model in prior tool results, and unknown-ID
    # tool calls already get a graceful tool_error from the handlers below.
    # Populating it would also make the tool schema grow every round as more
    # chunks are discovered, breaking the identical-tools-per-round rule above.
    final_tool = submit_ranking_tool(top_k=top_k, strict_top_k=strict_top_k)
    return [
        overview_search_tool(),
        search_corpus_tool(),
        filter_chunks_tool(),
        grep_tool(),
        read_document_tool(),
        get_chunks_tool(),
        prune_context_tool(),
        final_tool,
    ]


def estimate_messages_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, default=str)
    return harness_config.count_text_tokens(serialized)


@dataclass(slots=True)
class _PendingToolCall:
    """One remote tool call dispatched this round, awaiting its outcome."""

    tool_call: Any
    kind: str
    trace_event: dict[str, Any]
    execute: Awaitable[Any]
    metadata_validation: dict[str, Any] = field(default_factory=dict)


async def _handle_searcher_tool_calls(
    tool_calls: Sequence[Any],
    *,
    agent_iteration: int,
    messages: list[dict[str, Any]],
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    initial_metadata_facets: Mapping[str, Any] | None = None,
    top_k: int | None = None,
    strict_top_k: bool = False,
    context_tokens_baseline: int | None = None,
) -> SearcherToolRound:
    timing_started = time.perf_counter()
    tool_messages: dict[str, dict[str, Any]] = {}
    queries_made: list[dict[str, Any]] = []
    tool_trace: list[dict[str, Any]] = []
    final_result: Any | None = None
    pending: list[_PendingToolCall] = []
    prune_chunk_keys: set[tuple[str, str, int]] = set()
    prune_document_keys: set[tuple[str, str]] = set()

    tool_call_count = 0
    group_size = len(tool_calls)
    # The dispatch loop creates the tool coroutines eagerly; if it raises
    # before the gather, close them so nothing is silently dropped un-awaited.
    try:
        for call_index, tool_call in enumerate(tool_calls, 1):
            trace_event = start_tool_call_trace(
                agent="fast_searcher",
                iteration=agent_iteration,
                tool_call=tool_call,
                call_index=call_index,
                group_size=group_size,
            )
            tool_trace.append(trace_event)
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            if tool_call_count >= MAX_PARALLEL_TOOL_CALLS:
                error_message = (
                    f"Too many parallel tool calls requested. Maximum is {MAX_PARALLEL_TOOL_CALLS}."
                )
                tool_messages[tool_call.id] = tool_error(tool_call.id, error_message)
                finish_tool_call_trace(
                    trace_event,
                    status="error",
                    error=error_message,
                )
                continue
            tool_call_count += 1

            if tool_name == "submit_ranking":
                try:
                    parsed_ranking = parse_ranking(raw_arguments)
                    validate_ranked_chunk_ids(
                        parsed_ranking,
                        index,
                        top_k=top_k,
                        strict_top_k=strict_top_k,
                    )
                    final_result = parsed_ranking
                    finish_tool_call_trace(
                        trace_event,
                        output={"ranking": ranking_trace_payload(parsed_ranking, index)},
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    tool_messages[tool_call.id] = tool_error(
                        tool_call.id,
                        f"Failed to parse ranking: {exc}",
                    )
                    finish_tool_call_trace(
                        trace_event,
                        status="error",
                        error=f"Failed to parse ranking: {exc}",
                    )
                continue

            parsed = parse_tool_args(tool_call, _SEARCHER_ARG_SCHEMAS.get(tool_name))
            if isinstance(parsed, dict) and "error" in parsed:
                tool_messages[tool_call.id] = tool_error(tool_call.id, parsed["error"])
                finish_tool_call_trace(
                    trace_event,
                    status="error",
                    error=parsed["error"],
                )
                continue
            trace_event["parsed_arguments"] = jsonable(parsed)

            if tool_name == "search_corpus":
                validation = validate_metadata_filter_args(
                    parsed.model_dump(mode="json"),
                    registry=build_metadata_registry(
                        initial_metadata_facets=initial_metadata_facets,
                        index=index,
                    ),
                )
                if not validation.valid:
                    validation_metadata = validation.trace_metadata()
                    payload = {
                        "tool": "search_corpus",
                        "error": "Invalid metadata filters; use only verified metadata fields and values.",
                        "metadata_validation": validation_metadata,
                    }
                    tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
                    finish_tool_call_trace(
                        trace_event,
                        status="error",
                        output=payload,
                        metadata=validation_metadata,
                        error="Invalid metadata filters",
                    )
                    continue
                trace_event["parsed_arguments"] = jsonable(validation.args)
                pending.append(
                    _PendingToolCall(
                        tool_call=tool_call,
                        kind="search_corpus",
                        trace_event=trace_event,
                        execute=execute_search_corpus(
                            validation.args,
                            index=index,
                            store_identifiers=store_identifiers,
                            client=client,
                            api_key=api_key,
                            api_key_env=api_key_env,
                        ),
                        metadata_validation=validation.trace_metadata(),
                    )
                )
                continue

            if tool_name == "filter_chunks":
                validation = validate_metadata_filter_args(
                    parsed.model_dump(mode="json"),
                    registry=build_metadata_registry(
                        initial_metadata_facets=initial_metadata_facets,
                        index=index,
                    ),
                )
                if not validation.valid:
                    validation_metadata = validation.trace_metadata()
                    payload = {
                        "tool": "filter_chunks",
                        "error": "Invalid metadata filters; use only verified metadata fields and values.",
                        "metadata_validation": validation_metadata,
                    }
                    tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
                    finish_tool_call_trace(
                        trace_event,
                        status="error",
                        output=payload,
                        metadata=validation_metadata,
                        error="Invalid metadata filters",
                    )
                    continue
                trace_event["parsed_arguments"] = jsonable(validation.args)
                pending.append(
                    _PendingToolCall(
                        tool_call=tool_call,
                        kind="filter_chunks",
                        trace_event=trace_event,
                        execute=execute_filter_chunks(
                            validation.args,
                            index=index,
                            store_identifiers=store_identifiers,
                            client=client,
                            api_key=api_key,
                            api_key_env=api_key_env,
                        ),
                        metadata_validation=validation.trace_metadata(),
                    )
                )
                continue

            if tool_name == "grep":
                validation = validate_metadata_filter_args(
                    parsed.model_dump(mode="json"),
                    registry=build_metadata_registry(
                        initial_metadata_facets=initial_metadata_facets,
                        index=index,
                    ),
                )
                if not validation.valid:
                    validation_metadata = validation.trace_metadata()
                    payload = {
                        "tool": "grep",
                        "error": "Invalid metadata filters; use only verified metadata fields and values.",
                        "metadata_validation": validation_metadata,
                    }
                    tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
                    finish_tool_call_trace(
                        trace_event,
                        status="error",
                        output=payload,
                        metadata=validation_metadata,
                        error="Invalid metadata filters",
                    )
                    continue
                trace_event["parsed_arguments"] = jsonable(validation.args)
                pending.append(
                    _PendingToolCall(
                        tool_call=tool_call,
                        kind="grep",
                        trace_event=trace_event,
                        execute=execute_grep(
                            validation.args,
                            index=index,
                            store_identifiers=store_identifiers,
                            client=client,
                            api_key=api_key,
                            api_key_env=api_key_env,
                        ),
                        metadata_validation=validation.trace_metadata(),
                    )
                )
                continue

            if tool_name == "overview_search":
                pending.append(
                    _PendingToolCall(
                        tool_call=tool_call,
                        kind="overview_search",
                        trace_event=trace_event,
                        execute=execute_overview_search(
                            parsed.model_dump(mode="json"),
                            index=index,
                            store_identifiers=store_identifiers,
                            client=client,
                            api_key=api_key,
                            api_key_env=api_key_env,
                        ),
                    )
                )
                continue

            if tool_name == "read_document":
                pending.append(
                    _PendingToolCall(
                        tool_call=tool_call,
                        kind="read_document",
                        trace_event=trace_event,
                        execute=execute_read_document(
                            parsed.model_dump(mode="json"),
                            index=index,
                            client=client,
                            api_key=api_key,
                            api_key_env=api_key_env,
                        ),
                    )
                )
                continue

            if tool_name == "get_chunks":
                pending.append(
                    _PendingToolCall(
                        tool_call=tool_call,
                        kind="get_chunks",
                        trace_event=trace_event,
                        execute=execute_get_chunks(
                            parsed.model_dump(mode="json"),
                            index=index,
                            client=client,
                            api_key=api_key,
                            api_key_env=api_key_env,
                        ),
                    )
                )
                continue

            if tool_name == "prune_context":
                try:
                    turn_chunk_keys = {
                        index.refs.chunk_key_for_id(chunk_id) for chunk_id in parsed.chunk_ids
                    }
                    turn_document_keys = {
                        index.refs.document_key_for_id(document_id)
                        for document_id in parsed.document_ids
                    }
                except ValueError as exc:
                    tool_messages[tool_call.id] = tool_error(tool_call.id, str(exc))
                    finish_tool_call_trace(
                        trace_event,
                        status="error",
                        error=str(exc),
                    )
                    continue
                prune_chunk_keys.update(turn_chunk_keys)
                prune_document_keys.update(turn_document_keys)
                index.mark_pruned(
                    chunk_keys=turn_chunk_keys,
                    document_keys=turn_document_keys,
                )
                payload = {
                    "tool": "prune_context",
                    "chunk_ids": parsed.chunk_ids,
                    "document_ids": parsed.document_ids,
                }
                tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
                finish_tool_call_trace(trace_event, output=payload)
                continue

            tool_messages[tool_call.id] = tool_error(tool_call.id, f"Unknown tool: {tool_name}")
            finish_tool_call_trace(
                trace_event,
                status="error",
                error=f"Unknown tool: {tool_name}",
            )

    except BaseException:
        for entry in pending:
            entry.execute.close()
        raise
    tools_dispatched = time.perf_counter()
    outcomes = (
        await asyncio.gather(*(entry.execute for entry in pending), return_exceptions=True)
        if pending
        else []
    )
    result_processing_started = time.perf_counter()
    for entry, result in zip(pending, outcomes, strict=True):
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            # Cancellation (and friends) must abort the rollout, never
            # degrade into model-visible tool feedback.
            raise result
        tool_call = entry.tool_call
        trace_event = entry.trace_event
        if isinstance(result, Exception):
            tool_messages[tool_call.id] = tool_error(tool_call.id, str(result))
            finish_tool_call_trace(
                trace_event,
                status="error",
                error=str(result),
                error_kind=error_kind(result),
            )
            continue

        if isinstance(result, ToolOutcome):
            payload, metadata = result.payload, result.query
            metadata["source"] = f"searcher_{entry.kind}"
            metadata.update(entry.metadata_validation)
            metadata["zero_result_filtered_search_count"] = zero_result_filtered_search_count(
                metadata
            )
            queries_made.append(metadata)
            tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
            finish_tool_call_trace(
                trace_event,
                output=payload,
                metadata=metadata,
            )
            continue

        tool_messages[tool_call.id] = tool_message(tool_call.id, result)
        payload_error = agent_caused_payload_error(result)
        finish_tool_call_trace(
            trace_event,
            status="error" if payload_error else "success",
            output=result,
            error=payload_error,
            error_kind=AGENT_ERROR_KIND if payload_error else None,
        )
    result_processing_seconds = time.perf_counter() - result_processing_started

    tools_collected = time.perf_counter()

    baseline_count_started = time.perf_counter()
    context_baseline = (
        max(context_tokens_baseline, await _estimate_messages_tokens_off_loop(messages))
        if context_tokens_baseline is not None
        else 0
    )
    baseline_count_finished = time.perf_counter()

    truncation_started = time.perf_counter()
    if context_tokens_baseline is not None:
        # Truncation re-serializes and re-counts every clipped payload: CPU
        # work, kept off the event loop. Sequential await, so the mutations it
        # makes to tool_messages/trace/index are race-free.
        await asyncio.to_thread(
            _truncate_round_tool_messages,
            tool_calls,
            tool_messages,
            tool_trace=tool_trace,
            index=index,
            context_tokens_baseline=context_baseline,
        )
    truncation_finished = time.perf_counter()

    append_started = time.perf_counter()
    media_messages: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        message = tool_messages.get(tool_call.id)
        if message is not None:
            messages.append(message)
            media_messages.extend(media_messages_for_tool_message(message))
    messages.extend(media_messages)
    append_finished = time.perf_counter()

    redaction_started = time.perf_counter()
    if prune_chunk_keys or prune_document_keys:
        redact_messages(
            messages,
            refs=index.refs,
            chunk_keys=prune_chunk_keys,
            document_keys=prune_document_keys,
        )
    redaction_finished = time.perf_counter()

    emit_bridge_timing(
        "tool_results_prepared",
        agent="fast_searcher",
        iteration=agent_iteration,
        tool_call_count=len(tool_calls),
        remote_tool_count=len(pending),
        pruned_context=bool(prune_chunk_keys or prune_document_keys),
        message_count=len(messages),
        setup_and_dispatch_ms=round((tools_dispatched - timing_started) * 1000, 3),
        await_and_collect_ms=round((tools_collected - tools_dispatched) * 1000, 3),
        result_processing_ms=round(result_processing_seconds * 1000, 3),
        baseline_token_count_ms=round((baseline_count_finished - baseline_count_started) * 1000, 3),
        truncate_and_token_count_ms=round((truncation_finished - truncation_started) * 1000, 3),
        append_tool_messages_ms=round((append_finished - append_started) * 1000, 3),
        redact_pruned_context_ms=round((redaction_finished - redaction_started) * 1000, 3),
        total_ms=round((redaction_finished - timing_started) * 1000, 3),
    )

    return SearcherToolRound(
        final=final_result,
        queries=queries_made,
        trace=tool_trace,
        pruned_context=bool(prune_chunk_keys or prune_document_keys),
    )


def _truncate_round_tool_messages(
    tool_calls: Sequence[Any],
    tool_messages: dict[str, dict[str, Any]],
    *,
    tool_trace: list[dict[str, Any]],
    index: ChunkIndex,
    context_tokens_baseline: int,
) -> None:
    """Clip the round's tool payloads so the next prompt stays under the hard limit.

    Runs before payloads enter the transcript, so the model only ever sees the
    truncated version. Trace events are rewritten to match what the model saw;
    truncation is flagged in trace metadata, not as a tool error.
    """
    parsed: list[tuple[str, dict[str, Any]]] = []
    for tool_call in tool_calls:
        message = tool_messages.get(tool_call.id)
        content = message.get("content") if message is not None else None
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            parsed.append((tool_call.id, payload))
    if not parsed:
        return

    context_headroom = SEARCHER_PROMPT_TOKEN_LIMIT - context_tokens_baseline
    stats = truncate_round_payloads(
        [payload for _, payload in parsed],
        index=index,
        remaining_tokens=min(TURN_TOOL_PAYLOAD_TOKEN_BUDGET, context_headroom),
        turn_capped=context_headroom > TURN_TOOL_PAYLOAD_TOKEN_BUDGET,
    )
    events_by_call_id = {event.get("call_id"): event for event in tool_trace}
    for (call_id, payload), payload_stats in zip(parsed, stats, strict=True):
        if payload_stats is None:
            continue
        tool_messages[call_id]["content"] = json.dumps(payload, ensure_ascii=False, default=str)
        event = events_by_call_id.get(call_id)
        if event is None:
            continue
        event["output"] = jsonable(payload)
        output_summary = summarize_tool_output(event["output"])
        if output_summary:
            event["output_summary"] = output_summary
        metadata = event.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata.update({"payload_truncated": True, **payload_stats})
        event["metadata"] = metadata


_SEARCHER_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "filter_chunks": FilterChunksArgs,
    "grep": GrepArgs,
    "search_corpus": SearchCorpusArgs,
    "overview_search": OverviewSearchArgs,
    "read_document": ReadDocumentArgs,
    "get_chunks": GetChunksArgs,
    "prune_context": PruneContextArgs,
}


def _fast_searcher_prompt_snapshot(
    *,
    messages: Sequence[Mapping[str, Any]],
    additional_instructions: str | None,
) -> dict[str, Any]:
    return {
        "kind": "fast_searcher_initial_messages",
        "messages": jsonable(messages),
        "additional_instructions": additional_instructions,
    }
