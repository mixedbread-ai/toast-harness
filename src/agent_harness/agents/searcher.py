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

``answer_mode`` (``schemas.AnswerMode``) picks how the episode ends: with a
``submit_ranking`` call ("none"), with a ``submit_ranking`` call that carries
a required answer ("submit_ranking"), or with a plain-text turn and no final
tool at all ("plain_text").
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    ForcedSubmission,
    TokenUsage,
    completion_reasoning_tokens,
    extend_responses_api_trace,
    force_answer,
    force_ranking,
    generation_failed,
    parse_ranking,
    require_generation_fn,
    response_message_to_dict,
    responses_api_trace_payload,
    wire_forces_tool_call,
)
from agent_harness.metadata_guard import (
    build_metadata_registry,
    validate_metadata_filter_args,
    zero_result_filtered_search_count,
)
from agent_harness.prompts import (
    force_answer_message,
    force_submit_message,
    over_budget_message,
    round_notice_message,
)
from agent_harness.retrieval import AsyncRetrievalClient
from agent_harness.schemas import (
    AnsweredRankedChunkList,
    AnswerMode,
    FilterChunksArgs,
    GetChunksArgs,
    GrepArgs,
    OverviewSearchArgs,
    PruneContextArgs,
    RankedChunkList,
    ReadDocumentArgs,
    SearchCorpusArgs,
    validate_answer_mode,
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

AGENT_NAME = "fast_searcher"
FINAL_TOOL_NAME = "submit_ranking"
# The trace name of a forced plain-text answer, which no tool call carries.
FINAL_ANSWER_TRACE_NAME = "final_answer"


@dataclass(slots=True)
class FastAgenticSearchResult:
    """Structured fast-searcher result before rollout-record wrapping.

    ``answer`` carries the final answer under a non-default ``answer_mode``:
    the submitted ``answer`` argument ("submit_ranking") or the final
    plain-text turn ("plain_text"); ``None`` under ``"none"``. In plain-text
    mode ``ranking`` is always ``None`` and ``chunks`` always ``[]`` -- there
    is no ranking to finalize -- and ``forced_ranking`` marks a forced
    *answer*, keeping the field's one meaning: the final submission did not
    come voluntarily.
    """

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
    answer: str | None = None
    answer_mode: AnswerMode = "none"

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
            "answer": self.answer,
            "ranking_strategy": self.ranking.ranking_strategy if self.ranking else None,
            "top_k": self.top_k,
            "strict_top_k": self.strict_top_k,
        }

    def to_record(self) -> dict[str, Any]:
        """Return the ``fast_agentic_search`` record payload."""
        result = {
            "ranking": self.ranking,
            "ranking_strategy": self.ranking.ranking_strategy if self.ranking else None,
            "answer": self.answer,
            "answer_mode": self.answer_mode,
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

    def record_forced(self, forced: ForcedSubmission[Any], *, prompt_tokens: int) -> None:
        """Add every forced attempt, preferring provider counts over the estimate."""
        self.usage = self.usage + forced.usage
        self.reasoning_tokens += forced.reasoning_tokens
        self.max_input_tokens = max(self.max_input_tokens, forced.max_input_tokens or prompt_tokens)
        self.final_submit_input_tokens = forced.final_input_tokens or prompt_tokens


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


@dataclass(frozen=True, slots=True)
class _RoundConfig:
    """What every tool round of one rollout reads and never changes.

    The corpus bindings, the result index the rollout accumulates into, the
    facets the metadata guard vets filters against, the ranking shape the
    final call must satisfy, and the answer protocol the episode ends on.
    """

    index: ChunkIndex
    store_identifiers: Sequence[str]
    client: AsyncRetrievalClient | None
    api_key: str | None
    api_key_env: str | None
    initial_metadata_facets: Mapping[str, Any] | None
    top_k: int | None
    strict_top_k: bool
    answer_mode: AnswerMode = "none"

    @property
    def answers_in_text(self) -> bool:
        return self.answer_mode == "plain_text"

    @property
    def requires_answer(self) -> bool:
        return self.answer_mode == "submit_ranking"

    @property
    def final_tool_name(self) -> str | None:
        """The tool that ends the episode, or ``None`` when a prose turn does."""
        return None if self.answers_in_text else FINAL_TOOL_NAME

    @property
    def client_bindings(self) -> dict[str, Any]:
        return {"client": self.client, "api_key": self.api_key, "api_key_env": self.api_key_env}

    @property
    def corpus_bindings(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "store_identifiers": self.store_identifiers,
            **self.client_bindings,
        }

    @property
    def filtered_search_executors(self) -> dict[str, _Executor]:
        """The tools whose metadata filters are vetted before they run.

        Bound on access rather than at import so a test that patches an
        executor on this module is honored.
        """
        corpus = self.corpus_bindings
        return {
            "search_corpus": lambda args: execute_search_corpus(args, **corpus),
            "filter_chunks": lambda args: execute_filter_chunks(args, **corpus),
            "grep": lambda args: execute_grep(args, **corpus),
        }

    @property
    def plain_retrieval_executors(self) -> dict[str, _Executor]:
        """The tools that run as parsed: one corpus preview and two index lookups."""
        corpus = self.corpus_bindings
        lookup = {"index": self.index, **self.client_bindings}
        return {
            "overview_search": lambda args: execute_overview_search(args, **corpus),
            "read_document": lambda args: execute_read_document(args, **lookup),
            "get_chunks": lambda args: execute_get_chunks(args, **lookup),
        }


_Executor = Callable[[Mapping[str, Any]], Awaitable[Any]]


@dataclass(slots=True)
class _SearcherRun:
    """What one searcher run accumulates: rounds, cost, trace, and the final submission.

    ``final`` is the submitted ``RankedChunkList``, or the plain-text answer
    under answer_mode="plain_text", or ``None`` until the forced tail compels
    one. The round loop fills it; the forced tail extends it in place.
    """

    max_rounds: int
    final: RankedChunkList | str | None = None
    rounds_executed: int = 0
    totals: RolloutTotals = field(default_factory=RolloutTotals)
    queries: list[dict[str, Any]] = field(default_factory=list)
    tool_call_iterations: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    openai_responses: list[dict[str, Any]] = field(default_factory=list)

    def summarize_iteration(
        self, tool_calls: Sequence[Any], *, over_budget_without_prune: bool = False
    ) -> None:
        self.tool_call_iterations.append(
            summarize_tool_call_iteration(
                agent=AGENT_NAME,
                iteration=self.rounds_executed,
                tool_calls=tool_calls,
                over_budget_without_prune=over_budget_without_prune,
            )
        )

    def merge(self, tool_round: SearcherToolRound) -> None:
        self.queries.extend(tool_round.queries)
        self.tool_trace.extend(tool_round.trace)
        if isinstance(tool_round.final, RankedChunkList):
            self.final = tool_round.final


@dataclass(frozen=True, slots=True)
class _AssistantTurn:
    """One model turn the loop accepted: its message, its cost, and when it arrived."""

    message: Any
    usage: TokenUsage
    returned_at: float

    @property
    def tool_calls(self) -> list[Any]:
        return list(self.message.tool_calls or [])

    @property
    def text(self) -> str:
        content = self.message.content
        return content.strip() if isinstance(content, str) else ""


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
    answer_mode: AnswerMode = "none",
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
        answer_mode=answer_mode,
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
    answer_mode: AnswerMode = "none",
) -> FastAgenticSearchResult:
    """Run the fast searcher and return structured loop state.

    ``as_of`` pins the runtime-context date instead of the UTC wall clock
    (see ``prompts._runtime_context``). ``tuning`` and ``media_content`` are
    scoped to this rollout; both scopes are no-ops when left unset.

    ``answer_mode`` picks the answer protocol (``schemas.AnswerMode``):
    ``"none"`` is the default submit_ranking episode, byte-identical prompts
    and tools; ``"submit_ranking"`` adds a required ``answer`` argument to the
    final call; ``"plain_text"`` removes submit_ranking entirely -- the episode
    ends on a plain-text turn with no tool calls, whose text is the answer, so
    ``top_k`` and ``strict_top_k`` have no ranking to shape and are rejected.
    """
    validate_answer_mode(answer_mode)
    if answer_mode == "plain_text" and (top_k is not None or strict_top_k):
        raise ValueError("answer_mode='plain_text' has no ranking; top_k/strict_top_k do not apply")
    with (
        harness_config.tuning_setting(tuning),
        harness_config.media_content_setting(media_content),
    ):
        return await _run_scoped_fast_agentic_search(
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
            answer_mode=answer_mode,
        )


async def _run_scoped_fast_agentic_search(
    user_text: str,
    *,
    store_identifiers: Sequence[str],
    top_k: int | None,
    strict_top_k: bool,
    client: AsyncRetrievalClient | None,
    api_key: str | None,
    api_key_env: str | None,
    additional_instructions: str | None,
    include_prompt_snapshot: bool,
    generation_fn: AsyncGenerationFn | None,
    as_of: date | None,
    answer_mode: AnswerMode,
) -> FastAgenticSearchResult:
    """The rollout proper: bootstrap, the round loop, the forced tail, the result."""
    generate = require_generation_fn(generation_fn)
    # Every budget below measures through config.count_text_tokens: install the
    # policy tokenizer behind it before the first prompt is built. The first
    # install loads the tokenizer (possibly a hub download), so it runs off the
    # event loop; later calls are an O(1) resolved-model check.
    await asyncio.to_thread(ensure_rollout_token_counter, searcher_agent_config().get("model"))
    effective_top_k, effective_strict_top_k = _resolve_ranking_shape(top_k, strict_top_k)

    index = ChunkIndex()
    bootstrap = await _fetch_initial_context(
        user_text,
        index=index,
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    messages = fast_searcher_messages(
        user_text=user_text,
        initial_search_results=bootstrap.search_results,
        initial_metadata_facets=bootstrap.metadata_facets,
        top_k=effective_top_k,
        strict_top_k=effective_strict_top_k,
        additional_instructions=additional_instructions,
        as_of=as_of,
        answer_mode=answer_mode,
    )
    messages.extend(media_messages_for_payload(bootstrap.search_results))
    prompt_snapshot = (
        _fast_searcher_prompt_snapshot(
            messages=messages,
            additional_instructions=additional_instructions,
        )
        if include_prompt_snapshot
        else None
    )

    config = _RoundConfig(
        index=index,
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
        initial_metadata_facets=bootstrap.metadata_facets,
        top_k=effective_top_k,
        strict_top_k=effective_strict_top_k,
        answer_mode=answer_mode,
    )
    run = await _run_searcher_rounds(generate=generate, messages=messages, config=config)
    forced_ranking = run.final is None
    if forced_ranking:
        force = _force_final_answer if config.answers_in_text else _force_final_ranking
        run.final = await force(messages, run, generate=generate, config=config)

    return _build_search_result(
        messages=messages,
        bootstrap=bootstrap,
        run=run,
        config=config,
        forced_ranking=forced_ranking,
        additional_instructions=additional_instructions,
        prompt_snapshot=prompt_snapshot,
    )


def _resolve_ranking_shape(top_k: int | None, strict_top_k: bool) -> tuple[int | None, bool]:
    """Normalize the requested ranking shape.

    A strict top-k with no k falls back to the default k; strictness without a
    k to enforce is meaningless and is dropped.
    """
    effective_top_k = normalize_top_k(top_k)
    if strict_top_k and effective_top_k is None:
        effective_top_k = AGENTIC_SEARCH_DEFAULT_K
    return effective_top_k, bool(strict_top_k and effective_top_k is not None)


def _build_search_result(
    *,
    messages: list[dict[str, Any]],
    bootstrap: InitialSearchContext,
    run: _SearcherRun,
    config: _RoundConfig,
    forced_ranking: bool,
    additional_instructions: str | None,
    prompt_snapshot: dict[str, Any] | None,
) -> FastAgenticSearchResult:
    """Finalize the ranking and assemble the structured result."""
    clock = _PhaseClock()
    final_ranking, final_answer = _split_final_submission(run.final)
    chunks = finalize_chunks(
        config.index, final_ranking, top_k=config.top_k, strict_top_k=config.strict_top_k
    )
    clock.mark("chunks_finalized")
    deleted_chunk_keys = sorted(config.index.deleted_chunk_keys)
    result = FastAgenticSearchResult(
        messages=messages,
        ranking=final_ranking,
        chunks=chunks,
        top_k=config.top_k,
        strict_top_k=config.strict_top_k,
        media_content=harness_config.MEDIA_CONTENT,
        additional_instructions=additional_instructions,
        queries_made=[
            {**bootstrap.metadata_query, "source": "initial_metadata_facets"},
            {**bootstrap.search_query, "source": "initial_original_query"},
            *run.queries,
        ],
        initial_search_results=bootstrap.search_results,
        initial_metadata_facets=bootstrap.metadata_facets,
        input_tokens=run.totals.usage.input_tokens,
        output_tokens=run.totals.usage.output_tokens,
        reasoning_tokens=run.totals.reasoning_tokens,
        max_input_tokens=run.totals.max_input_tokens,
        final_submit_input_tokens=run.totals.final_submit_input_tokens,
        rounds_executed=run.rounds_executed,
        forced_ranking=forced_ranking,
        ranking_unresolved=ranking_unresolved(config.index, final_ranking),
        tool_call_iterations=run.tool_call_iterations,
        tool_trace=[*bootstrap.trace_events, *run.tool_trace],
        openai_responses=run.openai_responses,
        id_mapping=config.index.refs.snapshot(),
        deleted_chunk_ids=[
            {"store_id": key[0], "file_id": key[1], "chunk_index": key[2]}
            for key in deleted_chunk_keys
        ],
        deleted_chunk_refs=[
            {"chunk_id": config.index.refs.chunk_id_for_key(key)} for key in deleted_chunk_keys
        ],
        prompt_snapshot=prompt_snapshot,
        answer=final_answer,
        answer_mode=config.answer_mode,
    )
    clock.mark("assembled")
    emit_bridge_timing(
        "search_result_construction",
        forced=forced_ranking,
        chunk_count=len(chunks),
        finalize_chunks_ms=clock.ms("start", "chunks_finalized"),
        snapshot_and_dataclass_ms=clock.ms("chunks_finalized", "assembled"),
        total_ms=clock.ms("start", "assembled"),
    )
    return result


def _split_final_submission(
    final: RankedChunkList | str | None,
) -> tuple[RankedChunkList | None, str | None]:
    """The ranking and the answer one final submission carries, either possibly absent."""
    if isinstance(final, str):
        return None, final
    answer = final.answer if isinstance(final, AnsweredRankedChunkList) else None
    return final, answer


async def _run_searcher_rounds(
    *,
    generate: AsyncGenerationFn,
    messages: list[dict[str, Any]],
    config: _RoundConfig,
) -> _SearcherRun:
    """Drive generate/tool-call rounds until a final submission or the round cap.

    A turn with no tool calls ends the loop. Under answer_mode="plain_text" it
    ends it *successfully* -- that prose is the episode's answer, and no final
    tool exists for it to have called instead. Everywhere else, and when the
    turn came back empty, the caller's forced tail compels the submission.
    """
    run = _SearcherRun(max_rounds=searcher_max_rounds())
    prompt_tokens_estimate = 0
    while run.rounds_executed < run.max_rounds:
        run.rounds_executed += 1
        estimated_tokens = max(
            prompt_tokens_estimate, await _estimate_messages_tokens_off_loop(messages)
        )
        over_budget = _append_round_preamble(
            messages,
            round_index=run.rounds_executed,
            max_rounds=run.max_rounds,
            estimated_tokens=estimated_tokens,
            final_tool_name=config.final_tool_name,
        )
        turn = await _generate_turn(generate, messages, config=config, run=run)
        if turn is None:
            break
        prompt_tokens_estimate = turn.usage.input_tokens

        tool_calls = turn.tool_calls
        if not tool_calls:
            run.summarize_iteration([])
            if config.answers_in_text and turn.text:
                run.final = turn.text
                run.totals.final_submit_input_tokens = turn.usage.input_tokens
            break
        if config.final_tool_name is not None and _names_final_tool(tool_calls):
            run.totals.final_submit_input_tokens = turn.usage.input_tokens
        run.summarize_iteration(
            tool_calls,
            over_budget_without_prune=over_budget_round_missing_prune(
                over_budget, tool_calls, final_tool_name=config.final_tool_name
            ),
        )
        response_adapted = time.perf_counter()

        tool_round = await _handle_searcher_tool_calls(
            tool_calls,
            config,
            iteration=run.rounds_executed,
            messages=messages,
            context_tokens_baseline=turn.usage.total_tokens,
        )
        tool_handler_finished = time.perf_counter()
        run.merge(tool_round)
        if tool_round.pruned_context:
            prompt_tokens_estimate = await _estimate_messages_tokens_off_loop(messages)
        if run.final is not None:
            emit_bridge_timing(
                "final_response_processing",
                forced=False,
                response_trace_and_adaptation_ms=_elapsed_ms(turn.returned_at, response_adapted),
                ranking_handler_ms=_elapsed_ms(response_adapted, tool_handler_finished),
                total_ms=_elapsed_ms(turn.returned_at, tool_handler_finished),
            )
            break
    return run


def _append_round_preamble(
    messages: list[dict[str, Any]],
    *,
    round_index: int,
    max_rounds: int,
    estimated_tokens: int,
    final_tool_name: str | None,
) -> bool:
    """Label the round and, past the prune trigger, ask for a prune.

    Returns whether the round is over budget, which the iteration summary
    records against the calls the model then makes.
    """
    if round_index > 1:
        messages.append(
            round_notice_message(round_index, max_rounds, final_tool_name=final_tool_name)
        )
    over_budget = estimated_tokens >= SEARCHER_PRUNE_REMINDER_TOKENS
    if over_budget:
        messages.append(over_budget_message(estimated_tokens, final_tool_name=final_tool_name))
    return over_budget


async def _generate_turn(
    generate: AsyncGenerationFn,
    messages: list[dict[str, Any]],
    *,
    config: _RoundConfig,
    run: _SearcherRun,
) -> _AssistantTurn | None:
    """One model turn: generate, trace, account for it, and append it to the transcript.

    ``None`` when the seam reported a failed generation or an empty response;
    the loop ends there and the forced tail takes over. The tools schema stays
    identical across every round -- over-budget rounds included; a missing
    prune is recorded after generation instead of swapping in a reduced tool
    list, so every round of one rollout presents the same tool surface.
    """
    response = await generate(
        messages,
        tools=_round_tools(config),
        completion_config=_rollout_completion_config(config.answer_mode),
    )
    returned_at = time.perf_counter()
    extend_responses_api_trace(
        run.openai_responses,
        response,
        agent=AGENT_NAME,
        iteration=run.rounds_executed,
        phase="generation",
    )
    if generation_failed(response):
        return None
    usage = run.totals.record_turn(response)
    message = response.choices[0].message if response.choices else None
    if message is None:
        return None
    messages.append(response_message_to_dict(message))
    return _AssistantTurn(message=message, usage=usage, returned_at=returned_at)


def _names_final_tool(tool_calls: Sequence[Any]) -> bool:
    return any(
        getattr(getattr(tool_call, "function", None), "name", "") == FINAL_TOOL_NAME
        for tool_call in tool_calls
    )


def _rollout_completion_config(answer_mode: AnswerMode) -> dict[str, Any]:
    """The searcher's completion config for this rollout's answer protocol.

    Plain-text answer mode ends the episode on a turn with no tool calls, so
    both the harness-side ``require_tool_calls`` policy and a wire-level forced
    ``tool_choice`` are neutralized. Every other mode gets the config
    untouched, which keeps answer_mode="none" identical on the wire. Always a
    fresh copy: a generation seam that rewrites its config in place must not
    reach the shared default.
    """
    config = dict(searcher_agent_config())
    if answer_mode != "plain_text":
        return config
    config["require_tool_calls"] = False
    if wire_forces_tool_call(config.get("tool_choice")):
        config["tool_choice"] = "auto"
    return config


def _round_tools(config: _RoundConfig) -> list[dict[str, Any]]:
    return _searcher_tools(
        top_k=config.top_k, strict_top_k=config.strict_top_k, answer_mode=config.answer_mode
    )


@dataclass(slots=True)
class _ForcedTail:
    """Bookkeeping around one forced final turn: its iteration, prompt cost, and trace.

    The forced turn counts as one past the last executed round; its cost and
    trace land on ``run`` like any other turn's.
    """

    run: _SearcherRun
    trace_name: str
    prompt_tokens: int

    @classmethod
    async def start(
        cls,
        messages: Sequence[Mapping[str, Any]],
        run: _SearcherRun,
        *,
        trace_name: str,
    ) -> _ForcedTail:
        """Measure the forced prompt after its instruction has been appended."""
        return cls(run, trace_name, await _estimate_messages_tokens_off_loop(messages))

    @property
    def iteration(self) -> int:
        return self.run.rounds_executed + 1

    @property
    def trace_metadata(self) -> dict[str, Any]:
        return {"agent": AGENT_NAME, "iteration": self.iteration}

    def record_invalid_attempt(self, attempt: int, validation_error: str) -> None:
        self.run.tool_trace.append(
            synthetic_tool_call_trace(
                agent=AGENT_NAME,
                iteration=self.iteration,
                name=self.trace_name,
                metadata={"forced": True, "attempt": attempt},
                status="error",
                error=validation_error,
                error_kind=AGENT_ERROR_KIND,
                attempt=attempt,
            )
        )

    def finish(
        self,
        forced: ForcedSubmission[Any],
        *,
        arguments: Any,
        output: Any,
        failure: str,
    ) -> None:
        self.run.totals.record_forced(forced, prompt_tokens=self.prompt_tokens)
        succeeded = forced.submission is not None
        self.run.tool_trace.append(
            synthetic_tool_call_trace(
                agent=AGENT_NAME,
                iteration=self.iteration,
                name=self.trace_name,
                arguments=arguments,
                output=output,
                metadata={
                    "forced": True,
                    "input_tokens": forced.usage.input_tokens,
                    "output_tokens": forced.usage.output_tokens,
                },
                status="success" if succeeded else "error",
                error=None if succeeded else failure,
            )
        )


async def _force_final_ranking(
    messages: list[dict[str, Any]],
    run: _SearcherRun,
    *,
    generate: AsyncGenerationFn,
    config: _RoundConfig,
) -> RankedChunkList | None:
    """Compel the ranking the round loop never received."""
    messages.append(
        force_submit_message(
            top_k=config.top_k,
            strict_top_k=config.strict_top_k,
            require_answer=config.requires_answer,
            round_index=run.rounds_executed,
            max_rounds=run.max_rounds,
        )
    )
    tail = await _ForcedTail.start(messages, run, trace_name=FINAL_TOOL_NAME)
    forced = await force_ranking(
        messages,
        tools=_round_tools(config),
        completion_config=_rollout_completion_config(config.answer_mode),
        validate=lambda ranking: validate_ranked_chunk_ids(
            ranking, config.index, top_k=config.top_k, strict_top_k=config.strict_top_k
        ),
        require_answer=config.requires_answer,
        responses_trace=run.openai_responses,
        response_trace_metadata=tail.trace_metadata,
        generation_fn=generate,
        on_invalid_attempt=tail.record_invalid_attempt,
    )
    tail.finish(
        forced,
        arguments=forced.submission,
        output={"ranking": ranking_trace_payload(forced.submission, config.index)},
        failure="forced ranking failed",
    )
    return forced.submission


async def _force_final_answer(
    messages: list[dict[str, Any]],
    run: _SearcherRun,
    *,
    generate: AsyncGenerationFn,
    config: _RoundConfig,
) -> str | None:
    """Compel the plain-text answer the round loop never received."""
    messages.append(
        force_answer_message(round_index=run.rounds_executed, max_rounds=run.max_rounds)
    )
    tail = await _ForcedTail.start(messages, run, trace_name=FINAL_ANSWER_TRACE_NAME)
    forced = await force_answer(
        messages,
        tools=_round_tools(config),
        completion_config=_rollout_completion_config(config.answer_mode),
        responses_trace=run.openai_responses,
        response_trace_metadata=tail.trace_metadata,
        generation_fn=generate,
        on_invalid_attempt=tail.record_invalid_attempt,
    )
    tail.finish(
        forced,
        arguments={"answer": forced.submission},
        output={"answer": forced.submission},
        failure="forced answer failed",
    )
    return forced.submission


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
        agent=AGENT_NAME,
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
    answer_mode: AnswerMode = "none",
) -> list[dict[str, Any]]:
    """The tool surface one rollout offers on every round.

    Deliberately not passing chunk_ids/document_ids here: nothing enforces
    this enum (no strict/guided decoding on these tools), the IDs are already
    visible to the model in prior tool results, and unknown-ID tool calls
    already get a graceful tool_error from the handlers below. Populating it
    would also make the tool schema grow every round as more chunks are
    discovered, breaking the identical-tools-per-round rule above.

    Under answer_mode="plain_text" the episode ends on a plain-text turn, so
    there is no final tool to advertise.
    """
    tools = [
        overview_search_tool(),
        search_corpus_tool(),
        filter_chunks_tool(),
        grep_tool(),
        read_document_tool(),
        get_chunks_tool(),
        prune_context_tool(),
    ]
    if answer_mode != "plain_text":
        tools.append(
            submit_ranking_tool(
                top_k=top_k,
                strict_top_k=strict_top_k,
                require_answer=answer_mode == "submit_ranking",
            )
        )
    return tools


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


@dataclass(slots=True)
class _RoundState:
    """What one round's phases accumulate."""

    iteration: int
    tool_messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    queries_made: list[dict[str, Any]] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    pending: list[_PendingToolCall] = field(default_factory=list)
    final_result: Any | None = None
    prune_chunk_keys: set[tuple[str, str, int]] = field(default_factory=set)
    prune_document_keys: set[tuple[str, str]] = field(default_factory=set)

    @property
    def pruned_context(self) -> bool:
        return bool(self.prune_chunk_keys or self.prune_document_keys)


def _fail_tool_call(
    state: _RoundState,
    tool_call: Any,
    trace_event: dict[str, Any],
    error: str,
    *,
    error_kind: str | None = None,
) -> None:
    """Answer one failed call: an error message for the model, an error event
    for the trace."""
    state.tool_messages[tool_call.id] = tool_error(tool_call.id, error)
    finish_tool_call_trace(trace_event, status="error", error=error, error_kind=error_kind)


async def _handle_searcher_tool_calls(
    tool_calls: Sequence[Any],
    config: _RoundConfig,
    *,
    iteration: int,
    messages: list[dict[str, Any]],
    context_tokens_baseline: int | None = None,
) -> SearcherToolRound:
    """Run one round of model tool calls end to end.

    Dispatch decodes every call and starts the remote ones, the gather awaits
    them, and the tail prepares the messages the model reads next round:
    truncation, append, redaction. Each phase's timing lands on the
    ``tool_results_prepared`` bridge event.
    """
    clock = _PhaseClock()
    state = _RoundState(iteration=iteration)
    _dispatch_tool_calls(tool_calls, config, state)
    clock.mark("dispatched")

    outcomes = (
        await asyncio.gather(*(entry.execute for entry in state.pending), return_exceptions=True)
        if state.pending
        else []
    )
    clock.mark("gathered")
    _record_tool_outcomes(outcomes, state)
    clock.mark("recorded")

    context_baseline = (
        max(context_tokens_baseline, await _estimate_messages_tokens_off_loop(messages))
        if context_tokens_baseline is not None
        else 0
    )
    clock.mark("baseline_counted")
    if context_tokens_baseline is not None:
        # Truncation re-serializes and re-counts every clipped payload: CPU
        # work, kept off the event loop. Sequential await, so the mutations it
        # makes to tool_messages/trace/index are race-free.
        await asyncio.to_thread(
            _truncate_round_tool_messages,
            tool_calls,
            state.tool_messages,
            tool_trace=state.tool_trace,
            index=config.index,
            context_tokens_baseline=context_baseline,
        )
    clock.mark("truncated")
    _append_round_messages(messages, tool_calls, state)
    clock.mark("appended")
    _redact_pruned_context(messages, config.index, state)
    clock.mark("redacted")

    emit_bridge_timing(
        "tool_results_prepared",
        agent=AGENT_NAME,
        iteration=iteration,
        tool_call_count=len(tool_calls),
        remote_tool_count=len(state.pending),
        pruned_context=state.pruned_context,
        message_count=len(messages),
        setup_and_dispatch_ms=clock.ms("start", "dispatched"),
        await_and_collect_ms=clock.ms("dispatched", "recorded"),
        result_processing_ms=clock.ms("gathered", "recorded"),
        baseline_token_count_ms=clock.ms("recorded", "baseline_counted"),
        truncate_and_token_count_ms=clock.ms("baseline_counted", "truncated"),
        append_tool_messages_ms=clock.ms("truncated", "appended"),
        redact_pruned_context_ms=clock.ms("appended", "redacted"),
        total_ms=clock.ms("start", "redacted"),
    )

    return SearcherToolRound(
        final=state.final_result,
        queries=state.queries_made,
        trace=state.tool_trace,
        pruned_context=state.pruned_context,
    )


def _dispatch_tool_calls(
    tool_calls: Sequence[Any],
    config: _RoundConfig,
    state: _RoundState,
) -> None:
    """Decode every call the model made this round and start the remote ones.

    Final calls, prune calls, and every rejection are answered inline; the
    remote executors land in ``state.pending`` for the caller to gather.
    """
    accepted_calls = 0
    # The dispatch loop creates the tool coroutines eagerly; if it raises
    # before the gather, close them so nothing is silently dropped un-awaited.
    try:
        for call_index, tool_call in enumerate(tool_calls, 1):
            trace_event = start_tool_call_trace(
                agent=AGENT_NAME,
                iteration=state.iteration,
                tool_call=tool_call,
                call_index=call_index,
                group_size=len(tool_calls),
            )
            state.tool_trace.append(trace_event)
            if accepted_calls >= MAX_PARALLEL_TOOL_CALLS:
                _fail_tool_call(
                    state,
                    tool_call,
                    trace_event,
                    f"Too many parallel tool calls requested. Maximum is {MAX_PARALLEL_TOOL_CALLS}.",
                )
                continue
            accepted_calls += 1
            _dispatch_tool_call(tool_call, trace_event, config=config, state=state)
    except BaseException:
        for entry in state.pending:
            entry.execute.close()
        raise


def _dispatch_tool_call(
    tool_call: Any,
    trace_event: dict[str, Any],
    *,
    config: _RoundConfig,
    state: _RoundState,
) -> None:
    """Route one accepted call to its handler by tool name."""
    tool_name = tool_call.function.name
    if tool_name == FINAL_TOOL_NAME:
        _record_final_ranking(tool_call, trace_event, config=config, state=state)
        return

    parsed = parse_tool_args(tool_call, _SEARCHER_ARG_SCHEMAS.get(tool_name))
    if isinstance(parsed, dict) and "error" in parsed:
        _fail_tool_call(state, tool_call, trace_event, parsed["error"])
        return
    trace_event["parsed_arguments"] = jsonable(parsed)

    if tool_name == "prune_context":
        _apply_prune_call(parsed, tool_call, trace_event, config=config, state=state)
        return
    filtered_search = config.filtered_search_executors.get(tool_name)
    if filtered_search is not None:
        _dispatch_filtered_search(
            filtered_search, parsed, tool_call, trace_event, config=config, state=state
        )
        return
    plain_retrieval = config.plain_retrieval_executors.get(tool_name)
    if plain_retrieval is not None:
        state.pending.append(
            _PendingToolCall(
                tool_call=tool_call,
                kind=tool_name,
                trace_event=trace_event,
                execute=plain_retrieval(parsed.model_dump(mode="json")),
            )
        )
        return
    _fail_tool_call(state, tool_call, trace_event, f"Unknown tool: {tool_name}")


def _record_final_ranking(
    tool_call: Any,
    trace_event: dict[str, Any],
    *,
    config: _RoundConfig,
    state: _RoundState,
) -> None:
    if config.answers_in_text:
        # The tool was never offered this round, so a call naming it is the
        # model reproducing a shape it no longer has. Point it at the turn
        # that actually ends the episode rather than parsing the payload.
        _fail_tool_call(
            state,
            tool_call,
            trace_event,
            "submit_ranking is not available; reply with your final answer "
            "as plain text with no tool calls.",
        )
        return
    try:
        ranking = parse_ranking(tool_call.function.arguments, require_answer=config.requires_answer)
        validate_ranked_chunk_ids(
            ranking, config.index, top_k=config.top_k, strict_top_k=config.strict_top_k
        )
        state.final_result = ranking
        finish_tool_call_trace(
            trace_event, output={"ranking": ranking_trace_payload(ranking, config.index)}
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail_tool_call(state, tool_call, trace_event, f"Failed to parse ranking: {exc}")


def _dispatch_filtered_search(
    executor: _Executor,
    parsed: BaseModel,
    tool_call: Any,
    trace_event: dict[str, Any],
    *,
    config: _RoundConfig,
    state: _RoundState,
) -> None:
    """Vet the call's metadata filters against the corpus registry, then start it."""
    validation = validate_metadata_filter_args(
        parsed.model_dump(mode="json"),
        registry=build_metadata_registry(
            initial_metadata_facets=config.initial_metadata_facets, index=config.index
        ),
    )
    if not validation.valid:
        _reject_metadata_filters(tool_call, trace_event, validation.trace_metadata(), state)
        return
    trace_event["parsed_arguments"] = jsonable(validation.args)
    state.pending.append(
        _PendingToolCall(
            tool_call=tool_call,
            kind=tool_call.function.name,
            trace_event=trace_event,
            execute=executor(validation.args),
            metadata_validation=validation.trace_metadata(),
        )
    )


def _reject_metadata_filters(
    tool_call: Any,
    trace_event: dict[str, Any],
    validation_metadata: dict[str, Any],
    state: _RoundState,
) -> None:
    payload = {
        "tool": tool_call.function.name,
        "error": "Invalid metadata filters; use only verified metadata fields and values.",
        "metadata_validation": validation_metadata,
    }
    state.tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
    finish_tool_call_trace(
        trace_event,
        status="error",
        output=payload,
        metadata=validation_metadata,
        error="Invalid metadata filters",
    )


def _apply_prune_call(
    parsed: PruneContextArgs,
    tool_call: Any,
    trace_event: dict[str, Any],
    *,
    config: _RoundConfig,
    state: _RoundState,
) -> None:
    """Resolve the prune's handles and mark them pruned in the index.

    The transcript is redacted once, after the round's messages are appended,
    so the keys are collected on ``state`` rather than applied here.
    """
    refs = config.index.refs
    try:
        chunk_keys = {refs.chunk_key_for_id(chunk_id) for chunk_id in parsed.chunk_ids}
        document_keys = {
            refs.document_key_for_id(document_id) for document_id in parsed.document_ids
        }
    except ValueError as exc:
        _fail_tool_call(state, tool_call, trace_event, str(exc))
        return
    state.prune_chunk_keys.update(chunk_keys)
    state.prune_document_keys.update(document_keys)
    config.index.mark_pruned(chunk_keys=chunk_keys, document_keys=document_keys)
    payload = {
        "tool": "prune_context",
        "chunk_ids": parsed.chunk_ids,
        "document_ids": parsed.document_ids,
    }
    state.tool_messages[tool_call.id] = tool_message(tool_call.id, payload)
    finish_tool_call_trace(trace_event, output=payload)


def _record_tool_outcomes(outcomes: Sequence[Any], state: _RoundState) -> None:
    """Turn each gathered outcome into a tool message and a finished trace event."""
    for entry, result in zip(state.pending, outcomes, strict=True):
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            # Cancellation (and friends) must abort the rollout, never
            # degrade into model-visible tool feedback.
            raise result
        if isinstance(result, Exception):
            _fail_tool_call(
                state,
                entry.tool_call,
                entry.trace_event,
                str(result),
                error_kind=error_kind(result),
            )
        elif isinstance(result, ToolOutcome):
            _record_search_outcome(entry, result, state)
        else:
            _record_payload(entry, result, state)


def _record_search_outcome(
    entry: _PendingToolCall, outcome: ToolOutcome, state: _RoundState
) -> None:
    """A corpus search's payload for the model, and its query for the rollout record."""
    query = outcome.query
    query["source"] = f"searcher_{entry.kind}"
    query.update(entry.metadata_validation)
    query["zero_result_filtered_search_count"] = zero_result_filtered_search_count(query)
    state.queries_made.append(query)
    state.tool_messages[entry.tool_call.id] = tool_message(entry.tool_call.id, outcome.payload)
    finish_tool_call_trace(entry.trace_event, output=outcome.payload, metadata=query)


def _record_payload(entry: _PendingToolCall, payload: Any, state: _RoundState) -> None:
    """A lookup's payload for the model; a model-caused failure inside it is an error."""
    state.tool_messages[entry.tool_call.id] = tool_message(entry.tool_call.id, payload)
    payload_error = agent_caused_payload_error(payload)
    finish_tool_call_trace(
        entry.trace_event,
        status="error" if payload_error else "success",
        output=payload,
        error=payload_error,
        error_kind=AGENT_ERROR_KIND if payload_error else None,
    )


def _append_round_messages(
    messages: list[dict[str, Any]],
    tool_calls: Sequence[Any],
    state: _RoundState,
) -> None:
    """Append the round's tool messages in call order, then the media they carry."""
    media_messages: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        message = state.tool_messages.get(tool_call.id)
        if message is None:
            continue
        messages.append(message)
        media_messages.extend(media_messages_for_tool_message(message))
    messages.extend(media_messages)


def _redact_pruned_context(
    messages: list[dict[str, Any]],
    index: ChunkIndex,
    state: _RoundState,
) -> None:
    """Strip the content the round pruned from the history the model reads next."""
    if not state.pruned_context:
        return
    redact_messages(
        messages,
        refs=index.refs,
        chunk_keys=state.prune_chunk_keys,
        document_keys=state.prune_document_keys,
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


class _PhaseClock:
    """Named wall-clock marks for one bridge-timing event."""

    def __init__(self) -> None:
        self._marks: dict[str, float] = {"start": time.perf_counter()}

    def mark(self, name: str) -> None:
        self._marks[name] = time.perf_counter()

    def ms(self, start: str, end: str) -> float:
        return _elapsed_ms(self._marks[start], self._marks[end])


def _elapsed_ms(started: float, finished: float) -> float:
    return round((finished - started) * 1000, 3)


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
