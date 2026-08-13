"""Execution policies that emit OpenAI Responses API rollout records.

The async entry point (``run_searcher_async``) is the native; the policy class
and module-level ``run_searcher`` provide the sync surface by adapting sync
seams onto it and blocking on the result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any

from agent_harness.agents.searcher import fast_agentic_search
from agent_harness.agents.tool_trace import jsonable
from agent_harness.concurrency import run_coroutine_sync
from agent_harness.config import (
    STRICT_TOPK,
    HarnessTuning,
    MediaContentInput,
    token_counter_mode,
)
from agent_harness.errors import PROVIDER_ERROR_KIND
from agent_harness.llm import (
    RESPONSES_API_TRACE_SCHEMA_VERSION,
    AsyncGenerationFn,
    GenerationFn,
    require_generation_fn,
    sync_generation_as_async,
)
from agent_harness.retrieval import (
    AsyncRetrievalClient,
    RetrievalClient,
    SyncRetrievalClientAdapter,
)
from agent_harness.versions import build_version_manifest


def _adapt_client(client: RetrievalClient | None) -> AsyncRetrievalClient | None:
    return SyncRetrievalClientAdapter(client) if client is not None else None


async def run_searcher_async(
    input_text: str,
    *,
    store_identifiers: Sequence[str],
    generation_fn: AsyncGenerationFn | None,
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    query_id: str | None = None,
    query_index: int | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Async native: input -> fixed searcher -> Responses API rollout record."""
    require_generation_fn(generation_fn)
    started = perf_counter()
    result = await fast_agentic_search(
        input_text,
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
    runtime_s = perf_counter() - started
    return build_rollout_result(
        input_text=input_text,
        execution_policy="searcher",
        agent_name="searcher",
        result=result,
        runtime_s=runtime_s,
        query_id=query_id,
        query_index=query_index,
        store_identifiers=store_identifiers,
    )


@dataclass(slots=True)
class SearcherExecutionPolicy:
    """Run the fixed fast-searcher pipeline and return a Responses API rollout."""

    generation_fn: GenerationFn | None = None

    def __post_init__(self) -> None:
        require_generation_fn(self.generation_fn)

    def run(
        self,
        input_text: str,
        *,
        store_identifiers: Sequence[str],
        top_k: int | None = None,
        strict_top_k: bool = STRICT_TOPK,
        client: RetrievalClient | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        additional_instructions: str | None = None,
        query_id: str | None = None,
        query_index: int | None = None,
        include_prompt_snapshot: bool = False,
        media_content: MediaContentInput = None,
        tuning: HarnessTuning | None = None,
        as_of: date | None = None,
    ) -> dict[str, Any]:
        return run_coroutine_sync(
            run_searcher_async(
                input_text,
                store_identifiers=store_identifiers,
                generation_fn=sync_generation_as_async(self.generation_fn),
                top_k=top_k,
                strict_top_k=strict_top_k,
                client=_adapt_client(client),
                api_key=api_key,
                api_key_env=api_key_env,
                additional_instructions=additional_instructions,
                query_id=query_id,
                query_index=query_index,
                include_prompt_snapshot=include_prompt_snapshot,
                media_content=media_content,
                tuning=tuning,
                as_of=as_of,
            )
        )


def run_searcher(
    input_text: str,
    *,
    store_identifiers: Sequence[str],
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    client: RetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    query_id: str | None = None,
    query_index: int | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    generation_fn: GenerationFn | None = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Convenience function for input -> fixed searcher -> Responses API rollout."""
    return SearcherExecutionPolicy(generation_fn=generation_fn).run(
        input_text,
        store_identifiers=store_identifiers,
        top_k=top_k,
        strict_top_k=strict_top_k,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
        additional_instructions=additional_instructions,
        query_id=query_id,
        query_index=query_index,
        include_prompt_snapshot=include_prompt_snapshot,
        media_content=media_content,
        tuning=tuning,
        as_of=as_of,
    )


def build_rollout_result(
    *,
    input_text: str,
    execution_policy: str,
    agent_name: str,
    result: Mapping[str, Any],
    runtime_s: float,
    query_id: str | None,
    query_index: int | None,
    store_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Return the rollout in OpenAI Responses API format plus retrieval output."""
    retrieval = build_retrieval(result)
    openai_responses = result.get("openai_responses")
    if isinstance(openai_responses, Mapping):
        openai = jsonable(openai_responses)
    else:
        openai = {
            "schema_version": RESPONSES_API_TRACE_SCHEMA_VERSION,
            "api": "responses",
            "turns": [],
        }
    existing_metadata = openai.get("metadata")
    openai["metadata"] = {
        **(dict(existing_metadata) if isinstance(existing_metadata, Mapping) else {}),
        **build_rollout_metadata(
            input_text=input_text,
            execution_policy=execution_policy,
            agent_name=agent_name,
            result=result,
            runtime_s=runtime_s,
            query_id=query_id,
            query_index=query_index,
            store_identifiers=store_identifiers,
            retrieval=retrieval,
        ),
    }
    return {
        "openai": openai,
        "retrieval": retrieval,
    }


def build_retrieval(result: Mapping[str, Any]) -> dict[str, Any]:
    return _drop_none(
        {
            "ranked_ids": ranked_ids(result),
            "chunks": list(result.get("chunks") or []),
            "ranking_strategy": result.get("ranking_strategy"),
            "top_k": result.get("top_k"),
            "strict_top_k": result.get("strict_top_k"),
        }
    )


def build_rollout_metadata(
    *,
    input_text: str,
    execution_policy: str,
    agent_name: str,
    result: Mapping[str, Any],
    runtime_s: float,
    query_id: str | None,
    query_index: int | None,
    store_identifiers: Sequence[str],
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    tool_trace = list(result.get("tool_trace") or [])
    versions = build_version_manifest()
    metadata: dict[str, Any] = {
        "input": input_text,
        "execution_policy": execution_policy,
        "harness_version": versions["harness"],
        "versions": versions,
        "store_identifiers": [str(store_id) for store_id in store_identifiers],
        "runtime_s": runtime_s,
        "media_content": result.get("media_content"),
        # Stamped by the rollout; recomputed for records assembled by other callers.
        "token_counter_mode": result.get("token_counter_mode") or token_counter_mode(),
        "agent": {
            "name": agent_name,
            "rounds_executed": result.get("rounds_executed"),
            "forced_ranking": result.get("forced_ranking"),
            "ranking_unresolved": result.get("ranking_unresolved"),
            "total_tokens": result.get("total_tokens"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "agent_token_usage": result.get("agent_token_usage"),
            "trace_counts": count_trace_events(tool_trace),
            "provider_failure_count": count_provider_failures(result),
            "tool_call_iterations": list(result.get("tool_call_iterations") or []),
            "tool_trace": tool_trace,
            "queries_made": result.get("queries_made") or [],
            "id_mapping": result.get("id_mapping"),
        },
        "retrieval": {
            "ranked_ids": list(retrieval.get("ranked_ids") or []),
            "ranking_strategy": retrieval.get("ranking_strategy"),
            "top_k": retrieval.get("top_k"),
            "strict_top_k": retrieval.get("strict_top_k"),
        },
    }
    monitoring = result.get("monitoring")
    if isinstance(monitoring, Mapping):
        metadata["monitoring"] = jsonable(monitoring)
    if query_id is not None:
        metadata["query_id"] = query_id
    if query_index is not None:
        metadata["query_index"] = query_index
    return _drop_none(metadata)


def count_trace_events(tool_trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_agent = Counter(str(event.get("agent") or "unknown") for event in tool_trace)
    by_tool = Counter(str(event.get("name") or "unknown") for event in tool_trace)
    by_status = Counter(str(event.get("status") or "unknown") for event in tool_trace)
    return {
        "tool_calls": len(tool_trace),
        "errors": sum(1 for event in tool_trace if event.get("status") == "error"),
        "provider_errors": sum(
            1 for event in tool_trace if event.get("error_kind") == PROVIDER_ERROR_KIND
        ),
        "by_agent": dict(sorted(by_agent.items())),
        "by_tool": dict(sorted(by_tool.items())),
        "by_status": dict(sorted(by_status.items())),
    }


def count_provider_failures(result: Mapping[str, Any]) -> int:
    """Count Mixedbread-side failures across tool calls and bootstrap fetches.

    Tool-call failures are tagged on ``tool_trace`` events; initial metadata and
    search bootstrap failures only appear as ``queries_made`` entries, so both
    sources are counted.
    """
    events = [*(result.get("tool_trace") or []), *(result.get("queries_made") or [])]
    return sum(
        1
        for event in events
        if isinstance(event, Mapping) and event.get("error_kind") == PROVIDER_ERROR_KIND
    )


def ranked_ids(result: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    for chunk in result.get("chunks") or []:
        if not isinstance(chunk, Mapping):
            continue
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            ids.append(str(chunk_id))
    return ids


def _drop_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value
