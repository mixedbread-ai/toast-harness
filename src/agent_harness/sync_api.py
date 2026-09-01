"""The synchronous compatibility surface over the async-native harness.

Every public sync callable takes a sync ``RetrievalClient`` (the SDK included)
and a sync ``GenerationFn`` where one applies. Each call adapts those seams --
``SyncRetrievalClientAdapter`` and ``SyncGenerationAdapter`` run them on worker
threads so parallel tool calls still overlap -- and blocks on the async native
via ``run_coroutine_sync``.

Async callers skip this module entirely: ``agent_harness.aio`` for the agent
entry points, or the coroutines in ``agent_harness.tools.functions`` and
``agent_harness.agents`` directly.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from typing import Any

from agent_harness.agents import searcher as _searcher
from agent_harness.agents.searcher import FastAgenticSearchResult
from agent_harness.concurrency import run_coroutine_sync
from agent_harness.config import STRICT_TOPK, HarnessTuning, MediaContentInput
from agent_harness.llm import GenerationFn, sync_generation_as_async
from agent_harness.retrieval import (
    AsyncRetrievalClient,
    RetrievalClient,
    SyncRetrievalClientAdapter,
)
from agent_harness.schemas import AnswerMode
from agent_harness.tools import functions as _functions
from agent_harness.tools.functions import prune_context, submit_ranking

__all__ = [
    "TOOL_FUNCTIONS",
    "fast_agentic_search",
    "filter_chunks",
    "get_chunk",
    "get_chunks",
    "grep",
    "inspect_metadata",
    "overview_search",
    "prune_context",
    "read_document",
    "run_fast_agentic_search",
    "search_corpus",
    "submit_ranking",
]


def _adapt_client(client: RetrievalClient | None) -> AsyncRetrievalClient | None:
    return SyncRetrievalClientAdapter(client) if client is not None else None


def _sync_tool(async_fn: Callable[..., Awaitable[Any]]) -> Callable[..., Any]:
    """Wrap an async tool function; ``client`` here is the sync protocol."""

    @functools.wraps(async_fn)
    def wrapper(*args: Any, client: RetrievalClient | None = None, **kwargs: Any) -> Any:
        return run_coroutine_sync(async_fn(*args, client=_adapt_client(client), **kwargs))

    return wrapper


search_corpus = _sync_tool(_functions.search_corpus)
grep = _sync_tool(_functions.grep)
filter_chunks = _sync_tool(_functions.filter_chunks)
inspect_metadata = _sync_tool(_functions.inspect_metadata)
overview_search = _sync_tool(_functions.overview_search)
read_document = _sync_tool(_functions.read_document)
get_chunk = _sync_tool(_functions.get_chunk)
get_chunks = _sync_tool(_functions.get_chunks)


def run_fast_agentic_search(
    user_text: str,
    *,
    store_identifiers: Sequence[str],
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    client: RetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    generation_fn: GenerationFn | None = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
    answer_mode: AnswerMode = "none",
) -> FastAgenticSearchResult:
    """Sync ``agents.searcher.run_fast_agentic_search`` with adapted seams."""
    return run_coroutine_sync(
        _searcher.run_fast_agentic_search(
            user_text,
            store_identifiers=store_identifiers,
            top_k=top_k,
            strict_top_k=strict_top_k,
            client=_adapt_client(client),
            api_key=api_key,
            api_key_env=api_key_env,
            additional_instructions=additional_instructions,
            include_prompt_snapshot=include_prompt_snapshot,
            media_content=media_content,
            generation_fn=sync_generation_as_async(generation_fn),
            tuning=tuning,
            as_of=as_of,
            answer_mode=answer_mode,
        )
    )


def fast_agentic_search(
    user_text: str,
    *,
    store_identifiers: Sequence[str],
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    client: RetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    generation_fn: GenerationFn | None = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
    answer_mode: AnswerMode = "none",
) -> dict[str, Any]:
    """Run the fast searcher and return the plain-dict record payload."""
    return run_fast_agentic_search(
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
    ).to_record()


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "inspect_metadata": inspect_metadata,
    "filter_chunks": filter_chunks,
    "grep": grep,
    "search_corpus": search_corpus,
    "read_document": read_document,
    "get_chunks": get_chunks,
    "overview_search": overview_search,
    "submit_ranking": submit_ranking,
    "prune_context": prune_context,
}
