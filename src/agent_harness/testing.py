"""Test helpers for callers binding the harness's seams.

Shipped with the package so any in-process integration
can test its binding without copying fixtures or touching a GPU:

- ``tool_call`` / ``response``: build chat-completion-shaped fakes in the
  exact shape the loops read.
- ``ScriptedGeneration``: a sync ``GenerationFn`` that plays back a list of
  turn builders and fails loudly when over-consumed; pass it to the sync
  entry points, or through ``sync_generation_as_async`` for the async ones.
- ``verify_retrieval_client``: drive one real rollout against a caller's
  ``AsyncRetrievalClient`` with scripted generation, exercising the seam
  (bootstrap search + metadata facets + type-sampling ``list_chunks``, a
  ``grep``, and ``files.retrieve`` via ``get_chunks`` when the store has
  content) and asserting the contract held. ``files.list`` is not on the
  fast searcher's path and is not exercised.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

from agent_harness.agents.searcher import (
    FastAgenticSearchResult,
    run_fast_agentic_search,
)
from agent_harness.retrieval import AsyncRetrievalClient

__all__ = [
    "ScriptedGeneration",
    "response",
    "tool_call",
    "verify_retrieval_client",
]

_CHUNK_ID_PATTERN = re.compile(r'"chunk_id":\s*"([^"]+)"')


def tool_call(call_id: str, name: str, arguments: Any) -> SimpleNamespace:
    """Build one model-requested tool call in the shape the loops read."""
    serialized = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=serialized),
    )


def response(
    tool_calls: list[SimpleNamespace],
    *,
    response_id: str,
    content: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> SimpleNamespace:
    """Build one chat-completion-shaped model response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        id=response_id,
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


TurnBuilder = Callable[[list[dict[str, Any]]], SimpleNamespace]


class ScriptedGeneration:
    """Consume one turn builder per call; fail loudly when over-consumed."""

    def __init__(self, script: Sequence[TurnBuilder]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        turn = len(self.calls)
        self.calls.append(
            {
                "turn": turn + 1,
                "force_submit": force_submit,
                "forced_tool_name": forced_tool_name,
                "tool_count": len(tools),
            }
        )
        if turn >= len(self._script):
            msg = f"generation script exhausted at turn {turn + 1}"
            raise AssertionError(msg)
        return self._script[turn](messages)


class _VerificationGeneration:
    """Three scripted turns over whatever the caller's store returns.

    Turn 1 runs before any tool errors exist, so scraping chunk handles out
    of the bootstrap payload messages is safe there (later turns could echo
    invalid handles back inside error payloads).
    """

    def __init__(self, grep_pattern: str) -> None:
        self._grep_pattern = grep_pattern
        self.handles: list[str] = []
        self._turn = 0

    async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        self._turn += 1
        if self._turn == 1:
            self.handles = _bootstrap_handles(messages)[:2]
            if self.handles:
                return response(
                    [tool_call("verify-get", "get_chunks", {"chunk_ids": self.handles})],
                    response_id="verify-resp-1",
                )
            return response(
                [tool_call("verify-grep", "grep", {"pattern": self._grep_pattern})],
                response_id="verify-resp-1",
            )
        if self._turn == 2 and self.handles:
            return response(
                [tool_call("verify-grep", "grep", {"pattern": self._grep_pattern})],
                response_id="verify-resp-2",
            )
        return response(
            [
                tool_call(
                    "verify-submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "seam verification",
                        "chunks": [
                            {"chunk_id": handle, "relevance_score": 1.0} for handle in self.handles
                        ],
                    },
                )
            ],
            response_id=f"verify-resp-{self._turn}",
        )


def _bootstrap_handles(messages: list[dict[str, Any]]) -> list[str]:
    handles: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for handle in _CHUNK_ID_PATTERN.findall(content):
            if handle not in handles:
                handles.append(handle)
    return handles


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"retrieval seam verification failed: {message}")


async def verify_retrieval_client(
    client: AsyncRetrievalClient,
    *,
    store_identifiers: Sequence[str],
    query: str = "retrieval seam verification probe",
    grep_pattern: str = "the",
    expect_content: bool = True,
) -> FastAgenticSearchResult:
    """Drive one rollout against ``client`` and assert the seam contract held.

    Runs the real fast-searcher loop with scripted generation: the bootstrap
    exercises ``search``, ``metadata_facets`` and the type-sampling
    ``list_chunks``; the turns exercise ``grep`` and, when the store returned
    content, ``files.retrieve`` via ``get_chunks`` and a voluntary
    ``submit_ranking`` over handles minted from the client's own results.
    Raises ``AssertionError`` with a specific message on any contract
    violation; returns the result for further caller assertions.

    The kit fails CLOSED: a rollout that yields zero usable hits is
    indistinguishable from a binding whose response shapes hide the hits, so by
    default it is a failure. Pass ``expect_content=False`` only when the store
    really is empty -- that verifies the plumbing but not the content mapping.
    """
    generation = _VerificationGeneration(grep_pattern)
    result = await run_fast_agentic_search(
        query,
        store_identifiers=store_identifiers,
        client=client,
        generation_fn=generation,
    )

    _check(
        "error" not in result.initial_metadata_facets,
        f"bootstrap metadata_facets errored: {result.initial_metadata_facets.get('error')}",
    )
    _check(
        "error" not in result.initial_search_results,
        f"bootstrap search errored: {result.initial_search_results.get('error')}",
    )
    provider_errors = [
        event for event in result.tool_trace if event.get("error_kind") == "provider"
    ]
    _check(not provider_errors, f"provider-side tool failures: {provider_errors}")
    agent_errors = [event for event in result.tool_trace if event.get("status") == "error"]
    _check(not agent_errors, f"tool calls errored: {agent_errors}")
    _check(
        not result.forced_ranking,
        "the voluntary submission was rejected -- handles minted from the client's own "
        "results did not validate",
    )
    grep_queries = [record for record in result.queries_made if record.get("tool") == "grep"]
    _check(bool(grep_queries), "no grep query was recorded")
    if expect_content:
        _check(
            bool(generation.handles),
            "the rollout yielded no usable hits: either the store is actually empty "
            "(pass expect_content=False to verify an empty store) or the binding's "
            "response shapes hide its hits from the harness",
        )
    if generation.handles:
        _check(
            bool(result.chunks),
            "the store returned content but the final ranking resolved no chunks",
        )
        for chunk in result.chunks:
            _check(
                bool(chunk.get("chunk_id")),
                f"finalized chunk is missing its handle: {chunk}",
            )
        # get_chunks per-result errors (e.g. files.retrieve returning the wrong
        # chunks) finalize as trace successes, so the files.retrieve leg needs
        # its own assertion on the payload.
        get_events = [event for event in result.tool_trace if event.get("name") == "get_chunks"]
        _check(bool(get_events), "get_chunks was scripted but left no trace event")
        payload = get_events[0].get("output") or {}
        _check(
            bool(payload.get("restored_chunk_ids")),
            "files.retrieve resolved none of the handles minted from the client's own "
            "search results -- the file-retrieve contract does not round-trip",
        )
        errored = [entry for entry in payload.get("results") or [] if entry.get("error")]
        _check(not errored, f"get_chunks results carry errors: {errored}")
    return result
