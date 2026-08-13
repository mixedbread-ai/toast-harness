"""The aio surface's own behaviors: event stream, cancellation, typed seams.

Record parity between the sync and async surfaces lives in
``test_record_parity.py`` over the same scenarios; this file pins what aio
adds on top: an ordered typed progress-event stream ending in exactly one
terminal event, task cancellation that aborts the rollout and its in-flight
seam await, typed requests arriving whole at the seam, and the injected client
staying authoritative.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import replay_scenarios as rs
from mixedbread import APIConnectionError

import agent_harness.config as harness_config
from agent_harness import aio
from agent_harness.agents.searcher import FastAgenticSearchResult
from agent_harness.retrieval import MetadataFacetsRequest
from agent_harness.tools import functions as tool_functions

QUERY = "Which contract governs the 2019 Nike distribution agreement?"


class AsyncFiles:
    def __init__(self, files: Any) -> None:
        self._files = files

    async def retrieve(self, request: Any) -> Any:
        return self._files.retrieve(**request.to_kwargs())

    async def list(self, request: Any) -> Any:
        return self._files.list(**request.to_kwargs())


class AsyncStores:
    def __init__(self, stores: Any) -> None:
        self._stores = stores
        self.files = AsyncFiles(stores.files)

    async def search(self, request: Any) -> Any:
        return self._stores.search(**request.to_kwargs())

    async def metadata_facets(self, request: Any) -> Any:
        return self._stores.metadata_facets(**request.to_kwargs())

    async def grep(self, request: Any) -> Any:
        return self._stores.grep(**request.to_kwargs())

    async def list_chunks(self, request: Any) -> Any:
        return self._stores.list_chunks(**request.to_kwargs())


class AsyncScriptedClient:
    """The scripted sync fake behind async seams: identical data, async surface."""

    def __init__(self, sync_client: rs.ScriptedRetrievalClient) -> None:
        self.sync_client = sync_client
        self.stores = AsyncStores(sync_client.stores)


class AsyncScriptedGeneration:
    def __init__(self, scripted: rs.ScriptedGeneration) -> None:
        self.scripted = scripted

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        return self.scripted(
            messages,
            tools=tools,
            completion_config=completion_config,
            force_submit=force_submit,
            forced_tool_name=forced_tool_name,
        )


def _seams(scenario: rs.Scenario) -> tuple[AsyncScriptedGeneration, AsyncScriptedClient]:
    return (
        AsyncScriptedGeneration(scenario.generation()),
        AsyncScriptedClient(scenario.client()),
    )


_TERMINAL_TYPES = frozenset({"rollout_completed", "rollout_failed", "rollout_cancelled"})


def _terminal_types(events: list[Any]) -> list[str]:
    return [event.type for event in events if event.type in _TERMINAL_TYPES]


def _assert_sequences_increase(events: list[Any]) -> None:
    sequences = [event.seq for event in events]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


class FailingGeneration:
    """Replay the scripted turns, except ``fail_on_turn`` which raises ``exc``."""

    def __init__(self, scenario: rs.Scenario, exc: BaseException, *, fail_on_turn: int = 2) -> None:
        self.scripted = rs.ScriptedGeneration(scenario.build_script())
        self.exc = exc
        self.fail_on_turn = fail_on_turn
        self.turns = 0

    async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.turns += 1
        if self.turns == self.fail_on_turn:
            raise self.exc
        return self.scripted(messages, **kwargs)


async def _collect(stream: AsyncIterator[Any], events: list[Any]) -> None:
    async for event in stream:
        events.append(event)


def _rollout_tasks() -> list[asyncio.Task[Any]]:
    return [
        task
        for task in asyncio.all_tasks()
        if task.get_name().startswith("agent-harness-aio") and not task.done()
    ]


async def _wait_for_rollout_tasks_to_exit(timeout: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while _rollout_tasks():
        if asyncio.get_running_loop().time() > deadline:
            names = [task.get_name() for task in _rollout_tasks()]
            msg = f"rollout tasks still alive: {names}"
            raise AssertionError(msg)
        await asyncio.sleep(0.05)


async def test_event_stream_shape() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    generation, client = _seams(scenario)

    events = [
        event
        async for event in aio.stream_fast_agentic_search(
            QUERY,
            store_identifiers=[rs.STORE_ID],
            client=client,
            generation_fn=generation,
        )
    ]

    assert events[0].type == "rollout_started"
    assert events[-1].type == "rollout_completed"
    assert _terminal_types(events) == ["rollout_completed"]
    _assert_sequences_increase(events)

    generation_started = [event for event in events if event.type == "generation_started"]
    generation_completed = [event for event in events if event.type == "generation_completed"]
    assert [event.turn for event in generation_started] == [1, 2, 3]
    assert [event.turn for event in generation_completed] == [1, 2, 3]
    assert all(event.ok for event in generation_completed)

    retrieval_started = [event for event in events if event.type == "retrieval_started"]
    retrieval_completed = [event for event in events if event.type == "retrieval_completed"]
    assert len(retrieval_started) == len(retrieval_completed)
    calls = {event.call for event in retrieval_started}
    assert {"search", "metadata_facets", "grep"} <= calls
    greps = [event for event in retrieval_started if event.call == "grep"]
    assert any(event.pattern == "New York law" for event in greps)
    searches = [event for event in retrieval_started if event.call == "search"]
    assert any(event.query == "governing law" for event in searches)

    result = events[-1].result
    assert isinstance(result, FastAgenticSearchResult)
    assert result.rounds_executed == 3
    assert not result.forced_ranking


async def test_failed_rollout_emits_rollout_failed_then_reraises() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    _, client = _seams(scenario)
    generation = FailingGeneration(scenario, RuntimeError("policy backend exploded"))
    events: list[Any] = []

    stream = aio.stream_fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
    )
    with pytest.raises(RuntimeError, match="policy backend exploded"):
        await _collect(stream, events)

    assert [event.type for event in events[-2:]] == ["generation_completed", "rollout_failed"]
    assert events[-2].ok is False
    assert events[-2].turn == 2
    assert events[-1].error_kind == "agent"
    assert "policy backend exploded" in events[-1].error
    assert _terminal_types(events) == ["rollout_failed"]
    _assert_sequences_increase(events)


async def test_provider_failure_reaches_the_terminal_event_as_provider_kind() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    _, client = _seams(scenario)
    request = httpx.Request("POST", "https://api.mixedbread.test/v1/chat/completions")
    generation = FailingGeneration(scenario, APIConnectionError(request=request))
    events: list[Any] = []

    stream = aio.stream_fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
    )
    with pytest.raises(APIConnectionError):
        await _collect(stream, events)

    assert events[-1].type == "rollout_failed"
    assert events[-1].error_kind == "provider"


async def test_external_rollout_cancellation_ends_the_stream_cleanly() -> None:
    """An externally cancelled rollout must not cancel the consumer's own task."""
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    _, client = _seams(scenario)
    first_turn = rs.ScriptedGeneration(scenario.build_script())
    hanging = asyncio.Event()

    async def generation(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        if not first_turn.calls:
            return first_turn(messages, **kwargs)
        hanging.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    events: list[Any] = []
    stream = aio.stream_fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
    )
    consumer = asyncio.create_task(_collect(stream, events))
    await asyncio.wait_for(hanging.wait(), timeout=5)
    rollouts = _rollout_tasks()
    assert len(rollouts) == 1
    rollouts[0].cancel()

    await asyncio.wait_for(consumer, timeout=5)

    assert consumer.cancelled() is False
    assert events[-1].type == "rollout_cancelled"
    assert _terminal_types(events) == ["rollout_cancelled"]
    _assert_sequences_increase(events)
    await _wait_for_rollout_tasks_to_exit()


async def test_cancellation_aborts_rollout_task_and_seam_call() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    _, client = _seams(scenario)
    first_turn = rs.ScriptedGeneration(scenario.build_script())

    class HangingGeneration:
        def __init__(self) -> None:
            self.hanging = asyncio.Event()
            self.observed_cancellation = False

        async def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            if not first_turn.calls:
                return first_turn(messages, **kwargs)
            self.hanging.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.observed_cancellation = True
                raise
            raise AssertionError("unreachable")

    generation = HangingGeneration()

    async def consume() -> None:
        async for _ in aio.stream_fast_agentic_search(
            QUERY,
            store_identifiers=[rs.STORE_ID],
            client=client,
            generation_fn=generation,
        ):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(generation.hanging.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The seam coroutine's cancellation is scheduled on the loop and the
    # rollout task unwinds on it; both finish shortly after the consumer does.
    deadline = asyncio.get_running_loop().time() + 5
    while not generation.observed_cancellation:
        assert asyncio.get_running_loop().time() < deadline, "seam coroutine never cancelled"
        await asyncio.sleep(0.01)
    await _wait_for_rollout_tasks_to_exit()


async def test_metadata_facets_receives_the_full_typed_request() -> None:
    """The reduced-signature fallback is the sync adapter's concern only:
    an async implementation always gets the complete typed request."""
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    generation, client = _seams(scenario)
    facet_requests: list[Any] = []

    class RecordingStores(AsyncStores):
        async def metadata_facets(self, request: Any) -> Any:  # type: ignore[override]
            facet_requests.append(request)
            return {"metadata_fields": {"year": {"values": [2019]}}}

    client.stores = RecordingStores(client.sync_client.stores)

    result = await aio.run_fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
    )

    assert not result.forced_ranking
    assert len(facet_requests) == 1
    request = facet_requests[0]
    assert isinstance(request, MetadataFacetsRequest)
    assert request.store_identifiers == (rs.STORE_ID,)
    assert request.top_k == 100
    assert request.return_metadata is False


async def test_parallel_tool_round_bridges_concurrent_calls() -> None:
    def build_script() -> list[rs.TurnBuilder]:
        def two_searches(messages: list[dict[str, Any]]) -> Any:
            return rs.response(
                [
                    rs.tool_call("call-a", "search_corpus", {"query": "angle a"}),
                    rs.tool_call("call-b", "search_corpus", {"query": "angle b"}),
                ],
                response_id="resp-1",
            )

        return [
            two_searches,
            rs._submit_turn("resp-2", strategy="parallel round", input_tokens=150),
        ]

    scenario = rs.Scenario(name="parallel_round", build_script=build_script)
    generation, client = _seams(scenario)

    result = await aio.run_fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
    )

    assert result.rounds_executed == 2
    agent_queries = {
        query.get("query")
        for query in result.queries_made
        if query.get("source") == "searcher_search_corpus"
    }
    assert agent_queries == {"angle a", "angle b"}


async def test_ambient_media_content_reaches_the_rollout_thread() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    generation, client = _seams(scenario)

    with harness_config.media_content_setting("never"):
        result = await aio.run_fast_agentic_search(
            QUERY,
            store_identifiers=[rs.STORE_ID],
            client=client,
            generation_fn=generation,
        )

    assert result.media_content == "never"


async def test_run_searcher_returns_rollout_record() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    generation, client = _seams(scenario)

    record = await aio.run_searcher(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
        query_id="q-1",
    )

    assert record["retrieval"]["ranked_ids"] == list(rs.SEEDED_CHUNK_IDS)
    assert record["openai"]["metadata"]["agent"]["forced_ranking"] is False


class _SDKClientConstructed(BaseException):
    """BaseException so the harness's blanket ``except Exception`` cannot eat it."""


async def test_bridged_client_never_falls_back_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def poisoned(**kwargs: Any) -> Any:
        raise _SDKClientConstructed("SDK client must not be constructed")

    monkeypatch.setattr(tool_functions, "get_mixedbread_client", poisoned)

    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    generation, client = _seams(scenario)

    result = await aio.run_fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=client,
        generation_fn=generation,
    )

    assert [chunk["chunk_id"] for chunk in result.chunks] == list(rs.SEEDED_CHUNK_IDS)
