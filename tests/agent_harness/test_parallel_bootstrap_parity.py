"""Deterministic parity tests for the concurrent fast-search bootstrap.

The bootstrap overlaps its metadata and seed-search fetches with
``asyncio.gather`` and unpacks them in a fixed metadata-then-search order.
These tests pin that the rendered prompt state is byte-identical to a
sequential reference regardless of which concurrent fetch finishes first,
that the setup-failure retry contract survives, and that
``inspect_metadata``'s nested facets/type-sample concurrency keeps its
payload and silent-fallback behavior.
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import UTC
from datetime import datetime as RealDateTime
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import config as harness_config
from agent_harness import searcher_prompts
from agent_harness.agents import searcher as searcher_runtime
from agent_harness.agents.shared import media_messages_for_payload
from agent_harness.prompts import initial_metadata_facets_message
from agent_harness.search import ChunkIndex, ToolOutcome, serialize_agent_chunks
from agent_harness.tools import functions

_GATE_TIMEOUT = 2.0


class _FrozenDateTime(RealDateTime):
    @classmethod
    def now(cls, tz: Any = None) -> _FrozenDateTime:
        value = cls(2026, 8, 5, 12, 34, 56, tzinfo=UTC)
        return value if tz is not None else value.replace(tzinfo=None)


class _OrderedGate:
    """Force two concurrent calls to overlap, then complete in a chosen order.

    Each participant blocks until the other has arrived, so a completed run
    proves the two fetches were genuinely in flight together; ``first`` then
    decides which one finishes first.
    """

    def __init__(self, first: str | None = None) -> None:
        self.first = first
        self.order: list[str] = []
        self._reached: set[str] = set()
        self._both_reached = asyncio.Event()
        self._first_done = asyncio.Event()

    async def reach(self, name: str) -> None:
        if self.first is None:
            self.order.append(name)
            return

        self._reached.add(name)
        if len(self._reached) == 2:
            self._both_reached.set()
        await asyncio.wait_for(self._both_reached.wait(), timeout=_GATE_TIMEOUT)
        if name == self.first:
            self.order.append(name)
            self._first_done.set()
            return

        await asyncio.wait_for(self._first_done.wait(), timeout=_GATE_TIMEOUT)
        self.order.append(name)


def _prompt_bytes(value: Any) -> bytes:
    """Serialize exactly in insertion order, as prompt JSON is rendered."""

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _context_tuple(
    context: searcher_runtime.InitialSearchContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        context.metadata_facets,
        context.metadata_query,
        context.search_results,
        context.search_query,
    )


async def _sequential_initial_context(
    user_text: str,
    *,
    index: ChunkIndex,
    store_identifiers: list[str],
    client: Any | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> searcher_runtime.InitialSearchContext:
    resolved = client or searcher_runtime.resolve_async_retrieval_client(
        None,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    _ = resolved.stores
    metadata_outcome = await searcher_runtime._fetch_initial_metadata_facets(
        store_identifiers=store_identifiers,
        client=resolved,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    search_outcome = await searcher_runtime._fetch_initial_search_results(
        user_text,
        index=index,
        store_identifiers=store_identifiers,
        client=resolved,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    return searcher_runtime.InitialSearchContext(
        metadata_facets=metadata_outcome.payload,
        metadata_query=metadata_outcome.query,
        search_results=search_outcome.payload,
        search_query=search_outcome.query,
    )


def _render_initial_state(
    context: searcher_runtime.InitialSearchContext,
    *,
    index: ChunkIndex,
) -> dict[str, Any]:
    metadata_facets, metadata_query, search_results, search_query = _context_tuple(context)
    messages = searcher_runtime.fast_searcher_messages(
        user_text="find café launch image",
        initial_search_results=search_results,
        initial_metadata_facets=metadata_facets,
        top_k=10,
        strict_top_k=True,
        additional_instructions="Preserve naïve Unicode exactly.",
    )
    messages.extend(media_messages_for_payload(search_results))
    snapshot = searcher_runtime._fast_searcher_prompt_snapshot(
        messages=messages,
        additional_instructions="Preserve naïve Unicode exactly.",
    )
    return {
        "initial_metadata_facets": metadata_facets,
        "initial_search_results": search_results,
        "queries_made": [
            {**metadata_query, "source": "initial_metadata_facets"},
            {**search_query, "source": "initial_original_query"},
        ],
        "messages": messages,
        "prompt_snapshot": snapshot,
        "id_mapping": index.refs.snapshot(),
        "visible_chunk_ids": index.visible_chunk_ids(),
    }


def _install_initial_fetch_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata_succeeds: bool,
    search_succeeds: bool,
    gate: _OrderedGate,
    chunk_id_holder: dict[str, str] | None = None,
) -> list[bool]:
    shared_client = SimpleNamespace(stores=object())
    shared_client_checks: list[bool] = []

    async def fake_metadata(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        shared_client_checks.append(kwargs.get("client") is shared_client)
        await gate.reach("metadata")
        if not metadata_succeeds:
            raise RuntimeError("metadata boom Ω")
        return ToolOutcome(
            {
                "store_identifiers": ["store-a"],
                "metadata_field_count": 1,
                "metadata_fields": {
                    "year": {
                        "type": "integer",
                        "representative_values": [2026, 2025],
                    }
                },
            },
            {
                "tool": "inspect_metadata",
                "metadata_field_count": 1,
                "store_identifiers": ["store-a"],
            },
        )

    async def fake_search(
        args: dict[str, Any],
        *,
        index: ChunkIndex,
        store_identifiers: list[str],
        top_k: int,
        client: Any | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
    ) -> ToolOutcome:
        del store_identifiers, top_k, api_key, api_key_env
        shared_client_checks.append(client is shared_client)
        await gate.reach("search")
        if not search_succeeds:
            raise RuntimeError("search boom β")
        raw_chunks = [
            {
                "store_id": "store-a",
                "file_id": "file-a",
                "chunk_index": 7,
                "score": 0.9,
                "text": "café launch",
                "image_url": "https://example.test/cafe.png",
            },
            {
                "store_id": "store-a",
                "file_id": "file-b",
                "chunk_index": 2,
                "score": 0.8,
                "text": "naïve campaign",
            },
        ]
        added = index.ingest_search_results(raw_chunks, max_new_chunks=2)
        serialized = serialize_agent_chunks(added, refs=index.refs)
        if chunk_id_holder is not None:
            chunk_id_holder["chunk_id"] = serialized[0]["chunk_id"]
        query = str(args["query"])
        return ToolOutcome(
            {
                "tool": "search_corpus",
                "query": query,
                "new_unseen_results": serialized,
            },
            {
                "tool": "search_corpus",
                "query": query,
                "k": 2,
                "new_chunks_added": 2,
            },
        )

    monkeypatch.setattr(
        searcher_runtime,
        "resolve_async_retrieval_client",
        lambda client, **kwargs: client if client is not None else shared_client,
    )
    monkeypatch.setattr(searcher_runtime, "execute_inspect_metadata", fake_metadata)
    monkeypatch.setattr(searcher_runtime, "execute_search_corpus", fake_search)
    return shared_client_checks


@pytest.mark.parametrize("metadata_succeeds", [True, False])
@pytest.mark.parametrize("search_succeeds", [True, False])
@pytest.mark.parametrize("first", ["metadata", "search"])
async def test_parallel_initial_context_is_byte_identical_to_sequential_for_all_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    metadata_succeeds: bool,
    search_succeeds: bool,
    first: str,
) -> None:
    monkeypatch.setattr(searcher_prompts, "datetime", _FrozenDateTime)

    sequential_gate = _OrderedGate()
    sequential_client_checks = _install_initial_fetch_fakes(
        monkeypatch,
        metadata_succeeds=metadata_succeeds,
        search_succeeds=search_succeeds,
        gate=sequential_gate,
    )
    kwargs = {
        "store_identifiers": ["store-a"],
        "api_key": "key",
        "api_key_env": "KEY_ENV",
    }
    sequential_index = ChunkIndex()
    with harness_config.media_content_setting("always"):
        sequential = await _sequential_initial_context(
            "find café launch image",
            index=sequential_index,
            **kwargs,
        )
        expected = _prompt_bytes(_render_initial_state(sequential, index=sequential_index))
    assert sequential_gate.order == ["metadata", "search"]
    assert sequential_client_checks == [True, True]

    parallel_gate = _OrderedGate(first)
    parallel_client_checks = _install_initial_fetch_fakes(
        monkeypatch,
        metadata_succeeds=metadata_succeeds,
        search_succeeds=search_succeeds,
        gate=parallel_gate,
    )
    parallel_index = ChunkIndex()
    with harness_config.media_content_setting("always"):
        parallel = await searcher_runtime._fetch_initial_context(
            "find café launch image",
            index=parallel_index,
            **kwargs,
        )
        actual = _prompt_bytes(_render_initial_state(parallel, index=parallel_index))

    second = "search" if first == "metadata" else "metadata"
    assert parallel_gate.order == [first, second]
    assert parallel_client_checks == [True, True]
    assert actual == expected


@pytest.mark.parametrize("setup_failure", ["client", "stores"])
@pytest.mark.parametrize("search_succeeds", [True, False])
async def test_initial_context_setup_failure_preserves_sequential_retry_contract(
    monkeypatch: pytest.MonkeyPatch,
    setup_failure: str,
    search_succeeds: bool,
) -> None:
    class StatefulClient:
        def __init__(self) -> None:
            self.stores_accesses = 0

        @property
        def stores(self) -> object:
            self.stores_accesses += 1
            if setup_failure == "stores" and self.stores_accesses == 1:
                raise RuntimeError("stores setup boom")
            return object()

    client = StatefulClient()
    client_calls = {"count": 0}

    def resolve(explicit: Any | None, **kwargs: Any) -> StatefulClient:
        del kwargs
        if explicit is not None:
            return explicit
        client_calls["count"] += 1
        if setup_failure == "client" and client_calls["count"] == 1:
            raise RuntimeError("client setup boom")
        return client

    def resolved_client(explicit: Any | None) -> StatefulClient:
        value = resolve(explicit)
        _ = value.stores
        return value

    async def fake_inspect(
        args: dict[str, Any],
        *,
        store_identifiers: list[str],
        client: Any | None = None,
        **kwargs: Any,
    ) -> ToolOutcome:
        del args, kwargs
        resolved_client(client)
        return ToolOutcome(
            {"store_identifiers": store_identifiers, "metadata_fields": {}},
            {"tool": "inspect_metadata", "store_identifiers": store_identifiers},
        )

    async def fake_search(
        args: dict[str, Any],
        *,
        index: ChunkIndex,
        store_identifiers: list[str],
        client: Any | None = None,
        **kwargs: Any,
    ) -> ToolOutcome:
        del index, kwargs
        resolved_client(client)
        if not search_succeeds:
            raise RuntimeError("search boom")
        return ToolOutcome(
            {"tool": "search_corpus", "query": args["query"], "new_unseen_results": []},
            {
                "tool": "search_corpus",
                "query": args["query"],
                "store_identifiers": store_identifiers,
            },
        )

    monkeypatch.setattr(searcher_runtime, "resolve_async_retrieval_client", resolve)
    monkeypatch.setattr(searcher_runtime, "execute_inspect_metadata", fake_inspect)
    monkeypatch.setattr(searcher_runtime, "execute_search_corpus", fake_search)

    def reset() -> None:
        client_calls["count"] = 0
        client.stores_accesses = 0

    reset()
    metadata_outcome = await searcher_runtime._fetch_initial_metadata_facets(
        store_identifiers=["store-a"]
    )
    search_outcome = await searcher_runtime._fetch_initial_search_results(
        "find café",
        index=ChunkIndex(),
        store_identifiers=["store-a"],
    )
    expected = (
        metadata_outcome.payload,
        metadata_outcome.query,
        search_outcome.payload,
        search_outcome.query,
    )
    expected_calls = (client_calls["count"], client.stores_accesses)

    reset()
    actual = await searcher_runtime._fetch_initial_context(
        "find café",
        index=ChunkIndex(),
        store_identifiers=["store-a"],
    )

    assert _prompt_bytes(_context_tuple(actual)) == _prompt_bytes(expected)
    assert (client_calls["count"], client.stores_accesses) == expected_calls


def _fake_submit_response(chunk_id: str) -> Any:
    tool_call = SimpleNamespace(
        id="submit-1",
        type="function",
        function=SimpleNamespace(
            name="submit_ranking",
            arguments=json.dumps(
                {
                    "ranking_strategy": "stable",
                    "chunks": [{"chunk_id": chunk_id, "relevance_score": 1.0}],
                }
            ),
        ),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))],
        usage=SimpleNamespace(input_tokens=123, output_tokens=17),
    )


async def _run_fast_search_for_prompt_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fetch_impl: Any,
    gate: _OrderedGate,
) -> dict[str, Any]:
    chunk_id_holder: dict[str, str] = {}
    _install_initial_fetch_fakes(
        monkeypatch,
        metadata_succeeds=True,
        search_succeeds=True,
        gate=gate,
        chunk_id_holder=chunk_id_holder,
    )
    monkeypatch.setattr(searcher_runtime, "_fetch_initial_context", fetch_impl)
    captured_requests: list[list[dict[str, Any]]] = []

    async def generate(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del kwargs
        captured_requests.append(copy.deepcopy(messages))
        return _fake_submit_response(chunk_id_holder["chunk_id"])

    result = await searcher_runtime.run_fast_agentic_search(
        "find café",
        store_identifiers=["store-a"],
        top_k=1,
        strict_top_k=True,
        additional_instructions="naïve",
        include_prompt_snapshot=True,
        media_content="always",
        generation_fn=generate,
    )
    return {
        "first_request_messages": captured_requests[0],
        "prompt_snapshot": result.prompt_snapshot,
        "full_messages": result.messages,
        "initial_metadata_facets": result.initial_metadata_facets,
        "initial_search_results": result.initial_search_results,
        "queries_made": result.queries_made,
        "id_mapping": result.id_mapping,
    }


@pytest.mark.parametrize("first", ["metadata", "search"])
async def test_full_fast_search_prompt_snapshot_matches_sequential_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    first: str,
) -> None:
    monkeypatch.setattr(searcher_prompts, "datetime", _FrozenDateTime)
    actual_parallel_fetch = searcher_runtime._fetch_initial_context

    sequential_gate = _OrderedGate()
    expected = _prompt_bytes(
        await _run_fast_search_for_prompt_state(
            monkeypatch,
            fetch_impl=_sequential_initial_context,
            gate=sequential_gate,
        )
    )
    assert sequential_gate.order == ["metadata", "search"]

    parallel_gate = _OrderedGate(first)
    actual = _prompt_bytes(
        await _run_fast_search_for_prompt_state(
            monkeypatch,
            fetch_impl=actual_parallel_fetch,
            gate=parallel_gate,
        )
    )
    second = "search" if first == "metadata" else "metadata"
    assert parallel_gate.order == [first, second]
    assert actual == expected


def _install_nested_metadata_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    facets_mode: str,
    type_sample_fails: bool,
    gate: _OrderedGate,
) -> tuple[Any, list[bool], dict[str, int]]:
    facet_calls = {"count": 0}
    sampled_with_shared_client: list[bool] = []

    class Stores:
        async def metadata_facets(self, request: Any) -> dict[str, Any]:
            facet_calls["count"] += 1
            if facet_calls["count"] == 1:
                await gate.reach("facets")
            if facets_mode == "fail":
                raise RuntimeError("facets explode Ω")
            return {
                "facets": {
                    "year": {"2026": 4, "2025": 2},
                    "active": {"true": 3, "false": 1},
                }
            }

    class Client:
        stores = Stores()

    shared_client = Client()

    async def list_chunks_raw(**kwargs: Any) -> list[dict[str, Any]]:
        sampled_with_shared_client.append(kwargs.get("client") is shared_client)
        await gate.reach("sample")
        if type_sample_fails:
            raise RuntimeError("type sample unavailable β")
        return [{"metadata": {"year": 2026, "active": True}}]

    monkeypatch.setattr(
        functions,
        "resolve_async_retrieval_client",
        lambda client, **kwargs: client if client is not None else shared_client,
    )
    monkeypatch.setattr(functions, "list_chunks_raw", list_chunks_raw)
    return shared_client, sampled_with_shared_client, facet_calls


@pytest.mark.parametrize("facets_mode", ["ok", "fail"])
@pytest.mark.parametrize("type_sample_fails", [False, True])
@pytest.mark.parametrize("first", ["facets", "sample"])
async def test_nested_metadata_parallelism_preserves_payload_and_silent_sample_fallback(
    monkeypatch: pytest.MonkeyPatch,
    facets_mode: str,
    type_sample_fails: bool,
    first: str,
) -> None:
    sequential_gate = _OrderedGate()
    _, sequential_client_checks, sequential_facet_calls = _install_nested_metadata_fakes(
        monkeypatch,
        facets_mode=facets_mode,
        type_sample_fails=type_sample_fails,
        gate=sequential_gate,
    )
    sequential = await searcher_runtime._fetch_initial_metadata_facets(
        store_identifiers=["store-a"]
    )
    expected = _prompt_bytes(
        {
            "fetch": [sequential.payload, sequential.query],
            "prompt_message": initial_metadata_facets_message(sequential.payload),
        }
    )
    assert sequential_gate.order == ["facets", "sample"]
    assert sequential_client_checks == [True]
    assert sequential_facet_calls["count"] == 1

    parallel_gate = _OrderedGate(first)
    _, parallel_client_checks, parallel_facet_calls = _install_nested_metadata_fakes(
        monkeypatch,
        facets_mode=facets_mode,
        type_sample_fails=type_sample_fails,
        gate=parallel_gate,
    )
    parallel = await searcher_runtime._fetch_initial_metadata_facets(store_identifiers=["store-a"])
    actual = _prompt_bytes(
        {
            "fetch": [parallel.payload, parallel.query],
            "prompt_message": initial_metadata_facets_message(parallel.payload),
        }
    )

    second = "sample" if first == "facets" else "facets"
    assert parallel_gate.order == [first, second]
    assert parallel_client_checks == [True]
    assert parallel_facet_calls["count"] == 1
    assert actual == expected

    if type_sample_fails and facets_mode != "fail":
        payload = parallel.payload
        assert "error" not in payload
        samples = payload["metadata_fields"]["year"]
        assert all("type" not in sample for sample in samples)
