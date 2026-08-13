"""The shipped conformance kit verifies bindings the way integrators will use it."""

from __future__ import annotations

from typing import Any

import pytest
import replay_scenarios as rs
from test_aio import AsyncScriptedClient

from agent_harness.testing import verify_retrieval_client

STORES = [rs.STORE_ID]


async def test_verify_passes_for_a_conforming_client() -> None:
    client = AsyncScriptedClient(rs.ScriptedRetrievalClient())

    result = await verify_retrieval_client(client, store_identifiers=STORES)

    # The scripted corpus has content, so the full path ran: handles were
    # minted from the bootstrap, fetched back via files.retrieve, grepped,
    # and voluntarily submitted.
    assert [chunk["chunk_id"] for chunk in result.chunks] == list(rs.SEEDED_CHUNK_IDS)
    calls = client.sync_client.call_names()
    assert "metadata_facets" in calls
    assert "search" in calls
    assert "grep" in calls
    assert not result.forced_ranking


async def test_verify_fails_when_search_is_broken() -> None:
    broken = AsyncScriptedClient(
        rs.ScriptedRetrievalClient(failing_queries=frozenset({"retrieval seam verification probe"}))
    )

    with pytest.raises(AssertionError, match="bootstrap search errored"):
        await verify_retrieval_client(broken, store_identifiers=STORES)


async def test_verify_reports_agent_visible_tool_failures() -> None:
    class GrepBrokenStores:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.files = inner.files

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

        async def grep(self, request: Any) -> Any:
            raise RuntimeError("grep backend exploded")

    client = AsyncScriptedClient(rs.ScriptedRetrievalClient())
    client.stores = GrepBrokenStores(client.stores)

    with pytest.raises(AssertionError, match="tool calls errored"):
        await verify_retrieval_client(client, store_identifiers=STORES)


async def test_verify_fails_when_file_retrieve_does_not_round_trip() -> None:
    """A files.retrieve that resolves none of the minted handles must fail
    verification even though the rollout itself completes."""

    class WrongChunksFiles:
        def retrieve(self, **kwargs: Any) -> Any:
            return {
                "id": kwargs["file_identifier"],
                "store_id": kwargs["store_identifier"],
                "filename": "contract.pdf",
                "chunks": [],
            }

        def list(self, **kwargs: Any) -> Any:
            return {"data": [], "pagination": {}}

    sync_client = rs.ScriptedRetrievalClient()
    sync_client.stores.files = WrongChunksFiles()
    client = AsyncScriptedClient(sync_client)

    with pytest.raises(AssertionError, match=r"file-retrieve contract|carry errors"):
        await verify_retrieval_client(client, store_identifiers=STORES)


async def test_bootstrap_fetches_lead_the_tool_trace() -> None:
    client = AsyncScriptedClient(rs.ScriptedRetrievalClient())

    result = await verify_retrieval_client(client, store_identifiers=STORES)

    heads = [
        (event["name"], event["iteration"], event["forced"], event["status"])
        for event in result.tool_trace[:2]
    ]
    assert heads == [
        ("inspect_metadata", 0, False, "success"),
        ("search_corpus", 0, False, "success"),
    ]
    assert all("duration_ms" in event for event in result.tool_trace[:2])


class _ReshapedStores:
    """Delegate every seam method, but re-nest search hits outside "data"."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def search(self, request: Any) -> Any:
        response = await self._inner.search(request)
        hits = response.get("data") if isinstance(response, dict) else response.data
        return {"results": list(hits or [])}


class _EmptyStores:
    """Delegate every seam method, but return valid empty result containers."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def search(self, request: Any) -> Any:
        del request
        return {"data": []}

    async def grep(self, request: Any) -> Any:
        del request
        return {"data": []}


class _WrappedClient:
    def __init__(self, inner: Any, stores: Any) -> None:
        self._inner = inner
        self.stores = stores

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


async def test_verify_fails_a_binding_that_hides_hits_outside_data() -> None:
    """The kit's whole job: a wrong-shaped binding must not verify.

    Hits nested under anything but "data" fail the tool call loudly (instead of
    reading as an empty store), so the kit surfaces the shape violation rather
    than passing a binding that would silently zero retrieval in production."""
    inner = AsyncScriptedClient(rs.ScriptedRetrievalClient())
    wrong = _WrappedClient(inner, _ReshapedStores(inner.stores))

    with pytest.raises(AssertionError, match="errored"):
        await verify_retrieval_client(wrong, store_identifiers=STORES)


async def test_verify_fails_closed_when_the_rollout_yields_no_hits() -> None:
    """Zero usable hits is indistinguishable from a shape bug: fail by default,
    verify plumbing-only through the explicit expect_content=False opt-out."""
    inner = AsyncScriptedClient(rs.ScriptedRetrievalClient())
    empty = _WrappedClient(inner, _EmptyStores(inner.stores))

    with pytest.raises(AssertionError, match="no usable hits"):
        await verify_retrieval_client(empty, store_identifiers=STORES)

    result = await verify_retrieval_client(empty, store_identifiers=STORES, expect_content=False)
    assert result.chunks == []


async def test_every_trace_event_carries_the_same_agent_label() -> None:
    """The bootstrap fetches must not mint a phantom agent label:
    trace_counts.by_agent groups on it, and "fast" beside "fast_searcher"
    breaks any consumer keying on the agent."""
    client = AsyncScriptedClient(rs.ScriptedRetrievalClient())

    result = await verify_retrieval_client(client, store_identifiers=STORES)

    assert {event["agent"] for event in result.tool_trace} == {"fast_searcher"}
