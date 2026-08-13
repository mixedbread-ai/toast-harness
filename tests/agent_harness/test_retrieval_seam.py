"""An injected retrieval client serves a whole rollout: the in-process route.

The SDK client factory is poisoned for the duration, so any path that resolved a
network client instead of the injected one fails the test rather than silently
falling back to the public API.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness
from agent_harness import RetrievalClient
from agent_harness.retrieval import (
    GrepRequest,
    MetadataFacetsRequest,
    SyncRetrievalClientAdapter,
)
from agent_harness.tools import functions as tool_functions
from agent_harness.tools.functions import search_raw

_CHUNK_ID_PATTERN = re.compile(r'"chunk_id":\s*"([^"]+)"')

STORE_ID = "in-process-store"
FILE_ID = "file-1"


def _chunk(index: int) -> dict[str, Any]:
    return {
        "id": f"chunk-{index}",
        "file_id": FILE_ID,
        "store_id": STORE_ID,
        "chunk_index": index,
        "filename": "contract.pdf",
        "text": f"clause {index}: the distribution agreement is governed by New York law",
        "score": 1.0 - index / 100,
        "generated_metadata": {"year": 2019},
    }


class _FakeSearchResults:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class _FakeStoreFiles:
    def retrieve(self, *, file_identifier: str, store_identifier: str, return_chunks: Any) -> Any:
        indices = return_chunks if isinstance(return_chunks, list) else [0, 1]
        return {
            "id": file_identifier,
            "store_id": store_identifier,
            "filename": "contract.pdf",
            "chunks": [_chunk(index) for index in indices],
        }

    def list(self, **kwargs: Any) -> Any:
        return {"data": [], "pagination": {}}


class _FakeStores:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]]) -> None:
        self._calls = calls
        self.files = _FakeStoreFiles()

    def search(self, **kwargs: Any) -> _FakeSearchResults:
        self._calls.append(("search", kwargs))
        return _FakeSearchResults([_chunk(0), _chunk(1)])

    def metadata_facets(self, **kwargs: Any) -> Any:
        self._calls.append(("metadata_facets", kwargs))
        return {"metadata_fields": {"year": {"values": [2019]}}}

    def list_chunks(self, **kwargs: Any) -> Any:
        self._calls.append(("list_chunks", kwargs))
        return {"data": [_chunk(0)]}


class InProcessRetrievalClient:
    """An embedding host binds in the store layer already resident in the process."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stores = _FakeStores(self.calls)

    def post(self, path: str, *, cast_to: Any, body: Any) -> Any:
        self.calls.append(("post", {"path": path, "body": dict(body)}))
        return {"data": []}

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(tool_calls: list[SimpleNamespace]) -> SimpleNamespace:
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(
        id="resp-1",
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
    )


class _StubGeneration:
    """One search_corpus round, then submit_ranking over what the fake returned."""

    def __init__(self) -> None:
        self.rounds = 0

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        self.rounds += 1
        if self.rounds == 1:
            return _response([_tool_call("call-1", "search_corpus", {"query": "governing law"})])
        chunk_ids = _visible_chunk_ids(messages)
        return _response(
            [
                _tool_call(
                    "call-2",
                    "submit_ranking",
                    {
                        "ranking_strategy": "relevance to the governing-law question",
                        "chunks": [
                            {"chunk_id": chunk_id, "relevance_score": 0.9} for chunk_id in chunk_ids
                        ],
                    },
                )
            ]
        )


def _visible_chunk_ids(messages: list[dict[str, Any]]) -> list[str]:
    """Collect the handles the harness minted for the fake client's chunks."""
    found: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for chunk_id in _CHUNK_ID_PATTERN.findall(content):
            if chunk_id not in found:
                found.append(chunk_id)
    return found


class _SDKClientConstructed(BaseException):
    """Deliberately not an ``Exception``: the harness's blanket handlers must not eat it.

    Every retrieval call site degrades provider failures to an error payload
    under ``except Exception``, so a poison raising ``AssertionError`` is
    swallowed and the fallback it exists to catch still reports a green test.
    """


@pytest.fixture
def no_sdk_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def poisoned(**kwargs: Any) -> Any:
        raise _SDKClientConstructed("SDK client must not be constructed")

    # Patch the module that binds the factory by name, not just its
    # definition. The loops resolve clients through
    # tool_functions.resolve_async_retrieval_client, which reads this binding.
    monkeypatch.setattr(tool_functions, "get_mixedbread_client", poisoned)


def test_injected_client_serves_the_whole_rollout(no_sdk_client: None) -> None:
    client = InProcessRetrievalClient()
    assert isinstance(client, RetrievalClient)
    generate = _StubGeneration()

    result = agent_harness.run_searcher(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[STORE_ID],
        client=client,
        generation_fn=generate,
    )

    assert client.call_names().count("search") >= 2  # bootstrap + the agent's own call
    assert "metadata_facets" in client.call_names()
    assert any(
        kwargs.get("query") == "governing law" for name, kwargs in client.calls if name == "search"
    )
    assert result["retrieval"]["ranked_ids"]
    assert result["openai"]["metadata"]["agent"]["forced_ranking"] is False
    assert generate.rounds == 2


def test_injected_client_failure_never_falls_back_to_the_sdk(no_sdk_client: None) -> None:
    """One failure resolving the injected client must not become public-API egress.

    The bootstrap's failure contract gives the initial search its own chance
    after metadata-client setup fails; that retry must still carry the injected
    client rather than constructing an SDK client.
    """

    class FlakyStoresClient(InProcessRetrievalClient):
        def __init__(self) -> None:
            self._stores_reads = 0
            super().__init__()  # its stores assignment lands on the setter below

        @property  # type: ignore[override]
        def stores(self) -> Any:
            self._stores_reads += 1
            if self._stores_reads == 1:
                msg = "stores resource unavailable on first resolution"
                raise RuntimeError(msg)
            return self._stores

        @stores.setter
        def stores(self, value: Any) -> None:
            self._stores = value

    client = FlakyStoresClient()
    generate = _StubGeneration()

    result = agent_harness.run_searcher(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[STORE_ID],
        client=client,
        generation_fn=generate,
    )

    # The count is what binds: restoring the fallback drops the bootstrap search
    # off the injected client, leaving only the agent's own call. Mere presence
    # of a search survives that mutation and pins nothing.
    assert client.call_names().count("search") == 2
    assert result["retrieval"]["ranked_ids"]


async def test_adapter_grep_prefers_a_typed_sync_method() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class TypedGrepStores:
        def grep(self, **kwargs: Any) -> Any:
            calls.append(("grep", kwargs))
            return {"data": []}

    class TypedGrepClient:
        stores = TypedGrepStores()

        def post(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a typed stores.grep must win over the post escape hatch")

    adapter = SyncRetrievalClientAdapter(TypedGrepClient())
    request = GrepRequest(
        store_identifiers=(STORE_ID,),
        pattern="x",
        targets=("text",),
        case_sensitive=False,
        top_k=10,
    )

    result = await adapter.stores.grep(request)

    assert result == {"data": []}
    assert calls == [("grep", request.to_kwargs())]


async def test_adapter_grep_falls_back_to_the_sdk_post_escape_hatch() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class GrepLessStores:
        pass

    class GrepLessClient:
        stores = GrepLessStores()

        def post(self, path: str, *, cast_to: Any, body: Any) -> Any:
            del cast_to
            calls.append((path, dict(body)))
            return {"data": []}

    adapter = SyncRetrievalClientAdapter(GrepLessClient())
    request = GrepRequest(
        store_identifiers=(STORE_ID,),
        pattern="x",
        targets=("text",),
        case_sensitive=False,
        top_k=10,
    )

    result = await adapter.stores.grep(request)

    assert result == {"data": []}
    assert calls == [("/v1/stores/grep", request.to_kwargs())]


async def test_adapter_metadata_facets_sends_the_full_request_first() -> None:
    calls: list[dict[str, Any]] = []

    class ModernStores:
        def metadata_facets(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return {"metadata_fields": {}}

    adapter = SyncRetrievalClientAdapter(SimpleNamespace(stores=ModernStores()))
    request = MetadataFacetsRequest(store_identifiers=(STORE_ID,), facets=("year",))

    await adapter.stores.metadata_facets(request)

    assert calls == [request.to_kwargs()]


async def test_adapter_metadata_facets_falls_back_to_the_reduced_signature() -> None:
    """The older-SDK reduced call is the adapter's concern, never the async seam's."""
    calls: list[dict[str, Any]] = []

    class LegacyStores:
        def metadata_facets(self, *, store_identifiers: Any, facets: Any = None) -> Any:
            calls.append({"store_identifiers": store_identifiers, "facets": facets})
            return {"metadata_fields": {}}

    adapter = SyncRetrievalClientAdapter(SimpleNamespace(stores=LegacyStores()))
    request = MetadataFacetsRequest(store_identifiers=(STORE_ID,), facets=("year",))

    await adapter.stores.metadata_facets(request)

    assert calls == [{"store_identifiers": [STORE_ID], "facets": ["year"]}]


async def test_search_accepts_mapping_shaped_results() -> None:
    """Every seam method returns hits under "data" -- search included, so an
    in-process binding needs no attribute-bearing wrapper class."""

    class MappingStores:
        async def search(self, request: Any) -> Any:
            return {"data": [_chunk(0)]}

    class MappingClient:
        stores = MappingStores()

    chunks = await search_raw(
        "governing law", 5, store_identifiers=[STORE_ID], client=MappingClient()
    )

    assert chunks[0]["file_id"] == FILE_ID


async def test_search_accepts_an_explicit_null_data_container() -> None:
    """``data: None`` is a present-but-empty container, not a shape violation."""

    class NullDataStores:
        async def search(self, request: Any) -> Any:
            return {"data": None}

    class NullDataClient:
        stores = NullDataStores()

    chunks = await search_raw(
        "governing law", 5, store_identifiers=[STORE_ID], client=NullDataClient()
    )

    assert chunks == []


async def test_search_rejects_a_response_with_no_data_container() -> None:
    """A mis-shaped binding must fail the call loudly, not read as empty.

    Hits nested anywhere but "data" would otherwise zero retrieval on every
    rollout while each tool call reports success -- silently corrupted rollouts
    with no detector anywhere in the stack."""

    class WrongShapeStores:
        async def search(self, request: Any) -> Any:
            return {"results": [_chunk(0)]}

    class WrongShapeClient:
        stores = WrongShapeStores()

    with pytest.raises(TypeError, match="no 'data' items"):
        await search_raw(
            "governing law", 5, store_identifiers=[STORE_ID], client=WrongShapeClient()
        )
