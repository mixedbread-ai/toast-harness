"""The retrieval seam's structural contract."""

from __future__ import annotations

from typing import Any

from mixedbread import Mixedbread

from agent_harness import RetrievalClient


class _Files:
    def retrieve(self, *, file_identifier: str, store_identifier: str, return_chunks: Any) -> Any:
        return {"id": file_identifier, "store_id": store_identifier, "chunks": []}

    def list(self, **kwargs: Any) -> Any:
        return {"data": [], "pagination": {}}


class _Stores:
    def __init__(self) -> None:
        self.files = _Files()

    def search(self, **kwargs: Any) -> Any:
        return type("Results", (), {"data": []})()

    def metadata_facets(self, **kwargs: Any) -> Any:
        return {}

    def list_chunks(self, **kwargs: Any) -> Any:
        return {"data": []}


class _SDKShapedClient:
    """The member shape the Mixedbread SDK client presents to the harness."""

    def __init__(self) -> None:
        self.stores = _Stores()

    def post(self, path: str, *, cast_to: Any, body: Any) -> Any:
        return {"data": []}


class _NoPostClient:
    def __init__(self) -> None:
        self.stores = _Stores()


def test_sdk_shaped_client_is_a_retrieval_client() -> None:
    assert isinstance(_SDKShapedClient(), RetrievalClient)


def test_client_without_the_grep_escape_hatch_is_not_a_retrieval_client() -> None:
    assert not isinstance(_NoPostClient(), RetrievalClient)
    assert not isinstance(object(), RetrievalClient)


def test_real_sdk_client_satisfies_the_protocol() -> None:
    assert isinstance(Mixedbread(api_key="test-key"), RetrievalClient)
