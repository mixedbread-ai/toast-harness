"""AGENT_HARNESS_SEARCH_RERANK must reach every stores.search call, and only when set."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import config
from agent_harness.retrieval import SyncRetrievalClientAdapter
from agent_harness.tools import functions as search_provider


@pytest.fixture
def fresh_rerank_cache() -> Any:
    config.search_rerank.cache_clear()
    yield
    config.search_rerank.cache_clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("1", True),
        ("true", True),
        ("YES", True),
        (
            "mixedbread-ai/mxbai-rerank-v3-listwise",
            {"model": "mixedbread-ai/mxbai-rerank-v3-listwise"},
        ),
    ],
)
def test_search_rerank_parses_env(
    monkeypatch: pytest.MonkeyPatch,
    fresh_rerank_cache: Any,
    raw: str | None,
    expected: bool | dict[str, Any],
) -> None:
    if raw is None:
        monkeypatch.delenv("AGENT_HARNESS_SEARCH_RERANK", raising=False)
    else:
        monkeypatch.setenv("AGENT_HARNESS_SEARCH_RERANK", raw)
    assert config.search_rerank() == expected


@pytest.mark.parametrize(
    ("rerank", "expected_options"),
    [
        (False, {"return_metadata": True}),
        (True, {"return_metadata": True, "rerank": True}),
        (
            {"model": "mixedbread-ai/mxbai-rerank-v3-listwise"},
            {
                "return_metadata": True,
                "rerank": {"model": "mixedbread-ai/mxbai-rerank-v3-listwise"},
            },
        ),
    ],
)
async def test_search_raw_sends_rerank_option(
    monkeypatch: pytest.MonkeyPatch,
    rerank: bool | dict[str, Any],
    expected_options: dict[str, Any],
) -> None:
    class FakeStores:
        kwargs: dict[str, Any]

        def search(self, **kwargs: Any) -> SimpleNamespace:
            self.kwargs = kwargs
            return SimpleNamespace(data=[])

    stores = FakeStores()
    client = SyncRetrievalClientAdapter(SimpleNamespace(stores=stores))
    monkeypatch.setattr(search_provider, "search_rerank", lambda: rerank)

    await search_provider.search_raw(
        "rerank wiring",
        5,
        store_identifiers=["browsecomp-plus"],
        client=client,
    )

    assert stores.kwargs["search_options"] == expected_options
