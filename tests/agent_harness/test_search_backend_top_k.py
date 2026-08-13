from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import config
from agent_harness import search as search_execution
from agent_harness.retrieval import SyncRetrievalClientAdapter
from agent_harness.tools import functions as search_provider


class _FakeIndex:
    refs = SimpleNamespace()

    def ingest_search_results(
        self,
        candidates: list[dict[str, Any]],
        *,
        max_new_chunks: int,
        stats: dict[str, int],
    ) -> list[dict[str, Any]]:
        stats["skipped_existing_or_deleted"] = max(0, len(candidates) - max_new_chunks)
        return candidates[:max_new_chunks]


@pytest.mark.parametrize("backend_top_k", [10, 15, 20, 25])
async def test_search_corpus_backend_top_k_does_not_change_visible_limit(
    monkeypatch: pytest.MonkeyPatch,
    backend_top_k: int,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_search_corpus(query: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"results": [{"id": index} for index in range(kwargs["top_k"])]}

    monkeypatch.setattr(search_execution, "corpus_backend_top_k", lambda: backend_top_k)
    monkeypatch.setattr(search_execution, "search_corpus", fake_search_corpus)
    monkeypatch.setattr(
        search_execution,
        "serialize_agent_chunks",
        lambda chunks, **kwargs: chunks,
    )

    outcome = await search_execution.execute_search_corpus(
        {"query": "latency validation"},
        index=_FakeIndex(),
        store_identifiers=["browsecomp-plus"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert captured["top_k"] == backend_top_k
    assert payload["requested_top_k"] == 5
    assert len(payload["new_unseen_results"]) == 5
    assert payload["search_top_k"] == backend_top_k
    assert metadata["search_top_k"] == backend_top_k


def test_backend_top_k_validates_on_first_use_not_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad deployment value is a config error at the first search, never an import error."""
    monkeypatch.setenv("AGENT_HARNESS_CORPUS_BACKEND_TOP_K", "1")
    # The import half only means something in a fresh interpreter: this one has
    # already imported the package, so it can no longer fail at import.
    imported = subprocess.run(
        [sys.executable, "-c", "import agent_harness"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert imported.returncode == 0, imported.stderr

    config._env_corpus_backend_top_k.cache_clear()
    try:
        with pytest.raises(ValueError, match="AGENT_HARNESS_CORPUS_BACKEND_TOP_K"):
            config.corpus_backend_top_k()
    finally:
        config._env_corpus_backend_top_k.cache_clear()


@pytest.mark.parametrize("backend_top_k", [10, 15, 20, 25])
async def test_search_raw_sends_exact_top_k_to_mixedbread(backend_top_k: int) -> None:
    class FakeStores:
        kwargs: dict[str, Any]

        def search(self, **kwargs: Any) -> SimpleNamespace:
            self.kwargs = kwargs
            return SimpleNamespace(data=[])

    stores = FakeStores()
    client = SyncRetrievalClientAdapter(SimpleNamespace(stores=stores))

    await search_provider.search_raw(
        "latency validation",
        backend_top_k,
        store_identifiers=["browsecomp-plus"],
        client=client,
    )

    assert stores.kwargs == {
        "query": "latency validation",
        "store_identifiers": ["browsecomp-plus"],
        "top_k": backend_top_k,
        "search_options": {"return_metadata": True},
    }
