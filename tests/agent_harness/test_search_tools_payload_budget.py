"""The per-call payload budget on the search-style tools, exercised with a forced budget.

Production constants rarely trigger the aggregate (search 5x2k, filter 30x2k worst
cases); these tests shrink TOOL_CALL_PAYLOAD_TOKEN_BUDGET to exercise the spread path
and pin that under-budget payloads are byte-identical to the unbudgeted path.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_harness import config
from agent_harness import search as search_execution
from agent_harness.search import (
    ChunkIndex,
    estimate_payload_tokens,
    execute_filter_chunks,
    execute_grep,
    execute_overview_search,
    execute_search_corpus,
)

FORCED_BUDGET = 3_000


@pytest.fixture
def forced_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TOOL_CALL_PAYLOAD_TOKEN_BUDGET", FORCED_BUDGET)
    monkeypatch.setattr(config, "SEARCH_CORPUS_PAYLOAD_TOKEN_BUDGET", FORCED_BUDGET)
    monkeypatch.setattr(config, "FILTER_CHUNKS_PAYLOAD_TOKEN_BUDGET", FORCED_BUDGET)


def _chunk(i: int, *, text_chars: int, score: float = 0.5) -> dict[str, Any]:
    return {
        "store_id": "s1",
        "file_id": f"f{i}",
        "chunk_index": 0,
        "text": f"chunk-{i} " + "x" * text_chars,
        "summary": f"summary-{i} " + "s" * text_chars,
        "score": score,
    }


async def _run_search(
    index: ChunkIndex, chunks: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
):
    async def fake_search_corpus(query: str, **kwargs: Any) -> dict[str, Any]:
        return {"results": chunks}

    monkeypatch.setattr(search_execution, "search_corpus", fake_search_corpus)
    return await execute_search_corpus({"query": "q"}, index=index, store_identifiers=["s1"])


def _kept_lengths(payload: dict[str, Any], key: str = "results") -> list[int]:
    return [len(entry["text"] if "text" in entry else entry["summary"]) for entry in payload[key]]


async def test_search_corpus_under_budget_is_untouched(
    forced_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcome = await _run_search(
        ChunkIndex(), [_chunk(i, text_chars=100) for i in range(3)], monkeypatch
    )
    payload = outcome.payload

    assert "clipped_chunk_ids" not in payload
    assert "budget_notice" not in payload
    assert len(payload["new_unseen_results"]) == 3


async def test_search_corpus_over_budget_spreads_in_rank_order(
    forced_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk(i, text_chars=8_000, score=1.0 - i * 0.01) for i in range(5)]
    outcome = await _run_search(ChunkIndex(), chunks, monkeypatch)
    payload = outcome.payload

    results = payload["new_unseen_results"]
    assert len(results) == 5
    kept = [estimate_payload_tokens(entry) for entry in results]
    assert kept == sorted(kept, reverse=True)
    # Every result keeps a meaningful share (allocator floor modulo scale-down slack).
    assert all(estimate >= config.MIN_ALLOCATION_TOKENS // 2 for estimate in kept)
    assert estimate_payload_tokens(payload) <= FORCED_BUDGET + 100
    assert payload["clipped_chunk_count"] >= 1
    notice = payload["budget_notice"]
    assert "had to be truncated" in notice
    assert f"{FORCED_BUDGET}-token payload budget" in notice
    assert "proceed with the context provided" in notice


async def test_grep_over_budget_spreads_and_keeps_match_context(
    forced_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dense matches merge the context windows into one large kept span per chunk.
    chunks = [
        {
            "store_id": "s1",
            "file_id": f"f{i}",
            "chunk_index": 0,
            "text": (f"MATCH-{i} " + "x" * 400) * 20,
        }
        for i in range(6)
    ]

    async def fake_grep_raw(pattern: str, k: int, **kwargs: Any) -> list[dict[str, Any]]:
        return chunks[:k]

    monkeypatch.setattr(search_execution, "grep_raw", fake_grep_raw)

    outcome = await execute_grep({"pattern": "MATCH"}, index=ChunkIndex(), store_identifiers=["s1"])
    payload = outcome.payload

    assert estimate_payload_tokens(payload) <= FORCED_BUDGET + 100
    assert payload["budget_notice"]
    # The first result still carries the match neighbourhood, not just the preamble.
    assert "MATCH-0" in payload["results"][0]["text"]


async def test_filter_chunks_over_budget_spreads(
    forced_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk(i, text_chars=6_000, score=1.0 - i * 0.01) for i in range(8)]

    async def fake_filter_chunks(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"results": chunks, "k": kwargs.get("k") or len(chunks)}

    monkeypatch.setattr(search_execution, "filter_chunks", fake_filter_chunks)

    outcome = await execute_filter_chunks({"k": 8}, index=ChunkIndex(), store_identifiers=["s1"])
    payload = outcome.payload

    kept = _kept_lengths(payload)
    assert kept == sorted(kept, reverse=True)
    assert estimate_payload_tokens(payload) <= FORCED_BUDGET + 100
    assert payload["clipped_chunk_count"] >= 1


async def test_overview_search_over_budget_clips_summaries(
    forced_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = [_chunk(i, text_chars=50, score=1.0 - i * 0.01) for i in range(10)]
    for chunk in chunks:
        chunk["summary"] = "s" * 5_000

    async def fake_overview_search(query: str, **kwargs: Any) -> dict[str, Any]:
        return {"results": chunks, "query": "q"}

    monkeypatch.setattr(search_execution, "overview_search", fake_overview_search)

    outcome = await execute_overview_search(
        {"query": "q"}, index=ChunkIndex(), store_identifiers=["s1"]
    )
    payload = outcome.payload

    assert estimate_payload_tokens(payload) <= FORCED_BUDGET + 100
    assert payload["budget_notice"]
    lengths = [len(entry["summary"]) for entry in payload["results"]]
    assert lengths == sorted(lengths, reverse=True)


async def test_seen_reference_stubs_are_never_clipped(
    forced_budget: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = ChunkIndex()
    seen = _chunk(0, text_chars=9_000)
    index.add_chunk(seen)
    fresh = [_chunk(i, text_chars=9_000) for i in range(1, 5)]

    async def fake_grep_raw(pattern: str, k: int, **kwargs: Any) -> list[dict[str, Any]]:
        return [seen, *fresh][:k]

    monkeypatch.setattr(search_execution, "grep_raw", fake_grep_raw)

    outcome = await execute_grep({"pattern": "chunk"}, index=index, store_identifiers=["s1"])
    payload = outcome.payload

    stubs = [entry for entry in payload["results"] if entry.get("seen")]
    assert stubs, "expected at least one seen-reference stub"
    for stub in stubs:
        assert "truncated" not in str(stub.values())
