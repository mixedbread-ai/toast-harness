"""grep's post-fetch serialization is CPU-bound; it must never run on the event loop."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from agent_harness import search as search_execution
from agent_harness.config import GREP_DEFAULT_K
from agent_harness.search import ChunkIndex, chunk_key

# Long enough to clear the grep clip window, so the clip pass actually runs.
MATCHING_TEXT = "x" * 600 + "needle" + "y" * 600
# `(a+)+$` against a homogeneous run backtracks catastrophically; at this length
# the clip's 2 s bound is what stops it, which is exactly the stall being pinned.
BACKTRACKING_TEXT = "a" * 1200 + "b"
BACKTRACKING_PATTERN = r"(a+)+$"


def _chunk(i: int, text: str) -> dict[str, Any]:
    return {
        "store_id": "store-a",
        "file_id": f"file-{i}",
        "chunk_index": 0,
        "text": text,
        "search_score": 1.0 - i * 0.01,
    }


def _stub_grep_raw(monkeypatch: pytest.MonkeyPatch, chunks: list[dict[str, Any]]) -> None:
    async def fake_grep_raw(pattern: str, k: int, **kwargs: Any) -> list[dict[str, Any]]:
        del pattern, k, kwargs
        return [dict(chunk) for chunk in chunks]

    monkeypatch.setattr(search_execution, "grep_raw", fake_grep_raw)


def test_grep_result_building_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [_chunk(i, MATCHING_TEXT) for i in range(4)]
    _stub_grep_raw(monkeypatch, chunks)

    build_thread_ids: list[int] = []
    unwrapped_build = search_execution._build_grep_results

    def spy(*args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        build_thread_ids.append(threading.get_ident())
        return unwrapped_build(*args, **kwargs)

    monkeypatch.setattr(search_execution, "_build_grep_results", spy)

    loop_thread_ids: list[int] = []

    async def drive() -> search_execution.ToolOutcome:
        loop_thread_ids.append(threading.get_ident())
        return await search_execution.execute_grep(
            {"pattern": "needle"},
            index=ChunkIndex(),
            store_identifiers=["store-a"],
        )

    outcome = asyncio.run(drive())

    assert len(build_thread_ids) == 1
    assert build_thread_ids[0] != loop_thread_ids[0]

    # Same inputs through the unwrapped helper: hoisting it changed no output.
    expected_index = ChunkIndex()
    expected_new = expected_index.ingest_search_results(chunks, max_new_chunks=GREP_DEFAULT_K)
    expected_results, _ = unwrapped_build(
        chunks,
        index=expected_index,
        new_chunk_keys={chunk_key(chunk) for chunk in expected_new},
        requested_k=GREP_DEFAULT_K,
        clip_focus=search_execution._clip_regex.compile(
            "needle", search_execution._clip_regex.IGNORECASE
        ),
    )
    results = outcome.payload["results"]
    assert results == expected_results
    assert len(results) == len(chunks)
    for entry in results:
        assert "needle" in entry["text"]
        assert len(entry["text"]) < len(MATCHING_TEXT)


async def test_backtracking_grep_pattern_does_not_freeze_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_grep_raw(monkeypatch, [_chunk(0, BACKTRACKING_TEXT)])

    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    ticks = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    started = time.perf_counter()
    outcome = await search_execution.execute_grep(
        {"pattern": BACKTRACKING_PATTERN},
        index=ChunkIndex(),
        store_identifiers=["store-a"],
    )
    elapsed = time.perf_counter() - started
    stop.set()
    await ticks

    # The clip really did burn its bound: without the offload this is loop time.
    assert elapsed > 0.5
    assert gaps
    assert max(gaps) < 0.5
    assert len(outcome.payload["results"]) == 1
