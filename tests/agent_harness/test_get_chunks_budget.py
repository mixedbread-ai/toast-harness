"""get_chunks payload bounds: per-chunk clip, per-call spread budget, schema cap.

``execute_get_chunks`` returns fuller text than search on purpose, so it cannot
borrow the search-side ``SEARCH_CHUNK_TOKEN_LIMIT``. Without its own bounds a
single call could return an arbitrarily large payload -- the downstream prompt
truncation measures it with a heuristic that undercounts retrieval JSON.
Oversized calls are spread-truncated in requested order; nothing is deferred.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_harness import config
from agent_harness import search as search_module
from agent_harness.schemas import GetChunksArgs
from agent_harness.search import (
    ChunkIndex,
    chunk_key,
    estimate_payload_tokens,
    execute_get_chunks,
    serialize_agent_chunk,
)
from agent_harness.tools.shared import get_chunks_tool


def _chunk(i: int, *, text_chars: int, **extra: Any) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "store_id": "s1",
        "file_id": f"f{i}",
        "chunk_index": 0,
        "text": f"chunk-{i} " + "x" * text_chars,
        "score": 0.5,
    }
    chunk.update(extra)
    return chunk


@pytest.fixture
def provider_chunks() -> dict[tuple[str, str, int], dict[str, Any]]:
    return {}


@pytest.fixture
def fetched() -> list[tuple[str, str, int]]:
    return []


@pytest.fixture
def index(
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
    fetched: list[tuple[str, str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> ChunkIndex:
    async def fake_get_chunk(
        *, file_id: str, store_id: str, chunk_index: int, **_: Any
    ) -> dict[str, Any]:
        key = (store_id, file_id, chunk_index)
        fetched.append(key)
        return dict(provider_chunks[key])

    monkeypatch.setattr(search_module, "get_chunk", fake_get_chunk)
    return ChunkIndex()


def _visible(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> list[str]:
    """Make ``chunks`` reachable by handle and serveable by the stubbed provider."""
    chunk_ids = []
    for chunk in chunks:
        index.add_chunk(chunk)
        key = chunk_key(chunk)
        provider_chunks[key] = chunk
        chunk_ids.append(index.refs.chunk_id_for_key(key))
    return chunk_ids


async def test_small_request_payload_is_unchanged(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    chunks = [_chunk(i, text_chars=800) for i in range(3)]
    chunk_ids = _visible(index, provider_chunks, chunks)

    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)

    assert list(payload) == ["tool", "requested_chunk_ids", "results", "restored_chunk_ids"]
    assert payload["restored_chunk_ids"] == chunk_ids
    assert payload["results"] == [serialize_agent_chunk(chunk, refs=index.refs) for chunk in chunks]


async def test_oversized_call_spreads_truncation_in_requested_order(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
    fetched: list[tuple[str, str, int]],
) -> None:
    chunks = [_chunk(i, text_chars=30_000) for i in range(5)]
    chunk_ids = _visible(index, provider_chunks, chunks)

    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)

    # Everything requested comes back -- no deferral -- in requested order.
    assert payload["restored_chunk_ids"] == chunk_ids
    assert [result["chunk_id"] for result in payload["results"]] == chunk_ids
    assert len(fetched) == len(chunk_ids)
    # Earlier ids keep more text; every chunk keeps at least the floor. The head
    # chunk fits its share unclipped, so the budget truncates the remaining four.
    kept_lengths = [len(result["text"]) for result in payload["results"]]
    assert kept_lengths == sorted(kept_lengths, reverse=True)
    assert all(length >= config.MIN_ALLOCATION_TOKENS for length in kept_lengths)
    assert payload["clipped_chunk_count"] == 4
    assert estimate_payload_tokens(payload) <= config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET + 100
    notice = payload["budget_notice"]
    assert "4 chunks had to be truncated" in notice
    assert f"{config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET}-token payload budget" in notice
    # Proceed-first wording: continuing is the default, escalation the exception.
    assert "proceed with the context provided" in notice
    assert "prune_context" in notice
    assert "if needed" in notice
    assert "re-requested alone is shown in full" in notice


async def test_single_oversized_chunk_is_clipped_with_a_quantified_marker(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    oversized = _chunk(0, text_chars=400_000)
    chunk_ids = _visible(index, provider_chunks, [oversized, _chunk(1, text_chars=800)])

    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)

    assert payload["clipped_chunk_ids"] == [chunk_ids[0]]
    assert payload["clipped_chunk_count"] == 1
    clipped_text = payload["results"][0]["text"]
    assert "[... truncated: showing first " in clipped_text
    assert clipped_text.endswith(f" of {len(oversized['text'])} characters]")
    assert len(clipped_text) < len(oversized["text"])
    # Clipping the pathological chunk is what leaves the sibling affordable;
    # the per-call budget never engages, so there is no budget notice.
    assert payload["restored_chunk_ids"] == chunk_ids
    assert "budget_notice" not in payload
    assert estimate_payload_tokens(payload) <= config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET


async def test_metadata_monster_is_elided_and_flagged(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    monster = _chunk(0, text_chars=800, metadata={"blob": "m" * 400_000, "keep": "v"})
    chunk_ids = _visible(index, provider_chunks, [monster, _chunk(1, text_chars=800)])

    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)

    result = payload["results"][0]
    assert result["metadata_clipped"] is True
    metadata = result["metadata"]
    # The string-clip pass converges on the cap, so the blob stays a (shortened) string.
    assert isinstance(metadata["blob"], str)
    assert len(metadata["blob"]) < 400_000
    assert metadata["blob"].endswith("characters]")
    assert metadata["keep"] == "v"
    assert estimate_payload_tokens(payload) <= config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET


async def test_long_tail_of_small_metadata_values_cannot_break_the_budget(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    # 45 values just under the clip floor each: unclippable individually, but the
    # elision and field-drop passes must still land the call under budget.
    chunks = [
        _chunk(i, text_chars=800, metadata={f"field_{j}": "v" * 190 for j in range(45)})
        for i in range(20)
    ]
    chunk_ids = _visible(index, provider_chunks, chunks)

    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)

    assert estimate_payload_tokens(payload) <= config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET + 100
    assert len(payload["results"]) == len(chunk_ids)  # nothing dropped, everything bounded


async def test_clipped_chunk_is_recovered_in_full_by_a_single_id_rerequest(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    chunks = [_chunk(i, text_chars=60_000) for i in range(5)]
    chunk_ids = _visible(index, provider_chunks, chunks)

    first = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)
    assert chunk_ids[2] in first["clipped_chunk_ids"]

    # The marker's invitation must be satisfiable: narrowing to one id waives the
    # per-chunk cap, so the escalation returns the full text.
    second = await execute_get_chunks({"chunk_ids": [chunk_ids[2]]}, index=index)
    assert second["results"][0]["text"] == chunks[2]["text"]
    assert "clipped_chunk_ids" not in second
    assert "budget_notice" not in second


async def test_single_id_over_the_call_budget_gets_the_truthful_ceiling_notice(
    index: ChunkIndex,
    provider_chunks: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    giant = _chunk(0, text_chars=400_000)
    chunk_ids = _visible(index, provider_chunks, [giant])

    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)

    assert payload["clipped_chunk_ids"] == chunk_ids
    notice = payload["budget_notice"]
    assert "no get_chunks request can show more" in notice
    assert "read_document" in notice
    assert estimate_payload_tokens(payload) <= config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET + 100


def test_chunk_ids_bound_is_enforced_and_documented() -> None:
    function_schema = get_chunks_tool()["function"]
    chunk_ids_schema = function_schema["parameters"]["properties"]["chunk_ids"]

    assert chunk_ids_schema["maxItems"] == config.GET_CHUNKS_MAX_CHUNK_IDS
    assert str(config.GET_CHUNKS_MAX_CHUNK_IDS) in function_schema["description"]

    at_limit = [f"c{i}" for i in range(config.GET_CHUNKS_MAX_CHUNK_IDS)]
    assert GetChunksArgs(chunk_ids=at_limit).chunk_ids == at_limit
    with pytest.raises(ValidationError):
        GetChunksArgs(chunk_ids=[*at_limit, "c-one-too-many"])
