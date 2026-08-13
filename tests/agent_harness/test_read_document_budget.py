"""read_document's 32k payload budget: anchor-priority spread and the x bound."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from agent_harness import config
from agent_harness import search as search_execution
from agent_harness.schemas import ReadDocumentArgs
from agent_harness.search import (
    ChunkIndex,
    chunk_key,
    estimate_payload_tokens,
    execute_read_document,
)
from agent_harness.tools.shared import READ_DOCUMENT_TOOL


def _seed_document(
    index: ChunkIndex,
    provider: dict[tuple[str, str, int], dict[str, Any]],
    *,
    n_chunks: int,
    text_chars: int,
    file_id: str = "doc1",
    register: bool = True,
) -> list[dict[str, Any]]:
    """Seed the provider. ``register=False`` leaves the chunks unseen by the index,
    so a window over them reports them as new."""
    chunks = []
    for i in range(n_chunks):
        chunk = {
            "store_id": "s1",
            "file_id": file_id,
            "chunk_index": i,
            "text": f"chunk-{i} " + "x" * text_chars,
            "score": 0.5,
        }
        if register:
            index.add_chunk(chunk)
        provider[chunk_key(chunk)] = chunk
        chunks.append(chunk)
    return chunks


@pytest.fixture
def provider() -> dict[tuple[str, str, int], dict[str, Any]]:
    return {}


@pytest.fixture
def stub_provider(provider, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_read_document(*, file_id: str, store_id: str, chunk_indices: list[int], **_):
        return {
            "chunks": [
                dict(provider[(store_id, file_id, chunk_index)])
                for chunk_index in chunk_indices
                if (store_id, file_id, chunk_index) in provider
            ],
            "filename": "doc1.txt",
            "status": "completed",
        }

    monkeypatch.setattr(search_execution, "read_document", fake_read_document)


def _handles(index: ChunkIndex, chunks: list[dict[str, Any]], anchor: int) -> tuple[str, str]:
    anchor_key = chunk_key(chunks[anchor])
    document_id = index.refs.document_id_for_key((anchor_key[0], anchor_key[1]))
    return document_id, index.refs.chunk_id_for_key(anchor_key)


async def test_small_window_is_unchanged(provider, stub_provider) -> None:
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=3, text_chars=800)
    document_id, anchor_id = _handles(index, chunks, 1)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 1}, index=index
    )

    assert "budget_notice" not in payload
    assert "clipped_chunk_ids" not in payload
    # chunks[] is the only text carrier; no duplicate document blob.
    assert "content" not in payload
    assert len(payload["chunks"]) == 3
    assert all("truncated" not in str(chunk.get("text", "")) for chunk in payload["chunks"])


async def test_over_budget_window_spreads_anchor_first(provider, stub_provider) -> None:
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=11, text_chars=20_000)
    document_id, anchor_id = _handles(index, chunks, 5)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 5}, index=index
    )

    assert estimate_payload_tokens(payload) <= config.READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET + 150
    returned = payload["chunks"]
    assert len(returned) == 11  # nothing deferred, nothing dropped
    # Anchor keeps the most text; kept length is non-increasing in distance from it.
    kept = {chunk["chunk_index"]: len(chunk["text"]) for chunk in returned}
    distances = sorted(kept, key=lambda position: (abs(position - 5), position))
    assert [kept[position] for position in distances] == sorted(
        [kept[position] for position in distances], reverse=True
    )
    assert payload["budget_notice"]
    assert "content" not in payload


async def test_far_over_budget_window_keeps_every_chunk(provider, stub_provider) -> None:
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=3, text_chars=60_000)
    document_id, anchor_id = _handles(index, chunks, 1)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 1}, index=index
    )

    assert estimate_payload_tokens(payload) <= config.READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET + 150
    # No blob to sacrifice first, so the spread alone has to fit the budget
    # without starving any chunk out of the window.
    assert len(payload["chunks"]) == 3
    assert all(chunk["text"] for chunk in payload["chunks"])


async def test_budget_is_spent_on_chunks_not_on_duplicate_carriers(provider, stub_provider) -> None:
    """chunks[] is the only text carrier, so nearly the whole budget reaches the model."""
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=5, text_chars=40_000)
    document_id, anchor_id = _handles(index, chunks, 2)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 2}, index=index
    )

    budget = config.READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET
    assert estimate_payload_tokens(payload) <= budget + 150
    shown = sum(len(chunk["text"]) for chunk in payload["chunks"])
    # No blob and no mirror: the envelope is the only thing between the budget and
    # the text, so at least 3/4 of the budget lands in chunks[].
    assert shown > budget * config.TOKEN_ESTIMATE_CHARS_PER_TOKEN * 3 // 4


async def test_new_window_chunks_are_reported_as_ids_only(provider, stub_provider) -> None:
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=3, text_chars=800, register=False)
    index.add_chunk(chunks[1])  # only the anchor arrived via search
    document_id, anchor_id = _handles(index, chunks, 1)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 1}, index=index
    )

    new_ids = payload["new_unseen_chunk_ids"]
    window_ids = [chunk["chunk_id"] for chunk in payload["chunks"]]
    assert all(isinstance(chunk_id, str) for chunk_id in new_ids)
    # The neighbours are new; the anchor was already seen. Every id points into chunks[].
    assert set(new_ids) == set(window_ids) - {anchor_id}
    # The signal costs ids, not a second copy of the text.
    wire = json.dumps(payload)
    assert all(wire.count(chunk["text"][:40]) == 1 for chunk in payload["chunks"])


def test_window_bound_is_enforced_and_documented() -> None:
    assert ReadDocumentArgs(document_id="d1", chunk_id="c1", x=config.READ_DOCUMENT_MAX_WINDOW)
    with pytest.raises(ValidationError):
        ReadDocumentArgs(document_id="d1", chunk_id="c1", x=config.READ_DOCUMENT_MAX_WINDOW + 1)

    x_schema = READ_DOCUMENT_TOOL["function"]["parameters"]["properties"]["x"]
    assert x_schema["maximum"] == config.READ_DOCUMENT_MAX_WINDOW
    assert str(config.READ_DOCUMENT_MAX_WINDOW) in x_schema["description"]


async def test_raw_only_fields_do_not_break_the_budget_math(provider, stub_provider) -> None:
    # generated_metadata rides the stored chunk but is dropped by serialization;
    # measuring the raw chunk would chase phantom mass.
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=11, text_chars=12_000)
    for chunk in chunks:
        chunk["generated_metadata"] = {"extraction": "g" * 20_000}
        provider[chunk_key(chunk)] = chunk
        index.add_chunk(chunk)
    document_id, anchor_id = _handles(index, chunks, 5)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 5}, index=index
    )

    assert estimate_payload_tokens(payload) <= config.READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET + 150
    assert payload["budget_notice"]


async def test_empty_window_with_document_metadata_is_still_bounded(
    provider, stub_provider
) -> None:
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=3, text_chars=400)
    index.mark_pruned(chunk_keys={chunk_key(chunk) for chunk in chunks}, document_keys=())
    document_id, anchor_id = _handles(index, chunks, 1)

    async def fat_metadata_read(*, file_id: str, store_id: str, chunk_indices: list[int], **_):
        return {
            "chunks": [],
            "filename": "doc1.txt",
            "status": "completed",
            "metadata": {"blob": "m" * 400_000},
        }

    provider.clear()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(search_execution, "read_document", fat_metadata_read)
    try:
        payload = await execute_read_document(
            {"document_id": document_id, "chunk_id": anchor_id, "x": 1}, index=index
        )
    finally:
        monkeypatch.undo()

    assert estimate_payload_tokens(payload) <= config.READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET + 150


async def test_pruned_anchor_allocates_from_its_neighbour(provider, stub_provider) -> None:
    index = ChunkIndex()
    chunks = _seed_document(index, provider, n_chunks=11, text_chars=20_000)
    anchor_key = chunk_key(chunks[5])
    index.mark_pruned(chunk_keys={anchor_key}, document_keys=())
    document_id, anchor_id = _handles(index, chunks, 5)

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": anchor_id, "x": 5}, index=index
    )

    returned = {chunk["chunk_index"]: chunk for chunk in payload["chunks"]}
    assert 5 not in returned  # the pruned anchor is not in the window
    # The anchor's immediate neighbours keep the most text, not chunk 0.
    assert len(returned[4]["text"]) >= len(returned[0]["text"])
    assert len(returned[6]["text"]) >= len(returned[10]["text"])
