"""Unresolvable chunk/document ids must surface as agent-caused tool errors.

``execute_get_chunks``/``execute_read_document`` report bad ids inside their
return payload instead of raising, so the trace event used to be finalized as
``status="success"``, so a consumer scoring the trace never saw them -- unlike
``prune_context``, which reports the same mistake as an error.
"""

from __future__ import annotations

from typing import Any

from agent_harness.agents.shared import agent_caused_payload_error
from agent_harness.search import ChunkIndex, execute_get_chunks, execute_read_document


def _chunk(store: str = "s1", file: str = "f1", index: int = 0) -> dict[str, Any]:
    return {
        "store_id": store,
        "file_id": file,
        "chunk_index": index,
        "text": "hello world",
        "score": 0.5,
    }


def _index_with_one_chunk() -> tuple[ChunkIndex, str]:
    index = ChunkIndex()
    chunk = _chunk()
    index.add_chunk(chunk)
    chunk_id = index.refs.chunk_id_for_key(("s1", "f1", 0))
    return index, chunk_id


async def test_get_chunks_flags_unknown_id_as_invalid() -> None:
    index, _ = _index_with_one_chunk()

    payload = await execute_get_chunks({"chunk_ids": ["c-nope"]}, index=index)

    assert payload["invalid_chunk_ids"] == ["c-nope"]
    assert agent_caused_payload_error(payload) is not None


async def test_get_chunks_without_bad_ids_reports_no_error() -> None:
    index, _ = _index_with_one_chunk()

    payload = await execute_get_chunks({"chunk_ids": []}, index=index)

    assert "invalid_chunk_ids" not in payload
    assert agent_caused_payload_error(payload) is None


async def test_read_document_flags_unknown_id_as_invalid_request() -> None:
    index, _ = _index_with_one_chunk()

    payload = await execute_read_document(
        {"document_id": "d-nope", "chunk_id": "c-nope"},
        index=index,
    )

    assert payload["invalid_request"] is True
    assert agent_caused_payload_error(payload) == payload["error"]


async def test_read_document_flags_chunk_document_mismatch() -> None:
    index = ChunkIndex()
    index.add_chunk(_chunk(file="f1", index=0))
    index.add_chunk(_chunk(file="f2", index=0))
    document_id = index.refs.document_id_for_key(("s1", "f1"))
    other_chunk_id = index.refs.chunk_id_for_key(("s1", "f2", 0))

    payload = await execute_read_document(
        {"document_id": document_id, "chunk_id": other_chunk_id},
        index=index,
    )

    assert payload["invalid_request"] is True
    assert "does not belong" in payload["error"]


def test_store_side_chunk_not_found_is_not_the_models_mistake() -> None:
    """A resolvable, visible id whose chunk the store cannot return is not agent-caused."""
    payload = {
        "tool": "get_chunks",
        "results": [{"chunk_id": "c01", "error": "Chunk not found"}],
    }

    assert agent_caused_payload_error(payload) is None


def test_helper_ignores_non_mapping_payloads() -> None:
    assert agent_caused_payload_error(None) is None
    assert agent_caused_payload_error("boom") is None
