"""The per-turn payload budget: min(TURN, headroom) at the round-truncation seam."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import config
from agent_harness import search as search_execution
from agent_harness.agents import searcher as searcher_module
from agent_harness.search import (
    ChunkIndex,
    chunk_key,
    estimate_payload_tokens,
    execute_get_chunks,
    truncate_round_payloads,
)


def _chunk(i: int, *, text_chars: int, file_id: str = "f") -> dict[str, Any]:
    return {
        "store_id": "s1",
        "file_id": file_id,
        "chunk_index": i,
        "text": f"chunk-{i} " + "x" * text_chars,
        "score": 0.5,
    }


@pytest.fixture
def stub_get_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_get_chunk(*, file_id: str, store_id: str, chunk_index: int, **_: Any):
        return _chunk(chunk_index, text_chars=8_000, file_id=file_id)

    monkeypatch.setattr(search_execution, "get_chunk", fake_get_chunk)


async def _restored_get_chunks_payload(index: ChunkIndex, monkeypatch: pytest.MonkeyPatch):
    """A get_chunks payload whose chunks were pruned, then restored by the call."""
    chunks = [_chunk(i, text_chars=8_000, file_id="g") for i in range(4)]
    for chunk in chunks:
        index.add_chunk(chunk)
    index.mark_pruned(chunk_keys={chunk_key(chunk) for chunk in chunks}, document_keys=())
    provider = {chunk_key(chunk): chunk for chunk in chunks}

    async def fake_get_chunk(*, file_id: str, store_id: str, chunk_index: int, **_: Any):
        return dict(provider[(store_id, file_id, chunk_index)])

    monkeypatch.setattr(search_execution, "get_chunk", fake_get_chunk)
    chunk_ids = [index.refs.chunk_id_for_key(chunk_key(chunk)) for chunk in chunks]
    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)
    assert payload["restored_chunk_ids"] == chunk_ids
    return payload, chunk_ids


async def test_single_id_get_chunks_is_exempt_from_the_turn_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The sanctioned recovery escalation must survive the round pass intact.
    index = ChunkIndex()
    chunk = _chunk(0, text_chars=40_000, file_id="g")
    index.add_chunk(chunk)

    async def fake_get_chunk(*, file_id: str, store_id: str, chunk_index: int, **_: Any):
        return dict(chunk)

    monkeypatch.setattr(search_execution, "get_chunk", fake_get_chunk)
    chunk_id = index.refs.chunk_id_for_key(chunk_key(chunk))
    payload = await execute_get_chunks({"chunk_ids": [chunk_id]}, index=index)
    snapshot = json.dumps(payload, sort_keys=True)
    other = {
        "tool": "search_corpus",
        "new_unseen_results": [
            {"chunk_id": "x1", "text": "t" * 20_000},
            {"chunk_id": "x2", "text": "t" * 20_000},
        ],
    }

    stats = truncate_round_payloads(
        [payload, other], index=index, remaining_tokens=4_000, turn_capped=True
    )

    assert stats[0] is None
    assert json.dumps(payload, sort_keys=True) == snapshot
    assert stats[1] is not None  # the sibling still absorbs the truncation


def test_turn_cap_notice_names_the_turn_budget(stub_get_chunk) -> None:
    index = ChunkIndex()
    payloads = [
        {
            "tool": "search_corpus",
            "new_unseen_results": [
                {"chunk_id": "x1", "text": "t" * 20_000},
                {"chunk_id": "x2", "text": "t" * 20_000},
            ],
        }
    ]

    stats = truncate_round_payloads(payloads, index=index, remaining_tokens=4_000, turn_capped=True)

    payload = payloads[0]
    assert stats[0] is not None
    assert stats[0]["budget_kind"] == "turn"
    notice = payload["truncation_notice"]
    assert "tool payload budget" in notice
    # The round-free remedy leads; re-searching is framed as the exception.
    assert "prune_context" in notice
    assert "runs in parallel" in notice
    assert "if needed" in notice
    assert "some of your calls" in notice
    assert len(payload["new_unseen_results"]) == 1  # top entry always survives


def test_context_cap_notice_is_unchanged(stub_get_chunk) -> None:
    index = ChunkIndex()
    payloads = [
        {
            "tool": "search_corpus",
            "new_unseen_results": [
                {"chunk_id": "x1", "text": "t" * 20_000},
                {"chunk_id": "x2", "text": "t" * 20_000},
            ],
        }
    ]

    truncate_round_payloads(payloads, index=index, remaining_tokens=4_000, turn_capped=False)

    assert "context token limit" in payloads[0]["truncation_notice"]
    assert "prune_context" in payloads[0]["truncation_notice"]


async def test_dropped_restored_chunks_return_to_pruned_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ChunkIndex()
    payload, chunk_ids = await _restored_get_chunks_payload(index, monkeypatch)
    payloads = [payload]

    stats = truncate_round_payloads(payloads, index=index, remaining_tokens=2_000, turn_capped=True)

    assert stats[0] is not None
    kept_ids = set(payload["restored_chunk_ids"])
    dropped_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in kept_ids]
    assert dropped_ids, "expected the turn pass to drop restored entries"
    for chunk_id in dropped_ids:
        key = index.refs.chunk_key_for_id(chunk_id)
        assert key in index.deleted_chunk_keys
        assert key not in index.restored_chunk_keys
    assert len(payload["restored_chunk_ids"]) == len(payload["results"])


async def test_context_mode_also_reprunes_restored_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    index = ChunkIndex()
    payload, chunk_ids = await _restored_get_chunks_payload(index, monkeypatch)
    payloads = [payload]

    truncate_round_payloads(payloads, index=index, remaining_tokens=2_000, turn_capped=False)

    kept_ids = set(payload["restored_chunk_ids"])
    dropped_ids = [chunk_id for chunk_id in chunk_ids if chunk_id not in kept_ids]
    assert dropped_ids
    for chunk_id in dropped_ids:
        key = index.refs.chunk_key_for_id(chunk_id)
        assert key in index.deleted_chunk_keys
        assert key not in index.restored_chunk_keys


async def test_dropped_never_pruned_chunks_keep_their_index_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ChunkIndex()
    chunks = [_chunk(i, text_chars=8_000, file_id="g") for i in range(4)]
    for chunk in chunks:
        index.add_chunk(chunk)  # visible, never pruned
    provider = {chunk_key(chunk): chunk for chunk in chunks}

    async def fake_get_chunk(*, file_id: str, store_id: str, chunk_index: int, **_: Any):
        return dict(provider[(store_id, file_id, chunk_index)])

    monkeypatch.setattr(search_execution, "get_chunk", fake_get_chunk)
    chunk_ids = [index.refs.chunk_id_for_key(chunk_key(chunk)) for chunk in chunks]
    payload = await execute_get_chunks({"chunk_ids": chunk_ids}, index=index)
    payloads = [payload]

    truncate_round_payloads(payloads, index=index, remaining_tokens=2_000, turn_capped=True)

    dropped_ids = set(chunk_ids) - set(payload["restored_chunk_ids"])
    assert dropped_ids, "expected the turn pass to drop entries"
    for chunk_id in dropped_ids:
        key = index.refs.chunk_key_for_id(chunk_id)
        # Dropped before the model saw them, but they were never pruned: no blacklist.
        assert key not in index.deleted_chunk_keys


async def test_round_tool_messages_uses_the_turn_cap_with_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ChunkIndex()
    payload, _ = await _restored_get_chunks_payload(index, monkeypatch)
    second = {
        "tool": "search_corpus",
        "new_unseen_results": [{"chunk_id": "y1", "text": "t" * 30_000}],
    }
    tool_calls = [SimpleNamespace(id="call-1"), SimpleNamespace(id="call-2")]
    tool_messages = {
        "call-1": {"content": json.dumps(payload)},
        "call-2": {"content": json.dumps(second)},
    }
    tool_trace = [
        {"call_id": "call-1", "output": {}, "metadata": {}},
        {"call_id": "call-2", "output": {}, "metadata": {}},
    ]
    monkeypatch.setattr(searcher_module, "TURN_TOOL_PAYLOAD_TOKEN_BUDGET", 4_000)
    monkeypatch.setattr(config, "TURN_TOOL_PAYLOAD_TOKEN_BUDGET", 4_000)

    searcher_module._truncate_round_tool_messages(
        tool_calls,
        tool_messages,
        tool_trace=tool_trace,
        index=index,
        context_tokens_baseline=0,  # headroom (100k) far above the turn cap
    )

    first = json.loads(tool_messages["call-1"]["content"])
    assert "tool payload budget" in first["truncation_notice"]
    assert tool_trace[0]["metadata"]["payload_truncated"] is True
    assert tool_trace[0]["metadata"]["budget_kind"] == "turn"
    total = estimate_payload_tokens(first) + estimate_payload_tokens(
        json.loads(tool_messages["call-2"]["content"])
    )
    assert total <= 4_000 + 200


def test_round_tool_messages_keeps_context_semantics_without_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = ChunkIndex()
    payload = {
        "tool": "search_corpus",
        "new_unseen_results": [
            {"chunk_id": "y1", "text": "t" * 20_000},
            {"chunk_id": "y2", "text": "t" * 20_000},
        ],
    }
    tool_calls = [SimpleNamespace(id="call-1")]
    tool_messages = {"call-1": {"content": json.dumps(payload)}}
    tool_trace = [{"call_id": "call-1", "output": {}, "metadata": {}}]

    searcher_module._truncate_round_tool_messages(
        tool_calls,
        tool_messages,
        tool_trace=tool_trace,
        index=index,
        context_tokens_baseline=config.SEARCHER_PROMPT_TOKEN_LIMIT - 2_000,
    )

    first = json.loads(tool_messages["call-1"]["content"])
    assert "context token limit" in first["truncation_notice"]
    assert tool_trace[0]["metadata"]["budget_kind"] == "context"
