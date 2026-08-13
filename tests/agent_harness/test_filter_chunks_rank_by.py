"""Tests for filter_chunks rank_by fallback when a field is not numeric."""

from __future__ import annotations

import pytest

from agent_harness.search import ChunkIndex, execute_filter_chunks
from agent_harness.tools import functions
from agent_harness.tools.functions import filter_chunks


@pytest.fixture
def stub_chunks(monkeypatch: pytest.MonkeyPatch):
    """Serve a fixed chunk list from list_chunks_raw, recording request kwargs."""

    def install(chunks: list[dict]) -> list[dict]:
        calls: list[dict] = []

        async def fake_list_chunks_raw(**kwargs):
            calls.append(kwargs)
            return [dict(chunk) for chunk in chunks]

        monkeypatch.setattr(functions, "list_chunks_raw", fake_list_chunks_raw)
        return calls

    return install


def _chunk(file_id: str, chunk_index: int, metadata: dict) -> dict:
    return {
        "store_id": "store-a",
        "file_id": file_id,
        "chunk_index": chunk_index,
        "metadata": metadata,
    }


async def test_non_numeric_rank_by_falls_back_to_deterministic_order(stub_chunks) -> None:
    stub_chunks(
        [
            _chunk("file-c", 0, {"ad_id": "1456043385917139", "fatigue_tier": "severe"}),
            _chunk("file-a", 1, {"ad_id": "1456043385917140", "fatigue_tier": "severe"}),
            _chunk("file-b", 0, {"ad_id": "1456043385917141", "fatigue_tier": "severe"}),
        ]
    )

    out = await filter_chunks(
        filter_by=[{"key": "fatigue_tier", "operator": "eq", "value": "severe"}],
        filter_mode="all",
        rank_by="ad_id",
        direction="asc",
        k=5,
        store_identifiers=["nike-demo"],
    )

    assert [chunk["file_id"] for chunk in out["results"]] == ["file-a", "file-b", "file-c"]
    assert out["rank_by"] == "ad_id"
    assert out["rank_by_applied"] is False
    assert out["rank_by_non_numeric_count"] == 3
    assert out["candidate_count"] == 3


async def test_partially_numeric_rank_by_keeps_non_numeric_remainder(stub_chunks) -> None:
    stub_chunks(
        [
            _chunk("file-z", 0, {"spend": 100.0}),
            _chunk("file-b", 0, {"spend": "n/a"}),
            _chunk("file-y", 0, {"spend": 900.0}),
            _chunk("file-a", 2, {}),
            _chunk("file-a", 1, {}),
        ]
    )

    out = await filter_chunks(
        filter_by=[],
        rank_by="spend",
        direction="desc",
        k=5,
        store_identifiers=["store-a"],
    )

    assert [(chunk["file_id"], chunk["chunk_index"]) for chunk in out["results"]] == [
        ("file-y", 0),
        ("file-z", 0),
        ("file-a", 1),
        ("file-a", 2),
        ("file-b", 0),
    ]
    assert out["rank_by_applied"] is True
    assert out["rank_by_non_numeric_count"] == 3
    assert out["candidate_count"] == 5


async def test_numeric_rank_by_orders_numerically(stub_chunks) -> None:
    stub_chunks(
        [
            _chunk("file-a", 0, {"ad_name": "low", "spend": 100.0}),
            _chunk("file-b", 0, {"ad_name": "high", "spend": 900.0}),
            _chunk("file-c", 0, {"ad_name": "mid", "spend": 500.0}),
        ]
    )

    out = await filter_chunks(
        filter_by=[],
        rank_by="spend",
        direction="desc",
        k=2,
        store_identifiers=["store-a"],
    )

    assert [chunk["metadata"]["ad_name"] for chunk in out["results"]] == ["high", "mid"]
    assert out["direction"] == "desc"
    assert out["rank_by_applied"] is True
    assert out["rank_by_non_numeric_count"] == 0
    assert out["candidate_count"] == 3


async def test_omitted_rank_by_reports_no_rank_keys(stub_chunks) -> None:
    stub_chunks(
        [
            _chunk("file-c", 0, {"spend": 100.0}),
            _chunk("file-a", 0, {"spend": 900.0}),
        ]
    )

    out = await filter_chunks(
        filter_by=[],
        k=5,
        store_identifiers=["store-a"],
    )

    assert [chunk["file_id"] for chunk in out["results"]] == ["file-a", "file-c"]
    assert "rank_by" not in out
    assert "direction" not in out
    assert "rank_by_applied" not in out
    assert "rank_by_non_numeric_count" not in out


async def test_agent_payload_reports_ignored_rank_by(stub_chunks) -> None:
    stub_chunks(
        [
            _chunk("file-a", 0, {"ad_id": "1456043385917139", "fatigue_tier": "severe"}),
            _chunk("file-b", 0, {"ad_id": "1456043385917140", "fatigue_tier": "severe"}),
        ]
    )

    outcome = await execute_filter_chunks(
        {
            "filter_by": [{"key": "fatigue_tier", "operator": "eq", "value": "severe"}],
            "filter_mode": "all",
            "rank_by": "ad_id",
            "direction": "asc",
            "k": 5,
        },
        index=ChunkIndex(),
        store_identifiers=["nike-demo"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert payload["rank_by_applied"] is False
    assert payload["rank_by_non_numeric_count"] == 2
    assert metadata["rank_by_applied"] is False
    assert len(payload["results"]) == 2
