"""A narrowed grep that finds nothing says so when the other target bucket matches."""

from __future__ import annotations

import pytest

from agent_harness import search as search_runtime
from agent_harness.search import ChunkIndex, execute_grep


def _chunk(file_id: str, text: str) -> dict:
    return {
        "store_id": "gdp-pdf",
        "file_id": file_id,
        "chunk_index": 0,
        "ocr_text": text,
    }


@pytest.fixture
def stub_grep(monkeypatch: pytest.MonkeyPatch):
    """Serve grep hits per target bucket, recording every bucket that was queried."""

    def install(hits_by_target: dict[str, list[dict]]) -> list[list[str]]:
        queried: list[list[str]] = []

        async def fake_grep_raw(pattern, k, *, targets, **kwargs):
            queried.append(list(targets))
            hits: list[dict] = []
            for target in targets:
                hits.extend(hits_by_target.get(target) or [])
            return hits[:k]

        monkeypatch.setattr(search_runtime, "grep_raw", fake_grep_raw)
        return queried

    return install


async def test_empty_narrowed_grep_points_at_the_matching_bucket(stub_grep) -> None:
    queried = stub_grep({"generated": [_chunk("file-a", "Insurance")]})

    outcome = await execute_grep(
        {"pattern": "Insurance", "targets": ["text"]},
        index=ChunkIndex(),
        store_identifiers=["gdp-pdf"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert payload["results"] == []
    assert "targets=['generated']" in payload["targets_note"]
    assert metadata["targets_probe"] == "matched_other_bucket"
    assert queried == [["text"], ["generated"]]


async def test_empty_grep_on_both_buckets_adds_no_note(stub_grep) -> None:
    queried = stub_grep({})

    outcome = await execute_grep(
        {"pattern": "Insurance"},
        index=ChunkIndex(),
        store_identifiers=["gdp-pdf"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert "targets_note" not in payload
    # Nothing was narrowed away, so there is no other bucket to probe.
    assert queried == [["text", "generated"]]
    assert "targets_probe" not in metadata


async def test_empty_narrowed_grep_with_no_matches_anywhere_adds_no_note(stub_grep) -> None:
    stub_grep({})

    outcome = await execute_grep(
        {"pattern": "Insurance", "targets": ["text"]},
        index=ChunkIndex(),
        store_identifiers=["gdp-pdf"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert "targets_note" not in payload
    assert metadata["targets_probe"] == "no_match"


async def test_narrowed_grep_with_hits_does_not_probe_the_other_bucket(stub_grep) -> None:
    queried = stub_grep({"generated": [_chunk("file-a", "Insurance")]})

    outcome = await execute_grep(
        {"pattern": "Insurance", "targets": ["generated"]},
        index=ChunkIndex(),
        store_identifiers=["gdp-pdf"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert len(payload["results"]) == 1
    assert "targets_note" not in payload
    assert queried == [["generated"]]
    assert "targets_probe" not in metadata


async def test_probe_failure_returns_the_successful_empty_result(monkeypatch) -> None:
    async def fake_grep_raw(pattern, k, *, targets, **kwargs):
        if list(targets) == ["text"]:
            return []
        raise ConnectionError("provider down")

    monkeypatch.setattr(search_runtime, "grep_raw", fake_grep_raw)

    outcome = await execute_grep(
        {"pattern": "Insurance", "targets": ["text"]},
        index=ChunkIndex(),
        store_identifiers=["gdp-pdf"],
    )
    payload, metadata = outcome.payload, outcome.query

    assert payload["results"] == []
    assert "targets_note" not in payload
    assert metadata["targets_probe"] == "error"


async def test_probe_forwards_case_and_filters(monkeypatch) -> None:
    probed: list[dict] = []

    async def fake_grep_raw(pattern, k, *, targets, **kwargs):
        if list(targets) == ["text"]:
            return []
        probed.append({"k": k, **kwargs})
        return [_chunk("file-a", "Insurance")]

    monkeypatch.setattr(search_runtime, "grep_raw", fake_grep_raw)
    filter_by = [{"field": "year", "operator": "eq", "value": 2024}]

    outcome = await execute_grep(
        {
            "pattern": "Insurance",
            "targets": ["text"],
            "case_sensitive": True,
            "filter_by": filter_by,
            "filter_mode": "any",
        },
        index=ChunkIndex(),
        store_identifiers=["gdp-pdf"],
    )
    payload = outcome.payload

    assert "targets_note" in payload
    assert len(probed) == 1
    assert probed[0]["k"] == 1
    assert probed[0]["case_sensitive"] is True
    assert probed[0]["filter_by"] == filter_by
    assert probed[0]["filter_mode"] == "any"
