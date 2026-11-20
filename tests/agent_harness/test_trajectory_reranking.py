"""Final reranking uses the evidence Toast saw, independently of its submission."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import mixedbread
import pytest
import replay_scenarios as rs

from agent_harness.agents.searcher import run_fast_agentic_search
from agent_harness.llm import sync_generation_as_async
from agent_harness.reranking import SearchTrajectory, rerank_document
from agent_harness.retrieval import SyncRetrievalClientAdapter
from agent_harness.search import ChunkIndex
from agent_harness.testing import ScriptedGeneration, response, tool_call


async def run(*, strict=True):
    def prune(_messages):
        return response(
            [tool_call("prune", "prune_context", {"chunk_ids": ["c2"]})],
            content="c2 may contain the complete agreement.",
            response_id="first",
        )

    generation = ScriptedGeneration(
        [
            lambda _: response(
                [tool_call("read", "get_chunks", {"chunk_ids": ["c2"]})],
                response_id="read",
            ),
            prune,
            lambda _: response(
                [
                    tool_call(
                        "submit",
                        "submit_ranking",
                        rs.ranking_arguments(["c1"], strategy="enough evidence"),
                    )
                ],
                content="I have enough evidence.",
                response_id="last",
            ),
        ]
    )
    return await run_fast_agentic_search(
        "Which agreement applies?",
        store_identifiers=[rs.STORE_ID],
        top_k=1,
        strict_top_k=strict,
        client=SyncRetrievalClientAdapter(rs.ScriptedRetrievalClient()),
        generation_fn=sync_generation_as_async(generation),
    )


async def test_strict_reranks_pruned_evidence_and_retains_the_model_submission(listwise_requests):
    result = await run()

    assert {"chunk_id": "c2"} in result.deleted_chunk_refs
    assert len(listwise_requests) == 1
    request = listwise_requests[0]
    assert len(request["input"]) == 2
    assert "clause 1:" in request["input"][1]
    assert "c2 may contain the complete agreement." in request["query"]
    assert "I have enough evidence." in request["query"]
    assert '"input_index_zero_based": 1' in request["query"]
    assert "submit_ranking" not in request["query"]
    assert [chunk["chunk_id"] for chunk in result.chunks] == ["c2"]
    assert result.chunks[0]["relevance_score"] == 1.0
    assert [chunk.chunk_id for chunk in result.ranking.chunks] == ["c1"]
    assert result.rounds_executed == 3
    assert result.reranking.candidate_count == 2
    assert result.reranking.input_tokens == 42
    assert result.to_record()["reranking"]["model"] == "mixedbread-ai/mxbai-rerank-v3.1-listwise"


async def test_non_strict_does_not_call_the_reranker(listwise_requests):
    result = await run(strict=False)
    assert listwise_requests == []
    assert [chunk["chunk_id"] for chunk in result.chunks] == ["c1"]
    assert result.reranking is None


async def test_omitted_top_k_keeps_all_relevant_chunks_without_reranking(
    monkeypatch, listwise_requests
):
    client = rs.ScriptedRetrievalClient()
    monkeypatch.setattr(
        client.stores,
        "search",
        lambda **_: rs.ScriptedSearchResults([rs.corpus_chunk(i) for i in range(15)]),
    )
    chunk_ids = [f"c{i + 1}" for i in range(15)]
    generation = ScriptedGeneration(
        [
            lambda _: response(
                [
                    tool_call(
                        "read", "read_document", {"document_id": "d1", "chunk_id": "c1", "x": 14}
                    )
                ],
                response_id="read",
            ),
            lambda _: response(
                [
                    tool_call(
                        "submit",
                        "submit_ranking",
                        rs.ranking_arguments(chunk_ids, strategy="all chunks are relevant"),
                    )
                ],
                response_id="submit",
            ),
        ]
    )

    result = await run_fast_agentic_search(
        "Find the relevant agreements.",
        store_identifiers=[rs.STORE_ID],
        client=SyncRetrievalClientAdapter(client),
        generation_fn=sync_generation_as_async(generation),
    )

    assert result.top_k is None
    assert result.strict_top_k is False
    assert [chunk["chunk_id"] for chunk in result.chunks] == chunk_ids
    assert listwise_requests == []


@pytest.mark.parametrize("indices", [(0, 0), (0, 2), (0,)])
async def test_incomplete_or_invalid_reranker_output_is_rejected(indices, monkeypatch):
    mock_scores(monkeypatch, [(i, 0.5) for i in indices])
    with pytest.raises(ValueError, match="every candidate exactly once"):
        await run()


async def test_equal_scores_keep_first_seen_order(monkeypatch):
    mock_scores(monkeypatch, [(1, 0.5), (0, 0.5)])
    result = await run()
    assert result.chunks[0]["chunk_id"] == "c1"


async def test_cancellation_propagates(monkeypatch):
    monkeypatch.setattr(
        mixedbread.AsyncMixedbread, "rerank", AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await run()


async def test_empty_pool_never_calls_inference(listwise_requests):
    result = await SearchTrajectory(ChunkIndex()).rerank(
        "question", top_k=10, api_key=None, api_key_env=None
    )
    assert result == ([], None)
    assert listwise_requests == []


def test_only_presented_chunks_enter_the_pool_and_full_text_survives():
    index = ChunkIndex()
    index.ingest_search_results(
        [
            {**rs.corpus_chunk(0), "text": "full evidence beyond the rendered excerpt"},
            rs.corpus_chunk(1),
            rs.corpus_chunk(2),
        ]
    )
    index.discard_chunks([(rs.STORE_ID, rs.FILE_ID, 2)])
    trajectory = SearchTrajectory(index)
    trajectory.observe_request(
        [
            {
                "role": "tool",
                "content": '{"chunk_id": "c1", "text": "excerpt"} invalid c2, {"chunk_id": "c3"}',
            }
        ]
    )

    assert list(trajectory.candidates) == ["c1"]
    document = rerank_document(trajectory.candidates["c1"])
    assert "full evidence beyond the rendered excerpt" in document
    assert "2019" not in document


def mock_scores(monkeypatch, pairs):
    result = SimpleNamespace(
        data=[SimpleNamespace(index=index, score=score) for index, score in pairs],
        model="mixedbread-ai/mxbai-rerank-v3.1-listwise",
        usage=SimpleNamespace(prompt_tokens=42),
    )
    monkeypatch.setattr(mixedbread.AsyncMixedbread, "rerank", AsyncMock(return_value=result))


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
async def test_nonfinite_scores_are_rejected(score, monkeypatch):
    mock_scores(monkeypatch, [(0, score), (1, 0.5)])
    with pytest.raises(ValueError, match="finite"):
        await run()
