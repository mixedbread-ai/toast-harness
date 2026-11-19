"""Final reranking uses the evidence Toast saw, independently of its submission."""

from __future__ import annotations

import asyncio

import pytest
import replay_scenarios as rs

from agent_harness.agents.searcher import run_fast_agentic_search
from agent_harness.llm import sync_generation_as_async
from agent_harness.reranking import RerankRequest, RerankResult, RerankScore, SearchTrajectory
from agent_harness.retrieval import SyncRetrievalClientAdapter
from agent_harness.search import ChunkIndex
from agent_harness.testing import ScriptedGeneration, response, tool_call


class RecordingReranker:
    def __init__(self, *, scores=None, error=None):
        self.requests: list[RerankRequest] = []
        self.scores = scores
        self.error = error

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return RerankResult(
            scores=self.scores
            or (RerankScore(index=0, score=0.2), RerankScore(index=1, score=0.9)),
            model="test-listwise",
            input_tokens=42,
        )


async def run(reranker, *, strict=True):
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
        reranker=reranker,
    )


async def test_strict_reranks_pruned_evidence_and_retains_the_model_submission():
    reranker = RecordingReranker()
    result = await run(reranker)

    assert {"chunk_id": "c2"} in result.deleted_chunk_refs
    assert len(reranker.requests) == 1
    request = reranker.requests[0]
    assert len(request.documents) == 2
    assert "clause 1:" in request.documents[1]
    assert "c2 may contain the complete agreement." in request.query
    assert "I have enough evidence." in request.query
    assert '"input_index_zero_based": 1' in request.query
    assert "submit_ranking" not in request.query
    assert [chunk["chunk_id"] for chunk in result.chunks] == ["c2"]
    assert result.chunks[0]["relevance_score"] == 0.9
    assert [chunk.chunk_id for chunk in result.ranking.chunks] == ["c1"]
    assert result.rounds_executed == 3
    assert result.reranking.candidate_count == 2
    assert result.reranking.input_tokens == 42
    assert result.to_record()["reranking"]["model"] == "test-listwise"


async def test_non_strict_does_not_call_the_reranker():
    reranker = RecordingReranker()
    result = await run(reranker, strict=False)
    assert reranker.requests == []
    assert [chunk["chunk_id"] for chunk in result.chunks] == ["c1"]
    assert result.reranking is None


async def test_omitted_top_k_keeps_all_relevant_chunks_without_reranking(monkeypatch):
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
    reranker = RecordingReranker()

    result = await run_fast_agentic_search(
        "Find the relevant agreements.",
        store_identifiers=[rs.STORE_ID],
        client=SyncRetrievalClientAdapter(client),
        generation_fn=sync_generation_as_async(generation),
        reranker=reranker,
    )

    assert result.top_k is None
    assert result.strict_top_k is False
    assert [chunk["chunk_id"] for chunk in result.chunks] == chunk_ids
    assert reranker.requests == []


@pytest.mark.parametrize("indices", [(0, 0), (0, 2), (0,)])
async def test_incomplete_or_invalid_reranker_output_is_rejected(indices):
    reranker = RecordingReranker(scores=tuple(RerankScore(index=i, score=0.5) for i in indices))
    with pytest.raises(ValueError, match="every candidate exactly once"):
        await run(reranker)


async def test_equal_scores_keep_first_seen_order():
    reranker = RecordingReranker(
        scores=(RerankScore(index=1, score=0.5), RerankScore(index=0, score=0.5))
    )
    result = await run(reranker)
    assert result.chunks[0]["chunk_id"] == "c1"


async def test_cancellation_propagates():
    with pytest.raises(asyncio.CancelledError):
        await run(RecordingReranker(error=asyncio.CancelledError()))


async def test_empty_pool_never_calls_inference():
    reranker = RecordingReranker()
    result = await SearchTrajectory(ChunkIndex()).rerank("question", reranker, top_k=10)
    assert result == ([], None)
    assert reranker.requests == []


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

    request = trajectory.request("question")
    assert len(request.documents) == 1
    assert "full evidence beyond the rendered excerpt" in request.documents[0]
    assert "2019" not in request.documents[0]  # Generated metadata is not a rerank input.


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scores_are_rejected(score):
    with pytest.raises(ValueError, match="finite"):
        RerankScore(index=0, score=score)
