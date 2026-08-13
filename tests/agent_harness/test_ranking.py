from __future__ import annotations

from agent_harness.agents.ranking import finalize_chunks, ranking_unresolved
from agent_harness.schemas import RankedChunk, RankedChunkList
from agent_harness.search import ChunkIndex


def test_non_strict_finalization_keeps_distinct_chunks_from_same_media_file() -> None:
    index = ChunkIndex()
    chunks = [
        {
            "store_id": "store",
            "file_id": "video-file",
            "filename": "video.mp4",
            "chunk_index": 3,
            "mime_type": "video/mp4",
        },
        {
            "store_id": "store",
            "file_id": "video-file",
            "filename": "video.mp4",
            "chunk_index": 28,
            "mime_type": "video/mp4",
        },
    ]
    for chunk in chunks:
        assert index.add_chunk(chunk)

    ranking = RankedChunkList(
        chunks=[
            RankedChunk(
                chunk_id=index.refs.chunk_id_for_key(("store", "video-file", 3)),
                relevance_score=0.99,
            ),
            RankedChunk(
                chunk_id=index.refs.chunk_id_for_key(("store", "video-file", 28)),
                relevance_score=0.98,
            ),
        ]
    )

    finalized = finalize_chunks(index, ranking, top_k=10, strict_top_k=False)

    assert [chunk["chunk_index"] for chunk in finalized] == [3, 28]


def test_non_strict_finalization_keeps_chunks_sharing_an_asset_identity() -> None:
    # Gold is judged at chunk granularity: one ad or media asset can carry
    # several judged chunks, so sharing an ad_id or media URL must not collapse
    # submitted entries — the scored list has to be exactly what the agent ranked.
    index = ChunkIndex()
    chunks = [
        {
            "store_id": "store",
            "file_id": "image-file-a",
            "chunk_index": 0,
            "image_url": "https://example.com/image.png",
            "metadata": {"ad_id": "523461560789627"},
        },
        {
            "store_id": "store",
            "file_id": "image-file-b",
            "chunk_index": 0,
            "image_url": "https://example.com/image.png",
            "metadata": {"ad_id": "523461560789627"},
        },
    ]
    for chunk in chunks:
        assert index.add_chunk(chunk)

    ranking = RankedChunkList(
        chunks=[
            RankedChunk(
                chunk_id=index.refs.chunk_id_for_key(("store", "image-file-a", 0)),
                relevance_score=0.99,
            ),
            RankedChunk(
                chunk_id=index.refs.chunk_id_for_key(("store", "image-file-b", 0)),
                relevance_score=0.98,
            ),
        ]
    )

    finalized = finalize_chunks(index, ranking, top_k=10, strict_top_k=False)

    assert [chunk["file_id"] for chunk in finalized] == ["image-file-a", "image-file-b"]


def _index_with_text_chunks() -> ChunkIndex:
    index = ChunkIndex()
    for chunk_index in (0, 1):
        assert index.add_chunk(
            {
                "store_id": "store",
                "file_id": "doc",
                "filename": "doc.pdf",
                "chunk_index": chunk_index,
                "score": 0.9 - chunk_index * 0.1,
            }
        )
    return index


def test_missing_submission_finalizes_to_empty() -> None:
    index = _index_with_text_chunks()

    assert finalize_chunks(index, None, top_k=10, strict_top_k=False) == []


def test_unresolvable_ranking_finalizes_to_empty() -> None:
    index = _index_with_text_chunks()
    ranking = RankedChunkList(
        chunks=[RankedChunk(chunk_id="chunk_does_not_exist", relevance_score=0.9)]
    )

    assert finalize_chunks(index, ranking, top_k=10, strict_top_k=False) == []


def test_empty_submission_finalizes_to_empty() -> None:
    index = _index_with_text_chunks()

    assert finalize_chunks(index, RankedChunkList(chunks=[]), top_k=10, strict_top_k=False) == []


def test_ranking_unresolved_requires_a_submission_with_chunks() -> None:
    index = _index_with_text_chunks()

    assert ranking_unresolved(index, None) is False
    assert ranking_unresolved(index, RankedChunkList(chunks=[])) is False


def test_ranking_unresolved_when_no_ranked_chunk_maps_to_the_index() -> None:
    index = _index_with_text_chunks()
    ranking = RankedChunkList(
        chunks=[RankedChunk(chunk_id="chunk_does_not_exist", relevance_score=0.9)]
    )

    assert ranking_unresolved(index, ranking) is True
    # The finalized payload is [] here, indistinguishable from a deliberately
    # empty submission; the signal must come from the index, not from
    # finalize_chunks output.
    assert finalize_chunks(index, ranking, top_k=10, strict_top_k=False) == []


def test_ranking_unresolved_false_when_any_ranked_chunk_resolves() -> None:
    index = _index_with_text_chunks()
    ranking = RankedChunkList(
        chunks=[
            RankedChunk(chunk_id="chunk_does_not_exist", relevance_score=0.9),
            RankedChunk(
                chunk_id=index.refs.chunk_id_for_key(("store", "doc", 0)),
                relevance_score=0.8,
            ),
        ]
    )

    assert ranking_unresolved(index, ranking) is False
