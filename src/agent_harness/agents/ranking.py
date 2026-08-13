"""Shared final-ranking helpers for agent runtimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_harness.schemas import RankedChunk, RankedChunkList
from agent_harness.search import ChunkIndex, chunk_key, ranked_chunk_key

FinalChunkSubmission = RankedChunkList


def finalize_chunks(
    index: ChunkIndex,
    final_ranking: FinalChunkSubmission | None,
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> list[dict[str, Any]]:
    """Resolve the agent's final ranking into the chunks it submitted.

    Returns only what the agent itself ranked; the harness never substitutes
    a ranking of its own. ``[]`` therefore covers a deliberate empty answer,
    a missing submission, and a ranking whose chunks resolved to nothing --
    ``forced_ranking`` on the result records marks the missing-submission
    case, ``ranking_unresolved`` the resolved-to-nothing case.
    """
    if final_ranking is None or not final_ranking.chunks:
        return []

    # Dedup happens at chunk granularity only
    chunks: list[dict[str, Any]] = []
    selected: set[tuple[str, str, int]] = set()
    for ranked in final_ranking.chunks:
        try:
            key = ranked_chunk_key(ranked, index.refs)
        except ValueError:
            continue
        if key in selected:
            continue
        chunk = index.final_chunk(ranked)
        if chunk is None:
            continue
        payload = chunk_with_reference_ids(chunk, index)
        payload["relevance_score"] = ranked.relevance_score
        chunks.append(payload)
        selected.add(key)
        if strict_top_k and top_k is not None and len(chunks) >= top_k:
            break

    return chunks


def ranking_unresolved(
    index: ChunkIndex,
    final_ranking: FinalChunkSubmission | None,
) -> bool:
    """True when the agent ranked chunks and none of them resolved to one.

    That is id-space breakage -- the agent named handles the index could not
    map back to a corpus chunk -- not an empty answer. The finalized payload
    alone cannot show it: ``finalize_chunks`` returns ``[]`` here, which is
    indistinguishable from a deliberately empty submission. Unlike a missing
    submission, nothing forces the ranking here, so ``forced_ranking`` does
    not cover it either. Carried on result records beside ``forced_ranking``
    for observability.
    """
    if final_ranking is None or not final_ranking.chunks:
        return False
    for ranked in final_ranking.chunks:
        try:
            ranked_chunk_key(ranked, index.refs)
        except ValueError:
            continue
        if index.final_chunk(ranked) is not None:
            return False
    return True


def normalize_top_k(top_k: int | None) -> int | None:
    if top_k is None:
        return None
    value = int(top_k)
    if value < 1:
        raise ValueError("top_k must be >= 1")
    return value


def chunk_with_reference_ids(
    chunk: Mapping[str, Any],
    index: ChunkIndex,
) -> dict[str, Any]:
    payload = dict(chunk)
    key = chunk_key(payload)
    chunk_id, document_id = index.refs.ids_for_chunk_key(key)
    payload["chunk_id"] = chunk_id
    payload["document_id"] = document_id
    return payload


def validate_ranked_chunk_ids(
    ranking: FinalChunkSubmission,
    index: ChunkIndex,
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
    tool_name: str = "submit_ranking",
) -> None:
    if strict_top_k and top_k is not None and len(ranking.chunks) != top_k:
        raise ValueError(
            f"{tool_name}.chunks must contain exactly {top_k} chunks; got {len(ranking.chunks)}"
        )
    valid_chunk_ids = set(index.visible_chunk_ids())
    unavailable_ids: list[str] = []
    for ranked in ranking.chunks:
        if ranked.chunk_id not in valid_chunk_ids:
            unavailable_ids.append(ranked.chunk_id)
    if unavailable_ids:
        raise ValueError("Unknown or unavailable chunk_id values: " + ", ".join(unavailable_ids))


def ranking_trace_payload(
    ranking: FinalChunkSubmission | None,
    index: ChunkIndex,
) -> dict[str, Any] | None:
    if ranking is None:
        return None
    payload = ranking.model_dump(mode="json", exclude_none=True)
    payload["chunks"] = ranked_chunk_trace_payloads(ranking.chunks, index)
    return payload


def ranked_chunk_trace_payloads(
    chunks: Sequence[RankedChunk],
    index: ChunkIndex,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for ranked in chunks:
        payload = ranked.model_dump(mode="json", exclude_none=True)
        try:
            key = ranked_chunk_key(ranked, index.refs)
        except ValueError:
            payloads.append(payload)
            continue
        payload["file_id"] = key[1]
        payload["chunk_index"] = key[2]
        payloads.append(payload)
    return payloads
