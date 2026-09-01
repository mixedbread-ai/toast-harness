"""Chunk index, deduplication, and context pruning helpers."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import Lock, RLock
from typing import Any, get_args

import regex as _clip_regex

from . import config
from .config import (
    GREP_DEFAULT_K,
    OVERVIEW_SEARCH_TOP_K,
    SEARCH_CORPUS_MAX_QPS,
    SEARCH_CORPUS_TOP_K,
    SEARCH_OVERFETCH_FACTOR,
    corpus_backend_top_k,
)
from .errors import is_provider_failure
from .references import ReferenceRegistry
from .retrieval import AsyncRetrievalClient
from .schemas import (
    AgentChunkPayload,
    ChunkKey,
    DocumentKey,
    RankedChunk,
    StoreChunkGrepTarget,
    chunk_key_from_parts,
    document_key_from_parts,
)
from .tools.functions import (
    build_mixedbread_filters,
    chunk_content_text,
    distinct_metadata,
    filter_chunks,
    filter_metadata,
    get_chunk,
    grep_raw,
    inspect_metadata,
    metadata_lookup,
    overview_search,
    rank_metadata,
    read_document,
    search_corpus,
)

CONTENT_FIELDS = {
    "text",
    "context",
    "ocr_text",
    "transcription",
    "summary",
    "content",
    "image_url",
    "media_url",
    "audio_url",
    "video_url",
    "content_url",
}
BACKEND_ID_FIELDS = {"file_id", "store_id"}


class _RateLimiter:
    """Thread-safe token bucket bounding global search_corpus QPS across all rollouts.

    Deliberately a per-second rate, not a concurrency cap: the per-turn
    tool-call limit is unrelated to backend capacity, and a process-wide
    concurrency gate would serialize rollout collection. Concurrency is
    bounded only by (inflight rollouts x per-turn tool-call fan-out).
    """

    def __init__(self, qps: float) -> None:
        self._rate = float(qps)
        self._capacity = float(qps)
        self._tokens = float(qps)
        self._last = time.monotonic()
        self._lock = Lock()

    async def acquire(self) -> None:
        # The threading.Lock guards bucket state shared across event loops (one
        # process may run several sync-wrapped rollout loops); it is only held
        # for the arithmetic, never across an await.
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self._capacity, self._tokens + (now - self._last) * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


_search_corpus_rate_limiter = _RateLimiter(SEARCH_CORPUS_MAX_QPS)


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """One executed retrieval tool: the model-visible payload and its query record.

    ``payload`` is the wire-format dict serialized into the tool message;
    ``query`` is the record appended to ``queries_made``. Both stay dicts
    because they are JSON the moment they leave the harness; the pair itself is
    typed so rounds stop shuttling anonymous tuples.
    """

    payload: dict[str, Any]
    query: dict[str, Any]


@dataclass(slots=True)
class ChunkIndex:
    """Chunk memory for one agent, with shared compact references.

    Scoped search agents keep their own chunk/document storage so deduplication is
    per-agent. They share only the reference registry with the core agent, keeping
    handles like ``c05`` stable across agents without sharing each agent's seen set.
    """

    chunks_by_key: dict[ChunkKey, dict[str, Any]] = field(default_factory=dict)
    document_keys: set[DocumentKey] = field(default_factory=set)
    overview_only_chunk_keys: set[ChunkKey] = field(default_factory=set)
    deleted_chunk_keys: set[ChunkKey] = field(default_factory=set)
    restored_chunk_keys: set[ChunkKey] = field(default_factory=set)
    deleted_document_keys: set[DocumentKey] = field(default_factory=set)
    refs: ReferenceRegistry = field(default_factory=ReferenceRegistry)
    visible_chunk_keys: set[ChunkKey] | None = None
    visible_document_keys: set[DocumentKey] | None = None
    _lock: Any = field(default_factory=RLock, init=False, repr=False)

    def scoped_view(self) -> ChunkIndex:
        """Return a child index with independent seen state and shared references."""
        return ChunkIndex(
            refs=self.refs,
            visible_chunk_keys=set(),
            visible_document_keys=set(),
        )

    def is_scoped(self) -> bool:
        return self.visible_chunk_keys is not None

    def shares_storage_with(self, other: ChunkIndex) -> bool:
        return self.chunks_by_key is other.chunks_by_key and self.refs is other.refs

    def has_seen(self, key: ChunkKey) -> bool:
        with self._lock:
            return key in self.chunks_by_key or key in self.deleted_chunk_keys

    def expose_chunk_reference(self, chunk: Mapping[str, Any]) -> bool:
        """Make an already-seen local chunk selectable without re-adding content."""
        key = chunk_key(chunk)
        with self._lock:
            chunk_is_pruned = key in self.deleted_chunk_keys and key not in self.restored_chunk_keys
            if chunk_is_pruned or key not in self.chunks_by_key:
                return False
            self.refs.chunk_id_for_key(key)
            document_key_value = document_key_from_parts(key[0], key[1])
            self.document_keys.add(document_key_value)
            self._remember_visible_chunk_unlocked(key)
            self._remember_visible_document_unlocked(document_key_value)
            return True

    def can_add_from_search(self, key: ChunkKey) -> bool:
        with self._lock:
            return key not in self.deleted_chunk_keys and key not in self.chunks_by_key

    def add_chunk(self, chunk: Mapping[str, Any], *, restore: bool = False) -> bool:
        key = chunk_key(chunk)
        with self._lock:
            existing = self.chunks_by_key.get(key)
            can_upgrade = existing is not None and key in self.overview_only_chunk_keys
            if not restore and (
                key in self.deleted_chunk_keys or (existing is not None and not can_upgrade)
            ):
                return False
            self.refs.chunk_id_for_key(key)
            document_key_value = document_key_from_parts(key[0], key[1])
            self.document_keys.add(document_key_value)
            self.chunks_by_key[key] = merge_chunk_payload(existing, chunk)
            self.overview_only_chunk_keys.discard(key)
            if restore:
                self.restored_chunk_keys.add(key)
            self._remember_visible_chunk_unlocked(key)
            self._remember_visible_document_unlocked(document_key_value)
            return True

    def ingest_overview_results(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        max_new_chunks: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        overview_chunks: list[dict[str, Any]] = []
        for chunk in sorted(chunks, key=chunk_score, reverse=True):
            if stats is not None:
                stats["examined"] = stats.get("examined", 0) + 1
            key = chunk_key(chunk)
            overview_chunk = dict(chunk)
            stored = dict(chunk)
            stored.pop("summary", None)
            with self._lock:
                if not self.can_add_from_search(key):
                    if stats is not None:
                        stats["skipped_existing_or_deleted"] = (
                            stats.get("skipped_existing_or_deleted", 0) + 1
                        )
                    continue
                self.refs.chunk_id_for_key(key)
                document_key_value = document_key_from_parts(key[0], key[1])
                self.document_keys.add(document_key_value)
                self.chunks_by_key[key] = stored
                self.overview_only_chunk_keys.add(key)
                self._remember_visible_chunk_unlocked(key)
                self._remember_visible_document_unlocked(document_key_value)
                overview_chunks.append(overview_chunk)
            if max_new_chunks is not None and len(overview_chunks) >= max_new_chunks:
                break
        return overview_chunks

    def ingest_search_results(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        max_new_chunks: int | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        new_chunks: list[dict[str, Any]] = []
        for chunk in sorted(chunks, key=chunk_score, reverse=True):
            if stats is not None:
                stats["examined"] = stats.get("examined", 0) + 1
            key = chunk_key(chunk)
            stored = dict(chunk)
            with self._lock:
                existing = self.chunks_by_key.get(key)
                can_upgrade = existing is not None and key in self.overview_only_chunk_keys
                if key in self.deleted_chunk_keys or (existing is not None and not can_upgrade):
                    if stats is not None:
                        stats["skipped_existing_or_deleted"] = (
                            stats.get("skipped_existing_or_deleted", 0) + 1
                        )
                    continue
                self.refs.chunk_id_for_key(key)
                document_key_value = document_key_from_parts(key[0], key[1])
                self.document_keys.add(document_key_value)
                stored = merge_chunk_payload(existing, stored)
                self.chunks_by_key[key] = stored
                self.overview_only_chunk_keys.discard(key)
                self._remember_visible_chunk_unlocked(key)
                self._remember_visible_document_unlocked(document_key_value)
            new_chunks.append(stored)
            if max_new_chunks is not None and len(new_chunks) >= max_new_chunks:
                break
        return new_chunks

    def mark_pruned(
        self,
        *,
        chunk_keys: Iterable[ChunkKey],
        document_keys: Iterable[DocumentKey],
    ) -> None:
        with self._lock:
            chunk_key_set = set(chunk_keys)
            self.deleted_chunk_keys.update(chunk_key_set)
            self.restored_chunk_keys.difference_update(chunk_key_set)
            self.deleted_document_keys.update(document_keys)

    def discard_chunks(self, keys: Iterable[ChunkKey]) -> None:
        """Forget chunks whose payload was truncated away before the model saw it.

        Unlike ``mark_pruned`` the keys are not blacklisted, so a later search
        may legitimately re-surface the same chunk.
        """
        with self._lock:
            discarded_document_keys: set[DocumentKey] = set()
            for key in set(keys):
                if self.chunks_by_key.pop(key, None) is None:
                    continue
                self.overview_only_chunk_keys.discard(key)
                self.restored_chunk_keys.discard(key)
                if self.visible_chunk_keys is not None:
                    self.visible_chunk_keys.discard(key)
                discarded_document_keys.add(document_key_from_parts(key[0], key[1]))
            surviving_document_keys = {
                document_key_from_parts(key[0], key[1]) for key in self.chunks_by_key
            }
            for document_key_value in discarded_document_keys - surviving_document_keys:
                self.document_keys.discard(document_key_value)
                if self.visible_document_keys is not None:
                    self.visible_document_keys.discard(document_key_value)

    def get(self, key: ChunkKey) -> dict[str, Any] | None:
        with self._lock:
            if not self._is_chunk_visible_unlocked(key):
                return None
            return self.chunks_by_key.get(key)

    def register_document(self, document: Mapping[str, Any]) -> str:
        key = document_key(document)
        with self._lock:
            self.document_keys.add(key)
            self._remember_visible_document_unlocked(key)
            return self.refs.document_id_for_key(key)

    def is_overview_only(self, key: ChunkKey) -> bool:
        with self._lock:
            return key in self.overview_only_chunk_keys

    def is_visible_chunk(self, key: ChunkKey) -> bool:
        with self._lock:
            return self._is_chunk_visible_unlocked(key)

    def is_visible_document(self, key: DocumentKey) -> bool:
        with self._lock:
            return self._is_document_visible_unlocked(key)

    def final_chunk(self, ranked: RankedChunk) -> dict[str, Any] | None:
        key = ranked_chunk_key(ranked, self.refs)
        with self._lock:
            if not self._is_chunk_visible_unlocked(key):
                return None
            chunk = self.chunks_by_key.get(key)
            if chunk is None:
                return None
            if key in self.deleted_chunk_keys and key not in self.restored_chunk_keys:
                return redacted_chunk(chunk)
            return chunk

    def top_scored(self, top_k: int | None = None) -> list[Mapping[str, Any]]:
        with self._lock:
            keys = self._visible_chunk_keys_unlocked()
            chunks = [self.chunks_by_key[key] for key in keys if key in self.chunks_by_key]
            chunks = sorted(chunks, key=chunk_score, reverse=True)
            if top_k is None:
                return chunks
            return chunks[:top_k]

    def visible_chunk_ids(self) -> list[str]:
        with self._lock:
            keys = sorted(self._visible_chunk_keys_unlocked())
        return [self.refs.chunk_id_for_key(key) for key in keys]

    def visible_document_ids(self) -> list[str]:
        with self._lock:
            document_keys = self._visible_document_keys_unlocked()
        return [self.refs.document_id_for_key(key) for key in sorted(document_keys)]

    def _remember_visible_chunk_unlocked(self, key: ChunkKey) -> None:
        if self.visible_chunk_keys is not None:
            self.visible_chunk_keys.add(key)

    def _remember_visible_document_unlocked(self, key: DocumentKey) -> None:
        if self.visible_document_keys is not None:
            self.visible_document_keys.add(key)

    def _visible_chunk_keys_unlocked(self) -> set[ChunkKey]:
        if self.visible_chunk_keys is None:
            return set(self.chunks_by_key)
        return {key for key in self.visible_chunk_keys if key in self.chunks_by_key}

    def _visible_document_keys_unlocked(self) -> set[DocumentKey]:
        if self.visible_document_keys is None:
            document_keys = {document_key_from_parts(key[0], key[1]) for key in self.chunks_by_key}
            document_keys.update(self.document_keys)
            return document_keys

        document_keys = {
            document_key_from_parts(key[0], key[1])
            for key in self.visible_chunk_keys or set()
            if key in self.chunks_by_key
        }
        document_keys.update(self.visible_document_keys or set())
        return document_keys

    def _is_chunk_visible_unlocked(self, key: ChunkKey) -> bool:
        return key in self._visible_chunk_keys_unlocked()

    def _is_document_visible_unlocked(self, key: DocumentKey) -> bool:
        return key in self._visible_document_keys_unlocked()


async def execute_search_corpus(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    top_k: int = SEARCH_CORPUS_TOP_K,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    query = str(args["query"]).strip()
    filter_by = args.get("filter_by") or args.get("metadata_filters") or []
    # This provider fetch depth is intentionally independent of the five
    # chunks exposed to the agent and of the strict final-ranking cutoff.
    search_top_k = corpus_backend_top_k()
    await _search_corpus_rate_limiter.acquire()
    raw_result = await search_corpus(
        query,
        store_identifiers=store_identifiers,
        top_k=search_top_k,
        filter_by=filter_by,
        metadata_filter=args.get("metadata_filter"),
        filter_mode=args.get("filter_mode"),
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    candidates = raw_result.get("results") or []
    ingest_stats: dict[str, int] = {}
    new_chunks = index.ingest_search_results(
        candidates,
        max_new_chunks=top_k,
        stats=ingest_stats,
    )
    result = {
        "tool": "search_corpus",
        "query": query,
        "requested_top_k": top_k,
        "search_top_k": search_top_k,
        "filter_by": filter_by,
        "metadata_filters": raw_result.get("metadata_filters"),
        "metadata_filter": raw_result.get("metadata_filter"),
        "new_unseen_results": serialize_agent_chunks(
            new_chunks, refs=index.refs, clip_search_payload=True
        ),
        "deduped_existing_or_deleted": ingest_stats.get("skipped_existing_or_deleted", 0),
    }
    _enforce_call_payload_budget(result, budget=config.SEARCH_CORPUS_PAYLOAD_TOKEN_BUDGET)
    metadata = {
        "query": query,
        "k": top_k,
        "search_top_k": search_top_k,
        "filter_by": filter_by,
        "metadata_filters": raw_result.get("metadata_filters"),
        "metadata_filter": raw_result.get("metadata_filter"),
        "new_chunks_added": len(new_chunks),
    }
    return ToolOutcome(_drop_empty(result), _drop_empty(metadata))


async def execute_inspect_metadata(
    args: Mapping[str, Any],
    *,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    max_values = int(args.get("max_values_per_field") or 8)
    result = await inspect_metadata(
        max_values_per_field=max_values,
        facets=args.get("facets"),
        metadata_filter=args.get("metadata_filter"),
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    metadata = {
        "tool": "inspect_metadata",
        "metadata_field_count": result.get("metadata_field_count"),
        "store_identifiers": result.get("store_identifiers"),
        "requested_facets": result.get("requested_facets"),
        "metadata_filter": result.get("metadata_filter"),
    }
    return ToolOutcome(result, metadata)


async def execute_filter_metadata(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    result = await filter_metadata(
        args.get("metadata_filters") or [],
        metadata_filter=args.get("metadata_filter"),
        filter_mode=args.get("filter_mode"),
        limit=int(args.get("limit") or 20),
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    files = []
    for file_payload in result.get("files") or []:
        if not file_payload.get("file_id") or not file_payload.get("store_id"):
            continue
        document_id = index.register_document(file_payload)
        enriched = agent_file_payload(file_payload)
        enriched["document_id"] = document_id
        files.append(enriched)

    payload = dict(result)
    payload["files"] = files
    metadata = {
        "tool": "filter_metadata",
        "metadata_filter": result.get("metadata_filter"),
        "files_returned": len(files),
        "limit": result.get("limit"),
    }
    return ToolOutcome(_drop_empty(payload), _drop_empty(metadata))


async def execute_rank_metadata(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    result = await rank_metadata(
        str(args["sort_key"]).strip(),
        sort_order=args.get("sort_order") or "desc",
        metadata_filters=args.get("metadata_filters") or [],
        metadata_filter=args.get("metadata_filter"),
        filter_mode=args.get("filter_mode"),
        limit=int(args.get("limit") or 10),
        fetch_limit=int(args.get("fetch_limit") or 100),
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    include_chunks = bool(args.get("include_chunks", True))
    files = [
        await enrich_file_payload(
            file_payload,
            index=index,
            include_chunks=include_chunks,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        for file_payload in result.get("files") or []
    ]
    payload = dict(result)
    payload["files"] = [file_payload for file_payload in files if file_payload]
    metadata = {
        "tool": "rank_metadata",
        "sort_key": result.get("sort_key"),
        "sort_order": result.get("sort_order"),
        "metadata_filter": result.get("metadata_filter"),
        "files_returned": len(payload["files"]),
        "fetch_limit": result.get("fetch_limit"),
        "limit": result.get("limit"),
    }
    return ToolOutcome(_drop_empty(payload), _drop_empty(metadata))


async def execute_distinct_metadata(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    result = await distinct_metadata(
        str(args["distinct_key"]).strip(),
        metadata_filters=args.get("metadata_filters") or [],
        metadata_filter=args.get("metadata_filter"),
        filter_mode=args.get("filter_mode"),
        examples_per_value=int(args.get("examples_per_value") or 1),
        fetch_limit=int(args.get("fetch_limit") or 100),
        store_identifiers=store_identifiers,
        sort_key=args.get("sort_key"),
        sort_order=args.get("sort_order") or "desc",
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    include_chunks = bool(args.get("include_chunks", True))
    distinct_values: list[dict[str, Any]] = []
    for group in result.get("distinct_values") or []:
        enriched_group = dict(group)
        files = [
            await enrich_file_payload(
                file_payload,
                index=index,
                include_chunks=include_chunks,
                client=client,
                api_key=api_key,
                api_key_env=api_key_env,
            )
            for file_payload in group.get("files") or []
        ]
        enriched_group["files"] = [file_payload for file_payload in files if file_payload]
        enriched_group["file_count_returned"] = len(enriched_group["files"])
        distinct_values.append(enriched_group)
    payload = dict(result)
    payload["distinct_values"] = distinct_values
    metadata = {
        "tool": "distinct_metadata",
        "distinct_key": result.get("distinct_key"),
        "metadata_filter": result.get("metadata_filter"),
        "distinct_value_count": result.get("distinct_value_count"),
        "fetch_limit": result.get("fetch_limit"),
    }
    return ToolOutcome(_drop_empty(payload), _drop_empty(metadata))


async def execute_filter_chunks(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    filter_by = args.get("filter_by") or args.get("metadata_filters") or []
    result = await filter_chunks(
        filter_by,
        filter_mode=args.get("filter_mode") or "all",
        rank_by=args.get("rank_by"),
        direction=args.get("direction") or "desc",
        k=int(args.get("k") or 0),
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    candidates = result.get("results") or []
    new_chunks = index.ingest_search_results(
        candidates,
        max_new_chunks=int(result.get("k") or 0),
    )
    new_chunk_keys = {chunk_key(chunk) for chunk in new_chunks}
    results = [
        (
            serialize_agent_chunk(
                index.get(chunk_key(candidate)) or candidate,
                refs=index.refs,
                clip_search_payload=True,
            )
            if chunk_key(candidate) in new_chunk_keys
            else _seen_chunk_reference(candidate, index=index, rank_by=result.get("rank_by"))
        )
        for candidate in candidates
        if chunk_key(candidate) in new_chunk_keys or index.expose_chunk_reference(candidate)
    ]
    payload = {
        "tool": "filter_chunks",
        "filter_by": filter_by,
        "metadata_filter": result.get("metadata_filter"),
        "rank_by": result.get("rank_by"),
        "direction": result.get("direction"),
        "rank_by_applied": result.get("rank_by_applied"),
        "rank_by_non_numeric_count": result.get("rank_by_non_numeric_count"),
        "k": result.get("k"),
        "candidate_count": result.get("candidate_count"),
        "results": results,
    }
    _enforce_call_payload_budget(payload, budget=config.FILTER_CHUNKS_PAYLOAD_TOKEN_BUDGET)
    metadata = {
        "tool": "filter_chunks",
        "filter_by": filter_by,
        "metadata_filter": result.get("metadata_filter"),
        "rank_by": result.get("rank_by"),
        "direction": result.get("direction"),
        "rank_by_applied": result.get("rank_by_applied"),
        "rank_by_non_numeric_count": result.get("rank_by_non_numeric_count"),
        "candidate_count": result.get("candidate_count"),
        "returned_result_count": len(results),
        "new_chunks_added": len(new_chunks),
        "k": result.get("k"),
    }
    return ToolOutcome(_drop_empty(payload), _drop_empty(metadata))


def _build_grep_results(
    chunks: Sequence[Mapping[str, Any]],
    *,
    index: ChunkIndex,
    new_chunk_keys: set[ChunkKey],
    requested_k: int,
    clip_focus: re.Pattern[str] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Serialize grep hits, clip pass included; CPU-bound, so callers keep it off the loop."""
    results: list[dict[str, Any]] = []
    seen_candidate_keys: set[ChunkKey] = set()
    included_new_chunks = 0
    for candidate in chunks:
        key = chunk_key(candidate)
        if key in seen_candidate_keys:
            continue
        if key in new_chunk_keys:
            stored = index.get(key) or candidate
            results.append(
                serialize_agent_chunk(
                    stored,
                    refs=index.refs,
                    clip_search_payload=True,
                    clip_focus=clip_focus,
                )
            )
            included_new_chunks += 1
        elif index.expose_chunk_reference(candidate):
            results.append(_seen_chunk_reference(candidate, index=index, rank_by=None))
        seen_candidate_keys.add(key)
        if included_new_chunks >= len(new_chunk_keys) and len(results) >= requested_k:
            break
    return results, included_new_chunks


async def execute_grep(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    requested_k = GREP_DEFAULT_K
    fetch_k = requested_k * SEARCH_OVERFETCH_FACTOR
    targets = args.get("targets") or ["text", "generated"]
    filter_by = args.get("filter_by") or args.get("metadata_filters") or []
    case_sensitive = bool(args.get("case_sensitive", False))
    # Compiled once per grep call, not per chunk. An invalid regex still reaches
    # the provider (which owns pattern validation); clipping just falls back to
    # head truncation rather than failing the tool call locally.
    clip_focus: re.Pattern[str] | None
    try:
        # `regex`, not stdlib `re`: a model-authored pattern with nested quantifiers
        # backtracks catastrophically and stdlib matching cannot be interrupted;
        # `regex` accepts the same syntax and lets the clip pass bound each match.
        clip_focus = _clip_regex.compile(
            str(args["pattern"]).strip(), 0 if case_sensitive else _clip_regex.IGNORECASE
        )
    except (re.error, _clip_regex.error):
        clip_focus = None
    chunks = await grep_raw(
        str(args["pattern"]).strip(),
        fetch_k,
        store_identifiers=store_identifiers,
        targets=targets,
        case_sensitive=case_sensitive,
        filter_by=filter_by,
        filter_mode=args.get("filter_mode") or "all",
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    filters = build_mixedbread_filters(filter_by, args.get("filter_mode") or "all")
    new_chunks = index.ingest_search_results(chunks, max_new_chunks=requested_k)
    new_chunk_keys = {chunk_key(chunk) for chunk in new_chunks}
    results, _ = await asyncio.to_thread(
        _build_grep_results,
        chunks,
        index=index,
        new_chunk_keys=new_chunk_keys,
        requested_k=requested_k,
        clip_focus=clip_focus,
    )

    targets_note, targets_probe = await _grep_other_bucket_note(
        args,
        targets=targets,
        candidate_count=len(chunks),
        store_identifiers=store_identifiers,
        filter_by=filter_by,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    payload = {
        "tool": "grep",
        "pattern": str(args["pattern"]).strip(),
        "targets": list(dict.fromkeys(targets)),
        "case_sensitive": bool(args.get("case_sensitive", False)),
        "filter_by": filter_by,
        "metadata_filter": filters,
        "k": requested_k,
        "fetch_k": fetch_k,
        "targets_note": targets_note,
        "results": results,
    }
    _enforce_call_payload_budget(payload, budget=config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET)
    metadata = {
        "tool": "grep",
        "pattern": str(args["pattern"]).strip(),
        "targets": list(dict.fromkeys(targets)),
        "case_sensitive": bool(args.get("case_sensitive", False)),
        "filter_by": filter_by,
        "metadata_filter": filters,
        "candidate_count": len(chunks),
        "returned_result_count": len(results),
        "new_chunks_added": len(new_chunks),
        "k": requested_k,
        "fetch_k": fetch_k,
        "targets_probe": targets_probe,
    }
    return ToolOutcome(_drop_empty(payload), _drop_empty(metadata))


_GREP_TARGETS = get_args(StoreChunkGrepTarget)


async def _grep_other_bucket_note(
    args: Mapping[str, Any],
    *,
    targets: Sequence[str],
    candidate_count: int,
    store_identifiers: Sequence[str],
    filter_by: Sequence[Mapping[str, Any]],
    client: AsyncRetrievalClient | None,
    api_key: str | None,
    api_key_env: str | None,
) -> tuple[str | None, str | None]:
    """Warn when a narrowed grep found nothing but the other target bucket has matches.

    Stores whose documents are page images keep all content in the generated bucket
    (OCR, summaries, transcriptions) and text-extracted stores keep it in ``text``, so
    naming one bucket can return zero for a pattern the corpus does contain — which
    reads as "the corpus has no such content" rather than "you searched the wrong
    field". Only checked when a narrowed grep came back empty, so the extra provider
    call never touches the normal path.

    Returns the note (or None) plus the probe outcome for metadata: the probe is an
    extra provider call and must stay visible to trace/cost accounting.
    """
    remaining = [target for target in _GREP_TARGETS if target not in targets]
    if candidate_count or not remaining:
        return None, None
    try:
        other_bucket_hits = await grep_raw(
            str(args["pattern"]).strip(),
            1,
            store_identifiers=store_identifiers,
            targets=remaining,
            case_sensitive=bool(args.get("case_sensitive", False)),
            filter_by=filter_by,
            filter_mode=args.get("filter_mode") or "all",
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except Exception:
        # The note is best-effort: a probe failure must not turn a successful empty grep into a tool error.
        return None, "error"
    if not other_bucket_hits:
        return None, "no_match"
    return (
        f"No matches in targets={list(targets)}, but this pattern does match in "
        f"targets={remaining} in at least one searched store; re-run with those "
        "targets (or omit targets to search both) before concluding the corpus "
        "lacks the pattern."
    ), "matched_other_bucket"


async def execute_overview_search(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> ToolOutcome:
    query = str(args["query"]).strip()
    top_k = OVERVIEW_SEARCH_TOP_K
    # Half the search_corpus overfetch: dedup drops already-seen chunks, so a
    # bare top_k fetch returns fewer and fewer results as an episode goes on.
    # Overview payloads are summary-only, so they need less headroom than the
    # full-text search paths to still fill top_k after dedup.
    search_top_k = top_k * (SEARCH_OVERFETCH_FACTOR // 2)
    result = await overview_search(
        query,
        store_identifiers=store_identifiers,
        top_k=search_top_k,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    candidates = result.get("results") or []
    ingest_stats: dict[str, int] = {}
    overview_chunks = index.ingest_overview_results(
        candidates,
        max_new_chunks=top_k,
        stats=ingest_stats,
    )
    payload = {
        "tool": "overview_search",
        "query": result.get("query"),
        "requested_top_k": top_k,
        "search_top_k": search_top_k,
        "results": serialize_overview_agent_chunks(overview_chunks, refs=index.refs),
        "deduped_existing_or_deleted": ingest_stats.get("skipped_existing_or_deleted", 0),
    }
    _enforce_call_payload_budget(payload, budget=config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET)
    return ToolOutcome(
        _drop_empty(payload),
        {
            "query": query,
            "source": "overview_search",
            "k": top_k,
            "search_k": search_top_k,
            "overview_chunks_added": len(overview_chunks),
            "summaries_found": result.get("summaries_found"),
            "summaries_missing": result.get("summaries_missing"),
        },
    )


async def execute_get_chunks(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    chunk_ids = [str(chunk_id).strip() for chunk_id in args["chunk_ids"]]
    fetch = await _fetch_requested_chunks(
        chunk_ids, index=index, client=client, api_key=api_key, api_key_env=api_key_env
    )
    clipped_chunk_ids, budget_clipped_count = _clip_fetched_chunks_to_budget(
        fetch.fetched,
        requested_chunk_ids=chunk_ids,
        restored_chunk_ids=fetch.restored_chunk_ids,
        results=fetch.results,
    )

    payload: dict[str, Any] = {
        "tool": "get_chunks",
        "requested_chunk_ids": chunk_ids,
        "results": fetch.results,
        "restored_chunk_ids": fetch.restored_chunk_ids,
    }
    if fetch.invalid_chunk_ids:
        payload["invalid_chunk_ids"] = fetch.invalid_chunk_ids
    if clipped_chunk_ids:
        payload["clipped_chunk_ids"] = clipped_chunk_ids
        payload["clipped_chunk_count"] = len(clipped_chunk_ids)
    if len(fetch.fetched) == 1 and clipped_chunk_ids:
        # The escalation ceiling: a lone chunk still over the call budget cannot be
        # shown in full by ANY request — say so instead of inviting a retry.
        payload["budget_notice"] = (
            f"This chunk exceeds the {config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET}-token "
            "per-call display limit; no get_chunks request can show more of it than "
            "this. Use read_document with x=0 for a windowed view, or proceed with "
            "what is shown."
        )
    elif budget_clipped_count:
        payload["budget_notice"] = _payload_budget_notice(
            budget_clipped_count, config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET
        )
    return payload


@dataclass(slots=True)
class _FetchedChunks:
    """What resolving and fetching the requested chunk ids produced.

    ``invalid_chunk_ids`` are ids the model invented or cannot reach, reported
    separately from the per-result errors so the caller can mark the trace
    event agent-caused: a "Chunk not found" from the store is not the model's
    mistake, an unresolvable chunk_id is.
    """

    results: list[dict[str, Any]]
    restored_chunk_ids: list[str]
    invalid_chunk_ids: list[str]
    fetched: list[tuple[str, dict[str, Any]]]


async def _fetch_requested_chunks(
    chunk_ids: Sequence[str],
    *,
    index: ChunkIndex,
    client: AsyncRetrievalClient | None,
    api_key: str | None,
    api_key_env: str | None,
) -> _FetchedChunks:
    fetch = _FetchedChunks(results=[], restored_chunk_ids=[], invalid_chunk_ids=[], fetched=[])
    for chunk_id in chunk_ids:
        try:
            key = index.refs.chunk_key_for_id(chunk_id)
        except ValueError as exc:
            fetch.results.append({"chunk_id": chunk_id, "error": str(exc)})
            fetch.invalid_chunk_ids.append(chunk_id)
            continue
        if not index.is_visible_chunk(key):
            fetch.results.append(
                {
                    "chunk_id": chunk_id,
                    "error": "chunk_id is not available in this agent context",
                }
            )
            fetch.invalid_chunk_ids.append(chunk_id)
            continue

        result = await get_chunk(
            file_id=key[1],
            store_id=key[0],
            chunk_index=key[2],
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        if "error" in result:
            fetch.results.append({"chunk_id": chunk_id, "error": result["error"]})
            continue

        index.add_chunk(result, restore=True)
        fetch.restored_chunk_ids.append(chunk_id)
        serialized = serialize_agent_chunk(result, refs=index.refs)
        fetch.fetched.append((chunk_id, serialized))
        fetch.results.append(serialized)
    return fetch


def _clip_fetched_chunks_to_budget(
    fetched: list[tuple[str, dict[str, Any]]],
    *,
    requested_chunk_ids: list[str],
    restored_chunk_ids: list[str],
    results: list[dict[str, Any]],
) -> tuple[list[str], int]:
    """Clip the fetched chunks in place to the per-call payload budget.

    Returns the chunk ids that lost text and how many of them lost it to the
    call budget rather than to the per-chunk cap.
    """
    if not fetched:
        return [], 0

    sizes = [estimate_payload_tokens(item) for _, item in fetched]
    # A single-id request is an explicit escalation ("show me this one"), so the
    # per-chunk cap is waived and the chunk may fill the whole call budget.
    per_chunk = config.GET_CHUNKS_CHUNK_TOKEN_LIMIT if len(fetched) > 1 else 0
    # What each chunk will occupy after its per-chunk clip; the allocator splits the rest.
    effective = [
        min(size, per_chunk + _BUDGET_CLIP_SLACK_TOKENS) if 0 < per_chunk < size else size
        for size in sizes
    ]

    def envelope_tokens(reporting: Mapping[str, Any]) -> int:
        return estimate_payload_tokens(
            {
                "tool": "get_chunks",
                "requested_chunk_ids": requested_chunk_ids,
                "restored_chunk_ids": restored_chunk_ids,
                "results": [entry for entry in results if "error" in entry],
                **reporting,
            }
        )

    def allocate(item_budget: int) -> list[int]:
        caps = _spread_token_budget(effective, item_budget, floor=config.MIN_ALLOCATION_TOKENS)
        return [min(cap, per_chunk) if per_chunk > 0 else cap for cap in caps]

    reporting: dict[str, Any] = {}
    # The reporting keys consume payload budget too, so allocate against an
    # envelope that includes them; the notice count feeds the notice text,
    # so iterate the (tiny) fixpoint until it stabilises.
    for _ in range(3):
        item_budget = max(
            config.MIN_ALLOCATION_TOKENS,
            config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET - envelope_tokens(reporting),
        )
        final_caps = allocate(item_budget)
        planned_clipped = [
            chunk_id
            for (chunk_id, _), cap, size in zip(fetched, final_caps, sizes, strict=True)
            if size > cap
        ]
        planned_budget_count = sum(
            1
            for (_, _), cap, size in zip(fetched, final_caps, sizes, strict=True)
            if size > cap and cap < min(size, per_chunk if per_chunk > 0 else size)
        )
        next_reporting: dict[str, Any] = {}
        if planned_clipped:
            next_reporting = {
                "clipped_chunk_ids": planned_clipped,
                "clipped_chunk_count": len(planned_clipped),
            }
            if planned_budget_count:
                next_reporting["budget_notice"] = _payload_budget_notice(
                    planned_budget_count, config.TOOL_CALL_PAYLOAD_TOKEN_BUDGET
                )
        if next_reporting == reporting:
            break
        reporting = next_reporting

    clipped_chunk_ids: list[str] = []
    budget_clipped_count = 0
    for (chunk_id, item), cap, size in zip(fetched, final_caps, sizes, strict=True):
        if size <= cap:
            continue
        # Report from what actually clipped: a chunk whose oversize sits in
        # non-clippable fields keeps its text and is not counted.
        if _clip_chunk_to_cap(item, cap, estimated=size):
            clipped_chunk_ids.append(chunk_id)
            if cap < min(size, per_chunk if per_chunk > 0 else size):
                budget_clipped_count += 1
    return clipped_chunk_ids, budget_clipped_count


async def execute_read_document(
    args: Mapping[str, Any],
    *,
    index: ChunkIndex,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    requested_keys, invalid_request = _resolve_read_document_request(args, index)
    if invalid_request is not None:
        return invalid_request
    requested_doc_key, requested_chunk_key = requested_keys

    raw_window_size = args.get("x", 1)
    window_size = max(int(1 if raw_window_size is None else raw_window_size), 0)
    anchor_chunk_index = requested_chunk_key[2]
    start_chunk_index = max(0, anchor_chunk_index - window_size)
    end_chunk_index = anchor_chunk_index + window_size
    requested_chunk_indices = list(range(start_chunk_index, end_chunk_index + 1))
    result = await read_document(
        file_id=requested_doc_key[1],
        store_id=requested_doc_key[0],
        chunk_indices=requested_chunk_indices,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    chunks = sorted(result.get("chunks") or [], key=lambda chunk: chunk_key(chunk)[2])
    doc_key = document_key(result)
    document_id = index.refs.document_id_for_key(doc_key)
    document_was_pruned = doc_key in index.deleted_document_keys
    new_chunks = (
        [] if document_was_pruned else index.ingest_search_results(chunks, max_new_chunks=None)
    )
    new_chunk_keys = {chunk_key(new_chunk) for new_chunk in new_chunks}
    window_chunks = _visible_window_chunks(
        chunks,
        index=index,
        new_chunk_keys=new_chunk_keys,
        document_was_pruned=document_was_pruned,
    )
    returned_chunk_keys = {chunk_key(chunk) for chunk in window_chunks}
    omitted_count = sum(1 for chunk in chunks if chunk_key(chunk) not in returned_chunk_keys)
    payload = {
        "tool": "read_document",
        "document_id": document_id,
        "anchor_chunk_id": str(args["chunk_id"]),
        "anchor_chunk_index": anchor_chunk_index,
        "x": window_size,
        "requested_chunk_indices": requested_chunk_indices,
        "filename": result.get("filename"),
        "external_id": result.get("external_id"),
        "metadata": result.get("metadata"),
        "status": result.get("status"),
        "chunks": serialize_agent_chunks(window_chunks, refs=index.refs),
        # An id-only side channel into chunks[], like get_chunks' restored_chunk_ids:
        # every new chunk is already in the window, so re-serializing it here would
        # only buy the same text a second time.
        "new_unseen_chunk_ids": [
            index.refs.chunk_id_for_key(chunk_key(chunk)) for chunk in new_chunks
        ],
        "document_content_pruned": document_was_pruned,
    }
    if index.is_scoped():
        payload["omitted_deleted_or_pruned_chunks_count"] = omitted_count
    else:
        payload["omitted_deleted_or_pruned_chunks"] = [
            chunk_identifier(chunk, refs=index.refs)
            for chunk in chunks
            if chunk_key(chunk) not in returned_chunk_keys
        ]
    budget = config.READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET
    if estimate_payload_tokens(payload) > budget:
        _budget_read_document_payload(
            payload,
            window_chunks,
            anchor_chunk_index=anchor_chunk_index,
            new_chunk_keys=new_chunk_keys,
            index=index,
            budget=budget,
        )
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def _resolve_read_document_request(
    args: Mapping[str, Any],
    index: ChunkIndex,
) -> tuple[tuple[DocumentKey, ChunkKey] | None, dict[str, Any] | None]:
    """Resolve the request's ids to keys, or return the failure payload.

    ``invalid_request`` marks the failure as the model's own (an id it
    invented, cannot reach, or paired wrong) so the caller can mark the trace
    event agent-caused; provider failures raise out of the executor instead.
    """
    try:
        requested_doc_key = index.refs.document_key_for_id(str(args["document_id"]))
        requested_chunk_key = index.refs.chunk_key_for_id(str(args["chunk_id"]))
    except ValueError as exc:
        return None, {"tool": "read_document", "error": str(exc), "invalid_request": True}
    if not index.is_visible_document(requested_doc_key):
        return None, {
            "tool": "read_document",
            "document_id": str(args["document_id"]),
            "error": "document_id is not available in this agent context",
            "invalid_request": True,
        }
    if not index.is_visible_chunk(requested_chunk_key):
        return None, {
            "tool": "read_document",
            "chunk_id": str(args["chunk_id"]),
            "error": "chunk_id is not available in this agent context",
            "invalid_request": True,
        }
    if document_key_from_parts(requested_chunk_key[0], requested_chunk_key[1]) != requested_doc_key:
        return None, {
            "tool": "read_document",
            "document_id": str(args["document_id"]),
            "chunk_id": str(args["chunk_id"]),
            "error": "chunk_id does not belong to document_id",
            "invalid_request": True,
        }
    return (requested_doc_key, requested_chunk_key), None


def _visible_window_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    index: ChunkIndex,
    new_chunk_keys: set[ChunkKey],
    document_was_pruned: bool,
) -> list[dict[str, Any]]:
    """The window's chunks the agent may see: deduplicated, prunes dropped."""
    if document_was_pruned:
        return []
    window_chunks: list[dict[str, Any]] = []
    seen_window_keys: set[ChunkKey] = set()
    for chunk in chunks:
        key = chunk_key(chunk)
        chunk_is_pruned = key in index.deleted_chunk_keys and key not in index.restored_chunk_keys
        if key in seen_window_keys or chunk_is_pruned:
            continue
        seen_window_keys.add(key)
        if key in new_chunk_keys or index.expose_chunk_reference(chunk):
            window_chunks.append(index.get(key) or dict(chunk))
    return window_chunks


async def enrich_file_payload(
    file_payload: Mapping[str, Any],
    *,
    index: ChunkIndex,
    include_chunks: bool,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Attach local document/chunk handles to a file-level metadata result."""
    if not file_payload.get("file_id") or not file_payload.get("store_id"):
        return agent_file_payload(file_payload)

    enriched = agent_file_payload(file_payload)
    enriched["document_id"] = index.register_document(file_payload)
    if not include_chunks:
        return enriched

    try:
        document = await read_document(
            file_id=str(file_payload["file_id"]),
            store_id=str(file_payload["store_id"]),
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except Exception as exc:
        if is_provider_failure(exc):
            raise
        enriched["chunk_read_error"] = str(exc)
        return enriched

    chunks = document.get("chunks") or []
    index.ingest_search_results(chunks, max_new_chunks=None)
    visible_chunks: list[dict[str, Any]] = []
    visible_stored_chunks: list[dict[str, Any]] = []
    seen_chunk_keys: set[ChunkKey] = set()
    for chunk in chunks:
        key = chunk_key(chunk)
        if key in seen_chunk_keys or not index.is_visible_chunk(key):
            continue
        stored_chunk = index.get(key)
        if stored_chunk is None:
            continue
        visible_chunks.append(serialize_agent_chunk(stored_chunk, refs=index.refs))
        visible_stored_chunks.append(stored_chunk)
        seen_chunk_keys.add(key)

    if visible_chunks:
        enriched["chunks"] = visible_chunks
        enriched["content"] = agent_document_content_text(
            visible_stored_chunks,
            refs=index.refs,
        )
    return enriched


def agent_document_content_text(
    chunks: Sequence[Mapping[str, Any]],
    *,
    refs: ReferenceRegistry,
) -> str:
    """Join chunk content with short references for model-visible document text."""
    sections: list[str] = []
    for chunk in chunks:
        content = chunk_content_text(chunk)
        if not content:
            continue
        key = chunk_key(chunk)
        sections.append(f"[chunk_id={refs.chunk_id_for_key(key)} chunk_index={key[2]}]\n{content}")
    return "\n\n".join(sections)


def serialize_agent_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    refs: ReferenceRegistry,
    clip_search_payload: bool = False,
) -> list[dict[str, Any]]:
    return [
        serialize_agent_chunk(chunk, refs=refs, clip_search_payload=clip_search_payload)
        for chunk in chunks
    ]


def serialize_overview_agent_chunks(
    chunks: Sequence[Mapping[str, Any]],
    *,
    refs: ReferenceRegistry,
) -> list[dict[str, Any]]:
    """Serialize overview results without exposing metadata or full chunk internals."""
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        key = chunk_key(chunk)
        chunk_id, document_id = refs.ids_for_chunk_key(key)
        payload: dict[str, Any] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
        }
        filename = chunk.get("filename") or chunk.get("file_title")
        if filename:
            payload["filename"] = filename
        summary = chunk.get("summary")
        if isinstance(summary, str) and summary.strip():
            payload["summary"] = summary.strip()
        results.append(payload)
    return results


def serialize_agent_chunk(
    chunk: Mapping[str, Any],
    *,
    refs: ReferenceRegistry,
    clip_search_payload: bool = False,
    clip_focus: re.Pattern[str] | None = None,
) -> dict[str, Any]:
    key = chunk_key(chunk)
    chunk_id, document_id = refs.ids_for_chunk_key(key)
    payload = AgentChunkPayload.from_chunk(
        dict(chunk),
        chunk_id=chunk_id,
        document_id=document_id,
    ).model_dump(
        mode="json",
        exclude_none=True,
        # The SDK nests union-typed chunk models whose serializer mismatches are
        # harmless; formatting the warnings reprs whole payloads per chunk.
        warnings=False,
    )
    image_url = chunk.get("image_url")
    if image_url not in (None, "", []) and config.include_media_content_for_chunk(chunk):
        payload["image_url"] = image_url
    if clip_search_payload:
        if clip_focus is not None:
            # grep: always keep ~GREP_MATCH_WINDOW_TOKENS of context around every
            # occurrence of the pattern, independent of SEARCH_CHUNK_TOKEN_LIMIT.
            _clip_grep_match_windows(
                payload,
                window_tokens=config.GREP_MATCH_WINDOW_TOKENS,
                focus=clip_focus,
            )
        elif config.SEARCH_CHUNK_TOKEN_LIMIT > 0:
            _clip_chunk_text_fields(payload, max_tokens=config.SEARCH_CHUNK_TOKEN_LIMIT)
    return payload


def agent_file_payload(file_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a file/document payload without backend Mixedbread identifiers."""
    return {key: value for key, value in file_payload.items() if key not in BACKEND_ID_FIELDS}


def chunk_key(chunk: Mapping[str, Any]) -> ChunkKey:
    return chunk_key_from_parts(
        str(chunk.get("store_id", "")),
        str(chunk.get("file_id", "")),
        int(chunk.get("chunk_index", 0) or 0),
    )


def ranked_chunk_key(ranked: RankedChunk, refs: ReferenceRegistry) -> ChunkKey:
    return refs.chunk_key_for_id(ranked.chunk_id)


def document_key(document: Mapping[str, Any]) -> DocumentKey:
    return document_key_from_parts(
        str(document.get("store_id", "")),
        str(document.get("file_id", "")),
    )


def chunk_score(chunk: Mapping[str, Any]) -> float:
    return float(chunk.get("search_score", chunk.get("score", 0.0)) or 0.0)


def _drop_empty(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None and value != ""}


def chunk_identifier(
    chunk: Mapping[str, Any],
    *,
    refs: ReferenceRegistry,
) -> dict[str, Any]:
    key = chunk_key(chunk)
    return {"chunk_id": refs.chunk_id_for_key(key), "chunk_index": key[2]}


def _seen_chunk_reference(
    chunk: Mapping[str, Any],
    *,
    index: ChunkIndex,
    rank_by: str | None,
) -> dict[str, Any]:
    key = chunk_key(chunk)
    chunk_id, document_id = index.refs.ids_for_chunk_key(key)
    payload: dict[str, Any] = {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": key[2],
        "seen": True,
    }
    filename = chunk.get("filename")
    if filename is not None:
        payload["filename"] = filename
    if rank_by:
        value = metadata_lookup(chunk, rank_by)
        if value is not None:
            payload[rank_by] = value
    return payload


def document_identifier(
    document: Mapping[str, Any],
    *,
    refs: ReferenceRegistry,
) -> dict[str, Any]:
    key = document_key(document)
    return {"document_id": refs.document_id_for_key(key)}


def redacted_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in dict(chunk).items() if key not in CONTENT_FIELDS}
    payload["content_pruned"] = True
    return payload


def redact_chunk_media(chunk: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(chunk)
    for media_field in ("image_url", "media_url", "audio_url", "video_url", "content_url"):
        payload.pop(media_field, None)
    payload["media_pruned"] = True
    return payload


def merge_chunk_payload(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge a fuller chunk payload into an existing overview/full payload."""
    merged = dict(existing or {})
    for key, value in dict(incoming).items():
        if value not in (None, "", []):
            merged[key] = value
    return merged


_TRUNCATABLE_TOOLS = {
    "search_corpus",
    "grep",
    "filter_chunks",
    "overview_search",
    "read_document",
    "get_chunks",
}
_TRUNCATION_RESULT_FIELDS = ("new_unseen_results", "results", "chunks")
_TRUNCATION_TEXT_FIELDS = ("text", "content", "ocr_text", "transcription", "context", "summary")
# Budget layer (_clip_text_fields): the agent's context really is exhausted, and
# prune_context is the remedy — see _set_truncation_notice.
_TRUNCATION_TEXT_MARKER = "…[truncated: context token limit reached]"
# Per-chunk layer (_clip_chunk_text_fields): a fixed per-chunk cap unrelated to
# the agent's context budget, so it must not read as budget exhaustion. Nothing
# is freed by pruning; the full text is still one read_document/get_chunks away.
_CHUNK_CLIP_MARKER = "…[truncated: chunk payload shortened]"
_CHUNK_CLIP_PREFIX_MARKER = "[truncated: chunk payload shortened]…"
# Full-content layer (_clip_chunk_to_cap with the default marker): get_chunks is
# the call a model makes to recover text a search result already shortened, so
# its marker quantifies what is still missing instead of only saying that something is.
_CHUNK_TRUNCATION_CHARS_MARKER = "\n[... truncated: showing first {shown} of {total} characters]"
# Turn-budget layer: fires when one round's combined tool payloads cross
# TURN_TOOL_PAYLOAD_TOKEN_BUDGET with context headroom to spare, so the marker
# must not claim the context window is exhausted.
_TURN_TRUNCATION_TEXT_MARKER = "…[truncated: turn payload budget reached]"
_TRUNCATION_MIN_TEXT_CHARS = 200


def estimate_payload_tokens(payload: Any) -> int:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return config.count_text_tokens(serialized)


def truncate_round_payloads(
    payloads: Sequence[dict[str, Any]],
    *,
    index: ChunkIndex,
    remaining_tokens: int,
    turn_capped: bool = False,
) -> list[dict[str, Any] | None]:
    """Clip one round's tool payloads to the remaining prompt-token budget.

    Result entries are dropped whole from the tail of the score-sorted lists so
    surviving chunk payloads stay intact and parseable; every payload keeps at
    least its top entry. Dropped chunks the model never saw are discarded from
    ``index`` (and sibling-payload references to them scrubbed) so tool schemas
    and metrics only offer IDs that are actually in context; unlike pruned
    chunks they may be re-surfaced by a later search. ``turn_capped`` says the
    bound came from TURN_TOOL_PAYLOAD_TOKEN_BUDGET rather than context
    exhaustion, which changes the marker/notice wording and re-prunes (not
    discards) chunks a get_chunks call had restored. Returns one stats dict
    per payload, ``None`` where a payload was left untouched.
    """
    stats: list[dict[str, Any] | None] = [None] * len(payloads)
    estimates = [estimate_payload_tokens(payload) for payload in payloads]
    budget = max(0, remaining_tokens)
    if sum(estimates) <= budget:
        return stats

    truncatable = [
        payload_index
        for payload_index, payload in enumerate(payloads)
        if payload.get("tool") in _TRUNCATABLE_TOOLS
        # A single-id get_chunks call is the sanctioned recovery escalation; the
        # turn pass re-truncating it would undo exactly what the model asked for.
        and not (
            payload.get("tool") == "get_chunks"
            and len(payload.get("requested_chunk_ids") or []) == 1
        )
    ]
    if not truncatable:
        return stats
    fixed_cost = sum(
        estimate
        for payload_index, estimate in enumerate(estimates)
        if payload_index not in truncatable
    )
    budget = max(0, budget - fixed_cost)

    # Water-fill: small payloads keep everything, the remaining budget is split
    # evenly among the oversized ones so no single parallel call is sacrificed.
    allocations: dict[int, int] = {}
    for position, payload_index in enumerate(
        sorted(truncatable, key=lambda candidate: estimates[candidate])
    ):
        share = budget // (len(truncatable) - position)
        allocations[payload_index] = min(estimates[payload_index], share)
        budget -= allocations[payload_index]

    dropped_new_chunk_ids: set[str] = set()
    repruned_chunk_ids: set[str] = set()
    for payload_index in truncatable:
        payload = payloads[payload_index]
        if estimates[payload_index] <= allocations[payload_index]:
            continue
        payload_stats = _truncate_payload(
            payload,
            max_tokens=allocations[payload_index],
            dropped_new_chunk_ids=dropped_new_chunk_ids,
            repruned_chunk_ids=repruned_chunk_ids,
            turn_capped=turn_capped,
        )
        payload_stats["estimated_tokens_before"] = estimates[payload_index]
        payload_stats["estimated_tokens_after"] = estimate_payload_tokens(payload)
        payload_stats["budget_kind"] = "turn" if turn_capped else "context"
        stats[payload_index] = payload_stats

    if dropped_new_chunk_ids:
        for payload_index in truncatable:
            payload = payloads[payload_index]
            scrubbed = _scrub_chunk_references(payload, chunk_ids=dropped_new_chunk_ids)
            if not scrubbed:
                continue
            payload_stats = stats[payload_index] or {"results_omitted": 0}
            payload_stats["results_omitted"] += scrubbed
            _set_truncation_notice(
                payload, payload_stats["results_omitted"], turn_capped=turn_capped
            )
            stats[payload_index] = payload_stats
        _discard_dropped_chunks(index, dropped_new_chunk_ids)
    if repruned_chunk_ids:
        _reprune_dropped_chunks(index, repruned_chunk_ids)
    return stats


def _truncate_payload(
    payload: dict[str, Any],
    *,
    max_tokens: int,
    dropped_new_chunk_ids: set[str],
    repruned_chunk_ids: set[str],
    turn_capped: bool,
) -> dict[str, Any]:
    new_chunk_ids: set[str] | None = None
    if payload.get("tool") == "read_document":
        new_chunk_ids = set(payload.get("new_unseen_chunk_ids") or [])

    restored_ids: set[str] | None = None
    if payload.get("tool") == "get_chunks":
        restored_ids = set(payload.get("restored_chunk_ids") or [])

    dropped = 0
    entries = _primary_result_list(payload)
    # Track the estimate arithmetically while dropping; re-serializing the whole
    # payload per entry is O(entries x payload), which a real tokenizer makes visible.
    payload_estimate = estimate_payload_tokens(payload)
    while entries and len(entries) > 1 and payload_estimate > max_tokens:
        entry = entries.pop()
        dropped += 1
        payload_estimate -= estimate_payload_tokens(entry)
        chunk_id = _entry_chunk_id(entry)
        if not chunk_id:
            continue
        _drop_chunk_id(payload, "new_unseen_chunk_ids", chunk_id)
        if restored_ids is not None and chunk_id in restored_ids:
            # The model never saw this restored chunk: return it to pruned state.
            repruned_chunk_ids.add(chunk_id)
            _drop_chunk_id(payload, "restored_chunk_ids", chunk_id)
            continue
        is_new = (
            chunk_id in new_chunk_ids
            if new_chunk_ids is not None
            else not (isinstance(entry, Mapping) and entry.get("seen"))
        )
        if is_new:
            dropped_new_chunk_ids.add(chunk_id)

    content_clipped = False
    if estimate_payload_tokens(payload) > max_tokens:
        content_clipped = _clip_text_fields(
            payload,
            max_tokens=max_tokens,
            marker=_TURN_TRUNCATION_TEXT_MARKER if turn_capped else _TRUNCATION_TEXT_MARKER,
        )

    payload_stats: dict[str, Any] = {
        "results_omitted": dropped,
        "content_clipped": content_clipped,
    }
    if dropped:
        _set_truncation_notice(payload, dropped, turn_capped=turn_capped)
    return payload_stats


def _primary_result_list(payload: dict[str, Any]) -> list[Any] | None:
    for field_name in _TRUNCATION_RESULT_FIELDS:
        value = payload.get(field_name)
        if isinstance(value, list) and value:
            return value
    return None


def _entry_chunk_id(entry: Any) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    chunk_id = str(entry.get("chunk_id") or "").strip()
    return chunk_id or None


def _drop_chunk_id(payload: dict[str, Any], field: str, chunk_id: str) -> None:
    """Keep an id-only side channel in step with the results list it points into."""
    ids = payload.get(field)
    if isinstance(ids, list):
        payload[field] = [existing for existing in ids if existing != chunk_id]


def _clip_chunk_text_fields(payload: dict[str, Any], *, max_tokens: int) -> bool:
    """Clip one chunk payload's text fields to ``max_tokens`` (estimated).

    Per-chunk companion to ``_clip_text_fields``, applied at serialization time
    when ``config.SEARCH_CHUNK_TOKEN_LIMIT`` is set — for corpora whose chunks
    are whole pages, so a single search's payload stays proportional to top_k
    rather than page size. grep does not use this path; it always centres kept
    windows on the matches via ``_clip_grep_match_windows`` instead.
    """
    clipped = False
    for field_name in _TRUNCATION_TEXT_FIELDS:
        excess_tokens = estimate_payload_tokens(payload) - max_tokens
        if excess_tokens <= 0:
            break
        value = payload.get(field_name)
        if not isinstance(value, str) or len(value) <= _TRUNCATION_MIN_TEXT_CHARS:
            continue
        keep_chars = max(
            _TRUNCATION_MIN_TEXT_CHARS,
            len(value) - excess_tokens * config.TOKEN_ESTIMATE_CHARS_PER_TOKEN,
        )
        if keep_chars >= len(value):
            continue
        payload[field_name] = value[:keep_chars].rstrip() + _CHUNK_CLIP_MARKER
        clipped = True
    return clipped


# Marker + envelope slack so an item clipped to its cap still counts as fitting it.
_BUDGET_CLIP_SLACK_TOKENS = 64


def _spread_token_budget(sizes: Sequence[int], budget: int, *, floor: int) -> list[int]:
    """Per-item caps biased to the head of the list: linear rank weights, water-filled
    so small items keep their full size and release slack; every item keeps >= ``floor``."""
    n = len(sizes)
    if n == 0 or sum(sizes) <= budget:
        return list(sizes)
    weights = [n - i for i in range(n)]
    caps = [0] * n
    active = set(range(n))
    remaining = budget
    while active:
        total_weight = sum(weights[i] for i in active)
        progressed = False
        for i in sorted(active):
            share = remaining * weights[i] // total_weight
            if sizes[i] <= max(floor, share) and sizes[i] <= remaining:
                caps[i] = sizes[i]
                remaining -= sizes[i]
                active.discard(i)
                progressed = True
        if not progressed:
            break
    if active:
        total_weight = sum(weights[i] for i in active)
        for i in active:
            caps[i] = max(floor, remaining * weights[i] // total_weight)
        total_active = sum(caps[i] for i in active)
        if total_active > remaining:
            # Only reachable when budget < len(active) * floor; scale down in weight proportion.
            for i in active:
                caps[i] = max(1, caps[i] * remaining // total_active)
    return caps


def _enforce_call_payload_budget(payload: dict[str, Any], *, budget: int) -> None:
    """Spread-clip a retrieval payload's results when the call exceeds ``budget``;
    the payload is byte-identical when it fits."""
    entries = _primary_result_list(payload)
    if not entries:
        return
    total = estimate_payload_tokens(payload)
    if total <= budget:
        return
    item_positions = [
        position
        for position, entry in enumerate(entries)
        if isinstance(entry, dict) and not entry.get("seen")
    ]
    if not item_positions:
        return
    sizes = [estimate_payload_tokens(entries[position]) for position in item_positions]

    def envelope_tokens(reporting: Mapping[str, Any]) -> int:
        return total - sum(sizes) + (estimate_payload_tokens(reporting) if reporting else 0)

    # The reporting keys consume payload budget too, so allocate against an
    # envelope that includes them; iterate the (tiny) fixpoint until it stabilises.
    reporting: dict[str, Any] = {}
    caps = list(sizes)
    for _ in range(3):
        caps = _spread_token_budget(
            sizes,
            max(config.MIN_ALLOCATION_TOKENS, budget - envelope_tokens(reporting)),
            floor=config.MIN_ALLOCATION_TOKENS,
        )
        planned_ids = [
            str(entries[position].get("chunk_id") or f"result-{position}")
            for position, cap, size in zip(item_positions, caps, sizes, strict=True)
            if size > cap
        ]
        next_reporting: dict[str, Any] = {}
        if planned_ids:
            next_reporting = {
                "clipped_chunk_ids": planned_ids,
                "clipped_chunk_count": len(planned_ids),
                "budget_notice": _payload_budget_notice(len(planned_ids), budget),
            }
        if next_reporting == reporting:
            break
        reporting = next_reporting
    applied_ids: list[str] = []
    for position, cap, size in zip(item_positions, caps, sizes, strict=True):
        if size > cap and _clip_chunk_to_cap(
            entries[position], cap, estimated=size, quantified=False
        ):
            # Report from what actually clipped: an entry whose oversize sits in
            # non-clippable fields keeps its text and is not counted.
            applied_ids.append(str(entries[position].get("chunk_id") or f"result-{position}"))
    if applied_ids:
        payload["clipped_chunk_ids"] = applied_ids
        payload["clipped_chunk_count"] = len(applied_ids)
        payload["budget_notice"] = _payload_budget_notice(len(applied_ids), budget)


def _budget_read_document_payload(
    payload: dict[str, Any],
    window_chunks: list[dict[str, Any]],
    *,
    anchor_chunk_index: int,
    new_chunk_keys: set[ChunkKey],
    index: ChunkIndex,
    budget: int,
) -> None:
    """Spread-clip an over-budget read_document payload in place.

    The serialized entries are measured and clipped (fresh objects, so the index
    never mutates), anchor-first by distance. chunks[] is the only text carrier,
    so an entry's size is its whole cost and the budget spreads over sizes."""
    if window_chunks:
        entries = serialize_agent_chunks(window_chunks, refs=index.refs)
        new_ids = {index.refs.chunk_id_for_key(key) for key in new_chunk_keys}
        sizes = [estimate_payload_tokens(entry) for entry in entries]
        anchor_position = next(
            (
                position
                for position, entry in enumerate(entries)
                if entry.get("chunk_index") == anchor_chunk_index
            ),
            None,
        )
        if anchor_position is None:
            # A pruned anchor stays visible but out of the window: allocate from its neighbour.
            anchor_position = min(
                range(len(entries)),
                key=lambda p: abs(int(entries[p].get("chunk_index", 0)) - anchor_chunk_index),
            )
        priority = sorted(range(len(entries)), key=lambda i: (abs(i - anchor_position), i))
        envelope_tokens = estimate_payload_tokens(payload) - sum(sizes)
        item_budget = max(config.MIN_ALLOCATION_TOKENS, budget - envelope_tokens)
        ordered_caps = _spread_token_budget(
            [sizes[i] for i in priority],
            item_budget,
            floor=config.MIN_ALLOCATION_TOKENS * 2,
        )
        caps = [0] * len(entries)
        for position, cap in zip(priority, ordered_caps, strict=True):
            caps[position] = max(config.MIN_ALLOCATION_TOKENS, cap)
        clipped_chunk_ids: list[str] = []
        for entry, cap, size in zip(entries, caps, sizes, strict=True):
            if size <= cap:
                continue
            if _clip_chunk_to_cap(entry, cap, estimated=size):
                clipped_chunk_ids.append(str(entry["chunk_id"]))
        payload["chunks"] = entries
        payload["new_unseen_chunk_ids"] = [
            entry["chunk_id"] for entry in entries if entry["chunk_id"] in new_ids
        ]
        if clipped_chunk_ids:
            payload["clipped_chunk_ids"] = clipped_chunk_ids
            payload["clipped_chunk_count"] = len(clipped_chunk_ids)
            payload["budget_notice"] = _payload_budget_notice(len(clipped_chunk_ids), budget)
    if estimate_payload_tokens(payload) > budget:
        _clip_chunk_to_cap(payload, budget)


def _payload_budget_notice(truncated_count: int, budget: int) -> str:
    return (
        f"{truncated_count} chunks had to be truncated due to this call reaching a "
        f"{budget}-token payload budget. Please inspect the returned results and "
        "proceed with the context provided. You can call prune_context to free "
        "budget, and if needed you may use some of your calls to make more targeted "
        "requests — a truncated chunk re-requested alone is shown in full."
    )


def _clip_chunk_to_cap(
    payload: dict[str, Any],
    cap_tokens: int,
    *,
    estimated: int | None = None,
    quantified: bool = True,
) -> bool:
    """Clip one serialized chunk payload to ``cap_tokens``: text fields and metadata
    string values are shortened in proportion to their length (pristine totals in the
    marker), then oversize metadata values are elided. Returns whether anything changed."""
    estimated = estimate_payload_tokens(payload) if estimated is None else estimated
    if estimated <= cap_tokens:
        return False
    estimated, clipped, metadata_clipped = _shave_text_fields(
        payload, cap_tokens, estimated=estimated, quantified=quantified
    )
    elided = _elide_oversize_metadata(payload, cap_tokens, estimated=estimated)
    if metadata_clipped or elided:
        payload["metadata_clipped"] = True
    return clipped or elided


def _shave_text_fields(
    payload: dict[str, Any],
    cap_tokens: int,
    *,
    estimated: int,
    quantified: bool,
) -> tuple[int, bool, bool]:
    """Shorten the payload's string values in proportion to their length.

    Returns the resulting estimate and whether any text or metadata value was
    shortened. The chars-per-token ratio starts at the configured default and
    adapts to what each pass actually removed.
    """
    originals: dict[tuple[str, str], str] = {}
    keeps: dict[tuple[str, str], int] = {}
    chars_per_token = float(config.TOKEN_ESTIMATE_CHARS_PER_TOKEN)
    clipped = False
    metadata_clipped = False

    def candidates() -> list[tuple[tuple[str, str], str, int, int]]:
        found = []
        containers: list[tuple[str, Mapping[str, Any]]] = [("text", payload)]
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            containers.append(("metadata", metadata))
        for kind, container in containers:
            names = _TRUNCATION_TEXT_FIELDS if kind == "text" else list(container)
            for name in names:
                value = container.get(name)
                if not isinstance(value, str):
                    continue
                label = (kind, name)
                original = originals.setdefault(label, value)
                keep = keeps.get(label, len(original))
                floor = min(_TRUNCATION_MIN_TEXT_CHARS, len(original))
                if keep > floor:
                    found.append((label, original, floor, keep))
        return found

    for _ in range(5):  # bounded shave passes; the estimator is a heuristic
        if estimated <= cap_tokens:
            break
        found = candidates()
        total = sum(keep for _, _, _, keep in found)
        if not total:
            break
        excess_chars = int((estimated - cap_tokens) * chars_per_token)
        tokens_before = estimated
        removed_chars = 0
        for label, original, floor, keep in found:
            new_keep = max(floor, keep - max(1, excess_chars * keep // total))
            if new_keep >= keep:
                continue
            kept = original[:new_keep].rstrip()
            marked = kept + (
                _CHUNK_TRUNCATION_CHARS_MARKER.format(shown=len(kept), total=len(original))
                if quantified
                else _CHUNK_CLIP_MARKER
            )
            if label[0] == "text":
                payload[label[1]] = marked
            else:
                payload["metadata"][label[1]] = marked
                metadata_clipped = True
            removed_chars += keep - len(kept)
            keeps[label] = len(kept)
            clipped = True
        estimated = estimate_payload_tokens(payload)
        reduced = tokens_before - estimated
        if removed_chars and reduced > 0:
            # Adapt to the installed token counter: chars/4 is only the default.
            chars_per_token = min(16.0, max(2.0, removed_chars / reduced))
        elif removed_chars:
            chars_per_token = min(16.0, chars_per_token * 2)

    return estimated, clipped, metadata_clipped


def _elide_oversize_metadata(
    payload: dict[str, Any],
    cap_tokens: int,
    *,
    estimated: int,
) -> bool:
    """Replace metadata values with markers until the payload fits the cap.

    Returns whether any value was elided or dropped.
    """
    metadata = payload.get("metadata")
    if estimated <= cap_tokens or not isinstance(metadata, dict):
        return False
    changed = False
    # Elide the biggest values first; provider metadata is arbitrary JSON, so
    # a value can be larger than the whole cap on its own. Anything bigger than
    # its marker form is fair game when the payload is still over.
    while estimated > cap_tokens:
        by_size = sorted(
            metadata.items(),
            key=lambda item: len(json.dumps(item[1], ensure_ascii=False, default=str)),
            reverse=True,
        )
        elided = False
        for key, value in by_size:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
            marker = {"_truncated": {"original_json_chars": len(serialized)}}
            if len(serialized) <= len(json.dumps(marker)) + 16:
                continue
            metadata[key] = marker
            elided = True
            changed = True
            estimated = estimate_payload_tokens(payload)
            if estimated <= cap_tokens:
                break
        if not elided:
            break
    # Last resort: when even the marker forms overshoot, drop the biggest
    # remaining values entirely and leave a count so the model knows.
    dropped_fields = 0
    while estimated > cap_tokens and metadata:
        key = max(
            metadata,
            key=lambda k: len(k) + len(json.dumps(metadata[k], default=str)),
        )
        if len(key) + len(json.dumps(metadata[key], default=str)) < 24:
            break
        del metadata[key]
        dropped_fields += 1
        estimated = estimate_payload_tokens(payload)
    if dropped_fields:
        metadata["_truncated_fields"] = dropped_fields
        changed = True
    return changed


def _clip_grep_match_windows(
    payload: dict[str, Any],
    *,
    window_tokens: int,
    focus: re.Pattern[str],
) -> bool:
    """grep-only: keep ~``window_tokens`` of context around EVERY match of ``focus``.

    Applied unconditionally to every grep result chunk (unlike the budget-gated
    ``_clip_chunk_text_fields``): the matched text is the whole reason grep
    returned the chunk, so each occurrence gets its own centred window and the
    windows are merged when they overlap. Fields with no match head-truncate to
    the same size so a hit in one field (e.g. ``text``) never leaves a sibling
    field (e.g. ``generated``) returning a full page.
    """
    window_chars = max(
        _TRUNCATION_MIN_TEXT_CHARS,
        window_tokens * config.TOKEN_ESTIMATE_CHARS_PER_TOKEN,
    )
    clipped = False
    for field_name in _TRUNCATION_TEXT_FIELDS:
        value = payload.get(field_name)
        if not isinstance(value, str) or len(value) <= window_chars:
            continue
        focused = _grep_match_windows(value, window_chars=window_chars, focus=focus)
        payload[field_name] = (
            focused if focused is not None else value[:window_chars].rstrip() + _CHUNK_CLIP_MARKER
        )
        clipped = True
    return clipped


def _grep_match_windows(value: str, *, window_chars: int, focus: re.Pattern[str]) -> str | None:
    """Keep a ~``window_chars`` window around every match of ``focus``, merged.

    Returns ``None`` when the pattern does not match anywhere, leaving the caller
    to fall back to head truncation. Head truncation is otherwise wrong for grep:
    the tool's contract is "this chunk contains your literal pattern", so clipping
    every match away hands the model a hit whose text refutes the reason it was
    returned. Overlapping or adjacent windows are merged so dense matches read as
    one span rather than repeating boundary markers; gaps between kept windows are
    marked with the standard clip ellipsis.
    """
    try:
        try:
            # The clip runs off the event loop, and concurrent=True releases the GIL, so a
            # backtracking pattern burns one worker thread's 2 s budget, never the loop.
            matches = list(focus.finditer(value, timeout=2.0, concurrent=True))
        except TypeError:
            # A plain stdlib pattern from a caller-compiled focus: trusted, no timeout.
            matches = list(focus.finditer(value))
    except (re.error, _clip_regex.error, TimeoutError):
        return None
    if not matches:
        return None
    windows: list[list[int]] = []
    for match in matches:
        match_length = match.end() - match.start()
        lead = max(0, (window_chars - match_length) // 2)
        start = max(0, match.start() - lead)
        end = min(len(value), start + window_chars)
        # Re-extend backwards if the window hit the end of the string.
        start = max(0, min(start, end - window_chars))
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    segments = [value[start:end].strip() for start, end in windows]
    clipped = _CHUNK_CLIP_MARKER.join(segments)
    if windows[0][0] > 0:
        clipped = _CHUNK_CLIP_PREFIX_MARKER + clipped
    if windows[-1][1] < len(value):
        clipped = clipped + _CHUNK_CLIP_MARKER
    return clipped


def _clip_text_fields(
    payload: dict[str, Any], *, max_tokens: int, marker: str = _TRUNCATION_TEXT_MARKER
) -> bool:
    """Last resort once whole-entry drops are exhausted: clip long text values."""
    clipped = False
    # Estimate exactly once per round and adapt the chars/token ratio from the
    # observed delta: re-serializing per field is O(fields x payload), which a
    # real tokenizer makes visible, and chars/4 alone under-clips cheap tokenizers.
    chars_per_token = float(config.TOKEN_ESTIMATE_CHARS_PER_TOKEN)
    for _ in range(3):
        estimated = estimate_payload_tokens(payload)
        if estimated <= max_tokens:
            return clipped
        tokens_before = estimated
        removed_chars = 0
        for entry in reversed(_primary_result_list(payload) or []):
            if not isinstance(entry, dict):
                continue
            for field_name in _TRUNCATION_TEXT_FIELDS:
                if estimated <= max_tokens:
                    break
                value = entry.get(field_name)
                if not isinstance(value, str) or len(value) <= _TRUNCATION_MIN_TEXT_CHARS:
                    continue
                keep_chars = max(
                    _TRUNCATION_MIN_TEXT_CHARS,
                    len(value) - int((estimated - max_tokens) * chars_per_token),
                )
                if keep_chars >= len(value):
                    continue
                entry[field_name] = value[:keep_chars] + marker
                removed_chars += len(value) - keep_chars
                estimated -= int((len(value) - keep_chars) / chars_per_token)
                clipped = True
        if not removed_chars:
            break
        reduced = tokens_before - estimate_payload_tokens(payload)
        if reduced > 0:
            chars_per_token = min(16.0, max(2.0, removed_chars / reduced))
        else:
            chars_per_token = min(16.0, chars_per_token * 2)
    return clipped


def _set_truncation_notice(
    payload: dict[str, Any], omitted: int, *, turn_capped: bool = False
) -> None:
    payload["results_omitted"] = omitted
    if turn_capped:
        payload["truncation_notice"] = (
            f"{omitted} lower-ranked result(s) omitted to fit this turn's "
            f"{config.TURN_TOOL_PAYLOAD_TOKEN_BUDGET}-token tool payload budget. You can "
            "call prune_context to free budget — it runs in parallel with your other "
            "tool calls — and if needed you may use some of your calls to make more "
            "targeted requests; omitted chunks can be re-surfaced by a new search."
        )
        return
    payload["truncation_notice"] = (
        f"{omitted} lower-ranked result(s) omitted to fit the context token limit. "
        "Call prune_context to free budget; omitted chunks can be re-surfaced by a new search."
    )


def _scrub_chunk_references(payload: dict[str, Any], *, chunk_ids: set[str]) -> int:
    # Only the primary list counts as visible results; the other result lists are
    # scrubbed without double-counting.
    primary = _primary_result_list(payload)
    removed = 0
    for field_name in _TRUNCATION_RESULT_FIELDS:
        entries = payload.get(field_name)
        if not isinstance(entries, list):
            continue
        original_length = len(entries)
        entries[:] = [
            entry
            for entry in entries
            if (chunk_id := _entry_chunk_id(entry)) is None or chunk_id not in chunk_ids
        ]
        if entries is primary:
            removed += original_length - len(entries)
    # Id-only side channels carry no text, so they never count as removed results;
    # they just must not point at chunks that are no longer in the payload.
    ids = payload.get("new_unseen_chunk_ids")
    if isinstance(ids, list):
        ids[:] = [chunk_id for chunk_id in ids if chunk_id not in chunk_ids]
    return removed


def _reprune_dropped_chunks(index: ChunkIndex, chunk_ids: set[str]) -> None:
    """Return get_chunks-restored chunks to pruned state when the round pass drops
    them before the model saw them. Only chunks that were pruned to begin with:
    a still-visible chunk that is dropped from one payload keeps its index state."""
    keys: set[ChunkKey] = set()
    for chunk_id in chunk_ids:
        try:
            key = index.refs.chunk_key_for_id(chunk_id)
        except ValueError:
            continue
        if key in index.deleted_chunk_keys:
            keys.add(key)
    index.mark_pruned(chunk_keys=keys, document_keys=())


def _discard_dropped_chunks(index: ChunkIndex, chunk_ids: set[str]) -> None:
    keys: set[ChunkKey] = set()
    for chunk_id in chunk_ids:
        try:
            keys.add(index.refs.chunk_key_for_id(chunk_id))
        except ValueError:
            continue
    index.discard_chunks(keys)


def redact_messages(
    messages: list[dict[str, Any]],
    *,
    refs: ReferenceRegistry,
    chunk_keys: set[ChunkKey],
    document_keys: set[DocumentKey],
) -> None:
    if not chunk_keys and not document_keys:
        return
    chunk_ids = {refs.chunk_id_for_key(key) for key in chunk_keys}
    document_ids = {refs.document_id_for_key(key) for key in document_keys}

    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message["content"] = redact_media_content_parts(
                content,
                chunk_ids=chunk_ids,
                document_ids=document_ids,
            )
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        redacted = redact_payload(
            payload,
            chunk_ids=chunk_ids,
            document_ids=document_ids,
        )
        message["content"] = json.dumps(redacted, ensure_ascii=False)


def redact_payload(
    payload: Any,
    *,
    chunk_ids: set[str],
    document_ids: set[str],
    skip_chunk_redaction: bool = False,
    document_scope: str | None = None,
) -> Any:
    if isinstance(payload, list):
        return [
            redact_payload(
                item,
                chunk_ids=chunk_ids,
                document_ids=document_ids,
                skip_chunk_redaction=skip_chunk_redaction,
                document_scope=document_scope,
            )
            for item in payload
        ]

    if not isinstance(payload, dict):
        return payload

    local_skip = skip_chunk_redaction or payload.get("tool") == "overview_search"
    copied = dict(payload)

    if _looks_like_document(copied):
        doc_id = str(copied.get("document_id", "")).strip()
        if doc_id in document_ids:
            copied.pop("content", None)
            copied.pop("content_url", None)
            copied["document_content_pruned"] = True
            document_scope = doc_id

    if _looks_like_chunk(copied):
        chunk_id = str(copied.get("chunk_id", "")).strip()
        document_id = str(copied.get("document_id", "")).strip()
        should_redact_chunk = chunk_id in chunk_ids or document_scope == document_id
        if should_redact_chunk:
            copied = redact_chunk_media(copied) if local_skip else redacted_chunk(copied)

    for key, value in list(copied.items()):
        copied[key] = redact_payload(
            value,
            chunk_ids=chunk_ids,
            document_ids=document_ids,
            skip_chunk_redaction=local_skip,
            document_scope=document_scope,
        )

    return copied


def _looks_like_chunk(payload: Mapping[str, Any]) -> bool:
    return "chunk_id" in payload


def _looks_like_document(payload: Mapping[str, Any]) -> bool:
    return "document_id" in payload and "chunk_id" not in payload


def redact_media_content_parts(
    content: list[Any],
    *,
    chunk_ids: set[str],
    document_ids: set[str],
) -> Any:
    redacted: list[Any] = []
    removed_count = 0
    for part in content:
        if not isinstance(part, Mapping):
            redacted.append(part)
            continue
        chunk_id = str(part.get("chunk_id") or "").strip()
        document_id = str(part.get("document_id") or "").strip()
        if chunk_id in chunk_ids or document_id in document_ids:
            removed_count += 1
            continue
        redacted.append(dict(part))

    if any(
        isinstance(part, Mapping) and str(part.get("type") or "") in {"image_url", "input_image"}
        for part in redacted
    ):
        return redacted
    if removed_count:
        return "Retrieved image content pruned."
    return content
