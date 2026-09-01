"""Runtime implementations for the ``agent_harness.tools`` schemas.

Async-native: every function that reaches a store awaits the
``AsyncRetrievalClient`` seam. A sync ``RetrievalClient`` (the SDK included)
participates through ``SyncRetrievalClientAdapter``; the sync tool surface in
``agent_harness.sync_api`` wraps these coroutines for compatibility.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

import regex as _filter_regex
from dotenv import load_dotenv
from mixedbread import Mixedbread, UnprocessableEntityError

from agent_harness.bridge_timing import emit as emit_bridge_timing
from agent_harness.config import (
    FILTER_CHUNK_FILE_SCAN_LIMIT,
    FILTER_CHUNKS_DEFAULT_K,
    FILTER_CHUNKS_MAX_K,
    GREP_DEFAULT_K,
    METADATA_INSPECT_MAX_INTERNAL_VALUES_PER_FIELD,
    METADATA_TYPE_SAMPLE_TOP_K,
    OVERVIEW_SEARCH_TOP_K,
    SEARCH_CORPUS_TOP_K,
    search_rerank,
)
from agent_harness.retrieval import (
    AsyncRetrievalClient,
    FileListRequest,
    FileRetrieveRequest,
    GrepRequest,
    ListChunksRequest,
    MetadataFacetsRequest,
    SearchRequest,
    SyncRetrievalClientAdapter,
)
from agent_harness.schemas import FILTER_MODES, FILTER_OPERATORS

ChunkKey = tuple[str, str, int]
DocumentKey = tuple[str, str]

_STORE_IDENTIFIER_ENV_VARS = (
    "MXBAI_STORE_IDENTIFIERS",
    "MBREAD_STORE_IDENTIFIERS",
    "MIXEDBREAD_STORE_IDENTIFIERS",
    "STORE_IDENTIFIERS",
)

_API_KEY_ENV_VARS = (
    "MBREAD_API_KEY",
    "MXBAI_API_KEY",
    "MIXEDBREAD_API_KEY",
)

_BASE_URL_ENV_VARS = (
    "MXBAI_BASE_URL",
    "MBREAD_BASE_URL",
    "MIXEDBREAD_BASE_URL",
)

_OVERVIEW_SEARCH_OMITTED_CONTENT_FIELDS = (
    "text",
    "context",
    "ocr_text",
    "transcription",
    "image_url",
    "media_url",
    "audio_url",
    "video_url",
    "content_url",
)
_OVERVIEW_TEXT_FALLBACK_SUMMARY_MAX_CHARS = 240
_OVERVIEW_TEXT_FALLBACK_SUMMARY_MAX_SENTENCES = 2

_mxbai_clients: dict[tuple[str, str], Mixedbread] = {}
_mxbai_clients_lock = RLock()


def get_mixedbread_client(
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> Mixedbread:
    """Return a lazily initialized Mixedbread client."""
    resolved_api_key = resolve_mixedbread_api_key(
        api_key=api_key,
        api_key_env=api_key_env,
    )
    resolved_base_url = resolve_mixedbread_base_url()
    cache_key = (resolved_base_url or "", resolved_api_key or "__default__")
    with _mxbai_clients_lock:
        client = _mxbai_clients.get(cache_key)
        if client is None:
            kwargs: dict[str, Any] = {"api_key": resolved_api_key}
            if resolved_base_url:
                kwargs["base_url"] = resolved_base_url
            client = Mixedbread(**kwargs)
            _mxbai_clients[cache_key] = client
        return client


def resolve_mixedbread_api_key(
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> str | None:
    """Resolve an explicit API key or a comma-separated list of env var names."""
    if api_key:
        return api_key
    load_dotenv()
    env_names = (
        tuple(name.strip() for name in api_key_env.split(",") if name.strip())
        if api_key_env
        else _API_KEY_ENV_VARS
    )
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def resolve_mixedbread_base_url() -> str | None:
    """Resolve the Mixedbread base URL used by store tool clients."""
    load_dotenv()
    for env_name in _BASE_URL_ENV_VARS:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def resolve_async_retrieval_client(
    client: AsyncRetrievalClient | None,
    *,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> AsyncRetrievalClient:
    """Return the injected async client, else the SDK behind the sync adapter.

    The injected client stays authoritative: once a caller passes one, no code
    path may fall back to constructing an SDK client (that would silently send
    an in-process deployment's rollout to the public API).
    """
    if client is not None:
        return client
    return SyncRetrievalClientAdapter(
        get_mixedbread_client(api_key=api_key, api_key_env=api_key_env)
    )


@dataclass(frozen=True, slots=True)
class StoreFileListing:
    """One ``list_store_files`` pass: resolved stores, files, and pagination."""

    store_ids: list[str]
    files: list[dict[str, Any]]
    pagination: dict[str, Any]


async def search_raw(
    query: str,
    k: int,
    *,
    store_identifiers: Sequence[str] | None = None,
    filter_by: Sequence[Mapping[str, Any]] | None = None,
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    filter_mode: str | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> list[dict[str, Any]]:
    """Search Mixedbread stores and return full SDK chunks as JSON dictionaries."""
    store_ids = _resolve_store_identifiers(store_identifiers)
    effective_metadata_filters = filter_by if filter_by is not None else metadata_filters
    filters = build_mixedbread_filters(
        effective_metadata_filters,
        filter_mode,
        metadata_filter=metadata_filter,
    )
    request = SearchRequest(
        query=query,
        store_identifiers=tuple(store_ids),
        top_k=k,
        filters=filters,
        rerank=search_rerank(),
    )
    resolved = resolve_async_retrieval_client(client, api_key=api_key, api_key_env=api_key_env)
    results = await resolved.stores.search(request)
    # Like every other seam result, hits may arrive under a plain mapping's
    # "data" key or an SDK model's .data attribute.
    return [_model_to_json_dict(chunk) for chunk in _response_items(results)]


async def search_corpus(
    query: str,
    *,
    store_identifiers: Sequence[str] | None = None,
    top_k: int = SEARCH_CORPUS_TOP_K,
    filter_by: Sequence[Mapping[str, Any]] | None = None,
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    filter_mode: str | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Execute the ``search_corpus`` tool."""
    effective_metadata_filters = filter_by if filter_by is not None else metadata_filters
    filters = build_mixedbread_filters(
        effective_metadata_filters,
        filter_mode,
        metadata_filter=metadata_filter,
    )
    chunks = await search_raw(
        query.strip(),
        top_k,
        store_identifiers=store_identifiers,
        filter_by=effective_metadata_filters,
        metadata_filter=metadata_filter,
        filter_mode=filter_mode,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "results": [serialize_chunk(chunk) for chunk in chunks],
    }
    if filters:
        payload["metadata_filters"] = filters
        payload["metadata_filter"] = filters
    return payload


async def inspect_metadata(
    max_values_per_field: int = 8,
    *,
    facets: Sequence[str] | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    store_identifiers: Sequence[str] | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Return available metadata fields and representative values for stores."""
    store_ids = _resolve_store_identifiers(store_identifiers)
    max_values = min(
        max(int(max_values_per_field or 8), 1),
        METADATA_INSPECT_MAX_INTERNAL_VALUES_PER_FIELD,
    )
    filters = build_mixedbread_filters(None, None, metadata_filter=metadata_filter)
    facet_names = [str(facet).strip() for facet in (facets or []) if str(facet).strip()]
    request = MetadataFacetsRequest(
        store_identifiers=tuple(store_ids),
        facets=tuple(facet_names) if facet_names else None,
        filters=filters,
    )

    resolved_client = resolve_async_retrieval_client(
        client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    stores_client = resolved_client.stores

    # Facet aggregation and type sampling are independent remote operations,
    # overlapped here while their merge below stays deterministic.
    # return_exceptions keeps both joined before either failure propagates
    # (facets first) instead of leaving the sibling running detached.
    facets_response, value_types = await asyncio.gather(
        _timed_metadata_provider_call(
            "metadata_facets", lambda: stores_client.metadata_facets(request)
        ),
        _timed_metadata_provider_call(
            "list_chunks_type_sample",
            lambda: _sampled_metadata_value_types(
                store_identifiers=store_ids,
                client=resolved_client,
                api_key=api_key,
                api_key_env=api_key_env,
            ),
        ),
        return_exceptions=True,
    )
    _raise_gathered_failure(facets_response, value_types)
    fields = _facet_fields_from_response(facets_response, max_values=max_values)

    fields = {
        field_name: _typed_facet_samples(values, value_types.get(field_name))
        for field_name, values in fields.items()
    }

    return {
        "tool": "inspect_metadata",
        "store_identifiers": store_ids,
        "requested_facets": facet_names or None,
        "metadata_filter": filters,
        "max_values_per_field": max_values,
        "metadata_field_count": len(fields),
        "rankable_fields": _rankable_metadata_fields(fields, value_types),
        "field_types_sampled": bool(value_types),
        "metadata_fields": fields,
    }


def _rankable_metadata_fields(
    fields: Mapping[str, Any],
    value_types: Mapping[str, set[str]],
) -> list[str]:
    """Metadata fields that ``rank_by`` can actually order, named explicitly.

    Facet values arrive from the provider as JSON object keys, so an identifier field
    and a metric field are both strings there and only the sampled chunk types tell
    them apart. Per-value ``type`` annotations already carry that, but reading a
    field's rankability off its samples is a step agents skip: they guess a rank_by
    and burn a round on the failure.
    """
    rankable = [
        field_name
        for field_name in fields
        if (value_types.get(field_name, set()) - {"null"}) == {"number"}
    ]
    return sorted(rankable)


async def _timed_metadata_provider_call[ResultT](
    operation: str,
    call: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    """Run one provider call with its duration on the bridge-timing stream."""
    started = time.perf_counter()
    try:
        return await call()
    finally:
        emit_bridge_timing(
            "metadata_provider_call",
            operation=operation,
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def _raise_gathered_failure(*outcomes: Any) -> None:
    """Re-raise a gathered failure: cancellations and exits first, then in argument order."""
    for outcome in outcomes:
        if isinstance(outcome, BaseException) and not isinstance(outcome, Exception):
            raise outcome
    for outcome in outcomes:
        if isinstance(outcome, Exception):
            raise outcome


def _facet_fields_from_response(facets_response: Any, *, max_values: int) -> dict[str, Any]:
    """Normalize the provider's facet payload shapes into field-to-values."""
    raw = _model_to_json_dict(facets_response)
    if "facets" in raw:
        facet_data = raw["facets"]
    elif "data" in raw:
        facet_data = raw["data"]
    else:
        facet_data = raw

    fields: dict[str, Any] = {}
    if isinstance(facet_data, Mapping):
        for key, values in facet_data.items():
            fields[str(key)] = _compact_facet_values(values, max_values=max_values)
    elif isinstance(facet_data, list):
        for item in facet_data:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key") or item.get("field") or item.get("name")
            if not key:
                continue
            values = item.get("values") or item.get("facets") or item
            fields[str(key)] = _compact_facet_values(values, max_values=max_values)
    return fields


async def filter_metadata(
    metadata_filters: Sequence[Mapping[str, Any]],
    *,
    metadata_filter: Mapping[str, Any] | None = None,
    store_identifiers: Sequence[str] | None = None,
    filter_mode: str | None = None,
    limit: int = 20,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """List store files matching metadata filters without semantic search."""
    filters = build_mixedbread_filters(
        metadata_filters,
        filter_mode,
        metadata_filter=metadata_filter,
    )
    if not filters:
        raise ValueError("filter_metadata requires at least one valid metadata filter")

    max_files = min(max(int(limit or 20), 1), 100)
    listing = await list_store_files(
        store_identifiers=store_identifiers,
        metadata_filter=filters,
        limit=max_files,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )

    payload: dict[str, Any] = {
        "tool": "filter_metadata",
        "store_identifiers": listing.store_ids,
        "metadata_filter": filters,
        "limit": max_files,
        "files_returned": len(listing.files),
        "files": listing.files,
    }
    if listing.pagination:
        payload["pagination"] = listing.pagination
    return payload


async def rank_metadata(
    sort_key: str,
    *,
    sort_order: str = "desc",
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    filter_mode: str | None = None,
    limit: int = 10,
    fetch_limit: int = 100,
    store_identifiers: Sequence[str] | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """List files ranked by a metadata field."""
    filters = build_mixedbread_filters(
        metadata_filters,
        filter_mode,
        metadata_filter=metadata_filter,
    )
    max_files = min(max(int(fetch_limit or 100), 1), 100)
    output_limit = min(max(int(limit or 10), 1), 50)
    listing = await list_store_files(
        store_identifiers=store_identifiers,
        metadata_filter=filters,
        limit=max_files,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    ranked = sort_files_by_metadata(
        listing.files,
        sort_key=sort_key,
        sort_order=sort_order,
    )[:output_limit]
    payload: dict[str, Any] = {
        "tool": "rank_metadata",
        "store_identifiers": listing.store_ids,
        "metadata_filter": filters,
        "sort_key": sort_key,
        "sort_order": sort_order,
        "fetch_limit": max_files,
        "limit": output_limit,
        "files_returned": len(ranked),
        "files": ranked,
    }
    if listing.pagination:
        payload["pagination"] = listing.pagination
    return _drop_none(payload)


async def distinct_metadata(
    distinct_key: str,
    *,
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    metadata_filter: Mapping[str, Any] | None = None,
    filter_mode: str | None = None,
    examples_per_value: int = 1,
    fetch_limit: int = 100,
    store_identifiers: Sequence[str] | None = None,
    sort_key: str | None = None,
    sort_order: str = "desc",
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Return representative files grouped by a distinct metadata value."""
    filters = build_mixedbread_filters(
        metadata_filters,
        filter_mode,
        metadata_filter=metadata_filter,
    )
    max_files = min(max(int(fetch_limit or 100), 1), 100)
    examples = min(max(int(examples_per_value or 1), 1), 5)
    listing = await list_store_files(
        store_identifiers=store_identifiers,
        metadata_filter=filters,
        limit=max_files,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    store_ids, files, pagination = listing.store_ids, listing.files, listing.pagination
    candidates = (
        sort_files_by_metadata(files, sort_key=sort_key, sort_order=sort_order)
        if sort_key
        else files
    )
    groups: dict[str, dict[str, Any]] = {}
    for file_payload in candidates:
        value = metadata_lookup(file_payload, distinct_key)
        if value is None:
            continue
        value_key = _metadata_value_key(value)
        group = groups.setdefault(
            value_key,
            {
                "value": value,
                "files": [],
            },
        )
        if len(group["files"]) < examples:
            group["files"].append(file_payload)

    distinct_values = [
        {**group, "file_count_returned": len(group["files"])} for group in groups.values()
    ]
    payload: dict[str, Any] = {
        "tool": "distinct_metadata",
        "store_identifiers": store_ids,
        "metadata_filter": filters,
        "distinct_key": distinct_key,
        "examples_per_value": examples,
        "fetch_limit": max_files,
        "sort_key": sort_key,
        "sort_order": sort_order if sort_key else None,
        "distinct_value_count": len(distinct_values),
        "distinct_values": distinct_values,
    }
    if pagination:
        payload["pagination"] = pagination
    return _drop_none(payload)


async def filter_chunks(
    filter_by: Sequence[Mapping[str, Any]] | None = None,
    *,
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    filter_mode: str | None = None,
    rank_by: str | None = None,
    direction: str = "desc",
    k: int = FILTER_CHUNKS_DEFAULT_K,
    file_scan_limit: int = FILTER_CHUNK_FILE_SCAN_LIMIT,
    store_identifiers: Sequence[str] | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """List chunks matching metadata filters via Mixedbread's list-chunks endpoint."""
    effective_filters = filter_by if filter_by is not None else metadata_filters
    filters = build_mixedbread_filters(effective_filters, filter_mode)
    store_ids = _resolve_store_identifiers(store_identifiers)
    result_k = min(max(int(k or FILTER_CHUNKS_DEFAULT_K), 1), FILTER_CHUNKS_MAX_K)
    sort_direction = "asc" if str(direction or "desc").lower() == "asc" else "desc"

    chunks = await _list_chunks_with_sort_fallback(
        store_ids=store_ids,
        result_k=result_k,
        file_scan_limit=int(file_scan_limit),
        filters=filters,
        sort_by=[rank_by, sort_direction == "asc"] if rank_by else None,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )

    if rank_by:
        ranking = _rank_chunks_by_metadata(
            chunks, rank_by=rank_by, descending=sort_direction == "desc"
        )
        chunks = ranking.chunks
        rank_by_applied = ranking.applied
        non_numeric_count = ranking.non_numeric_count
    else:
        chunks = _deterministic_chunk_order(chunks)
        rank_by_applied = None
        non_numeric_count = None

    chunks = _normalize_chunk_scores_by_rank(chunks)
    results = chunks[:result_k]
    return _drop_none(
        {
            "tool": "filter_chunks",
            "store_identifiers": store_ids,
            "metadata_filter": filters,
            "rank_by": rank_by,
            "direction": sort_direction if rank_by else None,
            "rank_by_applied": rank_by_applied,
            "rank_by_non_numeric_count": non_numeric_count,
            "k": result_k,
            "candidate_count": len(chunks),
            "results": results,
        }
    )


async def _list_chunks_with_sort_fallback(
    *,
    store_ids: Sequence[str],
    result_k: int,
    file_scan_limit: int,
    filters: Mapping[str, Any] | None,
    sort_by: list[Any] | None,
    client: AsyncRetrievalClient | None,
    api_key: str | None,
    api_key_env: str | None,
) -> list[dict[str, Any]]:
    try:
        return await list_chunks_raw(
            store_identifiers=store_ids,
            top_k=result_k,
            metadata_filter=filters,
            sort_by=sort_by,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except UnprocessableEntityError:
        if sort_by is None:
            raise
        # Stores whose metadata index types numeric fields as strings reject
        # server-side sorts; fetch unsorted and rank client-side instead.
        return await list_chunks_raw(
            store_identifiers=store_ids,
            top_k=max(result_k, file_scan_limit),
            metadata_filter=filters,
            sort_by=None,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )


@dataclass(slots=True)
class _RankedByMetadata:
    chunks: list[dict[str, Any]]
    applied: bool
    non_numeric_count: int


def _rank_chunks_by_metadata(
    chunks: Sequence[Mapping[str, Any]],
    *,
    rank_by: str,
    descending: bool,
) -> _RankedByMetadata:
    """Order the chunks client-side by a numeric metadata field.

    rank_by is an ordering hint, not a filter: chunks with no numeric value keep
    the no-rank_by deterministic order behind the ranked ones instead of being
    dropped.
    """
    sortable_chunks: list[tuple[float, dict[str, Any]]] = []
    unrankable_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        value = metadata_lookup(chunk, rank_by)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            unrankable_chunks.append(dict(chunk))
            continue
        sortable_chunks.append((float(value), dict(chunk)))
    ranked = [
        chunk for _, chunk in sorted(sortable_chunks, key=lambda item: item[0], reverse=descending)
    ] + _deterministic_chunk_order(unrankable_chunks)
    return _RankedByMetadata(
        chunks=ranked,
        applied=bool(sortable_chunks),
        non_numeric_count=len(unrankable_chunks),
    )


def _deterministic_chunk_order(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order chunks the way filter_chunks orders them when rank_by is omitted."""
    return sorted(
        (dict(chunk) for chunk in chunks),
        key=lambda chunk: (
            str(chunk.get("store_id", "")),
            str(chunk.get("file_id", "")),
            int(chunk.get("chunk_index", 0) or 0),
        ),
    )


async def list_chunks_raw(
    *,
    store_identifiers: Sequence[str],
    top_k: int,
    metadata_filter: Mapping[str, Any] | None,
    sort_by: Any | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> list[dict[str, Any]]:
    """Call Mixedbread's list-chunks endpoint and return serialized chunks."""
    store_ids = _normalize_store_identifiers(store_identifiers)
    request_top_k = max(int(top_k or FILTER_CHUNKS_DEFAULT_K), 1)
    resolved = resolve_async_retrieval_client(client, api_key=api_key, api_key_env=api_key_env)
    stores_client = resolved.stores

    chunks: list[dict[str, Any]] = []
    for store_id in store_ids:
        request = ListChunksRequest(
            store_identifiers=(store_id,),
            top_k=request_top_k,
            filters=metadata_filter or None,
            sort_by=tuple(sort_by) if sort_by is not None else None,
        )
        response = await stores_client.list_chunks(request)
        chunks.extend(
            serialize_chunk(chunk, fallback_store_id=store_id)
            for chunk in _response_items(response)
        )

    return chunks


async def grep_raw(
    pattern: str,
    k: int,
    *,
    store_identifiers: Sequence[str],
    targets: Sequence[str],
    case_sensitive: bool,
    filter_by: Sequence[Mapping[str, Any]] | None = None,
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    filter_mode: str | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> list[dict[str, Any]]:
    """Call the Mixedbread grep endpoint and return matching chunks."""
    effective_filters = filter_by if filter_by is not None else metadata_filters
    filters = build_mixedbread_filters(effective_filters, filter_mode)
    request = GrepRequest(
        store_identifiers=tuple(_normalize_store_identifiers(store_identifiers)),
        pattern=pattern,
        targets=tuple(dict.fromkeys(targets or ["text", "generated"])),
        case_sensitive=bool(case_sensitive),
        top_k=max(int(k or GREP_DEFAULT_K), 1),
        filters=filters,
    )

    resolved = resolve_async_retrieval_client(client, api_key=api_key, api_key_env=api_key_env)
    response = await resolved.stores.grep(request)
    return [serialize_chunk(chunk) for chunk in _response_items(response)]


async def grep(
    pattern: str,
    *,
    targets: Sequence[str] | None = None,
    case_sensitive: bool = False,
    filter_by: Sequence[Mapping[str, Any]] | None = None,
    metadata_filters: Sequence[Mapping[str, Any]] | None = None,
    filter_mode: str | None = None,
    store_identifiers: Sequence[str] | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Find chunks whose literal text matches a regular expression."""
    store_ids = _resolve_store_identifiers(store_identifiers)
    result_k = GREP_DEFAULT_K
    effective_filters = filter_by if filter_by is not None else metadata_filters
    filters = build_mixedbread_filters(effective_filters, filter_mode)
    results = await grep_raw(
        pattern,
        result_k,
        store_identifiers=store_ids,
        targets=targets or ["text", "generated"],
        case_sensitive=case_sensitive,
        filter_by=effective_filters,
        filter_mode=filter_mode,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    return _drop_none(
        {
            "tool": "grep",
            "store_identifiers": store_ids,
            "pattern": pattern,
            "targets": list(dict.fromkeys(targets or ["text", "generated"])),
            "case_sensitive": bool(case_sensitive),
            "metadata_filter": filters,
            "k": result_k,
            "results": results[:result_k],
            "candidate_count": len(results),
        }
    )


def build_mixedbread_filters(
    metadata_filters: Sequence[Mapping[str, Any]] | None,
    filter_mode: str | None,
    *,
    metadata_filter: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Convert tool metadata filter arguments to Mixedbread filter syntax."""
    normalized_expression = _normalize_filter_expression(metadata_filter)
    if normalized_expression:
        return normalized_expression

    if not metadata_filters:
        return None

    clean_filters: list[dict[str, Any]] = []
    for raw_item in metadata_filters:
        condition = _normalize_filter_condition(raw_item)
        if condition:
            clean_filters.append(condition)

    if not clean_filters:
        return None
    mode = str(filter_mode or "all").strip()
    if mode not in FILTER_MODES:
        mode = "all"
    return {mode: clean_filters}


def _normalize_filter_expression(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping):
        return None

    expression: dict[str, Any] = {}
    for mode in FILTER_MODES:
        raw_group = value.get(mode)
        if not raw_group:
            continue
        if not isinstance(raw_group, Sequence) or isinstance(raw_group, (str, bytes)):
            continue
        conditions: list[dict[str, Any]] = []
        for raw_condition in raw_group:
            nested = _normalize_filter_expression(raw_condition)
            if nested:
                conditions.append(nested)
                continue
            condition = _normalize_filter_condition(raw_condition)
            if condition:
                conditions.append(condition)
        if conditions:
            expression[mode] = conditions
    return expression or None


def _normalize_filter_condition(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping):
        return None

    key = str(value.get("key") or "").strip()
    operator = str(value.get("operator") or "").strip()
    if not key or operator not in FILTER_OPERATORS:
        return None
    return {
        "key": key,
        "operator": operator,
        "value": _normalize_filter_value(value.get("value")),
    }


async def list_store_files(
    *,
    store_identifiers: Sequence[str] | None,
    metadata_filter: Mapping[str, Any] | None,
    limit: int,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> StoreFileListing:
    """List store files across store identifiers with optional metadata filtering."""
    store_ids = _resolve_store_identifiers(store_identifiers)
    max_files = min(max(int(limit or 20), 1), 100)
    files: list[dict[str, Any]] = []
    pagination: dict[str, Any] = {}
    stores_client = resolve_async_retrieval_client(
        client, api_key=api_key, api_key_env=api_key_env
    ).stores

    for store_id in store_ids:
        if len(files) >= max_files:
            break
        matched_for_store: list[dict[str, Any]] = []
        attempted_filters: list[Mapping[str, Any] | None] = [metadata_filter]
        if metadata_filter:
            # If the API-side file filter is unsupported or unexpectedly sparse,
            # fall back to a local pass over the first page of files.
            attempted_filters.append(None)

        for api_metadata_filter in attempted_filters:
            if len(matched_for_store) >= max_files - len(files):
                break
            remaining = max_files - len(files)
            if remaining <= 0:
                break
            request_limit = 100 if metadata_filter and api_metadata_filter is None else remaining
            try:
                response = await stores_client.files.list(
                    FileListRequest(
                        store_identifier=store_id,
                        limit=request_limit,
                        metadata_filter=api_metadata_filter or None,
                    )
                )
            except Exception:
                if api_metadata_filter:
                    continue
                raise
            raw_response = _model_to_json_dict(response)
            pagination_payload = raw_response.get("pagination")
            if not isinstance(pagination_payload, Mapping):
                pagination_payload = {}
            next_cursor = (
                raw_response.get("next_cursor")
                or raw_response.get("after")
                or raw_response.get("cursor")
                or pagination_payload.get("next_cursor")
                or pagination_payload.get("after")
            )
            if next_cursor:
                pagination[store_id] = {"next_cursor": next_cursor}

            for file_obj in _response_items(response):
                if not metadata_filter and len(files) + len(matched_for_store) >= max_files:
                    break
                file_payload = serialize_file_metadata(
                    file_obj,
                    fallback_store_id=store_id,
                )
                if metadata_filter and not metadata_payload_matches_filter(
                    file_payload,
                    metadata_filter,
                ):
                    continue
                file_key = document_key(file_payload)
                if any(document_key(existing) == file_key for existing in matched_for_store):
                    continue
                matched_for_store.append(file_payload)
                if len(files) + len(matched_for_store) >= max_files:
                    break

        files.extend(matched_for_store[: max_files - len(files)])

    return StoreFileListing(store_ids=store_ids, files=files, pagination=pagination)


async def _load_filtered_chunks(
    *,
    store_identifiers: Sequence[str],
    metadata_filter: Mapping[str, Any] | None,
    file_scan_limit: int,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper around Mixedbread's list-chunks endpoint."""
    return await list_chunks_raw(
        store_identifiers=store_identifiers,
        top_k=file_scan_limit,
        metadata_filter=metadata_filter,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )


def sort_files_by_metadata(
    files: Sequence[Mapping[str, Any]],
    *,
    sort_key: str | None,
    sort_order: str,
) -> list[dict[str, Any]]:
    """Sort files by a metadata field while keeping missing values last."""
    if not sort_key:
        return [dict(file_payload) for file_payload in files]
    present: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for file_payload in files:
        copied = dict(file_payload)
        value = metadata_lookup(copied, sort_key)
        if value is None:
            missing.append(copied)
            continue
        copied["rank_metadata_value"] = value
        present.append(copied)

    reverse = str(sort_order or "desc").lower() == "desc"
    present.sort(
        key=lambda file_payload: _sortable_metadata_value(file_payload.get("rank_metadata_value")),
        reverse=reverse,
    )
    return present + missing


def metadata_lookup(payload: Mapping[str, Any], key: str) -> Any:
    """Look up a top-level or metadata dot-path value."""
    if not key:
        return None
    parts = [part for part in str(key).split(".") if part]
    candidates: list[Any] = [payload]
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.append(metadata)
    generated_metadata = payload.get("generated_metadata")
    if isinstance(generated_metadata, Mapping):
        candidates.append(generated_metadata)

    for candidate in candidates:
        value: Any = candidate
        found = True
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value.get(part)
        if found and value not in (None, ""):
            return value
    return None


def metadata_payload_matches_filter(
    payload: Mapping[str, Any],
    metadata_filter: Mapping[str, Any] | None,
) -> bool:
    """Evaluate a Mixedbread-style metadata filter against serialized file/chunk metadata."""
    if not metadata_filter:
        return True
    return _evaluate_filter_expression(payload, metadata_filter)


def _evaluate_filter_expression(payload: Mapping[str, Any], expression: Any) -> bool:
    if hasattr(expression, "model_dump"):
        expression = expression.model_dump(mode="json", exclude_none=True)
    if expression is None:
        return True
    if isinstance(expression, Sequence) and not isinstance(expression, (str, bytes)):
        return all(_evaluate_filter_expression(payload, item) for item in expression)
    if not isinstance(expression, Mapping):
        return True
    if "key" in expression:
        return _evaluate_filter_condition(payload, expression)
    if expression.get("all") and not all(
        _evaluate_filter_expression(payload, item) for item in expression["all"]
    ):
        return False
    if expression.get("any") and not any(
        _evaluate_filter_expression(payload, item) for item in expression["any"]
    ):
        return False
    return not (
        expression.get("none")
        and any(_evaluate_filter_expression(payload, item) for item in expression["none"])
    )


def _evaluate_filter_condition(payload: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    key = str(condition.get("key") or "").strip()
    if not key:
        return True
    actual = metadata_lookup(payload, key)
    expected = condition.get("value")
    operator = str(condition.get("operator") or "").strip()

    if operator == "eq":
        return actual == expected
    if operator == "not_eq":
        return actual not in (None, "") and actual != expected
    if operator in {"gt", "gte", "lt", "lte"}:
        actual_value = _coerce_comparable_value(actual)
        expected_value = _coerce_comparable_value(expected)
        if actual_value is None or expected_value is None:
            return False
        try:
            if operator == "gt":
                return actual_value > expected_value
            if operator == "gte":
                return actual_value >= expected_value
            if operator == "lt":
                return actual_value < expected_value
            return actual_value <= expected_value
        except TypeError:
            return False
    if operator == "in":
        return (
            isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
            and actual in expected
        )
    if operator == "not_in":
        return (
            isinstance(expected, Sequence)
            and not isinstance(expected, (str, bytes))
            and actual not in (None, "")
            and actual not in expected
        )
    if operator == "like":
        return (
            isinstance(actual, str)
            and isinstance(expected, str)
            and expected.casefold() in actual.casefold()
        )
    if operator == "not_like":
        return (
            isinstance(actual, str)
            and isinstance(expected, str)
            and expected.casefold() not in actual.casefold()
        )
    if operator == "starts_with":
        return (
            isinstance(actual, str)
            and isinstance(expected, str)
            and actual.casefold().startswith(expected.casefold())
        )
    if operator == "regex":
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        try:
            # `regex`, not stdlib: the filter is model-authored, and stdlib matching
            # cannot be interrupted once started; concurrent=True releases the GIL.
            return _filter_regex.search(expected, actual, timeout=1.0, concurrent=True) is not None
        except (re.error, _filter_regex.error, TimeoutError):
            return False
    return False


def _coerce_comparable_value(value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return stripped
    return value


def _normalize_chunk_scores_by_rank(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []

    denominator = max(len(chunks) - 1, 1)
    normalized: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        score = 1.0 if len(chunks) == 1 else 1.0 - (index / denominator)
        copied = dict(chunk)
        copied["search_score"] = round(score, 4)
        normalized.append(copied)
    return normalized


async def read_document(
    file_id: str,
    store_id: str,
    *,
    chunk_indices: Sequence[int] | None = None,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Retrieve a store file with all chunks or a selected chunk-index window."""
    return_chunks: Any = True
    if chunk_indices is not None:
        normalized_indices = set()
        for index in chunk_indices:
            normalized_index = int(index)
            if normalized_index >= 0:
                normalized_indices.add(normalized_index)
        return_chunks = sorted(normalized_indices)
    resolved = resolve_async_retrieval_client(client, api_key=api_key, api_key_env=api_key_env)
    try:
        store_file = await resolved.stores.files.retrieve(
            FileRetrieveRequest(
                file_identifier=file_id,
                store_identifier=store_id,
                return_chunks=tuple(return_chunks)
                if isinstance(return_chunks, list)
                else return_chunks,
            )
        )
    except UnprocessableEntityError as exc:
        clamped_return_chunks = _clamp_return_chunks_for_invalid_indices(
            return_chunks,
            exc,
        )
        if clamped_return_chunks is None:
            raise
        store_file = await resolved.stores.files.retrieve(
            FileRetrieveRequest(
                file_identifier=file_id,
                store_identifier=store_id,
                return_chunks=tuple(clamped_return_chunks),
            )
        )
    return serialize_store_file(
        store_file,
        fallback_file_id=file_id,
        fallback_store_id=store_id,
    )


def _clamp_return_chunks_for_invalid_indices(
    return_chunks: Any,
    exc: UnprocessableEntityError,
) -> list[int] | None:
    if not isinstance(return_chunks, list):
        return None

    bounds = _invalid_chunk_index_bounds(exc)
    if bounds is None:
        return None

    min_index, max_index = bounds
    clamped = [
        chunk_index for chunk_index in return_chunks if min_index <= chunk_index <= max_index
    ]
    if not clamped or clamped == return_chunks:
        return None
    return clamped


def _invalid_chunk_index_bounds(
    exc: UnprocessableEntityError,
) -> tuple[int, int] | None:
    body = getattr(exc, "body", None)
    if not isinstance(body, Mapping):
        return None
    nested_error = body.get("error")
    if isinstance(nested_error, Mapping):
        body = nested_error
    if body.get("code") != "store_file_invalid_chunk_indices_error":
        return None

    message = str(body.get("message") or getattr(exc, "message", ""))
    match = re.search(r"between\s+(\d+)\s+and\s+(\d+)", message)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


async def get_chunk(
    file_id: str,
    store_id: str,
    chunk_index: int,
    *,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Retrieve a single chunk from a Mixedbread store file."""
    resolved = resolve_async_retrieval_client(client, api_key=api_key, api_key_env=api_key_env)
    store_file = await resolved.stores.files.retrieve(
        FileRetrieveRequest(
            file_identifier=file_id,
            store_identifier=store_id,
            return_chunks=(chunk_index,),
        )
    )
    file_payload = _model_to_json_dict(store_file)
    chunks = file_payload.get("chunks") or []
    for chunk in chunks:
        chunk_payload = _model_to_json_dict(chunk)
        if _int_or_zero(chunk_payload.get("chunk_index")) == chunk_index:
            serialized = serialize_chunk(
                chunk_payload,
                fallback_file_id=file_payload.get("id") or file_id,
                fallback_store_id=file_payload.get("store_id") or store_id,
                fallback_filename=file_payload.get("filename"),
            )
            serialized["content"] = chunk_content_text(serialized)
            return serialized

    return {
        "error": "Chunk not found",
        "file_id": file_id,
        "store_id": store_id,
        "chunk_index": chunk_index,
    }


async def get_chunks(
    file_id: str,
    store_id: str,
    chunk_indices: Sequence[int],
    *,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Retrieve multiple chunks from a Mixedbread store file."""
    normalized_indices = list(dict.fromkeys(int(index) for index in chunk_indices))
    document = await read_document(
        file_id=file_id,
        store_id=store_id,
        chunk_indices=normalized_indices,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    chunks_by_index = {
        int(chunk.get("chunk_index", 0) or 0): dict(chunk) for chunk in document.get("chunks") or []
    }
    results: list[dict[str, Any]] = []
    for chunk_index in normalized_indices:
        chunk = chunks_by_index.get(chunk_index)
        if chunk is None:
            results.append(
                {
                    "error": "Chunk not found",
                    "file_id": file_id,
                    "store_id": store_id,
                    "chunk_index": chunk_index,
                }
            )
            continue
        chunk["content"] = chunk_content_text(chunk)
        results.append(chunk)

    return {
        "file_id": file_id,
        "store_id": store_id,
        "chunk_indices": normalized_indices,
        "results": results,
    }


async def overview_search(
    query: str,
    *,
    store_identifiers: Sequence[str] | None = None,
    top_k: int = OVERVIEW_SEARCH_TOP_K,
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Execute a high-recall search and return Mixedbread per-chunk summaries."""
    chunks = await search_raw(
        query.strip(),
        top_k,
        store_identifiers=store_identifiers,
        client=client,
        api_key=api_key,
        api_key_env=api_key_env,
    )
    results: list[dict[str, Any]] = []
    missing_summary_count = 0

    for chunk in chunks:
        serialized = serialize_chunk(chunk)
        _populate_summary(serialized, allow_text_fallback=True)
        if not _summary_text(serialized.get("summary")):
            missing_summary_count += 1
        for field in _OVERVIEW_SEARCH_OMITTED_CONTENT_FIELDS:
            serialized.pop(field, None)
        results.append(serialized)

    return {
        "query": query,
        "top_k": top_k,
        "results": results,
        "summaries_found": len(results) - missing_summary_count,
        "summaries_missing": missing_summary_count,
    }


def submit_ranking(
    chunks: Sequence[Mapping[str, Any]],
    *,
    ranking_strategy: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """Normalize ``submit_ranking`` output into the final ranked-chunk payload."""
    result: dict[str, Any] = {
        "chunks": [normalize_ranked_chunk(chunk) for chunk in chunks],
    }
    if ranking_strategy is not None:
        result["ranking_strategy"] = ranking_strategy.strip()
    if answer is not None:
        result["answer"] = answer.strip()
    return result


def prune_context(
    chunks: Sequence[Mapping[str, Any]] | None = None,
    documents: Sequence[Mapping[str, Any]] | None = None,
    chunk_ids: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Normalize context-pruning requests into short chunk and document references."""
    return {
        "chunk_ids": [
            str(chunk_id).strip() for chunk_id in (chunk_ids or []) if str(chunk_id).strip()
        ]
        + [
            str(chunk.get("chunk_id", "")).strip()
            for chunk in (chunks or [])
            if str(chunk.get("chunk_id", "")).strip()
        ],
        "document_ids": [
            str(document_id).strip()
            for document_id in (document_ids or [])
            if str(document_id).strip()
        ]
        + [
            str(document.get("document_id", "")).strip()
            for document in (documents or [])
            if str(document.get("document_id", "")).strip()
        ],
    }


def serialize_store_file(
    store_file: Any,
    *,
    fallback_file_id: str | None = None,
    fallback_store_id: str | None = None,
) -> dict[str, Any]:
    """Serialize a Mixedbread StoreFile response for tool-call output."""
    payload = _model_to_json_dict(store_file)
    file_id = str(payload.get("id") or fallback_file_id or "")
    store_id = str(payload.get("store_id") or fallback_store_id or "")
    filename = payload.get("filename")
    content_url = payload.get("content_url")
    file_mime_type = payload.get("mime_type") or payload.get("type")
    chunks = [
        serialize_chunk(
            chunk,
            fallback_file_id=file_id,
            fallback_store_id=store_id,
            fallback_filename=filename,
        )
        for chunk in (payload.get("chunks") or [])
    ]
    if isinstance(content_url, str) and _looks_like_image_media(file_mime_type, content_url):
        for chunk in chunks:
            chunk.setdefault("media_url", content_url)
            if file_mime_type:
                chunk.setdefault("mime_type", file_mime_type)

    return _drop_none(
        {
            "file_id": file_id,
            "store_id": store_id,
            "filename": filename,
            "external_id": payload.get("external_id"),
            "metadata": payload.get("metadata"),
            "status": payload.get("status"),
            "last_error": payload.get("last_error"),
            "content_url": payload.get("content_url"),
            "chunks": chunks,
            "content": document_content_text(chunks),
        }
    )


def serialize_file_metadata(
    store_file: Any,
    *,
    fallback_store_id: str | None = None,
) -> dict[str, Any]:
    """Serialize file-level metadata for ``filter_metadata`` output."""
    payload = _model_to_json_dict(store_file)
    file_title = chunk_file_title(payload)
    return _drop_none(
        {
            "file_id": str(payload.get("id") or payload.get("file_id") or ""),
            "store_id": str(payload.get("store_id") or fallback_store_id or ""),
            "filename": payload.get("filename"),
            "external_id": payload.get("external_id"),
            "file_title": file_title,
            "metadata": _compact_metadata_payload(payload.get("metadata")),
            "generated_metadata": _compact_metadata_payload(payload.get("generated_metadata")),
            "status": payload.get("status"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "content_url": payload.get("content_url"),
            "mime_type": payload.get("mime_type") or payload.get("type"),
            "type": payload.get("type"),
        }
    )


def serialize_chunk(
    chunk: Any,
    *,
    fallback_file_id: str | None = None,
    fallback_store_id: str | None = None,
    fallback_filename: str | None = None,
) -> dict[str, Any]:
    """Serialize a Mixedbread search or file chunk into stable tool output."""
    payload = _model_to_json_dict(chunk)
    file_title = chunk_file_title(payload, fallback_filename=fallback_filename)
    result = {
        "chunk_index": _int_or_zero(payload.get("chunk_index")),
        "file_id": str(payload.get("file_id") or fallback_file_id or ""),
        "store_id": str(payload.get("store_id") or fallback_store_id or ""),
        "filename": payload.get("filename") or fallback_filename,
        "external_id": payload.get("external_id"),
        "file_title": file_title,
        "mime_type": payload.get("mime_type") or payload.get("type"),
        "type": payload.get("type"),
        "search_score": _rounded_score(payload.get("score")),
        "text": payload.get("text"),
        "context": payload.get("context"),
        "ocr_text": payload.get("ocr_text"),
        "transcription": payload.get("transcription"),
        "summary": payload.get("summary"),
        "metadata": payload.get("metadata"),
        "generated_metadata": payload.get("generated_metadata"),
        "image_url": payload.get("image_url"),
        "media_url": payload.get("media_url") or payload.get("url"),
    }
    _populate_summary(result)
    return _drop_none(result)


def chunk_content_text(chunk: Mapping[str, Any]) -> str:
    """Return the available human-readable content from a chunk."""
    parts: list[str] = []
    for field in (
        "file_title",
        "text",
        "context",
        "ocr_text",
        "transcription",
        "summary",
    ):
        value = chunk.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts)


def document_content_text(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Join chunk content into a readable document body."""
    sections: list[str] = []
    for chunk in chunks:
        content = chunk_content_text(chunk)
        if not content:
            continue
        sections.append(f"[chunk_index={chunk.get('chunk_index')}]\n{content}")
    return "\n\n".join(sections)


def normalize_ranked_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a ranking item returned by an agent."""
    return {
        "chunk_id": str(chunk.get("chunk_id", "")).strip(),
        "relevance_score": float(chunk.get("relevance_score", 0.0) or 0.0),
    }


def normalize_chunk_key(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a chunk identifier."""
    return {
        "chunk_id": str(chunk.get("chunk_id", "")).strip(),
    }


def normalize_document_key(document: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a document identifier."""
    return {
        "document_id": str(document.get("document_id", "")).strip(),
    }


def chunk_key(chunk: Mapping[str, Any]) -> ChunkKey:
    """Return a stable tuple key for a chunk."""
    return (
        str(chunk.get("store_id", "")),
        str(chunk.get("file_id", "")),
        _int_or_zero(chunk.get("chunk_index")),
    )


def document_key(document: Mapping[str, Any]) -> DocumentKey:
    """Return a stable tuple key for a document."""
    return (
        str(document.get("store_id", "")),
        str(document.get("file_id", "")),
    )


def chunk_file_title(
    payload: Mapping[str, Any],
    *,
    fallback_filename: str | None = None,
) -> str | None:
    """Return the best short title/name available for a file or chunk."""
    metadata = payload.get("metadata")
    metadata_values = metadata if isinstance(metadata, Mapping) else {}
    generated_metadata = payload.get("generated_metadata")
    generated_values = generated_metadata if isinstance(generated_metadata, Mapping) else {}
    title_candidates = (
        (payload, ("title", "file_title", "name")),
        (metadata_values, ("title", "file_title", "ad_name", "campaign_name", "name")),
        (generated_values, ("title", "file_title", "ad_name", "campaign_name", "name")),
        (payload, ("filename", "external_id")),
        (metadata_values, ("filename", "external_id")),
        (generated_values, ("filename", "external_id")),
    )
    for source, keys in title_candidates:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if fallback_filename:
        return str(fallback_filename).strip() or None
    return None


def _normalize_filter_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_filter_value(item) for item in value]
    return value


def _sortable_metadata_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return float(stripped)
        except ValueError:
            return stripped.casefold()
    return str(value).casefold()


def _metadata_value_key(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


_NO_DATA = object()


def _response_items(response: Any) -> list[Any]:
    """The ``data`` items of a seam response, loudly rejecting unknown shapes.

    Every retrieval seam result carries its items under ``data`` -- an SDK
    model's attribute or a plain mapping's key; an explicit ``data: None``
    counts as empty. A response with NO data container at all must fail the
    call: returning [] would make a mis-shaped in-process binding
    indistinguishable from an empty store, silently zeroing retrieval on every
    rollout while each tool call reports success.
    """
    if isinstance(response, Mapping):
        data = response.get("data", _NO_DATA)
    else:
        data = getattr(response, "data", _NO_DATA)
    if data is _NO_DATA:
        msg = (
            f"retrieval response of type {type(response).__name__!r} carries no 'data' "
            "items; in-process bindings must return their hits under 'data' (a mapping "
            "key or an attribute), even when empty"
        )
        raise TypeError(msg)
    return [] if data is None else list(data)


def _compact_facet_values(value: Any, *, max_values: int) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _compact_facet_value(item) for key, item in list(value.items())[:max_values]
        }
    if isinstance(value, list):
        return [_compact_facet_value(item) for item in value[:max_values]]
    return _compact_facet_value(value)


def _compact_facet_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(value, 240)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _compact_facet_value(item) for key, item in list(value.items())[:12]}
    if isinstance(value, list):
        return [_compact_facet_value(item) for item in value[:12]]
    return _truncate(str(value), 240)


async def _sampled_metadata_value_types(
    *,
    store_identifiers: Sequence[str],
    client: AsyncRetrievalClient | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, set[str]]:
    """Infer metadata field value types from real chunk payloads.

    The metadata-facets endpoint returns values as JSON object keys, so every
    value arrives stringified and the field's true type is lost. Chunk metadata
    preserves the original types.
    """
    try:
        chunks = await list_chunks_raw(
            store_identifiers=store_identifiers,
            top_k=METADATA_TYPE_SAMPLE_TOP_K,
            metadata_filter=None,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    except Exception:
        return {}
    types: dict[str, set[str]] = {}
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        for payload_key in ("metadata", "generated_metadata"):
            _collect_metadata_value_types(chunk.get(payload_key), prefix="", types=types)
    return types


def _collect_metadata_value_types(
    payload: Any,
    *,
    prefix: str,
    types: dict[str, set[str]],
) -> None:
    if not isinstance(payload, Mapping):
        return
    for key, value in payload.items():
        field_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _collect_metadata_value_types(value, prefix=field_name, types=types)
            continue
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, (Mapping, list)):
                    types.setdefault(field_name, set()).add(_json_type_name(item))
            continue
        types.setdefault(field_name, set()).add(_json_type_name(value))


def _json_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _typed_facet_samples(values: Any, field_types: set[str] | None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if isinstance(values, Mapping):
        for value, count in values.items():
            samples.append(_typed_facet_sample(value, count, field_types))
        return samples
    if isinstance(values, list):
        for item in values:
            if isinstance(item, Mapping):
                value = item.get("value")
                if value is None and "key" in item:
                    value = item.get("key")
                if value is None and "name" in item:
                    value = item.get("name")
                if value is None:
                    samples.append(dict(item))
                    continue
                count = next(
                    (
                        item[count_key]
                        for count_key in ("count", "doc_count", "frequency")
                        if isinstance(item.get(count_key), (int, float))
                    ),
                    None,
                )
                samples.append(_typed_facet_sample(value, count, field_types))
            else:
                samples.append(_typed_facet_sample(item, None, field_types))
        return samples
    if values in (None, ""):
        return samples
    samples.append(_typed_facet_sample(values, None, field_types))
    return samples


def _typed_facet_sample(
    value: Any,
    count: Any,
    field_types: set[str] | None,
) -> dict[str, Any]:
    typed_value = _coerce_facet_value(value, field_types)
    sample: dict[str, Any] = {"value": typed_value}
    if field_types is not None:
        sample["type"] = _json_type_name(typed_value)
    if isinstance(count, (int, float)):
        sample["count"] = count
    return sample


def _coerce_facet_value(value: Any, field_types: set[str] | None) -> Any:
    if not isinstance(value, str) or not field_types:
        return value
    non_null_types = field_types - {"null"}
    stripped = value.strip()
    if non_null_types == {"number"}:
        try:
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return value
    if non_null_types == {"boolean"}:
        lowered = stripped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


def _compact_metadata_payload(value: Any) -> Any:
    """Compact metadata values without dropping top-level field names."""
    if isinstance(value, Mapping):
        return {str(key): _compact_facet_value(item) for key, item in value.items()}
    return _compact_facet_value(value)


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "..."


def _resolve_store_identifiers(
    store_identifiers: Sequence[str] | None,
) -> list[str]:
    if store_identifiers is not None:
        return _normalize_store_identifiers(store_identifiers)

    for env_var in _STORE_IDENTIFIER_ENV_VARS:
        raw_value = os.getenv(env_var)
        if raw_value:
            return _normalize_store_identifiers(raw_value.split(","))

    raise ValueError(
        "store_identifiers must be provided or configured in one of: "
        + ", ".join(_STORE_IDENTIFIER_ENV_VARS)
    )


def _normalize_store_identifiers(store_identifiers: Sequence[str]) -> list[str]:
    if isinstance(store_identifiers, str):
        store_identifiers = [store_identifiers]
    store_ids = [str(store_id).strip() for store_id in store_identifiers if str(store_id).strip()]
    if not store_ids:
        raise ValueError("store_identifiers must contain at least one store id or name")
    return store_ids


def _model_to_json_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        # warnings=False: SDK responses nest union-typed chunk models whose
        # serializer mismatches are harmless, and formatting the warnings
        # reprs whole chunk payloads on every call.
        return value.model_dump(mode="json", exclude_none=True, warnings=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a mapping or pydantic model, got {type(value)!r}")


def _summary_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    summary = value.strip()
    return summary or None


def _populate_summary(
    chunk: dict[str, Any],
    *,
    allow_text_fallback: bool = False,
) -> None:
    mixedbread_summary = _summary_text(chunk.get("summary"))
    if mixedbread_summary:
        chunk["summary"] = mixedbread_summary
        return

    text_summary = _text_fallback_summary(chunk.get("text")) if allow_text_fallback else None
    if text_summary:
        chunk["summary"] = text_summary
    else:
        chunk.pop("summary", None)


def _text_fallback_summary(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return None
    sentences = re.split(r"(?<=[.!?])\s+", text)
    first_sentences = " ".join(
        sentence.strip()
        for sentence in sentences[:_OVERVIEW_TEXT_FALLBACK_SUMMARY_MAX_SENTENCES]
        if sentence.strip()
    )
    return _truncate(
        first_sentences or text,
        _OVERVIEW_TEXT_FALLBACK_SUMMARY_MAX_CHARS,
    )


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _rounded_score(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value or 0.0), 4)


def _int_or_zero(value: Any) -> int:
    return int(value or 0)


def _looks_like_image_media(mime_type: Any, url: str) -> bool:
    mime_type_text = str(mime_type or "").strip().lower()
    if mime_type_text.startswith("image/") or mime_type_text == "image_url":
        return True
    clean_url = url.split("?", 1)[0].lower()
    return clean_url.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))
