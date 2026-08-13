"""Helpers for serializing agent tool-call turns and trace events."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any


def summarize_tool_call(tool_call: Any) -> dict[str, Any]:
    function = getattr(tool_call, "function", None)
    raw_arguments = getattr(function, "arguments", "") or "{}"

    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except json.JSONDecodeError:
        arguments = raw_arguments

    summary = {
        "call_id": str(getattr(tool_call, "id", "") or ""),
        "name": getattr(function, "name", ""),
        "arguments": arguments,
    }
    if isinstance(raw_arguments, str):
        summary["raw_arguments"] = raw_arguments
    return summary


def summarize_tool_call_iteration(
    *,
    agent: str,
    iteration: int,
    tool_calls: Sequence[Any],
    over_budget_without_prune: bool = False,
) -> dict[str, Any]:
    group_size = len(tool_calls)
    summary: dict[str, Any] = {
        "agent": agent,
        "iteration": iteration,
        "execution_mode": "parallel" if group_size > 1 else "sequential",
        "calls": [summarize_tool_call(tool_call) for tool_call in tool_calls],
    }
    if over_budget_without_prune:
        summary["over_budget_without_prune"] = True
    return summary


def start_tool_call_trace(
    *,
    agent: str,
    iteration: int,
    tool_call: Any,
    call_index: int,
    group_size: int,
) -> dict[str, Any]:
    """Create a mutable trace event for one model-requested tool call."""
    summarized = summarize_tool_call(tool_call)
    call_id = summarized.get("call_id") or f"{agent}-{iteration}-{call_index}"
    event: dict[str, Any] = {
        "event_type": "tool_call",
        "agent": agent,
        "iteration": iteration,
        "call_id": call_id,
        "call_index": call_index,
        "group_size": group_size,
        "execution_mode": "parallel" if group_size > 1 else "sequential",
        "parallel_group_id": f"{agent}:{iteration}",
        "name": summarized.get("name"),
        "arguments": jsonable(summarized.get("arguments")),
        "raw_arguments": summarized.get("raw_arguments"),
        "status": "started",
        "started_at": _utc_now_iso(),
        "_started_perf": perf_counter(),
    }
    return event


def finish_tool_call_trace(
    event: dict[str, Any],
    *,
    status: str = "success",
    output: Any = None,
    metadata: Any = None,
    error: str | None = None,
    error_kind: str | None = None,
) -> None:
    """Finalize a trace event with output, metadata, timing, and error state."""
    event["status"] = status
    event["completed_at"] = _utc_now_iso()
    started = event.pop("_started_perf", None)
    if isinstance(started, (int, float)):
        event["duration_ms"] = round((perf_counter() - started) * 1000, 3)

    if metadata is not None:
        event["metadata"] = jsonable(metadata)
    if output is not None:
        jsonable_output = jsonable(output)
        event["output"] = jsonable_output
        output_summary = summarize_tool_output(jsonable_output)
        if output_summary:
            event["output_summary"] = output_summary
    if error is not None:
        event["error"] = str(error)
    if error_kind is not None:
        event["error_kind"] = error_kind


def synthetic_tool_call_trace(
    *,
    agent: str,
    iteration: int,
    name: str,
    arguments: Any = None,
    output: Any = None,
    metadata: Any = None,
    status: str = "success",
    error: str | None = None,
    error_kind: str | None = None,
    attempt: int | None = None,
    call_id: str | None = None,
    forced: bool = True,
    started_at: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """Return a trace event for tool calls that bypass normal dispatch.

    Covers forced final-tool calls and the bootstrap fetches (``forced=False``
    with explicit ``call_id``/timing). ``attempt`` distinguishes the individual
    tries of a forced-submit retry loop, which share an agent/iteration and
    would otherwise collide on ``call_id``.
    """
    if call_id is None:
        call_id = f"{agent}-forced-{name}-{iteration}"
        if attempt is not None:
            call_id = f"{call_id}-attempt-{attempt}"
    event: dict[str, Any] = {
        "event_type": "tool_call",
        "agent": agent,
        "iteration": iteration,
        "call_id": call_id,
        "call_index": 1,
        "group_size": 1,
        "execution_mode": "sequential",
        "parallel_group_id": f"{agent}:{iteration}",
        "name": name,
        "arguments": jsonable(arguments),
        "status": "started",
        "started_at": started_at or _utc_now_iso(),
        "forced": forced,
    }
    if attempt is not None:
        event["attempt"] = attempt
    finish_tool_call_trace(
        event,
        status=status,
        output=output,
        metadata=metadata,
        error=error,
        error_kind=error_kind,
    )
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    return event


def jsonable(value: Any) -> Any:
    """Convert SDK/Pydantic/runtime objects into JSON-compatible values."""
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except Exception:
            # Some runtime objects carry fields mode="json" cannot serialize
            # (e.g. raw buffers). Fall back to mode="python" and sanitize
            # recursively instead of crashing the rollout.
            return jsonable(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, set):
        return [jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize_tool_output(payload: Any) -> dict[str, Any]:
    payload = jsonable(payload)
    summary: dict[str, Any] = {}
    counts = _output_counts(payload)
    if counts:
        summary["counts"] = counts
    entities = _extract_entities(payload)
    if entities:
        summary["entities"] = entities
    return summary


_ENTITY_BUCKETS: dict[str, str] = {
    "query": "queries",
    "top_k": "top_k_values",
    "requested_top_k": "top_k_values",
    "search_top_k": "top_k_values",
    "k": "top_k_values",
    "filename": "file_names",
    "file_name": "file_names",
    "file_names": "file_names",
    "file_id": "file_ids",
    "file_ids": "file_ids",
    "external_id": "external_ids",
    "external_ids": "external_ids",
    "document_id": "document_ids",
    "document_ids": "document_ids",
    "chunk_id": "chunk_ids",
    "chunk_ids": "chunk_ids",
    "chunk_index": "chunk_indices",
    "chunk_indices": "chunk_indices",
    "store_id": "store_ids",
    "store_ids": "store_ids",
    "store_identifier": "store_identifiers",
    "store_identifiers": "store_identifiers",
}


def _extract_entities(payload: Any) -> dict[str, Any]:
    entities: dict[str, list[Any]] = {}
    _collect_entities(payload, entities, depth=0)
    return {bucket: values for bucket, values in sorted(entities.items()) if values}


def _collect_entities(value: Any, entities: dict[str, list[Any]], *, depth: int) -> None:
    if depth > 8:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            bucket = _ENTITY_BUCKETS.get(str(key))
            if bucket:
                _add_entity_value(entities, bucket, item)
            _collect_entities(item, entities, depth=depth + 1)
        return
    if isinstance(value, list):
        for item in value:
            _collect_entities(item, entities, depth=depth + 1)


def _add_entity_value(entities: dict[str, list[Any]], bucket: str, value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _add_entity_value(entities, bucket, item)
        return
    if isinstance(value, Mapping):
        return
    if value in (None, "", []):
        return
    values = entities.setdefault(bucket, [])
    normalized = jsonable(value)
    if normalized not in values:
        values.append(normalized)


def _output_counts(payload: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not isinstance(payload, Mapping):
        return counts
    for key in (
        "results",
        "files",
        "chunks",
        "new_unseen_results",
        "new_unseen_chunk_ids",
        "distinct_values",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            counts[key] = len(value)
    for key in (
        "files_returned",
        "metadata_field_count",
        "distinct_value_count",
        "deduped_existing_or_deleted",
    ):
        value = payload.get(key)
        if isinstance(value, int):
            counts[key] = value
    return counts


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
