"""Shared tool-call helpers for agent runtimes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from agent_harness import config


def parse_tool_args(tool_call: Any, schema: type[BaseModel] | None) -> Any:
    if schema is None:
        return {"error": f"Unknown tool: {tool_call.function.name}"}
    try:
        raw = json.loads(tool_call.function.arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        return {"error": f"Invalid JSON: {exc}"}
    try:
        return schema.model_validate(raw)
    except ValueError as exc:
        return {"error": f"Invalid {tool_call.function.name} arguments: {exc}"}


def tool_message(tool_call_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False, default=str),
    }


def tool_error(tool_call_id: str, error: str) -> dict[str, Any]:
    return tool_message(tool_call_id, {"error": error})


def over_budget_round_missing_prune(
    over_budget: bool,
    tool_calls: Sequence[Any],
    *,
    final_tool_name: str | None,
) -> bool:
    """Whether an over-budget round failed to prune without finishing.

    Once a prompt crosses the prune-reminder trigger the agent is asked to
    include ``prune_context`` (in parallel with any other tools). A round that
    ignores that -- neither pruning nor submitting the final ``final_tool_name``
    -- is recorded on the iteration summary for observability, without an
    error being surfaced back to the model. ``final_tool_name=None`` means the
    episode ends on a prose turn, which no tool call can be.
    """
    if not over_budget:
        return False
    names = {
        str(getattr(getattr(tool_call, "function", None), "name", "")) for tool_call in tool_calls
    }
    finished = final_tool_name is not None and final_tool_name in names
    return "prune_context" not in names and not finished


def agent_caused_payload_error(payload: Any) -> str | None:
    """The model-caused failure recorded inside a tool payload, if any.

    ``execute_get_chunks``/``execute_read_document`` report unresolvable ids in
    their return value rather than raising; this maps those payloads onto an
    ``error`` trace status the way ``prune_context`` errors on the same
    mistake. Only ids the model got wrong count; store-side and provider
    failures are classified separately.
    """
    if not isinstance(payload, Mapping):
        return None
    invalid_chunk_ids = payload.get("invalid_chunk_ids")
    if isinstance(invalid_chunk_ids, (list, tuple)) and invalid_chunk_ids:
        return "Unknown or unavailable chunk_id values: " + ", ".join(
            str(chunk_id) for chunk_id in invalid_chunk_ids
        )
    if payload.get("invalid_request"):
        error = payload.get("error")
        return str(error) if error else "Invalid tool arguments"
    return None


def media_messages_for_tool_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build chat-completions image-input messages from image chunks in a tool result."""
    if config.MEDIA_CONTENT == "never":
        return []
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return []
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    return media_messages_for_payload(payload)


def media_messages_for_payload(payload: Any) -> list[dict[str, Any]]:
    if config.MEDIA_CONTENT == "never":
        return []
    parts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for chunk in _iter_image_chunks(payload):
        if not config.include_media_content_for_chunk(chunk):
            continue
        image_url = _image_url_from_chunk(chunk)
        if not image_url:
            continue
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        document_id = str(chunk.get("document_id") or "").strip()
        dedupe_key = (chunk_id, document_id, image_url)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        label = _media_label_for_chunk(chunk)
        # chunk_id/document_id are internal keys used by media redaction; they are
        # stripped before the request reaches the model API.
        part_metadata = {
            "chunk_id": chunk_id,
            "document_id": document_id,
        }
        parts.append(
            {
                "type": "text",
                "text": label,
                **part_metadata,
            }
        )
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
                **part_metadata,
            }
        )

    if not parts:
        return []
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Retrieved image chunks attached for visual inspection. Use the labels to map each image back to its chunk_id.",
                },
                *parts,
            ],
        }
    ]


def _iter_image_chunks(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_image_chunks(item)
        return
    if not isinstance(payload, Mapping):
        return
    if "chunk_id" in payload and _image_url_from_chunk(payload):
        yield payload
    for value in payload.values():
        yield from _iter_image_chunks(value)


def _image_url_from_chunk(chunk: Mapping[str, Any]) -> str | None:
    image_url = chunk.get("image_url")
    if isinstance(image_url, Mapping):
        url = image_url.get("url") or image_url.get("image_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    if isinstance(image_url, str) and image_url.strip():
        return image_url.strip()

    return None


def _media_label_for_chunk(chunk: Mapping[str, Any]) -> str:
    fields = [
        ("chunk_id", chunk.get("chunk_id")),
        ("document_id", chunk.get("document_id")),
        ("chunk_index", chunk.get("chunk_index")),
        ("filename", chunk.get("filename")),
        ("file_title", chunk.get("file_title")),
    ]
    details = [f"{name}={value}" for name, value in fields if value not in (None, "", [])]
    return "Image for retrieved chunk: " + " ".join(details)
