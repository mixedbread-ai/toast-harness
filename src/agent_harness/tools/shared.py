from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from agent_harness.config import (
    GET_CHUNKS_MAX_CHUNK_IDS,
    OVERVIEW_SEARCH_TOP_K,
    READ_DOCUMENT_MAX_WINDOW,
)

FILTER_VALUE_SCALAR_TYPES: list[dict[str, Any]] = [
    {"type": "string"},
    {"type": "number"},
    {"type": "boolean"},
    {"type": "null"},
]


READ_DOCUMENT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": (
            "Retrieve a bounded chunk window from a document, centered on a known chunk_id. "
            "Returns the anchor chunk plus x chunk indices before and after it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The short document handle returned in search results, for example d3",
                },
                "chunk_id": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The short chunk handle to center the document window on, for example c12",
                },
                "x": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": READ_DOCUMENT_MAX_WINDOW,
                    "default": 1,
                    "description": (
                        "Number of chunk indices before and after chunk_id to return. "
                        "Defaults to 1, which returns the previous chunk, anchor chunk, "
                        f"and next chunk. At most {READ_DOCUMENT_MAX_WINDOW}."
                    ),
                },
            },
            "required": ["document_id", "chunk_id"],
            "additionalProperties": False,
        },
    },
}

# This tool restores exact chunks even if they were already seen or pruned.
GET_CHUNKS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_chunks",
        "description": (
            "Retrieve one or more already-seen chunks by their short chunk_id handles. "
            f"Accepts at most {GET_CHUNKS_MAX_CHUNK_IDS} chunk_ids per call. Chunks that do "
            "not fit the per-call payload budget are truncated to fit, earliest ids "
            "keeping the most text; requesting fewer ids shows more of each, and a "
            "single-id request shows the most a call can display."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": GET_CHUNKS_MAX_CHUNK_IDS,
                    "description": (
                        "The short chunk handles returned in search results, "
                        f'for example ["c12", "c13"]. At most {GET_CHUNKS_MAX_CHUNK_IDS} per call.'
                    ),
                    "items": {"type": "string", "minLength": 1},
                }
            },
            "required": ["chunk_ids"],
            "additionalProperties": False,
        },
    },
}

OVERVIEW_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "overview_search",
        "description": (
            f"Executes a semantic search query and returns a high-level overview of up to {OVERVIEW_SEARCH_TOP_K} "
            "relevant chunks not yet seen by the current agent. The overview contains a summary of each chunk's content. "
            "Overview results carry no metadata and no chunk text, so no attribute constraint can be "
            "verified from them: re-fetch the returned handles with get_chunks, or use filter_chunks "
            "or grep, before treating a constraint as checked."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "A search query you want a broad overview for, for example the user query itself",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


PRUNE_CONTEXT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "prune_context",
        "description": (
            "Prune the context by removing irrelevant or redundant chunk/document content. "
            "Before hard context budget, call this only in the same parallel tool turn as "
            "another useful tool; prune-only turns are for hard budget only. At least one "
            "chunk_id or document_id must be provided."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "default": [],
                    "description": (
                        "Short chunk handles to prune, for example c3. "
                        "This removes chunk content from your context, not overview summaries."
                    ),
                    "items": {"type": "string", "minLength": 1},
                },
                "document_ids": {
                    "type": "array",
                    "default": [],
                    "description": (
                        "Short document handles to prune, for example d2. "
                        "This removes document-window content retrieved with read_document."
                    ),
                    "items": {"type": "string", "minLength": 1},
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}


def overview_search_tool() -> dict[str, Any]:
    return deepcopy(OVERVIEW_SEARCH_TOOL)


def read_document_tool(
    *,
    document_ids: Sequence[str] = (),
    chunk_ids: Sequence[str] = (),
) -> dict[str, Any]:
    tool = deepcopy(READ_DOCUMENT_TOOL)
    _apply_string_enum(
        tool["function"]["parameters"]["properties"]["document_id"],
        document_ids,
    )
    _apply_string_enum(
        tool["function"]["parameters"]["properties"]["chunk_id"],
        chunk_ids,
    )
    return tool


def get_chunks_tool(*, chunk_ids: Sequence[str] = ()) -> dict[str, Any]:
    tool = deepcopy(GET_CHUNKS_TOOL)
    _apply_string_enum(
        tool["function"]["parameters"]["properties"]["chunk_ids"]["items"],
        chunk_ids,
    )
    return tool


def prune_context_tool(
    *,
    chunk_ids: Sequence[str] = (),
    document_ids: Sequence[str] = (),
) -> dict[str, Any]:
    tool = deepcopy(PRUNE_CONTEXT_TOOL)
    properties = tool["function"]["parameters"]["properties"]
    _apply_string_enum(properties["chunk_ids"]["items"], chunk_ids)
    _apply_string_enum(properties["document_ids"]["items"], document_ids)
    return tool


def _apply_string_enum(schema: dict[str, Any], values: Sequence[str]) -> None:
    cleaned = _clean_enum_values(values)
    if cleaned:
        schema["enum"] = cleaned


def _clean_enum_values(values: Sequence[str]) -> list[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})
