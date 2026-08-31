from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from agent_harness.config import (
    FILTER_CHUNKS_DEFAULT_K,
    FILTER_CHUNKS_MAX_K,
    GREP_DEFAULT_K,
    SEARCH_CORPUS_TOP_K,
)
from agent_harness.schemas import AGENTIC_SEARCH_FILTER_OPERATORS

from .shared import FILTER_VALUE_SCALAR_TYPES, _apply_string_enum


def _agentic_metadata_filter_condition_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "minLength": 1,
                "description": "Metadata field name. Dot notation is supported for nested fields.",
            },
            "operator": {
                "type": "string",
                "enum": list(AGENTIC_SEARCH_FILTER_OPERATORS),
            },
            "value": {
                "description": "Value to compare against. Use an array for in/not_in.",
                "anyOf": [
                    *FILTER_VALUE_SCALAR_TYPES,
                    {"type": "array", "items": {"anyOf": FILTER_VALUE_SCALAR_TYPES}},
                ],
            },
        },
        "required": ["key", "operator", "value"],
        "additionalProperties": False,
    }


def _filter_by_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "description": (
            "Optional metadata filters. Use only keys and values from provided "
            "metadata facets or prior result metadata."
        ),
        "items": _agentic_metadata_filter_condition_schema(),
    }


def _agentic_filter_mode_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": ["all", "any"],
        "default": "all",
        "description": "Combine filter_by conditions with AND (`all`) or OR (`any`).",
    }


SEARCH_CORPUS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Execute a meaning-based semantic search query and return chunks you have not "
            "already seen. Do not use for keyword, regex, or literal-string matching; use grep. "
            f"Returns up to {SEARCH_CORPUS_TOP_K} chunks. Optionally apply Mixedbread metadata filters "
            "when field names and values are known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Natural-language query for ONE meaning/aspect; avoid Boolean syntax, regex, "
                        "and keyword dumps."
                    ),
                },
                "filter_by": _filter_by_schema(),
                "filter_mode": _agentic_filter_mode_schema(),
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

FILTER_CHUNKS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "filter_chunks",
        "description": (
            "List chunks matching metadata filters, optionally ordered by a numeric metadata field. "
            "No semantic search, no reranker. Use when the query is best answered by structured metadata "
            "(filters and/or ranking) rather than by semantic match. With no rank_by, results come back in "
            "deterministic chunk-index order; with rank_by, results are ordered by that numeric field. "
            f"Returns up to k chunks; default {FILTER_CHUNKS_DEFAULT_K}, max {FILTER_CHUNKS_MAX_K}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filter_by": _filter_by_schema(),
                "filter_mode": _agentic_filter_mode_schema(),
                "rank_by": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Optional numeric metadata field path to rank by; paths target file metadata "
                        "fields, and facet samples show each field's type. A non-numeric field never "
                        "fails the call: chunks with no numeric value keep the deterministic order used "
                        "when rank_by is omitted and follow any ranked chunks. The response reports "
                        "rank_by_applied and rank_by_non_numeric_count. Omit to get deterministic "
                        "chunk-index order."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "default": "desc",
                    "description": "Rank direction. Only meaningful when rank_by is set.",
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": FILTER_CHUNKS_MAX_K,
                    "default": FILTER_CHUNKS_DEFAULT_K,
                    "description": (
                        "Number of chunks to return. Use the default unless the user asks for more "
                        "or the task needs broader candidate coverage."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
}

GREP_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Find chunks whose literal text matches a regular expression. No embeddings, no semantic match, "
            "no reranker - this is exact pattern matching. Use it when the user wants chunks containing "
            "a keyword, regex, exact token, code, identifier, function name, SKU, or literal phrase rather "
            "than a topic or meaning. "
            f"Returns up to {GREP_DEFAULT_K} chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1024,
                    "description": (
                        "Regular expression matched against chunk text. A chunk matches if the pattern is "
                        "found anywhere in any targeted field."
                    ),
                },
                "targets": {
                    "type": "array",
                    "description": (
                        "'text' matches original extracted text/context; 'generated' matches OCR "
                        "text, summaries, and transcriptions. Defaults to both, which is almost "
                        "always what you want: a store's content lives in only one of these buckets "
                        "(page-image and audio/video corpora have text-empty chunks and all content "
                        "under 'generated'), so naming a single target can return zero matches for a "
                        "pattern the corpus does contain. Narrow only to exclude a bucket on purpose."
                    ),
                    "items": {"type": "string", "enum": ["text", "generated"]},
                },
                "case_sensitive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the regex is case-sensitive. Defaults to case-insensitive.",
                },
                "filter_by": _filter_by_schema(),
                "filter_mode": _agentic_filter_mode_schema(),
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_RANKING_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_ranking",
        "description": (
            "Submit the final ranked list of all chunks you judge relevant when you "
            "have gathered enough evidence. Choose the number of chunks yourself "
            "based on the user's query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ranking_strategy": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Briefly state how you interpreted the query, which hard "
                        "constraints you applied, and how you ordered the final chunks."
                    ),
                },
                "chunks": {
                    "type": "array",
                    "minItems": 0,
                    "description": (
                        "List of chunks you judge relevant to the question, ranked by "
                        "relevance (most relevant first). Include the requested number "
                        "when the user asks for one; otherwise include every meaningfully "
                        "relevant chunk and do not pad to a fixed count. Use an empty "
                        "list only when no chunks are relevant."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Short chunk handle returned by search tools, for example c12",
                            },
                            "relevance_score": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": (
                                    "Your assessed relevance score from 0 to 1 "
                                    "(1 = highly relevant)."
                                ),
                            },
                        },
                        "required": [
                            "chunk_id",
                            "relevance_score",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["ranking_strategy", "chunks"],
            "additionalProperties": False,
        },
    },
}


def search_corpus_tool() -> dict[str, Any]:
    return deepcopy(SEARCH_CORPUS_TOOL)


def filter_chunks_tool() -> dict[str, Any]:
    return deepcopy(FILTER_CHUNKS_TOOL)


def grep_tool() -> dict[str, Any]:
    return deepcopy(GREP_TOOL)


def submit_ranking_tool(
    *,
    chunk_ids: Sequence[str] = (),
    strict: bool = False,
    top_k: int | None = None,
    strict_top_k: bool = False,
    require_answer: bool = False,
) -> dict[str, Any]:
    """The final tool's schema for one rollout.

    ``require_answer`` (answer_mode="submit_ranking") adds a mandatory
    ``answer`` string and says so in the description.
    """
    tool = deepcopy(SUBMIT_RANKING_TOOL)
    exact_top_k = strict_top_k and top_k is not None
    if exact_top_k:
        tool["function"]["description"] = f"Submit the final ranked list of exactly {top_k} chunks."
    if require_answer:
        _require_answer(tool, chunk_count=f"exactly {top_k} chunks" if exact_top_k else "chunks")
    chunks_schema = tool["function"]["parameters"]["properties"]["chunks"]
    if exact_top_k:
        chunks_schema["description"] = (
            f"Exactly {top_k} chunks ranked by relevance (most relevant first). "
            f"If fewer than {top_k} chunks are strongly relevant, fill the remaining "
            "slots with the next-best retrieved chunks."
        )
        chunks_schema["minItems"] = top_k
        chunks_schema["maxItems"] = top_k
    chunk_id_schema = chunks_schema["items"]["properties"]["chunk_id"]
    _apply_string_enum(chunk_id_schema, chunk_ids)
    if strict:
        tool["function"]["strict"] = True
    return tool


def _require_answer(tool: dict[str, Any], *, chunk_count: str) -> None:
    tool["function"]["description"] = (
        f"Submit the final ranked list of {chunk_count} and your final answer to the user query."
    )
    parameters = tool["function"]["parameters"]
    parameters["properties"]["answer"] = {
        "type": "string",
        "minLength": 1,
        "description": (
            "Your final answer to the original user query, based only on retrieved "
            "evidence. Required on every submit_ranking call: give your single best "
            "answer even when uncertain; if the evidence is insufficient to answer, say so."
        ),
    }
    parameters["required"].append("answer")
