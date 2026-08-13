"""Portable manifest for reproducing the Mixedbread searcher on another runtime."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import date
from importlib.resources import files
from typing import Any, Final

from agent_harness.config import (
    INITIAL_SEARCH_TOP_K,
    MAX_PARALLEL_TOOL_CALLS,
    METADATA_REPRESENTATIVE_VALUES_PER_FIELD,
    SEARCHER_PRUNE_REMINDER_TOKENS,
    searcher_max_rounds,
)
from agent_harness.searcher_prompts import (
    _runtime_context,
    build_fast_searcher_task_description,
    build_searcher_system_prompt,
)
from agent_harness.tools.searcher_only import (
    filter_chunks_tool,
    grep_tool,
    search_corpus_tool,
    submit_ranking_tool,
)
from agent_harness.tools.shared import (
    get_chunks_tool,
    overview_search_tool,
    prune_context_tool,
    read_document_tool,
)

SEARCHER_CONTRACT_SCHEMA_VERSION: Final = "mixedbread_searcher.v1"
SEARCHER_CONTRACT_SCHEMA_RESOURCE: Final = "schemas/mixedbread_searcher.v1.schema.json"
# Sentinel fed to the live runtime-context builder to derive its template: the
# two ISO strings it produces (2001-01-01 and 2000-12-31) cannot appear in the
# block's static wording, so replacing them yields exactly the placeholders.
_RUNTIME_CONTEXT_PIN: Final = date(2001, 1, 1)


def _runtime_context_template() -> str:
    """The live system prompt's runtime-context suffix, dates as placeholders.

    Derived from the live builder at export time, so a wording change lands in
    the exported contract (and trips the goldens) instead of drifting behind an
    unpinned suffix."""
    pinned = _runtime_context(_RUNTIME_CONTEXT_PIN)
    return pinned.replace("2001-01-01", "{utc_date}").replace("2000-12-31", "{utc_yesterday}")


FAST_INITIALIZATION_STAGES: Final = (
    "initial_metadata",
    "initial_search",
    "prompt_assembly",
)


def build_searcher_contract(
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> dict[str, Any]:
    """Return JSON-shaped prompt, tool, and policy semantics for parity tests."""

    final_tool = submit_ranking_tool(top_k=top_k, strict_top_k=strict_top_k)
    provider_tools = [
        overview_search_tool(),
        search_corpus_tool(),
        filter_chunks_tool(),
        grep_tool(),
        read_document_tool(),
        get_chunks_tool(),
        prune_context_tool(),
        final_tool,
    ]
    tools = [deepcopy(tool["function"]) for tool in provider_tools]
    task_description = build_fast_searcher_task_description(
        top_k=top_k,
        strict_top_k=strict_top_k,
    )
    system_prompt = build_searcher_system_prompt(
        task_description=task_description,
        top_k=top_k,
        strict_top_k=strict_top_k,
    )
    initialization_stages = FAST_INITIALIZATION_STAGES
    prompt = {
        "system": system_prompt,
        # ``system`` is built from the bare task description and stops before the
        # runtime-context suffix the live message appends. additional_instructions
        # are folded into the task description *before* the system prompt is
        # assembled, so they land mid-string rather than appended -- a host runtime
        # that concatenates them onto this value emits a byte-different prompt.
        "system_validity": (
            "prefix: the live system message is this string plus "
            "runtime_context.template with both dates substituted; a byte-exact "
            "rebuild appends that suffix, and non-empty additional_instructions "
            "must be folded into the task description before assembly"
        ),
        "runtime_context": {
            "template": _runtime_context_template(),
            "utc_date": "the evaluation runtime's current UTC date, YYYY-MM-DD",
            "utc_yesterday": "utc_date minus one day, YYYY-MM-DD",
        },
        "initial_metadata_label": "INITIAL_METADATA_FACETS",
        "user_query_label": "USER_QUERY",
        "initial_search_label": "INITIAL_SEARCH_RESULTS",
    }

    runtime_prompt_rules = {
        "runtime_date_suffix": {
            "clock": "UTC",
            "format": "YYYY-MM-DD",
            "source": "evaluation runtime",
        },
        "additional_instructions": {
            "position": "task description suffix",
            "heading": "ADDITIONAL INSTRUCTIONS",
            "trim": True,
        },
    }
    initial_metadata_wire = {
        "role": "user",
        "prefix": "INITIAL_METADATA_FACETS:\n",
        "projection": {
            "remove_top_level_fields": ["tool"],
            "metadata_fields": "map every field to sample_values only",
            "append_metadata_filter_syntax": True,
        },
    }
    bootstrap: dict[str, Any]
    message_wire: dict[str, Any]
    bootstrap = {
        "semantic_order": list(FAST_INITIALIZATION_STAGES),
        "execution": {
            "default": "parallel",
            "parallel_group": ["initial_metadata", "initial_search"],
            "join_before": "prompt_assembly",
            "sequential_fallback": True,
        },
        "initial_metadata": {
            "operation": "inspect_metadata",
            "arguments": {
                "max_values_per_field": METADATA_REPRESENTATIVE_VALUES_PER_FIELD,
            },
            "success_payload": {
                "type": "INITIAL_METADATA_FACETS",
                "merge": "inspect_metadata result",
            },
            "failure_payload": {
                "type": "INITIAL_METADATA_FACETS",
                "store_identifiers": "requested store identifiers",
                "metadata_fields": {},
                "note": "Initial metadata inspection failed.",
                "error": "provider error text",
            },
            "continue_after_operation_error": True,
        },
        "initial_search": {
            "operation": "search_corpus",
            "arguments": {
                "query": "verbatim user query",
                "top_k": INITIAL_SEARCH_TOP_K,
                "store_identifiers": "requested store identifiers",
            },
            "success_payload": {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "verbatim user query",
                "results": "new_unseen_results in returned order",
            },
            "failure_payload": {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "verbatim user query",
                "results": [],
                "error": "provider error text",
            },
            "continue_after_operation_error": True,
        },
        "prompt_assembly": {
            "message_order": [
                "system",
                "initial_metadata",
                "initial_search",
                "user_query",
                "initial_search_media_messages",
            ],
            **runtime_prompt_rules,
        },
    }
    message_wire = {
        "initial_metadata": initial_metadata_wire,
        "initial_search": {
            "role": "user",
            "prefix": "INITIAL_SEARCH_RESULTS:\n",
            "projection": {
                "base": "identity",
                "result_score_fields": ["score", "search_score"],
                "rule": (
                    "each result's own score and search_score are rendered at 2 "
                    "significant figures when finite, numeric, and non-zero; zero, "
                    "non-numeric, and nested metadata fields named score are untouched"
                ),
            },
        },
        "user_query": {
            "role": "user",
            "prefix": "USER_QUERY:\n",
            "projection": "verbatim",
        },
    }

    message_wire.update(
        {
            "serialization": {
                "format": "JSON",
                "ensure_ascii": False,
                "non_json_values": {
                    "portable_requirement": "JSON-safe by construction",
                    "live_python_legacy_fallback": "stringify",
                },
            },
            "tool_result": {
                "semantic_role": "tool_result",
                "provider_serialization": (
                    "chat-completions, emitted in-package by agent_harness.agents.shared."
                    "tool_message; a host runtime targeting another wire format owns the "
                    "translation"
                ),
                "call_id_field": "tool_call_id",
                "content": "JSON payload serialized with the same policy",
            },
            "media": {
                "source": "initial search and tool-result chunk payloads",
                "placement": "after the associated textual message batch",
                "internal_identity_fields_removed_before_provider": True,
            },
        }
    )

    return deepcopy(
        {
            "schema_version": SEARCHER_CONTRACT_SCHEMA_VERSION,
            "ownership": {
                "host_runtime": ["generation", "transport", "retry", "transcript"],
                "mixedbread_extension": [
                    "prompts",
                    "bootstrap",
                    "tool_contracts",
                    "search_behavior",
                ],
                "evaluation_export": "language-neutral projection",
            },
            "initialization_stages": list(initialization_stages),
            "prompt": prompt,
            "bootstrap": bootstrap,
            "message_wire": message_wire,
            "tools": tools,
            "tool_runtime": {
                "dispatch": {
                    "parallel": True,
                    "max_calls_per_turn": MAX_PARALLEL_TOOL_CALLS,
                    "result_messages_follow_model_tool_call_order": True,
                },
                "retrieval_state": {
                    "chunk_id": "stable handle for one exact chunk",
                    "document_id": "stable handle for one document",
                    "deduplicate_seen_chunks": True,
                    "pruned_content_remains_seen": True,
                    "get_chunks_can_restore_pruned_content": True,
                },
                "metadata_validation": {
                    "source_of_truth": (
                        "initial facets, inspected facets, and observed result metadata"
                    ),
                    "invalid_filter_returns_tool_error": True,
                },
                "result_contract": {
                    "search_operations": (
                        "return a JSON payload plus non-model monitoring metadata"
                    ),
                    "read_operations": "return one JSON payload",
                    "errors": "ordinary tool result with structured error",
                    "final_tool_is_not_echoed_as_tool_result": True,
                },
                "context_management": {
                    "prune_reminder_tokens": SEARCHER_PRUNE_REMINDER_TOKENS,
                    "truncate_before_next_provider_request": True,
                    "trace_matches_model_visible_truncated_payload": True,
                },
            },
            "policy": {
                "final_tool_name": "submit_ranking",
                "final_tool_must_be_only_call": True,
                "max_parallel_tool_calls": MAX_PARALLEL_TOOL_CALLS,
                # Through the tuned accessor, like the prompt builder: the prompt
                # absorbs a per-rollout HarnessTuning.searcher_max_rounds, so an
                # import-time constant here would let a tuned export contradict
                # its own system prompt.
                "max_rounds": searcher_max_rounds(),
                "top_k": top_k,
                "strict_top_k": strict_top_k,
            },
        }
    )


def load_searcher_contract_schema() -> dict[str, Any]:
    """Load the bundled language-neutral JSON Schema for the exported contract."""

    resource = files("agent_harness.searcher_spec").joinpath(SEARCHER_CONTRACT_SCHEMA_RESOURCE)
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("searcher contract schema must be a JSON object")
    return value


def dumps_searcher_contract(contract: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialize a contract deterministically without Python-specific fallbacks."""

    return json.dumps(
        contract,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def searcher_contract_digest(contract: dict[str, Any]) -> str:
    """Hash the exact language-neutral contract with deterministic JSON."""

    payload = dumps_searcher_contract(contract).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
