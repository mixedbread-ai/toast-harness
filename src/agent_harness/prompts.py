"""Shared prompt messages for the fast-searcher loop."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any


def initial_metadata_facets_message(metadata_facets: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            "INITIAL_METADATA_FACETS:\n"
            f"{json.dumps(_metadata_facets_for_prompt(metadata_facets), ensure_ascii=False, default=str)}"
        ),
    }


def _metadata_facets_for_prompt(metadata_facets: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(metadata_facets)
    # Facet payloads are produced by inspect_metadata, but searcher agents do not
    # have that tool. Keep the provenance in runtime telemetry rather than
    # presenting an unavailable tool name to the model.
    payload.pop("tool", None)
    fields = payload.get("metadata_fields")
    if isinstance(fields, Mapping):
        payload["metadata_fields"] = {
            str(field_name): {
                "sample_values": _facet_sample_values_for_prompt(values),
            }
            for field_name, values in fields.items()
        }
    payload["metadata_filter_syntax"] = {
        "filter_by": [
            {
                "key": "<metadata_fields field name>",
                "operator": "eq",
                "value": "<verified value>",
            }
        ],
        "note": (
            "In filter_by, key is the metadata field name, for example invoice_id. "
            "Facet samples below use value/type/count; they are examples, not complete "
            "enums. Match the sample type when filtering, for example numeric values "
            "for number fields. rankable_fields lists the fields rank_by can order; "
            "ranking by any other field returns an unranked deterministic order."
        ),
    }
    return payload


def _facet_sample_values_for_prompt(values: Any) -> list[dict[str, Any]]:
    if isinstance(values, Mapping):
        return [_facet_sample_for_prompt(value=key, metadata=item) for key, item in values.items()]
    if isinstance(values, list):
        return [_facet_sample_for_prompt(item) for item in values]
    if values in (None, ""):
        return []
    return [{"value": values}]


def _facet_sample_for_prompt(
    item: Any = None,
    *,
    value: Any = None,
    metadata: Any = None,
) -> dict[str, Any]:
    if value is not None:
        sample: dict[str, Any] = {"value": value}
        if isinstance(metadata, (int, float)):
            sample["count"] = metadata
        return sample
    if isinstance(item, Mapping):
        sample_value = item.get("value")
        if sample_value is None:
            sample_value = item.get("key")
        if sample_value is None:
            sample_value = item.get("name")
        sample = {"value": sample_value if sample_value is not None else dict(item)}
        type_name = item.get("type")
        if isinstance(type_name, str) and type_name:
            sample["type"] = type_name
        for count_key in ("count", "doc_count", "frequency"):
            count = item.get(count_key)
            if isinstance(count, (int, float)):
                sample["count"] = count
                break
        return sample
    return {"value": item}


def _runtime_context(as_of: date | None = None) -> str:
    """Runtime-context block anchoring the agent's relative-date reasoning.

    ``as_of`` pins "today" instead of reading the wall clock: a caller
    evaluating against a corpus with a fixed reference date can resolve
    "yesterday"/"last week" against that date, and a serving caller can pass
    the user's local date so relative queries resolve in the user's timezone
    rather than UTC's.
    """
    today = datetime.now(UTC).date() if as_of is None else as_of
    yesterday = today - timedelta(days=1)
    return (
        "\nRuntime context:\n"
        f"- Current UTC date: {today.isoformat()}.\n"
        "- Relative date queries use this UTC date unless the user gives another "
        f"timezone; yesterday is {yesterday.isoformat()}.\n"
    )


def over_budget_message(
    estimated_tokens: int,
    *,
    final_tool_name: str | None = "submit_ranking",
) -> dict[str, Any]:
    """Reminder sent on any round whose prompt is over the prune trigger.

    The same message fires whether the prompt is just over the trigger or
    already at the truncation ceiling; it asks for a parallel prune rather
    than a forced solo prune turn. ``final_tool_name=None`` is the plain-text
    answer protocol, which has no final tool to name.
    """
    finish_instruction = (
        f"or call {final_tool_name} if you are done."
        if final_tool_name is not None
        else "or reply with your final answer if you are done."
    )
    return {
        "role": "user",
        "content": (
            "Context budget notice: your current prompt is estimated at "
            f"{estimated_tokens} tokens, over your context budget. Include prune_context "
            "among your tool calls this round to remove content you no longer need -- it "
            f"may run in parallel with other tools -- {finish_instruction}"
        ),
    }


def round_label(round_index: int, max_rounds: int, *, label: str = "Search round") -> str:
    """Position marker for the round whose tool results precede it.

    "of max" rather than "of": the ceiling is a bound the agent may stop short
    of, not a quota to fill. Kept to bare state with no instruction -- the
    system prompt already tells the agent to end the episode itself.
    """
    return f"{label} {round_index} of max {max_rounds}."


def round_notice_message(
    round_index: int,
    max_rounds: int,
    *,
    label: str = "Search round",
    final_tool_name: str | None = "submit_ranking",
) -> dict[str, Any]:
    """Round header appended from round 2 on, after the previous round's tool results.

    Round 1 is skipped deliberately: its prompt is the prefix shared by every
    rollout of a query, and there are no prior tool results to label yet.

    AGENT_HARNESS_FINAL_ROUND_NOTICE=1 appends a protocol-aware pre-notice on
    the final round; ``final_tool_name=None`` is the plain-text answer protocol.
    """
    content = round_label(round_index, max_rounds, label=label)
    if round_index >= max_rounds and os.environ.get("AGENT_HARNESS_FINAL_ROUND_NOTICE", "") in (
        "1",
        "true",
    ):
        content += _final_round_notice(final_tool_name)
    return {
        "role": "user",
        "content": content,
    }


def _final_round_notice(final_tool_name: str | None) -> str:
    if final_tool_name is None:
        return (
            " This is your final search round: reply with your final answer now if "
            "the evidence is sufficient; otherwise prioritize confirming your best "
            "evidence with the tools available this round."
        )
    return (
        " This is your final search round: after these tool calls return, you "
        f"must make your {final_tool_name} call. Prioritize confirming your best "
        "candidates now."
    )


def force_submit_message(
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
    require_answer: bool = False,
    round_index: int | None = None,
    max_rounds: int | None = None,
    label: str = "Search round",
) -> dict[str, Any]:
    # Emphasise EXACTLY ONE submit_ranking call: the searcher loop rejects a turn
    # with any other tool or multiple calls, and thinking-enabled runs otherwise
    # tend to keep searching here instead of submitting.
    answer_clause = " and your final answer in the answer field" if require_answer else ""
    if strict_top_k and top_k is not None:
        content = (
            "You have reached the search limit. Do NOT search further. You must now make "
            f"EXACTLY ONE submit_ranking tool call (no other tools, no parallel calls) with "
            f"ranking_strategy and exactly {top_k} chunks{answer_clause}."
        )
    else:
        content = (
            "You have reached the search limit. Do NOT search further. You must now make "
            "EXACTLY ONE submit_ranking tool call (no other tools, no parallel calls) with "
            f"ranking_strategy and your final ranked chunk list{answer_clause}. Use an "
            "empty list only if no relevant chunks exist."
        )
    return _forced_turn_message(
        content, round_index=round_index, max_rounds=max_rounds, label=label
    )


def force_answer_message(
    *,
    round_index: int | None = None,
    max_rounds: int | None = None,
    label: str = "Search round",
) -> dict[str, Any]:
    """The plain-text counterpart of :func:`force_submit_message`."""
    content = (
        "You have reached the search limit. Do NOT search further. You must now reply "
        "with your final answer to the user query as plain text, with NO tool calls. "
        "Base it only on retrieved evidence; if the evidence is insufficient to answer, "
        "say so."
    )
    return _forced_turn_message(
        content, round_index=round_index, max_rounds=max_rounds, label=label
    )


def _forced_turn_message(
    content: str,
    *,
    round_index: int | None,
    max_rounds: int | None,
    label: str,
) -> dict[str, Any]:
    if round_index is not None and max_rounds is not None:
        content = f"{round_label(round_index, max_rounds, label=label)} {content}"
    return {
        "role": "user",
        "content": content,
    }
