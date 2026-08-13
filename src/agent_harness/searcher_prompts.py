"""Prompts for the fast-searcher runtime."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .config import (
    FILTER_CHUNKS_DEFAULT_K,
    FILTER_CHUNKS_MAX_K,
    GREP_DEFAULT_K,
    MAX_PARALLEL_TOOL_CALLS,
    SEARCH_CORPUS_TOP_K,
    searcher_max_rounds,
)
from .prompts import initial_metadata_facets_message


def build_fast_searcher_task_description(
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> str:
    if strict_top_k and top_k is not None:
        return (
            f"Given a user query and a set of search tools, report exactly {top_k} "
            "document chunks ordered by relevance."
        )
    return (
        "Given a user query and a set of search tools, report the document chunks "
        "relevant to the query ordered by relevance."
    )


def build_searcher_system_prompt(
    *,
    task_description: str,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> str:
    context_metadata_source = (
        "- INITIAL_METADATA_FACETS is the source of truth for valid filter keys, value formats,\n"
        "  rank fields, and representative sample values. Samples are incomplete and are not\n"
        "  exhaustive enums, especially for high-cardinality identifier fields such as invoice_id.\n"
        "  Result metadata may confirm more fields or values. Do not invent metadata."
    )
    first_turn_guidance = (
        "- If INITIAL_SEARCH_RESULTS already answer the query, call submit_ranking immediately.\n"
        "- Otherwise, choose matching tools: overview_search for orientation, search_corpus for\n"
        "  semantic meaning, grep for exact tokens, filter_chunks for metadata."
    )
    handle_guidance = "submit_ranking"
    metadata_confirmation_source = "facets or result metadata"
    prune_subject = "user query"
    final_rules = _final_tool_rules(
        top_k=top_k,
        strict_top_k=strict_top_k,
    )
    max_rounds = searcher_max_rounds()

    return f"""You are a specialized search agent in a document retrieval pipeline.

TASK:
{task_description}

If the task description conflicts with the tool, metadata, context, or output rules below, follow
the rules below.

CONTEXT:
{context_metadata_source}
- Search results expose short handles: chunk_id identifies an exact chunk, document_id identifies
  a document. Use these handles in later tool calls and {handle_guidance}.
- Use the runtime UTC date for relative date/recency requests unless the user gives another
  timezone. Prefer half-open timestamp ranges and only use date-only strings when facets/results
  show that format; check likely fields such as active_from, active_to, created_at, launch_date,
  launched_at, and days_active.
- For audio/video, judge all available text fields. For image chunks, if exist inspect attached
  images directly when visual details matter; otherwise rely on text fields.

WORKFLOW:
- Plan silently first. Classify the user query as semantic, long-description, broad-recall,
  metadata-constrained, metric-ordered, or exact-literal.
{first_turn_guidance}
- Do not wait for first results before choosing the initial diverse search set.
- Use at most {MAX_PARALLEL_TOOL_CALLS} total tool calls in one turn, including retrieval,
  document and pruning tools.
- Follow-up searches should pivot to new evidence gaps, not shallow paraphrases. Use get_chunks
  for exact already-seen chunks and read_document when nearby context around a chunk matters.
- You have at most {max_rounds} rounds (the final submit_ranking turn is one of them); from the second round on, a
  "Search round N of max {max_rounds}." line marks which round the tool results above
  came from. It is a ceiling, not a quota. End the episode yourself: call
  submit_ranking in its own turn as soon as the evidence you have supports it, at latest in
  your final round. Extra rounds are not free; do not search on after the evidence is sufficient,
  and never wait to be told to submit.

RETRIEVAL:
- overview_search is summary-only orientation; use summary to identify promising themes,
  terminology, files, and follow-up searches.
- search_corpus is for focused semantic search. Each call returns up to {SEARCH_CORPUS_TOP_K}
  new unseen chunks. Use natural language, one meaning/aspect per query, and preserve exact user
  clues, entities, scene details, and relationships. Avoid Boolean syntax, regex, quoted-term
  operators, and keyword dumps.
- For semantic/open-ended or long-description aspects, call multiple search_corpus queries that
  chase different facets, entities, dates, settings, roles, or remembered wording.
- For broad-recall aspects, pair overview_search with multiple focused search_corpus calls; avoid
  relying on one generic query.
- Put hard structured constraints in filter_by; keep the semantic query about meaning. Use
  metadata-filtered searches only when {metadata_confirmation_source} confirm the
  exact key, value, and format.
- Use filter_chunks for metadata-first list/category/status/date tasks. Default k={FILTER_CHUNKS_DEFAULT_K}; raise it
  only when the user asks for more or broader coverage is needed, max {FILTER_CHUNKS_MAX_K}.
- Omit rank_by unless the task asks for numeric ordering and the field is confirmed
  numeric by {metadata_confirmation_source}.
- Use grep for keywords, regex, exact tokens, codes, identifiers, function names, SKUs, or literal
  phrases. grep matches literal chunk/generated text, not meaning, and returns up to
  {GREP_DEFAULT_K} chunks.
- search_corpus and grep deduplicate already retrieved chunks. If the same focused query/pattern is still
  promising and needs more depth, repeat it; prefer a new focused variant when the evidence gap differs.

Context management:
- prune_context removes content, not IDs. Pruned chunks stay in your seen index and will not be
  returned again from normal searches. Only get_chunks can restore pruned chunk content again.
- Call prune_context to remove chunk content irrelevant to your {prune_subject}. Keep the context
  window small; you have a limited token budget.
- Once you receive a context budget notice, include prune_context among your tool calls that round
  to drop content you no longer need, or call submit_ranking if you are done. prune_context may
  run in parallel with searches in the same turn, so a budget notice never has to cost a search --
  you do not need a prune-only turn. prune_context counts toward the {MAX_PARALLEL_TOOL_CALLS} call
  limit; do not exceed that limit.
- prune_context must include at least one valid chunk_id or document_id.

EXAMPLES:
- "environmental and economic impact of solar energy" -> search_corpus with the original aspect
  plus diverse angles: lifecycle emissions, manufacturing impact, jobs/economy, land use/wildlife.
- Long remembered clip/scene description -> search_corpus with variants that preserve details,
  dates, setting, roles, and remembered wording.
- "articles by jane doe about renewable energy" -> use confirmed author/byline facets, then combine
  metadata-filtered semantic retrieval with any needed filter_chunks checks.

{final_rules}
"""


def _final_tool_rules(
    *,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> str:
    if strict_top_k and top_k is not None:
        count_rule = (
            f"- submit_ranking.chunks must contain exactly {top_k} chunks ranked most-relevant first;\n"
            "  relevance_score in [0, 1]. If fewer than "
            f"{top_k} chunks are strongly relevant, fill the remaining slots with the\n"
            "  next-best retrieved chunks."
        )
        submit_rule = (
            f"- Call submit_ranking with exactly {top_k} chunks when you have enough evidence."
        )
    else:
        count_rule = (
            "- submit_ranking.chunks must include all chunks relevant to the user query; rank most-relevant first;\n"
            "  relevance_score in [0, 1]. Use an empty chunk list only if no relevant chunks exist."
        )
        submit_rule = "- Call submit_ranking when you have enough evidence."
    return f"""RANKING:
- Before submit_ranking, compare the retrieved chunks with each other. Rank for the user's intent,
  not just search_score.
- For metric/order requests, compare the relevant metadata field or value stated in content and
  order by the requested direction. Example: "highest budget campaigns" ranks by budget descending.
{count_rule}
- ranking_strategy must state the interpretation, constraints, comparison basis, and final ordering rule.
- Use only chunk_id values that appeared in your tool results. Do not duplicate chunk_id values.

OUTPUT RULES:
- USE TOOLS ONLY. Never generate a plain text response.
{submit_rule}
- submit_ranking must be the only tool call in its turn.
- For audio/video chunks, trust the search score as the relevance signal (the full media is not available to you)."""


def fast_searcher_messages(
    *,
    user_text: str,
    initial_search_results: Mapping[str, Any] | None = None,
    initial_metadata_facets: Mapping[str, Any] | None = None,
    top_k: int | None = None,
    strict_top_k: bool = False,
    additional_instructions: str | None = None,
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    prompt = build_searcher_system_prompt(
        task_description=_with_additional_instructions(
            build_fast_searcher_task_description(top_k=top_k, strict_top_k=strict_top_k),
            additional_instructions,
        ),
        top_k=top_k,
        strict_top_k=strict_top_k,
    )
    messages = [
        {"role": "system", "content": prompt + _runtime_context(as_of)},
    ]
    if initial_metadata_facets is not None:
        messages.append(initial_metadata_facets_message(initial_metadata_facets))
    if initial_search_results is not None:
        messages.append(initial_search_results_message(initial_search_results))
    messages.append({"role": "user", "content": f"USER_QUERY:\n{user_text}"})
    return messages


def _with_additional_instructions(
    task_description: str,
    additional_instructions: str | None,
) -> str:
    instructions = (additional_instructions or "").strip()
    if not instructions:
        return task_description
    return f"{task_description}\n\nADDITIONAL INSTRUCTIONS:\n{instructions}"


def initial_search_results_message(
    initial_search_results: Mapping[str, Any],
) -> dict[str, Any]:
    # Prompt-only normalization: preserve full-precision provider scores in the
    # index/ranking path while preventing insignificant response jitter from
    # forking otherwise identical c3 prefixes. Restrict this to each retrieval
    # result's own score fields; nested domain metadata called "score" is data.
    prompt_results = dict(initial_search_results)
    results = initial_search_results.get("results")
    if isinstance(results, (list, tuple)):
        normalized_results: list[Any] = []
        for result in results:
            if not isinstance(result, Mapping):
                normalized_results.append(result)
                continue
            normalized = dict(result)
            for field_name in ("score", "search_score"):
                value = normalized.get(field_name)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and value != 0
                ):
                    normalized[field_name] = float(f"{float(value):.2g}")
            normalized_results.append(normalized)
        prompt_results["results"] = normalized_results
    return {
        "role": "user",
        "content": (
            "INITIAL_SEARCH_RESULTS:\n"
            f"{json.dumps(prompt_results, ensure_ascii=False, default=str)}"
        ),
    }


def _runtime_context(as_of: date | None = None) -> str:
    """Runtime-context block anchoring the agent's relative-date reasoning.

    ``as_of`` pins "today" instead of reading the wall clock -- see
    ``prompts._runtime_context``.
    """
    today = datetime.now(UTC).date() if as_of is None else as_of
    yesterday = today - timedelta(days=1)
    return (
        "\nRuntime context:\n"
        f"- Current UTC date: {today.isoformat()}.\n"
        "- Relative date queries use this UTC date unless the user gives another "
        f"timezone; yesterday is {yesterday.isoformat()}.\n"
    )
