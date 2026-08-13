"""Tests for prompt-overflow prevention: round-payload truncation and over-budget pruning."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import agent_harness.config as harness_config
from agent_harness.agents import searcher as searcher_runtime
from agent_harness.prompts import over_budget_message
from agent_harness.search import (
    ChunkIndex,
    ToolOutcome,
    estimate_payload_tokens,
    serialize_agent_chunks,
    truncate_round_payloads,
)
from agent_harness.sync_api import run_fast_agentic_search as sync_run_fast_agentic_search


def test_over_budget_message_offers_parallel_prune_or_final_submit() -> None:
    assert over_budget_message(41_000)["content"] == (
        "Context budget notice: your current prompt is estimated at 41000 tokens, "
        "over your context budget. Include prune_context among your tool calls this round "
        "to remove content you no longer need -- it may run in parallel with other tools -- "
        "or call submit_ranking if you are done."
    )


def _chunk(
    i: int,
    *,
    file_id: str | None = None,
    chunk_index: int = 0,
    text_chars: int = 800,
) -> dict[str, Any]:
    return {
        "store_id": "store-a",
        "file_id": file_id or f"file-{i}",
        "chunk_index": chunk_index,
        "text": f"chunk-{i} " + "x" * text_chars,
        "search_score": 1.0 - i * 0.01,
    }


def _search_payload(
    index: ChunkIndex,
    chunks: list[dict[str, Any]],
    *,
    query: str = "q",
) -> dict[str, Any]:
    new_chunks = index.ingest_search_results(chunks)
    return {
        "tool": "search_corpus",
        "query": query,
        "new_unseen_results": serialize_agent_chunks(new_chunks, refs=index.refs),
    }


def test_truncation_drops_whole_entries_and_discards_unseen_chunks() -> None:
    index = ChunkIndex()
    payload = _search_payload(index, [_chunk(i) for i in range(10)])
    entries_before = [dict(entry) for entry in payload["new_unseen_results"]]
    budget = estimate_payload_tokens(payload) // 2

    stats = truncate_round_payloads([payload], index=index, remaining_tokens=budget)

    assert stats[0] is not None
    assert stats[0]["results_omitted"] >= 1
    assert payload["results_omitted"] == stats[0]["results_omitted"]
    assert "truncation_notice" in payload
    assert estimate_payload_tokens(payload) <= budget

    kept = payload["new_unseen_results"]
    # Survivors are the untouched head of the score-sorted list.
    assert kept == entries_before[: len(kept)]
    json.dumps(payload)

    kept_ids = {entry["chunk_id"] for entry in kept}
    dropped_ids = {entry["chunk_id"] for entry in entries_before[len(kept) :]}
    visible = set(index.visible_chunk_ids())
    assert kept_ids <= visible
    assert not dropped_ids & visible
    # Documents whose only chunk was dropped disappear from the schema enums too.
    visible_documents = set(index.visible_document_ids())
    kept_document_ids = {entry["document_id"] for entry in kept}
    dropped_document_ids = {entry["document_id"] for entry in entries_before[len(kept) :]}
    assert kept_document_ids <= visible_documents
    assert not dropped_document_ids & visible_documents


def test_truncation_keeps_top_entry_even_with_no_budget() -> None:
    index = ChunkIndex()
    payload = _search_payload(index, [_chunk(i) for i in range(5)])
    top_id = payload["new_unseen_results"][0]["chunk_id"]

    stats = truncate_round_payloads([payload], index=index, remaining_tokens=0)

    assert stats[0] is not None
    assert [entry["chunk_id"] for entry in payload["new_unseen_results"]] == [top_id]
    assert index.visible_chunk_ids() == [top_id]


def test_discarded_chunks_can_be_reingested_later() -> None:
    index = ChunkIndex()
    chunks = [_chunk(i) for i in range(6)]
    payload = _search_payload(index, chunks)

    truncate_round_payloads(
        [payload],
        index=index,
        remaining_tokens=estimate_payload_tokens(payload) // 3,
    )

    kept_count = len(payload["new_unseen_results"])
    dropped_chunks = chunks[kept_count:]
    assert dropped_chunks
    reingested = index.ingest_search_results(dropped_chunks)
    assert len(reingested) == len(dropped_chunks)


def test_budget_is_shared_across_parallel_payloads() -> None:
    index = ChunkIndex()
    payloads = [
        _search_payload(
            index,
            [_chunk(i, file_id=f"g{group}-f{i}") for i in range(8)],
            query=f"q{group}",
        )
        for group in range(4)
    ]
    budget = sum(estimate_payload_tokens(payload) for payload in payloads) // 2

    truncate_round_payloads(payloads, index=index, remaining_tokens=budget)

    kept_counts = [len(payload["new_unseen_results"]) for payload in payloads]
    # Equal-sized payloads get equal shares: no parallel call is sacrificed.
    assert min(kept_counts) >= 1
    assert max(kept_counts) - min(kept_counts) <= 1
    assert sum(estimate_payload_tokens(payload) for payload in payloads) <= budget


def test_dropped_chunk_references_are_scrubbed_from_sibling_payloads() -> None:
    index = ChunkIndex()
    fat = _search_payload(index, [_chunk(i, text_chars=2000) for i in range(4)])
    fat_ids = [entry["chunk_id"] for entry in fat["new_unseen_results"]]
    sibling = {
        "tool": "grep",
        "pattern": "x",
        "results": [
            {"chunk_id": fat_ids[0], "seen": True},
            {"chunk_id": fat_ids[-1], "seen": True},
        ],
    }
    budget = estimate_payload_tokens(sibling) + 400

    stats = truncate_round_payloads([fat, sibling], index=index, remaining_tokens=budget)

    kept_ids = {entry["chunk_id"] for entry in fat["new_unseen_results"]}
    assert fat_ids[0] in kept_ids
    assert fat_ids[-1] not in kept_ids
    # The sibling's reference to the dropped chunk is scrubbed and counted.
    sibling_ids = [entry["chunk_id"] for entry in sibling["results"]]
    assert sibling_ids == [fat_ids[0]]
    assert stats[1] is not None
    assert stats[1]["results_omitted"] == 1
    assert fat_ids[-1] not in set(index.visible_chunk_ids())


def test_dropping_seen_reference_does_not_discard_indexed_chunk() -> None:
    index = ChunkIndex()
    [old] = index.ingest_search_results([_chunk(0)])
    old_id = serialize_agent_chunks([old], refs=index.refs)[0]["chunk_id"]
    [fresh] = index.ingest_search_results([_chunk(1)])
    payload = {
        "tool": "grep",
        "pattern": "x",
        "results": [
            *serialize_agent_chunks([fresh], refs=index.refs),
            {"chunk_id": old_id, "seen": True},
        ],
    }

    truncate_round_payloads(
        [payload],
        index=index,
        remaining_tokens=estimate_payload_tokens(payload) - 10,
    )

    assert len(payload["results"]) == 1
    assert payload["results"][0].get("seen") is not True
    # The dropped entry was only a reference; the chunk itself stays available.
    assert old_id in set(index.visible_chunk_ids())


def test_read_document_tail_chunks_dropped_with_mirror() -> None:
    index = ChunkIndex()
    doc_chunks = [_chunk(i, file_id="doc-1", chunk_index=i) for i in range(6)]
    new_chunks = index.ingest_search_results(doc_chunks)
    serialized = serialize_agent_chunks(new_chunks, refs=index.refs)
    payload = {
        "tool": "read_document",
        "document_id": "d01",
        "chunks": [dict(entry) for entry in serialized],
        "new_unseen_chunk_ids": [entry["chunk_id"] for entry in serialized],
    }
    budget = estimate_payload_tokens(payload) // 4

    truncate_round_payloads([payload], index=index, remaining_tokens=budget)

    assert len(payload["chunks"]) < len(serialized)
    kept_ids = [entry["chunk_id"] for entry in payload["chunks"]]
    # The id side channel never points at a chunk that is no longer in the payload.
    assert payload["new_unseen_chunk_ids"] == kept_ids
    dropped_ids = {entry["chunk_id"] for entry in serialized} - set(kept_ids)
    assert dropped_ids
    assert not dropped_ids & set(index.visible_chunk_ids())


def test_get_chunks_payloads_keep_their_top_entry() -> None:
    index = ChunkIndex()
    chunks = [_chunk(i, text_chars=4000) for i in range(2)]
    restored = index.ingest_search_results(chunks)
    payload = {
        "tool": "get_chunks",
        "requested_chunk_ids": ["c01", "c02"],
        "results": serialize_agent_chunks(restored, refs=index.refs),
        "restored_chunk_ids": ["c01", "c02"],
    }

    stats = truncate_round_payloads([payload], index=index, remaining_tokens=10)

    # Multi-id get_chunks payloads are round-truncatable but always keep the top
    # entry; single-id calls are exempt entirely (the recovery escalation).
    assert stats[0] is not None
    assert len(payload["results"]) == 1


def _fake_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _fake_response(
    tool_calls: list[Any],
    *,
    input_tokens: int = 10,
    output_tokens: int = 2,
) -> Any:
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _patch_initial_fetches(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
) -> None:
    async def fake_initial_metadata_facets(**kwargs: Any) -> ToolOutcome:
        return ToolOutcome(
            {"type": "INITIAL_METADATA_FACETS", "metadata_fields": {}},
            {"tool": "inspect_metadata"},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        fake_initial_metadata_facets,
    )

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        index.add_chunk(
            {
                "store_id": "store-a",
                "file_id": "seed",
                "chunk_index": 0,
                "text": "seed chunk",
            }
        )
        chunk_id = index.visible_chunk_ids()[0]
        captured["seed_chunk_id"] = chunk_id
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "seed",
                "results": [{"chunk_id": chunk_id, "text": "seed chunk"}],
            },
            {"tool": "search_corpus", "new_chunks_added": 1},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )


def test_over_budget_rounds_consume_search_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(searcher_runtime, "SEARCHER_PRUNE_REMINDER_TOKENS", 1)
    captured: dict[str, Any] = {}
    _patch_initial_fetches(monkeypatch, captured)
    loop_calls: list[tuple[list[str], bool, str]] = []

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del messages, completion_config
        if force_submit and forced_tool_name == "submit_ranking":
            return _fake_response(
                [
                    _fake_tool_call(
                        "call_forced",
                        "submit_ranking",
                        {
                            "ranking_strategy": "forced",
                            "chunks": [
                                {
                                    "chunk_id": captured["seed_chunk_id"],
                                    "relevance_score": 1.0,
                                }
                            ],
                        },
                    )
                ]
            )
        tool_names = [tool["function"]["name"] for tool in tools]
        loop_calls.append((tool_names, force_submit, forced_tool_name))
        return _fake_response(
            [
                _fake_tool_call(
                    f"call_prune_{len(loop_calls)}",
                    "prune_context",
                    {"chunk_ids": [captured["seed_chunk_id"]], "document_ids": []},
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find things",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    # Every round counts against the budget, over-budget rounds included, so
    # the loop stops at SEARCHER_MAX_ROUNDS and then forces a ranking.
    assert len(loop_calls) == harness_config.SEARCHER_MAX_ROUNDS
    # The tools schema stays identical to a normal round; the missing prune is
    # recorded after generation instead of shrinking the tool list, so token-level
    # prefix reuse never sees a schema change on over-budget turns.
    assert all("prune_context" in tool_names for tool_names, _, _ in loop_calls)
    assert all(len(tool_names) > 1 for tool_names, _, _ in loop_calls)
    assert all(force_submit is False for _, force_submit, _ in loop_calls)
    assert result.rounds_executed == harness_config.SEARCHER_MAX_ROUNDS
    assert result.forced_ranking is True


def test_over_budget_round_without_prune_is_flagged_and_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(searcher_runtime, "SEARCHER_PRUNE_REMINDER_TOKENS", 1)
    captured: dict[str, Any] = {}
    _patch_initial_fetches(monkeypatch, captured)

    async def fake_execute_search_corpus(
        args: Any,
        *,
        index: ChunkIndex,
        store_identifiers: Any,
        client: Any = None,
        api_key: Any = None,
        api_key_env: Any = None,
    ) -> ToolOutcome:
        del index, store_identifiers, client, api_key, api_key_env
        query = str(args["query"])
        return ToolOutcome(
            {"tool": "search_corpus", "query": query, "new_unseen_results": []},
            {"tool": "search_corpus", "query": query, "new_chunks_added": 0},
        )

    monkeypatch.setattr(searcher_runtime, "execute_search_corpus", fake_execute_search_corpus)
    attempts = 0

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        nonlocal attempts
        del messages, tools, completion_config, forced_tool_name
        assert force_submit is False
        attempts += 1
        ranking = {
            "ranking_strategy": "done over budget",
            "chunks": [{"chunk_id": captured["seed_chunk_id"], "relevance_score": 1.0}],
        }
        if attempts == 1:
            # Over budget with neither prune nor submit: the call runs instead of
            # being rejected, and the round is flagged in the iteration record.
            return _fake_response(
                [_fake_tool_call("call_search", "search_corpus", {"query": "still-runs"})]
            )
        # Prune and submit in the same parallel turn are both honored now.
        return _fake_response(
            [
                _fake_tool_call(
                    "call_prune",
                    "prune_context",
                    {"chunk_ids": [captured["seed_chunk_id"]], "document_ids": []},
                ),
                _fake_tool_call("call_parallel_submit", "submit_ranking", ranking),
            ]
        )

    result = sync_run_fast_agentic_search(
        "find things",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    assert attempts == 2
    assert result.forced_ranking is False
    assert result.ranking is not None
    # None of the calls were rejected -- the lone over-budget search executed...
    handled = {
        event["call_id"]: event
        for event in result.tool_trace
        if event.get("call_id") in {"call_search", "call_prune", "call_parallel_submit"}
    }
    assert handled.keys() == {"call_search", "call_prune", "call_parallel_submit"}
    assert all(event["status"] != "error" for event in handled.values())
    assert any(query.get("query") == "still-runs" for query in result.queries_made)
    # ...but exactly the round that skipped pruning while over budget is flagged,
    # and the prune+submit round is not.
    flagged = [
        iteration
        for iteration in result.tool_call_iterations
        if iteration.get("over_budget_without_prune")
    ]
    assert len(flagged) == 1
    assert [call["name"] for call in flagged[0]["calls"]] == ["search_corpus"]


def test_fast_search_truncates_oversized_round_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    _patch_initial_fetches(monkeypatch, captured)
    generated_ids: dict[str, list[str]] = {}

    async def fake_execute_search_corpus(
        args: Any,
        *,
        index: ChunkIndex,
        store_identifiers: Any,
        client: Any = None,
        api_key: Any = None,
        api_key_env: Any = None,
    ) -> ToolOutcome:
        del store_identifiers, client, api_key, api_key_env
        query = str(args["query"])
        chunks = [_chunk(i, file_id=f"{query}-f{i}", text_chars=1200) for i in range(8)]
        new_chunks = index.ingest_search_results(chunks)
        serialized = serialize_agent_chunks(new_chunks, refs=index.refs)
        generated_ids[query] = [entry["chunk_id"] for entry in serialized]
        return ToolOutcome(
            {"tool": "search_corpus", "query": query, "new_unseen_results": serialized},
            {"tool": "search_corpus", "query": query, "new_chunks_added": len(new_chunks)},
        )

    monkeypatch.setattr(searcher_runtime, "execute_search_corpus", fake_execute_search_corpus)

    turn_tool_names: list[list[str]] = []
    turn_over_budget: list[bool] = []

    def _visible_chunk_ids(messages: list[dict[str, Any]]) -> set[str]:
        # The model's only source of truth for which chunk_ids are still
        # addressable is what's actually in its context -- there's no enum
        # constraint on the tool schemas to consult instead.
        ids: set[str] = set()
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                payload = json.loads(message["content"])
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if payload.get("tool") != "search_corpus":
                continue
            for entry in payload.get("new_unseen_results") or []:
                ids.add(entry["chunk_id"])
        return ids

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del completion_config
        tool_names = [tool["function"]["name"] for tool in tools]
        turn_tool_names.append(tool_names)
        turn_over_budget.append(
            bool(messages)
            and messages[-1].get("role") == "user"
            and str(messages[-1].get("content", "")).startswith("Context budget notice:")
        )
        turn = len(turn_tool_names)
        if turn == 1:
            # Report near-limit real usage so the round's payloads must be clipped.
            return _fake_response(
                [
                    _fake_tool_call("call_a", "search_corpus", {"query": "alpha"}),
                    _fake_tool_call("call_b", "search_corpus", {"query": "beta"}),
                ],
                input_tokens=searcher_runtime.SEARCHER_PROMPT_TOKEN_LIMIT - 2_000,
            )
        if turn_over_budget[-1]:
            return _fake_response(
                [
                    _fake_tool_call(
                        f"call_prune_{turn}",
                        "prune_context",
                        {"chunk_ids": [captured["seed_chunk_id"]], "document_ids": []},
                    )
                ]
            )
        visible_ids = _visible_chunk_ids(messages)
        kept_alpha = next(
            chunk_id for chunk_id in generated_ids["alpha"] if chunk_id in visible_ids
        )
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "post-truncation",
                        "chunks": [{"chunk_id": kept_alpha, "relevance_score": 1.0}],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find things",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    assert result.forced_ranking is False
    assert result.ranking is not None
    # Turn 2 was an over-budget prune turn triggered by the reported usage.
    # The tools schema stays the same full/stable set as any other round --
    # a skipped prune is recorded without shrinking that list.
    assert turn_over_budget[1] is True
    assert "prune_context" in turn_tool_names[1]
    assert len(turn_tool_names[1]) > 1

    truncated_payloads = []
    for message in result.messages:
        if message.get("role") != "tool":
            continue
        payload = json.loads(message["content"])
        if payload.get("tool") == "search_corpus" and payload.get("results_omitted"):
            truncated_payloads.append(payload)
    assert truncated_payloads
    for payload in truncated_payloads:
        assert payload["new_unseen_results"]
        assert "truncation_notice" in payload

    trace_metadata = [
        event.get("metadata")
        for event in result.tool_trace
        if isinstance(event.get("metadata"), dict) and event["metadata"].get("payload_truncated")
    ]
    assert trace_metadata

    all_ids = set(generated_ids["alpha"]) | set(generated_ids["beta"])
    kept_ids = {
        entry["chunk_id"]
        for payload in truncated_payloads
        for entry in payload["new_unseen_results"]
    }
    dropped_ids = all_ids - kept_ids
    assert dropped_ids

    # The truncation must survive the record round-trip, which is what any
    # downstream consumer of the rollout actually reads.
    record_trace = result.to_record()["tool_trace"]
    assert any(
        isinstance(event.get("metadata"), dict) and event["metadata"].get("payload_truncated")
        for event in record_trace
    )
