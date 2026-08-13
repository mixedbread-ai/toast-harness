from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from mixedbread import RateLimitError

import agent_harness.config as harness_config
import agent_harness.llm as llm_runtime
from agent_harness import SearcherExecutionPolicy
from agent_harness.agents import searcher as searcher_runtime
from agent_harness.execution_policy import (
    build_rollout_result,
    count_provider_failures,
    run_searcher,
)
from agent_harness.llm import (
    extend_responses_api_trace,
    failed_generation_response,
    generation_failed,
    require_generation_fn,
    response_responses_api_turns,
    response_to_chat_completion,
    validate_required_tool_response,
)
from agent_harness.prompts import round_notice_message
from agent_harness.search import ToolOutcome
from agent_harness.sync_api import fast_agentic_search as sync_fast_agentic_search
from agent_harness.sync_api import run_fast_agentic_search as sync_run_fast_agentic_search
from agent_harness.versions import __version__ as embedded_harness_version
from agent_harness.versions import (
    assert_compatible_versions,
    check_version_compatibility,
    current_version_manifest,
    extract_version_manifest,
    harness_version,
)


async def _fake_initial_metadata_facets(**kwargs: Any) -> ToolOutcome:
    return ToolOutcome(
        {"type": "INITIAL_METADATA_FACETS", "metadata_fields": {}},
        {"tool": "inspect_metadata"},
    )


def test_searcher_policy_returns_rollout_state(monkeypatch: pytest.MonkeyPatch) -> None:
    def sentinel_generation_fn(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("sentinel_generation_fn should only be forwarded")

    async def fake_fast_agentic_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["additional_instructions"] == "extra framing"
        assert kwargs["media_content"] == "always"
        assert kwargs["include_prompt_snapshot"] is True
        assert kwargs["generation_fn"].generation_fn is sentinel_generation_fn
        return _sample_pipeline_result(agent="fast_searcher", monitoring=True)

    monkeypatch.setattr(
        "agent_harness.execution_policy.fast_agentic_search",
        fake_fast_agentic_search,
    )

    result = run_searcher(
        "find the launch ads",
        store_identifiers=["store-a"],
        additional_instructions="extra framing",
        include_prompt_snapshot=True,
        media_content="always",
        generation_fn=sentinel_generation_fn,
    )

    assert result["retrieval"]["ranked_ids"] == ["c1"]
    assert result["openai"]["turns"][-1]["response"]["id"] == "resp_1"
    metadata = result["openai"]["metadata"]
    assert metadata["execution_policy"] == "searcher"
    assert "execution_policy_version" not in metadata
    assert metadata["versions"]["harness"] == metadata["harness_version"]
    assert metadata["versions"] == current_version_manifest()
    assert "searcher_version" not in metadata
    assert metadata["agent"]["name"] == "searcher"
    assert metadata["agent"]["tool_trace"][0]["agent"] == "fast_searcher"
    prompt_snapshot = metadata["monitoring"]["prompt_snapshot"]
    assert prompt_snapshot["messages"][0]["role"] == "system"
    assert "agent" not in result
    assert "completion" not in result


def test_fast_searcher_uses_injected_generation_fn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        _fake_initial_metadata_facets,
    )

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        index.add_chunk(
            {
                "store_id": "store-a",
                "file_id": "file-a",
                "chunk_index": 0,
                "text": "sample launch ad",
            }
        )
        chunk_id = index.visible_chunk_ids()[0]
        captured["chunk_id"] = chunk_id
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "find the launch ads",
                "results": [{"chunk_id": chunk_id, "text": "sample launch ad"}],
            },
            {"tool": "search_corpus", "new_chunks_added": 1},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        captured["messages"] = messages
        captured["tool_names"] = [tool["function"]["name"] for tool in tools]
        captured["completion_config"] = completion_config
        captured["force_submit"] = force_submit
        captured["forced_tool_name"] = forced_tool_name
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "single injected result",
                        "chunks": [
                            {
                                "chunk_id": captured["chunk_id"],
                                "relevance_score": 1.0,
                            }
                        ],
                    },
                )
            ]
        )

    result = sync_fast_agentic_search(
        "find the launch ads",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    assert result["forced_ranking"] is False
    assert result["ranking_strategy"] == "single injected result"
    assert result["chunks"][0]["chunk_id"] == captured["chunk_id"]
    assert result["chunks"][0]["relevance_score"] == 1.0
    assert "submit_ranking" in captured["tool_names"]
    assert captured["completion_config"] is harness_config.SEARCHER_AGENT_CONFIG
    assert captured["force_submit"] is False
    # forced_tool_name is only meaningful when force_submit is True. Hard-budget
    # turns use the runtime verifier instead of forcing one specific tool.
    assert captured["forced_tool_name"] == "submit_ranking"
    assert any("INITIAL_SEARCH_RESULTS" in str(message) for message in captured["messages"])


def test_run_fast_agentic_search_returns_structured_loop_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        _fake_initial_metadata_facets,
    )

    captured: dict[str, Any] = {}

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        index.add_chunk(
            {
                "store_id": "store-a",
                "file_id": "file-a",
                "chunk_index": 0,
                "text": "sample launch ad",
            }
        )
        chunk_id = index.visible_chunk_ids()[0]
        captured["chunk_id"] = chunk_id
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "find the launch ads",
                "results": [{"chunk_id": chunk_id, "text": "sample launch ad"}],
            },
            {"tool": "search_corpus", "new_chunks_added": 1},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del tools, completion_config, force_submit, forced_tool_name
        assert any("INITIAL_SEARCH_RESULTS" in str(message) for message in messages)
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "structured helper",
                        "chunks": [{"chunk_id": captured["chunk_id"], "relevance_score": 1.0}],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find the launch ads",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )
    record = result.to_record()

    assert result.messages[-1]["role"] == "assistant"
    assert result.ranking is not None
    assert result.retrieval["ranked_ids"] == [result.chunks[0]["chunk_id"]]
    assert record["ranking_strategy"] == "structured helper"
    assert record["chunks"][0]["chunk_id"] == result.chunks[0]["chunk_id"]
    assert record["chunks"][0]["relevance_score"] == 1.0
    assert record["openai_responses"]["api"] == "responses"


def test_fast_search_records_text_only_turn_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        _fake_initial_metadata_facets,
    )

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        index.add_chunk(
            {
                "store_id": "store-a",
                "file_id": "file-a",
                "chunk_index": 0,
                "text": "sample launch ad",
            }
        )
        chunk_id = index.visible_chunk_ids()[0]
        captured["chunk_id"] = chunk_id
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "find the launch ads",
                "results": [{"chunk_id": chunk_id, "text": "sample launch ad"}],
            },
            {"tool": "search_corpus", "new_chunks_added": 1},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del messages, tools, completion_config, forced_tool_name
        if not force_submit:
            # A reasoning-only turn: text without any tool call ends the loop.
            return _fake_response([])
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "forced after text-only turn",
                        "chunks": [{"chunk_id": captured["chunk_id"], "relevance_score": 1.0}],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find the launch ads",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )
    record = result.to_record()

    # The wasted turn is recorded as an iteration with no calls so downstream
    # consumers can see it, and the ranking had to be forced.
    assert record["forced_ranking"] is True
    text_only_iterations = [
        iteration for iteration in record["tool_call_iterations"] if iteration.get("calls") == []
    ]
    assert len(text_only_iterations) == 1
    assert text_only_iterations[0]["iteration"] == 1


def test_fast_search_tags_provider_failures_in_tool_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        _fake_initial_metadata_facets,
    )

    captured: dict[str, Any] = {}

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        index.add_chunk(
            {
                "store_id": "store-a",
                "file_id": "file-a",
                "chunk_index": 0,
                "text": "sample launch ad",
            }
        )
        chunk_id = index.visible_chunk_ids()[0]
        captured["chunk_id"] = chunk_id
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "find the launch ads",
                "results": [{"chunk_id": chunk_id, "text": "sample launch ad"}],
            },
            {"tool": "search_corpus", "new_chunks_added": 1},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )

    async def rate_limited_search_corpus(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        request = httpx.Request("POST", "https://api.mixedbread.test/v1/stores/search")
        raise RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=request),
            body=None,
        )

    monkeypatch.setattr(searcher_runtime, "execute_search_corpus", rate_limited_search_corpus)

    searches_requested = {"count": 0}

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del messages, tools, completion_config, forced_tool_name
        if not force_submit and searches_requested["count"] == 0:
            searches_requested["count"] += 1
            return _fake_response(
                [_fake_tool_call("call_search", "search_corpus", {"query": "launch ads"})]
            )
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "after rate limit",
                        "chunks": [{"chunk_id": captured["chunk_id"], "relevance_score": 1.0}],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find the launch ads",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )
    record = result.to_record()

    error_events = [event for event in record["tool_trace"] if event.get("status") == "error"]
    assert [event.get("error_kind") for event in error_events] == ["provider"]
    provider_event = error_events[0]
    assert "rate limited" in provider_event["error"]
    assert count_provider_failures(record) == 1
    # The failure surfaced to the model as a tool message and the loop finished.
    assert result.ranking is not None


def _patch_fast_search_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed_chunks: int = 1,
) -> dict[str, Any]:
    """Stub the fast-searcher IO seams and return the seeded chunk ids."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        _fake_initial_metadata_facets,
    )

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        for chunk_index in range(seed_chunks):
            index.add_chunk(
                {
                    "store_id": "store-a",
                    "file_id": "file-a",
                    "chunk_index": chunk_index,
                    "text": f"sample chunk {chunk_index}",
                }
            )
        captured["chunk_ids"] = index.visible_chunk_ids()
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "seed",
                "results": [
                    {"chunk_id": chunk_id, "text": "sample"} for chunk_id in captured["chunk_ids"]
                ],
            },
            {"tool": "search_corpus", "new_chunks_added": seed_chunks},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )

    async def fake_execute_search_corpus(
        args: Any,
        *,
        index: Any,
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
    return captured


def test_three_search_rounds_then_a_voluntary_submit_on_the_final_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget holds three search rounds plus the submit turn, never forced."""
    captured = _patch_fast_search_io(monkeypatch)
    turns: list[int] = []

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del messages, tools, completion_config, forced_tool_name
        assert force_submit is False
        turns.append(len(turns) + 1)
        if len(turns) <= 3:
            return _fake_response(
                [
                    _fake_tool_call(
                        f"call_search_{len(turns)}",
                        "search_corpus",
                        {"query": f"angle {len(turns)}"},
                    )
                ]
            )
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "after three searches",
                        "chunks": [{"chunk_id": captured["chunk_ids"][0], "relevance_score": 1.0}],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find things",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    # Three searches then a voluntary submit fit inside the four-turn budget.
    assert harness_config.SEARCHER_MAX_ROUNDS == 4
    assert turns == [1, 2, 3, 4]
    assert result.rounds_executed == 4
    assert result.forced_ranking is False
    assert result.ranking is not None
    assert result.ranking.ranking_strategy == "after three searches"


def test_forced_path_engages_only_when_the_final_turn_does_not_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forcing happens only after the post-third-search response fails to submit."""
    captured = _patch_fast_search_io(monkeypatch)
    loop_turns: list[int] = []
    forced_after: list[int] = []

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del messages, tools, completion_config, forced_tool_name
        if force_submit:
            forced_after.append(len(loop_turns))
            return _fake_response(
                [
                    _fake_tool_call(
                        "call_forced",
                        "submit_ranking",
                        {
                            "ranking_strategy": "forced",
                            "chunks": [
                                {"chunk_id": captured["chunk_ids"][0], "relevance_score": 1.0}
                            ],
                        },
                    )
                ]
            )
        loop_turns.append(len(loop_turns) + 1)
        return _fake_response(
            [
                _fake_tool_call(
                    f"call_search_{len(loop_turns)}",
                    "search_corpus",
                    {"query": "again"},
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find things",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    # All four budget turns ran unforced; forcing engaged only after the last one.
    assert loop_turns == [1, 2, 3, 4]
    assert forced_after == [4]
    assert result.rounds_executed == 4
    assert result.forced_ranking is True
    assert result.ranking is not None


def test_final_round_notice_is_env_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_HARNESS_FINAL_ROUND_NOTICE", raising=False)
    assert round_notice_message(4, 4)["content"] == "Search round 4 of max 4."

    monkeypatch.setenv("AGENT_HARNESS_FINAL_ROUND_NOTICE", "1")
    final = round_notice_message(4, 4)["content"]
    assert final.startswith("Search round 4 of max 4.")
    assert "This is your final search round" in final
    assert "submit_ranking" in final
    # Earlier rounds never carry the notice, gated on or off.
    assert round_notice_message(2, 4)["content"] == "Search round 2 of max 4."

    monkeypatch.setenv("AGENT_HARNESS_FINAL_ROUND_NOTICE", "true")
    assert "This is your final search round" in round_notice_message(4, 4)["content"]

    monkeypatch.setenv("AGENT_HARNESS_FINAL_ROUND_NOTICE", "0")
    assert round_notice_message(4, 4)["content"] == "Search round 4 of max 4."


def test_final_round_notice_lands_only_on_the_last_loop_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_HARNESS_FINAL_ROUND_NOTICE", "1")
    captured = _patch_fast_search_io(monkeypatch)
    last_message_contents: list[str] = []

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del tools, completion_config, forced_tool_name
        assert force_submit is False
        last_message_contents.append(str(messages[-1].get("content")))
        if len(last_message_contents) <= 3:
            return _fake_response(
                [
                    _fake_tool_call(
                        f"call_search_{len(last_message_contents)}",
                        "search_corpus",
                        {"query": f"angle {len(last_message_contents)}"},
                    )
                ]
            )
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "after the notice",
                        "chunks": [{"chunk_id": captured["chunk_ids"][0], "relevance_score": 1.0}],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find things",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    # The pre-notice precedes the final generation only; earlier rounds get bare labels.
    notice_turns = [
        "This is your final search round" in content for content in last_message_contents
    ]
    assert notice_turns == [False, False, False, True]
    assert last_message_contents[3].startswith("Search round 4 of max 4.")
    assert result.forced_ranking is False


def test_a_number_in_the_query_does_not_truncate_the_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A number in the query ("just 2 feet wide") never clips the model's submission."""
    captured = _patch_fast_search_io(monkeypatch, seed_chunks=3)

    def generation_fn(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        del messages, tools, completion_config, force_submit, forced_tool_name
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "all three sculptures",
                        "chunks": [
                            {"chunk_id": chunk_id, "relevance_score": 1.0}
                            for chunk_id in captured["chunk_ids"]
                        ],
                    },
                )
            ]
        )

    result = sync_run_fast_agentic_search(
        "find sculptures just 2 feet wide",
        store_identifiers=["store-a"],
        generation_fn=generation_fn,
    )

    assert len(result.chunks) == 3
    # No guessed-count field appears in any reported payload.
    assert "requested_result_limit" not in result.retrieval
    assert "requested_result_limit" not in result.to_record()


def test_response_to_chat_completion_preserves_openai_responses_trace() -> None:
    raw_response = {
        "id": "resp_1",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "overview_search",
                "arguments": '{"query": "launch"}',
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    response = SimpleNamespace(
        id="resp_1",
        output=raw_response["output"],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        raw_response=raw_response,
    )
    request = {
        "model": "gpt-test",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "call_previous",
                "output": '{"tool": "previous"}',
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "overview_search",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "store": False,
    }

    completion = response_to_chat_completion(response, request=request)

    tool_call = completion.choices[0].message.tool_calls[0]
    assert tool_call.id == "call_1"
    assert tool_call.function.name == "overview_search"
    assert completion.raw_response["output"][0]["type"] == "reasoning"
    assert completion.responses_api["response"]["output"][0]["id"] == "rs_1"
    assert completion.responses_api["request"]["input"][0]["call_id"] == "call_previous"
    assert completion.responses_api["response"]["output"][1]["call_id"] == "call_1"


def test_prose_beside_tool_calls_is_accepted() -> None:
    """Narration next to a valid tool call is accepted, not a violation."""
    message = SimpleNamespace(
        content="Searching for the founding date first.", tool_calls=[object()]
    )
    assert (
        validate_required_tool_response(message, message.tool_calls, {"tool_choice": "required"})
        is None
    )
    nontext = SimpleNamespace(content=[{"type": "text"}], tool_calls=[object()])
    error = validate_required_tool_response(
        nontext, nontext.tool_calls, {"tool_choice": "required"}
    )
    assert error == "non-text content is not allowed in agentic tool responses"


def test_generation_fn_signals_a_failed_turn_through_the_trace() -> None:
    """The failure contract an injected generation_fn implements, end to end."""
    turns: list[dict[str, Any]] = []
    for response_id in ("resp_bad_1", "resp_bad_2"):
        response = _invalid_required_tool_response(response_id)
        message = response.choices[0].message
        validation_error = validate_required_tool_response(
            message,
            message.tool_calls,
            {"tool_choice": "required"},
        )
        assert validation_error == "required tool call missing"
        turns.extend(response_responses_api_turns(response))
    failed = failed_generation_response(response, turns, validation_error)

    assert generation_failed(failed)
    trace: list[dict[str, Any]] = []
    extend_responses_api_trace(trace, failed, phase="generation", agent="fast_searcher")
    assert [turn["response"]["id"] for turn in trace] == ["resp_bad_1", "resp_bad_2"]
    assert [turn["metadata"]["phase"] for turn in trace] == ["generation", "generation"]


def test_entry_points_require_an_injected_generation_fn() -> None:
    """Generation is injected, never bundled: fail at entry, naming the parameter."""
    with pytest.raises(ValueError, match="generation_fn is required"):
        SearcherExecutionPolicy()
    with pytest.raises(ValueError, match="generation_fn is required"):
        run_searcher("find the launch ads", store_identifiers=["store-a"])
    with pytest.raises(ValueError, match="generation_fn is required"):
        require_generation_fn(None)


async def test_forced_ranking_retry_includes_rejected_empty_response() -> None:
    responses = [
        _fake_response([]),
        _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {"ranking_strategy": "corrected", "chunks": []},
                )
            ]
        ),
    ]
    requests: list[list[dict[str, Any]]] = []

    async def generate(
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> Any:
        requests.append([dict(message) for message in messages])
        return responses.pop(0)

    messages = [{"role": "user", "content": "rank the results"}]
    forced = await llm_runtime.force_ranking(
        messages,
        tools=[],
        completion_config={},
        generation_fn=generate,
    )
    ranking = forced.submission

    assert ranking is not None
    assert [message["role"] for message in requests[1]] == [
        "user",
        "assistant",
        "user",
    ]
    assert requests[1][1] == {"role": "assistant", "content": None}
    assert requests[1][2]["content"].startswith("Your previous final submission was invalid")


async def test_forced_ranking_retry_closes_rejected_tool_call() -> None:
    responses = [
        _fake_response(
            [
                _fake_tool_call(
                    "call_invalid",
                    "submit_ranking",
                    {"ranking_strategy": "invalid", "chunks": []},
                )
            ]
        ),
        _fake_response(
            [
                _fake_tool_call(
                    "call_valid",
                    "submit_ranking",
                    {"ranking_strategy": "corrected", "chunks": []},
                )
            ]
        ),
    ]
    requests: list[list[dict[str, Any]]] = []

    async def generate(
        messages: list[dict[str, Any]],
        **_kwargs: Any,
    ) -> Any:
        requests.append([dict(message) for message in messages])
        return responses.pop(0)

    def validate(ranking: Any) -> None:
        if ranking.ranking_strategy == "invalid":
            raise ValueError("ranking rejected")

    forced = await llm_runtime.force_ranking(
        [{"role": "user", "content": "rank the results"}],
        tools=[],
        completion_config={},
        validate=validate,
        generation_fn=generate,
    )
    ranking = forced.submission

    assert ranking is not None
    assert [message["role"] for message in requests[1]] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert requests[1][2]["tool_call_id"] == "call_invalid"
    assert json.loads(requests[1][2]["content"]) == {"error": "ranking rejected"}


def test_rollout_lifts_openai_responses_sidecar() -> None:
    result = _sample_pipeline_result(agent="fast_searcher")
    result["openai_responses"] = {
        "schema_version": "openai_responses.v1",
        "api": "responses",
        "turns": [
            {
                "request": {"model": "gpt-test", "input": [], "store": False},
                "response": {"id": "resp_1", "output": []},
            }
        ],
    }

    record = build_rollout_result(
        input_text="find the launch ads",
        execution_policy="searcher",
        agent_name="searcher",
        result=result,
        runtime_s=1.0,
        query_id=None,
        query_index=None,
        store_identifiers=["store-a"],
    )

    assert record["openai"]["api"] == "responses"
    assert record["openai"]["turns"][0]["response"]["id"] == "resp_1"
    assert record["openai"]["metadata"]["execution_policy"] == "searcher"
    assert record["openai"]["metadata"]["versions"] == current_version_manifest()
    assert record["openai"]["metadata"]["agent"]["trace_counts"]["tool_calls"] == 1


def test_version_helpers_validate_current_and_legacy_rollout_records() -> None:
    expected = current_version_manifest()
    assert harness_version() == embedded_harness_version

    record = build_rollout_result(
        input_text="find the launch ads",
        execution_policy="searcher",
        agent_name="searcher",
        result=_sample_pipeline_result(agent="fast_searcher"),
        runtime_s=1.0,
        query_id=None,
        query_index=None,
        store_identifiers=["store-a"],
    )

    compatibility = assert_compatible_versions(record, expected=expected)

    assert compatibility["compatible"] is True
    assert extract_version_manifest(record) == expected

    metadata = dict(record["openai"]["metadata"])
    metadata.pop("versions")
    legacy_record = {
        "openai": {
            "schema_version": record["openai"]["schema_version"],
            "metadata": metadata,
        }
    }

    assert extract_version_manifest(legacy_record) == expected
    assert assert_compatible_versions(legacy_record, expected=expected)["compatible"] is True

    incompatible = check_version_compatibility(record, expected={"harness": "other"})
    assert incompatible["compatible"] is False
    assert incompatible["mismatched"]["harness"] == {
        "expected": "other",
        "actual": expected["harness"],
    }
    with pytest.raises(ValueError, match="mismatched versions"):
        assert_compatible_versions(record, expected={"harness": "other"})


def test_rollout_record_preserves_mixedbread_chunk_refs() -> None:
    result = _sample_pipeline_result(agent="fast_searcher")
    result["chunks"] = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "file_id": "file-456",
            "store_id": "store-789",
            "chunk_index": 3,
            "filename": "creative.png",
            "text": "sample",
        }
    ]

    record = build_rollout_result(
        input_text="find the creative",
        execution_policy="searcher",
        agent_name="searcher",
        result=result,
        runtime_s=1.0,
        query_id=None,
        query_index=None,
        store_identifiers=["store-789"],
    )

    assert record["retrieval"]["ranked_ids"] == ["c1"]
    assert record["retrieval"]["chunks"][0]["file_id"] == "file-456"
    assert record["retrieval"]["chunks"][0]["chunk_index"] == 3
    assert "ranked_sources" not in record["retrieval"]


async def test_fast_search_submit_ranking_trace_output_includes_mixedbread_refs() -> None:
    index = searcher_runtime.ChunkIndex()
    index.add_chunk(
        {
            "store_id": "store-a",
            "file_id": "file-a",
            "chunk_index": 7,
            "text": "sample",
        }
    )
    chunk_id = index.visible_chunk_ids()[0]
    tool_call = _fake_tool_call(
        "call_submit",
        "submit_ranking",
        {
            "ranking_strategy": "best match",
            "chunks": [{"chunk_id": chunk_id, "relevance_score": 0.9}],
        },
    )

    tool_round = await searcher_runtime._handle_searcher_tool_calls(
        [tool_call],
        agent_iteration=1,
        messages=[],
        index=index,
        store_identifiers=["store-a"],
        initial_metadata_facets={"type": "INITIAL_METADATA_FACETS", "metadata_fields": {}},
    )

    tool_trace = tool_round.trace
    output_chunk = tool_trace[0]["output"]["ranking"]["chunks"][0]
    assert output_chunk == {
        "chunk_id": chunk_id,
        "relevance_score": 0.9,
        "file_id": "file-a",
        "chunk_index": 7,
    }
    assert tool_trace[0]["output_summary"]["entities"]["file_ids"] == ["file-a"]
    assert tool_trace[0]["output_summary"]["entities"]["chunk_indices"] == [7]


async def test_tool_bridge_preserves_full_history_baseline_when_larger_than_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted_lengths: list[int] = []
    truncation_baselines: list[int] = []

    def fake_estimate(messages: list[dict[str, Any]]) -> int:
        counted_lengths.append(len(messages))
        return 700

    def capture_truncation(*args: Any, context_tokens_baseline: int, **kwargs: Any) -> None:
        del args, kwargs
        truncation_baselines.append(context_tokens_baseline)

    monkeypatch.setattr(searcher_runtime, "estimate_messages_tokens", fake_estimate)
    monkeypatch.setattr(
        searcher_runtime,
        "_truncate_round_tool_messages",
        capture_truncation,
    )
    messages = [{"role": "assistant", "content": "prior model turn"}]

    await searcher_runtime._handle_searcher_tool_calls(
        [_fake_tool_call("call_unknown", "unknown_tool", {})],
        agent_iteration=1,
        messages=messages,
        index=searcher_runtime.ChunkIndex(),
        store_identifiers=["store-a"],
        context_tokens_baseline=100,
    )

    assert counted_lengths == [1]
    assert truncation_baselines == [700]


async def test_tool_bridge_preserves_server_usage_baseline_when_larger_than_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counted_lengths: list[int] = []
    truncation_baselines: list[int] = []

    def fake_estimate(messages: list[dict[str, Any]]) -> int:
        counted_lengths.append(len(messages))
        return 7

    def capture_truncation(*args: Any, context_tokens_baseline: int, **kwargs: Any) -> None:
        del args, kwargs
        truncation_baselines.append(context_tokens_baseline)

    monkeypatch.setattr(searcher_runtime, "estimate_messages_tokens", fake_estimate)
    monkeypatch.setattr(
        searcher_runtime,
        "_truncate_round_tool_messages",
        capture_truncation,
    )
    messages = [{"role": "assistant", "content": "prior model turn"}]

    await searcher_runtime._handle_searcher_tool_calls(
        [_fake_tool_call("call_unknown", "unknown_tool", {})],
        agent_iteration=1,
        messages=messages,
        index=searcher_runtime.ChunkIndex(),
        store_identifiers=["store-a"],
        context_tokens_baseline=100,
    )

    assert counted_lengths == [1]
    assert truncation_baselines == [100]


async def test_tool_bridge_pruning_forces_full_history_recount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = searcher_runtime.ChunkIndex()
    index.add_chunk(
        {"store_id": "store-a", "file_id": "file-a", "chunk_index": 0, "text": "sample"}
    )
    chunk_id = index.visible_chunk_ids()[0]
    counted_lengths: list[int] = []

    def fake_estimate(messages: list[dict[str, Any]]) -> int:
        counted_lengths.append(len(messages))
        return len(messages) * 100

    monkeypatch.setattr(searcher_runtime, "estimate_messages_tokens", fake_estimate)
    messages = [{"role": "assistant", "content": "prior model turn"}]
    tool_round = await searcher_runtime._handle_searcher_tool_calls(
        [
            _fake_tool_call(
                "call_prune",
                "prune_context",
                {"chunk_ids": [chunk_id], "document_ids": []},
            )
        ],
        agent_iteration=1,
        messages=messages,
        index=index,
        store_identifiers=["store-a"],
        context_tokens_baseline=100,
    )

    assert tool_round.pruned_context is True
    assert counted_lengths == [1]


async def test_model_issued_searches_remain_independent_parallel_calls_in_prompt_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All three parties must be in flight together to pass the barrier, which
    # proves the round's searches run concurrently rather than sequentially.
    barrier = asyncio.Barrier(3)
    queries: list[str] = []

    async def fake_search(args: dict[str, Any], **kwargs: Any) -> ToolOutcome:
        del kwargs
        query = str(args["query"])
        queries.append(query)
        await asyncio.wait_for(barrier.wait(), timeout=2)
        return ToolOutcome(
            {"tool": "search_corpus", "query": query, "new_unseen_results": []},
            {"tool": "search_corpus", "query": query},
        )

    monkeypatch.setattr(searcher_runtime, "execute_search_corpus", fake_search)
    calls = [
        _fake_tool_call(f"call_{index}", "search_corpus", {"query": f"query {index}"})
        for index in range(3)
    ]
    messages: list[dict[str, Any]] = []

    await searcher_runtime._handle_searcher_tool_calls(
        calls,
        agent_iteration=1,
        messages=messages,
        index=searcher_runtime.ChunkIndex(),
        store_identifiers=["store-a"],
    )

    assert set(queries) == {"query 0", "query 1", "query 2"}
    assert [message["tool_call_id"] for message in messages] == [
        "call_0",
        "call_1",
        "call_2",
    ]
    assert [json.loads(message["content"])["query"] for message in messages] == [
        "query 0",
        "query 1",
        "query 2",
    ]


def _sample_pipeline_result(*, agent: str, monitoring: bool = False) -> dict[str, Any]:
    result = {
        "ranking": None,
        "ranking_strategy": "sample ranking",
        "chunks": [{"chunk_id": "c1", "text": "sample"}],
        "top_k": 1,
        "strict_top_k": False,
        "queries_made": [],
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "agent_token_usage": {},
        "rounds_executed": 1,
        "forced_ranking": False,
        "tool_call_iterations": [],
        "tool_trace": [
            {
                "event_type": "tool_call",
                "agent": agent,
                "iteration": 1,
                "call_id": "call_1",
                "name": "overview_search",
                "status": "success",
            }
        ],
        "openai_responses": {
            "schema_version": "openai_responses.v1",
            "api": "responses",
            "turns": [
                {
                    "request": {
                        "model": "gpt-test",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": "find launch ads",
                            }
                        ],
                        "store": False,
                    },
                    "response": {
                        "id": "resp_1",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_1",
                                "name": "overview_search",
                                "arguments": '{"query": "launch ads"}',
                            }
                        ],
                    },
                }
            ],
        },
        "id_mapping": {},
    }
    if monitoring:
        result["monitoring"] = {
            "prompt_snapshot": {
                "kind": "sample_initial_messages",
                "messages": [{"role": "system", "content": "sample"}],
            }
        }
    return result


def _invalid_required_tool_response(response_id: str) -> Any:
    return SimpleNamespace(
        id=response_id,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I should have called a tool.",
                    tool_calls=[],
                )
            )
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
        responses_api={
            "request": {"model": "gpt-test", "input": [], "store": False},
            "response": {"id": response_id, "output": []},
        },
    )


def _fake_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _fake_response(tool_calls: list[Any]) -> Any:
    message = SimpleNamespace(content=None, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )
