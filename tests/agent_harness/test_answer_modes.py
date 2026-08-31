"""The three searcher answer protocols.

``answer_mode`` picks how (and whether) the fast searcher answers:
``"none"`` is the default submit_ranking episode and must stay byte-identical
to the pre-answer-mode harness; ``"submit_ranking"`` adds a required
``answer`` argument to submit_ranking; ``"plain_text"`` drops submit_ranking
entirely and ends the episode on a plain-text turn.
"""

from __future__ import annotations

import inspect
import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
import replay_scenarios as rs

import agent_harness
from agent_harness import HarnessTuning, aio, sync_api
from agent_harness import config as harness_config
from agent_harness.agents.searcher import _searcher_tools
from agent_harness.llm import parse_ranking
from agent_harness.prompts import force_answer_message, force_submit_message
from agent_harness.searcher_prompts import (
    build_fast_searcher_task_description,
    build_searcher_system_prompt,
    fast_searcher_messages,
)
from agent_harness.testing import response, tool_call

QUERY = "Which contract governs the 2019 Acme distribution agreement?"
AS_OF = date(2026, 1, 1)
ANSWER_TEXT = "The 2019 Acme distribution agreement is governed by New York law."


def _submit_tool(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return next(tool for tool in tools if tool["function"]["name"] == "submit_ranking")


def _run(
    script: list[Any],
    *,
    answer_mode: str,
    top_k: int | None = None,
    strict_top_k: bool = False,
) -> dict[str, Any]:
    return agent_harness.fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        top_k=top_k,
        strict_top_k=strict_top_k,
        client=rs.ScriptedRetrievalClient(),
        generation_fn=rs.ScriptedGeneration(script),
        answer_mode=answer_mode,
    )


def _search_turn(response_id: str) -> Any:
    def build(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [tool_call(f"call-{response_id}", "search_corpus", {"query": "governing law"})],
            response_id=response_id,
        )

    return build


def _submit_turn(response_id: str, *, answer: str | None) -> Any:
    arguments: dict[str, Any] = rs.ranking_arguments(
        list(rs.SEEDED_CHUNK_IDS), strategy="relevance to the governing-law question"
    )
    if answer is not None:
        arguments["answer"] = answer

    def build(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [tool_call(f"call-{response_id}", "submit_ranking", arguments)],
            response_id=response_id,
        )

    return build


def _text_turn(response_id: str, *, content: str) -> Any:
    def build(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response([], response_id=response_id, content=content)

    return build


def _exhausting_search_turns() -> list[Any]:
    """Enough search turns to spend the round budget without a final submission."""
    return [_search_turn(f"resp-{index}") for index in range(harness_config.SEARCHER_MAX_ROUNDS)]


def _trace_events(record: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [event for event in record["tool_trace"] if event.get("name") == name]


# --- answer_mode="none": the default protocol is untouched -------------------


def test_default_mode_prompts_and_tools_are_unchanged() -> None:
    explicit = fast_searcher_messages(user_text=QUERY, as_of=AS_OF, answer_mode="none")
    assert fast_searcher_messages(user_text=QUERY, as_of=AS_OF) == explicit

    tool = _submit_tool(_searcher_tools())
    assert "answer" not in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["required"] == ["ranking_strategy", "chunks"]

    prompt = explicit[0]["content"]
    assert "submit_ranking.answer" not in prompt
    assert "USE TOOLS ONLY" in prompt


def test_default_mode_record_carries_null_answer() -> None:
    record = _run(
        [_search_turn("resp-1"), _submit_turn("resp-2", answer=None)],
        answer_mode="none",
    )
    assert record["answer"] is None
    assert record["answer_mode"] == "none"
    assert record["ranking"] is not None


def test_unknown_answer_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown answer_mode"):
        _run([], answer_mode="prose")


@pytest.mark.parametrize(
    "surface",
    [
        agent_harness.fast_agentic_search,
        agent_harness.run_searcher,
        agent_harness.SearcherExecutionPolicy.run,
        sync_api.run_fast_agentic_search,
        aio.run_fast_agentic_search,
        aio.stream_fast_agentic_search,
    ],
)
def test_every_public_surface_takes_answer_mode(surface: Any) -> None:
    parameter = inspect.signature(surface).parameters["answer_mode"]
    assert parameter.default == "none"


# --- answer_mode="submit_ranking": required answer argument ------------------


def test_submit_mode_schema_requires_answer() -> None:
    tool = _submit_tool(_searcher_tools(answer_mode="submit_ranking"))
    parameters = tool["function"]["parameters"]
    assert parameters["properties"]["answer"]["type"] == "string"
    assert parameters["required"] == ["ranking_strategy", "chunks", "answer"]

    strict = _submit_tool(_searcher_tools(top_k=2, strict_top_k=True, answer_mode="submit_ranking"))
    assert "answer" in strict["function"]["parameters"]["required"]
    assert "exactly 2 chunks" in strict["function"]["description"]


def test_submit_mode_prompt_and_force_message_mention_answer() -> None:
    prompt = build_searcher_system_prompt(
        task_description=build_fast_searcher_task_description(answer_mode="submit_ranking"),
        answer_mode="submit_ranking",
    )
    assert "submit_ranking.answer must give your final answer" in prompt
    assert "with your answer when you have enough evidence" in prompt

    forced = force_submit_message(require_answer=True)["content"]
    assert "your final answer in the answer field" in forced
    unforced = force_submit_message()["content"]
    assert "answer" not in unforced


def test_parse_ranking_enforces_required_answer() -> None:
    arguments = rs.ranking_arguments(["c1"], strategy="test")
    with pytest.raises(ValueError, match="missing answer"):
        parse_ranking(json.dumps(arguments), require_answer=True)
    arguments["answer"] = ANSWER_TEXT
    parsed = parse_ranking(json.dumps(arguments), require_answer=True)
    assert parsed.answer == ANSWER_TEXT


def test_submit_mode_records_the_submitted_answer() -> None:
    record = _run(
        [_search_turn("resp-1"), _submit_turn("resp-2", answer=ANSWER_TEXT)],
        answer_mode="submit_ranking",
    )
    assert record["answer"] == ANSWER_TEXT
    assert record["answer_mode"] == "submit_ranking"
    assert record["forced_ranking"] is False
    assert [chunk["chunk_id"] for chunk in record["chunks"]] == list(rs.SEEDED_CHUNK_IDS)


def test_submit_mode_missing_answer_gets_a_correction_round() -> None:
    record = _run(
        [
            _search_turn("resp-1"),
            _submit_turn("resp-2", answer=None),
            _submit_turn("resp-3", answer=ANSWER_TEXT),
        ],
        answer_mode="submit_ranking",
    )
    assert record["answer"] == ANSWER_TEXT
    errors = [
        event for event in _trace_events(record, "submit_ranking") if event.get("status") == "error"
    ]
    assert any("missing answer" in str(event.get("error")) for event in errors)


def test_submit_mode_forced_submission_carries_the_answer() -> None:
    script = [*_exhausting_search_turns(), _submit_turn("resp-forced", answer=ANSWER_TEXT)]
    record = _run(script, answer_mode="submit_ranking")
    assert record["forced_ranking"] is True
    assert record["answer"] == ANSWER_TEXT


# --- answer_mode="plain_text": no ranking, the answer is the final turn ------


def test_plain_mode_advertises_no_submit_tool() -> None:
    tools = _searcher_tools(answer_mode="plain_text")
    assert all(tool["function"]["name"] != "submit_ranking" for tool in tools)


def test_plain_mode_prompt_swaps_ranking_for_answer_rules() -> None:
    prompt = build_searcher_system_prompt(
        task_description=build_fast_searcher_task_description(answer_mode="plain_text"),
        answer_mode="plain_text",
    )
    assert "RANKING:" not in prompt
    assert "ANSWER:" in prompt
    assert "plain-text message with NO tool" in prompt
    assert "the final answer turn is one of them" in prompt
    assert "reply with your final answer immediately" in prompt
    assert "submit_ranking" not in prompt

    forced = force_answer_message()["content"]
    assert "final answer to the user query as plain text" in forced


def test_plain_mode_rejects_retrieval_shape() -> None:
    with pytest.raises(ValueError, match="plain_text"):
        _run([], answer_mode="plain_text", top_k=5)
    with pytest.raises(ValueError, match="plain_text"):
        _run([], answer_mode="plain_text", strict_top_k=True)


def test_plain_mode_voluntary_answer_ends_the_episode() -> None:
    record = _run(
        [_search_turn("resp-1"), _text_turn("resp-2", content=ANSWER_TEXT)],
        answer_mode="plain_text",
    )
    assert record["answer"] == ANSWER_TEXT
    assert record["answer_mode"] == "plain_text"
    assert record["forced_ranking"] is False
    assert record["ranking"] is None
    assert record["chunks"] == []
    assert record["rounds_executed"] == 2


def test_plain_mode_hallucinated_submit_gets_a_tool_error() -> None:
    record = _run(
        [
            _submit_turn("resp-1", answer=None),
            _text_turn("resp-2", content=ANSWER_TEXT),
        ],
        answer_mode="plain_text",
    )
    assert record["answer"] == ANSWER_TEXT
    errors = [
        event for event in _trace_events(record, "submit_ranking") if event.get("status") == "error"
    ]
    assert any("not available" in str(event.get("error")) for event in errors)


def test_plain_mode_forces_an_answer_after_the_budget() -> None:
    script: list[Any] = [
        *_exhausting_search_turns(),
        _search_turn("resp-invalid-forced"),
        _text_turn("resp-forced", content=ANSWER_TEXT),
    ]
    record = _run(script, answer_mode="plain_text")
    assert record["forced_ranking"] is True
    assert record["answer"] == ANSWER_TEXT
    forced_events = _trace_events(record, "final_answer")
    assert any(event.get("status") == "error" for event in forced_events)
    assert any(event.get("status") == "success" for event in forced_events)


def test_plain_mode_stops_requiring_tool_calls() -> None:
    """The turn plain text mode waits for is the one require_tool_calls corrects."""
    configs: list[dict[str, Any]] = []

    def capture(script: list[Any], **kwargs: Any) -> dict[str, Any]:
        scripted = rs.ScriptedGeneration(script)

        def generation_fn(messages: list[dict[str, Any]], **call: Any) -> Any:
            configs.append(call["completion_config"])
            return scripted(messages, **call)

        return agent_harness.fast_agentic_search(
            QUERY,
            store_identifiers=[rs.STORE_ID],
            client=rs.ScriptedRetrievalClient(),
            generation_fn=generation_fn,
            **kwargs,
        )

    capture(
        [_search_turn("resp-1"), _text_turn("resp-2", content=ANSWER_TEXT)],
        answer_mode="plain_text",
    )
    assert all(config["require_tool_calls"] is False for config in configs)

    configs.clear()
    capture([_search_turn("resp-1"), _submit_turn("resp-2", answer=None)], answer_mode="none")
    assert all(config["require_tool_calls"] is True for config in configs)


def test_plain_mode_neutralizes_a_wire_required_tool_choice() -> None:
    """A tool-free final answer must remain possible under per-run tuning."""
    configs: list[dict[str, Any]] = []
    scripted = rs.ScriptedGeneration([_text_turn("resp-1", content=ANSWER_TEXT)])

    def generation_fn(messages: list[dict[str, Any]], **call: Any) -> Any:
        configs.append(call["completion_config"])
        return scripted(messages, **call)

    record = agent_harness.fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=rs.ScriptedRetrievalClient(),
        generation_fn=generation_fn,
        answer_mode="plain_text",
        tuning=HarnessTuning(tool_choice="required"),
    )

    assert record["answer"] == ANSWER_TEXT
    assert configs
    assert all(config["require_tool_calls"] is False for config in configs)
    assert all(config["tool_choice"] == "auto" for config in configs)


def test_forced_default_mode_does_not_leak_config_into_plain_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-side force-submit mutation stays local when modes share a process."""
    isolated_config = dict(harness_config.SEARCHER_AGENT_CONFIG)
    monkeypatch.setattr(harness_config, "SEARCHER_AGENT_CONFIG", isolated_config)
    expected_config = dict(isolated_config)
    scripted = rs.ScriptedGeneration(
        [*_exhausting_search_turns(), _submit_turn("resp-forced", answer=None)]
    )

    def force_mutating_generation(messages: list[dict[str, Any]], **call: Any) -> Any:
        if call["force_submit"]:
            agent_harness.apply_force_submit(call["completion_config"], call["forced_tool_name"])
        return scripted(messages, **call)

    default_record = agent_harness.fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=rs.ScriptedRetrievalClient(),
        generation_fn=force_mutating_generation,
        answer_mode="none",
    )

    assert default_record["forced_ranking"] is True
    assert isolated_config == expected_config

    plain_configs: list[dict[str, Any]] = []
    plain_script = rs.ScriptedGeneration([_text_turn("resp-plain", content=ANSWER_TEXT)])

    def plain_generation(messages: list[dict[str, Any]], **call: Any) -> Any:
        plain_configs.append(call["completion_config"])
        return plain_script(messages, **call)

    plain_record = agent_harness.fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=rs.ScriptedRetrievalClient(),
        generation_fn=plain_generation,
        answer_mode="plain_text",
    )

    assert plain_record["answer"] == ANSWER_TEXT
    assert all(config["parallel_tool_calls"] is True for config in plain_configs)


# --- the rollout record -------------------------------------------------------


def test_rollout_record_carries_the_answer() -> None:
    record = agent_harness.run_searcher(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=rs.ScriptedRetrievalClient(),
        generation_fn=rs.ScriptedGeneration(
            [_search_turn("resp-1"), _text_turn("resp-2", content=ANSWER_TEXT)]
        ),
        answer_mode="plain_text",
    )
    assert record["answer"] == ANSWER_TEXT
    assert record["answer_mode"] == "plain_text"
    assert record["retrieval"]["answer"] == ANSWER_TEXT
    assert record["retrieval"]["ranked_ids"] == []
    assert record["openai"]["metadata"]["agent"]["answer_mode"] == "plain_text"


def test_rollout_record_omits_an_answer_never_asked_for() -> None:
    record = agent_harness.run_searcher(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        client=rs.ScriptedRetrievalClient(),
        generation_fn=rs.ScriptedGeneration(
            [_search_turn("resp-1"), _submit_turn("resp-2", answer=None)]
        ),
    )
    assert "answer" not in record
    assert "answer" not in record["retrieval"]
    assert record["answer_mode"] == "none"
    assert record["retrieval"]["ranked_ids"] == list(rs.SEEDED_CHUNK_IDS)
