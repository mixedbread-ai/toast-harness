"""Per-rollout tuning: typed overrides for env-var-configured knobs.

A multi-tenant service cannot vary process-global, read-once configuration per
request; ``HarnessTuning`` scopes the overrides to one rollout via the same
contextvar mechanism as ``media_content``, with the env vars as deployment
defaults.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
import replay_scenarios as rs

from agent_harness import HarnessTuning, run_searcher
from agent_harness import config as harness_config
from agent_harness.llm import response_message_to_dict
from agent_harness.sync_api import run_fast_agentic_search
from agent_harness.testing import ScriptedGeneration


def test_backend_top_k_override_is_context_scoped() -> None:
    default = harness_config.corpus_backend_top_k()
    with harness_config.tuning_setting(HarnessTuning(backend_top_k=17)):
        assert harness_config.corpus_backend_top_k() == 17
    assert harness_config.corpus_backend_top_k() == default


def test_backend_top_k_below_visible_limit_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="backend_top_k"):
        HarnessTuning(backend_top_k=harness_config.SEARCH_CORPUS_TOP_K - 1)


def test_agent_config_keeps_identity_without_an_override() -> None:
    assert harness_config.searcher_agent_config() is harness_config.SEARCHER_AGENT_CONFIG

    with harness_config.tuning_setting(HarnessTuning(tool_choice="required")):
        tuned = harness_config.searcher_agent_config()
        assert tuned is not harness_config.SEARCHER_AGENT_CONFIG
        assert tuned["tool_choice"] == "required"
        assert (
            tuned["require_tool_calls"]
            is harness_config.SEARCHER_AGENT_CONFIG["require_tool_calls"]
        )


def test_keep_reasoning_history_override_round_trips_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KEEP_REASONING_HISTORY", raising=False)
    message = SimpleNamespace(content="answer", reasoning_content="the thinking", tool_calls=None)

    assert "reasoning_content" not in response_message_to_dict(message)
    with harness_config.tuning_setting(HarnessTuning(keep_reasoning_history=True)):
        assert response_message_to_dict(message)["reasoning_content"] == "the thinking"
    assert "reasoning_content" not in response_message_to_dict(message)


def test_tuning_reaches_the_rollout_through_the_entry_point() -> None:
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    seen_tool_choices: list[Any] = []
    scripted = scenario.generation()

    def generation_fn(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        seen_tool_choices.append(kwargs["completion_config"]["tool_choice"])
        return scripted(messages, **kwargs)

    result = run_fast_agentic_search(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[rs.STORE_ID],
        client=scenario.client(),
        generation_fn=generation_fn,
        tuning=HarnessTuning(backend_top_k=17, tool_choice="required"),
    )

    assert seen_tool_choices == ["required"] * 3
    # The bootstrap and agent searches both fetched at the overridden depth.
    search_depths = {
        query["search_top_k"] for query in result.queries_made if "search_top_k" in query
    }
    assert search_depths == {17}
    # The override never leaks past the rollout.
    assert harness_config.searcher_agent_config() is harness_config.SEARCHER_AGENT_CONFIG


def test_tuning_reaches_the_rollout_through_the_sync_policy_entry_point() -> None:
    """The same knobs through ``agent_harness.run_searcher`` -- the sync
    rollout-record entry point."""
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    seen_tool_choices: list[Any] = []
    scripted = scenario.generation()

    def generation_fn(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        seen_tool_choices.append(kwargs["completion_config"]["tool_choice"])
        return scripted(messages, **kwargs)

    record = run_searcher(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[rs.STORE_ID],
        client=scenario.client(),
        generation_fn=generation_fn,
        tuning=HarnessTuning(backend_top_k=17, tool_choice="required"),
    )

    assert seen_tool_choices == ["required"] * 3
    # The bootstrap and agent searches both fetched at the overridden depth.
    queries_made = record["openai"]["metadata"]["agent"]["queries_made"]
    search_depths = {query["search_top_k"] for query in queries_made if "search_top_k" in query}
    assert search_depths == {17}
    # The override never leaks past the rollout.
    assert harness_config.searcher_agent_config() is harness_config.SEARCHER_AGENT_CONFIG


def test_searcher_max_rounds_override_is_context_scoped() -> None:
    assert harness_config.searcher_max_rounds() == harness_config.SEARCHER_MAX_ROUNDS
    with harness_config.tuning_setting(HarnessTuning(searcher_max_rounds=2)):
        assert harness_config.searcher_max_rounds() == 2
    assert harness_config.searcher_max_rounds() == harness_config.SEARCHER_MAX_ROUNDS


def test_searcher_max_rounds_below_one_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="searcher_max_rounds"):
        HarnessTuning(searcher_max_rounds=0)


def test_searcher_max_rounds_bounds_the_rollout() -> None:
    """A script that would search on gets cut at one round and forced to
    submit; the round budget also lands in the system prompt."""
    scripted = ScriptedGeneration(
        [
            rs._search_turn("resp-1", "governing law", input_tokens=120),
            rs._submit_turn("resp-2", strategy="forced by round budget", input_tokens=140),
        ]
    )
    force_submit_flags: list[bool] = []

    def generation_fn(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        force_submit_flags.append(kwargs["force_submit"])
        return scripted(messages, **kwargs)

    result = run_fast_agentic_search(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[rs.STORE_ID],
        client=rs.ScriptedRetrievalClient(),
        generation_fn=generation_fn,
        include_prompt_snapshot=True,
        tuning=HarnessTuning(searcher_max_rounds=1),
    )

    assert result.rounds_executed == 1
    assert result.forced_ranking is True
    assert force_submit_flags == [False, True]
    system_prompt = result.prompt_snapshot["messages"][0]["content"]
    assert "at most 1 rounds" in system_prompt


def test_as_of_survives_the_searcher_tuning_reentry() -> None:
    """tuning= and as_of= together must both reach the rollout.

    The tuning override re-enters the entry point, and a re-entry that forgets
    to forward as_of fails SILENTLY: prompts._runtime_context falls back to the
    UTC wall clock, so the model resolves relative dates against today instead
    of the pinned date. Assert on the model-visible system prompt.
    """
    scenario = rs.SCENARIOS_BY_NAME["submit_after_two_rounds"]
    seen_system_prompts: list[str] = []
    scripted = scenario.generation()

    def generation_fn(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        seen_system_prompts.append(messages[0]["content"])
        return scripted(messages, **kwargs)

    run_fast_agentic_search(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[rs.STORE_ID],
        client=scenario.client(),
        generation_fn=generation_fn,
        tuning=HarnessTuning(backend_top_k=17),
        as_of=date(1999, 12, 31),
    )

    assert seen_system_prompts
    assert all("Current UTC date: 1999-12-31." in prompt for prompt in seen_system_prompts)
