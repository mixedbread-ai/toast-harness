"""The generation seam that puts toast-harness's loop on the Completions API."""

from __future__ import annotations

import os
import sys

import agent_harness_on_api as example
import pytest
from fake_completions import ScriptedClient, response

TOOLS = [{"type": "function", "function": {"name": "search_corpus", "parameters": {}}}]
CONFIG = {"tool_choice": "auto", "parallel_tool_calls": True, "require_tool_calls": True}


def test_generation_fn_declares_only_harness_tools_and_names_the_forced_terminal() -> None:
    client = ScriptedClient([response(content="a"), response(content="b")])
    generate = example.build_generation_fn(client)
    messages = [{"role": "user", "content": "q"}]

    generate(messages, tools=TOOLS, completion_config=CONFIG)
    generate(messages, tools=TOOLS, completion_config=CONFIG, force_submit=True)

    free, forced = client.requests
    assert free["model"] == "toast-1"
    assert free["tools"] == TOOLS
    assert free["tool_choice"] == "auto"
    assert free["parallel_tool_calls"] is True
    assert "max_completion_tokens" not in free  # the example leaves the API default in place
    assert free["store"] is False
    assert forced["tool_choice"] == {"type": "function", "function": {"name": "submit_ranking"}}
    assert forced["parallel_tool_calls"] is False
    assert CONFIG == {
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "require_tool_calls": True,
    }


def _stub_rollout(*args: object, **kwargs: object) -> dict[str, object]:
    del args, kwargs
    return {
        "answer": None,
        "retrieval": {"ranked_ids": []},
        "openai": {"metadata": {"agent": {"rounds_executed": 0, "total_tokens": 0}}},
    }


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, "estimate"), ("/models/policy", "/models/policy")],
    ids=["unset", "configured"],
)
def test_main_budgets_on_the_estimate_unless_a_tokenizer_is_configured(
    monkeypatch: pytest.MonkeyPatch, configured: str | None, expected: str
) -> None:
    """The hosted model's tokenizer is not local: the script asks the harness for its
    estimate outright, so no checkpoint is looked up and nothing is warned; a
    configured tokenizer is left in charge."""
    monkeypatch.delenv("AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER", raising=False)
    if configured is None:
        monkeypatch.delenv("AGENT_HARNESS_TOKENIZER", raising=False)
    else:
        monkeypatch.setenv("AGENT_HARNESS_TOKENIZER", configured)
    monkeypatch.setattr(example.agent_harness, "run_searcher", _stub_rollout)
    monkeypatch.setattr(
        sys, "argv", ["agent_harness_on_api.py", "--store", "s", "--api-key", "k", "q"]
    )

    assert example.main() == 0

    assert os.environ["AGENT_HARNESS_TOKENIZER"] == expected
    assert "AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER" not in os.environ
