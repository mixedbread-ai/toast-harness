"""The generation seam that puts toast-harness's loop on the Completions API."""

from __future__ import annotations

import agent_harness_on_api as example
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
