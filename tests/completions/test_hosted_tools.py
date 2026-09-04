"""The hosted-tools example against a scripted model."""

from __future__ import annotations

import hosted_tools
from fake_completions import ScriptedClient, response

SEARCHED = [
    {
        "type": "search_corpus_call",
        "id": "ss_1",
        "status": "completed",
        "queries": ["Hoover HW500 run time", "robot vacuum battery minutes"],
        "results": [
            {
                "filename": "document-00009.txt",
                "chunk_index": 0,
                "score": 0.61,
                "text": "HW500 ...",
            },
            {"filename": "document-00024.txt", "chunk_index": 1, "score": 0.52, "text": "X50 ..."},
        ],
    },
    {
        "type": "grep_call",
        "id": "sg_1",
        "status": "completed",
        "pattern": "HW500",
        "results": None,
    },
]
QUESTION = "How long does the Hoover run?"


def test_one_request_declares_the_hosted_tools_and_asks_for_their_results() -> None:
    client = ScriptedClient(
        [response(content="It runs for 30 minutes.", hosted_tool_calls=SEARCHED)]
    )
    hosted_tools.ask(client, QUESTION, store="catalog")
    (request,) = client.requests
    assert request["messages"] == [{"role": "user", "content": QUESTION}]
    assert request["tools"] == [
        {"type": "search_corpus", "store_identifiers": ["catalog"]},
        {"type": "grep", "store_identifiers": ["catalog"]},
    ]
    assert "tool_choice" not in request
    assert request["store"] is False
    assert request["extra_body"] == {
        "include": ["search_corpus_call.results", "grep_call.results"],
        "context_management": {"edits": [{"type": "prune_context"}]},
    }


def test_the_record_is_the_answer_with_what_the_api_ran_and_retrieved() -> None:
    client = ScriptedClient(
        [response(content="It runs for 30 minutes.", hosted_tool_calls=SEARCHED)]
    )
    record = hosted_tools.ask(client, QUESTION, store="catalog")
    assert record == {
        "answer": "It runs for 30 minutes.",
        "hosted_calls": SEARCHED,
        "context_edits": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }


def test_applied_context_edits_reach_the_record() -> None:
    client = ScriptedClient(
        [
            response(
                content="It runs for 30 minutes.",
                context_management={
                    "applied_edits": [
                        {"type": "prune_context", "calls": 2, "cleared_input_tokens": 5_400}
                    ]
                },
            )
        ]
    )
    record = hosted_tools.ask(client, QUESTION, store="catalog")
    assert record["context_edits"] == [
        {"type": "prune_context", "calls": 2, "cleared_input_tokens": 5_400}
    ]


def test_an_answer_without_hosted_calls_is_still_a_record() -> None:
    client = ScriptedClient([response(content="Thirty minutes.")])
    record = hosted_tools.ask(client, QUESTION, store="catalog")
    assert record["answer"] == "Thirty minutes."
    assert record["hosted_calls"] == []
