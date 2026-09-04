"""The own-harness example against a scripted model: corpus, tools, loop, submitted ending."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import own_harness as harness
import pytest
from fake_completions import ScriptedClient, response, tool_call

SAMPLE_CORPUS = Path(__file__).resolve().parents[2] / "completions/sample_corpus"
QUERY = "Which firmware does the Nordhavn fleet need for Modbus TCP, and what does the module cost?"
ANSWER = "Firmware 3.2 or later with the ETH-1 module (ET-001), EUR 190."


def search(query: str, *, call_id: str) -> Any:
    return tool_call("bm25_search", {"query": query}, call_id=call_id)


def grep(pattern: str, *, call_id: str) -> Any:
    return tool_call("grep", {"pattern": pattern}, call_id=call_id)


def tool_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [message for message in request["messages"] if message["role"] == "tool"]


@pytest.fixture
def chunks() -> list[harness.Chunk]:
    return harness.load_corpus(SAMPLE_CORPUS)


# --- corpus and tools ---------------------------------------------------------------


def test_handles_are_minted_once_in_file_order(chunks: list[harness.Chunk]) -> None:
    assert [chunk.chunk_id for chunk in chunks] == [f"c{n}" for n in range(1, len(chunks) + 1)]
    assert harness.load_corpus(SAMPLE_CORPUS) == chunks
    overview = [chunk.chunk_index for chunk in chunks if chunk.filename == "company-overview.md"]
    assert overview == [0, 1, 2]


def test_a_heading_joins_the_paragraph_below_it(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Title\n\n## Section\n\nBody one.\n\nBody two.\n")
    (tmp_path / "ignored.py").write_text("print()\n")
    texts = [chunk.text for chunk in harness.load_corpus(tmp_path)]
    assert texts == ["# Title\n## Section\nBody one.", "Body two."]


def test_bm25_ranks_the_keyword_chunk_first_and_skips_seen(chunks: list[harness.Chunk]) -> None:
    tools = harness.Tools(chunks)
    first = tools.bm25_search("Nordhavn drift firmware", top_k=2)
    assert first["candidate_count"] == 2
    assert first["results"][0]["filename"] == "field-incident-2025-03.md"
    assert set(first["results"][0]) == {"chunk_id", "filename", "chunk_index", "text", "score"}
    again = tools.bm25_search("Nordhavn drift firmware", top_k=2)
    shown = {hit["chunk_id"] for hit in first["results"]}
    assert not shown & {hit["chunk_id"] for hit in again["results"]}
    assert set(tools.seen) == shown | {hit["chunk_id"] for hit in again["results"]}


def test_bm25_documents_mode_keeps_one_chunk_per_document(chunks: list[harness.Chunk]) -> None:
    hits = harness.Tools(chunks).bm25_search("firmware", top_k=20, mode="documents")["results"]
    filenames = [hit["filename"] for hit in hits]
    assert len(filenames) == len(set(filenames)) > 1


def test_grep_counts_matches_and_orders_by_them(chunks: list[harness.Chunk]) -> None:
    tools = harness.Tools(chunks)
    hits = tools.grep(r"KS-400-[AH]")
    counts = [hit["match_count"] for hit in hits["results"]]
    assert hits["pattern"] == r"KS-400-[AH]"
    assert counts == sorted(counts, reverse=True)
    assert counts[0] >= 2
    assert set(tools.seen) == {hit["chunk_id"] for hit in hits["results"]}


def test_grep_reports_chunks_bm25_already_showed(chunks: list[harness.Chunk]) -> None:
    tools = harness.Tools(chunks)
    top = tools.grep("ET-001", top_k=3)["results"][0]["chunk_id"]
    assert top in tools.seen
    assert tools.grep("ET-001", top_k=3)["results"][0]["chunk_id"] == top  # seen, still reported


@pytest.mark.parametrize(
    ("tool", "arguments", "error"),
    [
        ("bm25_search", {"query": "x", "top_k": 0}, "top_k must be"),
        ("bm25_search", {"query": "x", "mode": "pages"}, "mode must be"),
        ("grep", {"pattern": "("}, "invalid regular expression"),
        ("grep", {"pattern": "x", "extra": 1}, "unexpected keyword"),
        ("submit_answer", {"answer": "EUR 190."}, "chain_of_thought"),
        ("read_document", {}, "unknown tool"),
    ],
)
def test_tool_failures_come_back_as_data(
    chunks: list[harness.Chunk], tool: str, arguments: dict[str, Any], error: str
) -> None:
    result = harness._execute(harness.Tools(chunks), tool_call(tool, arguments))
    assert error in result["error"]


def test_schemas_come_from_docstrings_and_annotated_hints(chunks: list[harness.Chunk]) -> None:
    schema = harness.tool_schema(harness.Tools(chunks).bm25_search)["function"]
    assert schema["name"] == "bm25_search"
    assert schema["description"].startswith("Keyword-based BM25 search over the corpus.")
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["top_k"] == {
        "type": "integer",
        "description": "Number of chunks to return, max 20.",
        "default": 5,
    }
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["chunks", "documents"]
    submit = harness.tool_schema(harness.Tools(chunks).submit_answer)["function"]
    assert submit["parameters"]["required"] == ["chain_of_thought", "answer"]


# --- the loop -----------------------------------------------------------------------


def submit(answer: str = ANSWER, *, call_id: str = "call_submit") -> Any:
    arguments = {"chain_of_thought": "The procurement FAQ prices it.", "answer": answer}
    return tool_call("submit_answer", arguments, call_id=call_id)


def test_a_fan_out_round_then_a_submitted_answer() -> None:
    client = ScriptedClient(
        [
            response(
                tool_calls=[
                    search("Nordhavn Modbus TCP", call_id="call_a"),
                    grep("ET-001", call_id="call_b"),
                ],
                reasoning_content="planning",
                completion_id="cmpl_a",
            ),
            response(tool_calls=[submit()], completion_id="cmpl_b"),
        ]
    )
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    first, second = client.requests
    assert first["model"] == "toast-1"
    assert first["tool_choice"] == "required"  # every round calls a tool
    assert first["parallel_tool_calls"] is True
    assert "store" not in first  # stored by default: the next request continues it
    names = [tool["function"]["name"] for tool in first["tools"]]
    assert names == ["bm25_search", "grep", "submit_answer"]
    assert first["messages"] == [
        {"role": "system", "content": harness.SYSTEM_PROMPT},
        {"role": "user", "content": QUERY},
    ]
    assert first["extra_body"] == {"context_management": harness.CONTEXT_MANAGEMENT}
    # The continuation sends only the new messages: one tool message per call,
    # in the model's order, then the round label.
    assert second["extra_body"] == {
        "context_management": harness.CONTEXT_MANAGEMENT,
        "previous_completion_id": "cmpl_a",
    }
    assert [message["role"] for message in second["messages"]] == ["tool", "tool", "user"]
    assert [message["tool_call_id"] for message in tool_messages(second)] == ["call_a", "call_b"]
    grepped = json.loads(tool_messages(second)[1]["content"])
    assert grepped["pattern"] == "ET-001"
    assert second["messages"][-1] == {"role": "user", "content": "Search round 2 of 6."}
    assert record["answer"] == ANSWER
    assert record["chain_of_thought"] == "The procurement FAQ prices it."
    assert record["rounds"] == 2
    assert record["generations"] == 2
    assert record["completion_id"] == "cmpl_b"
    assert record["usage"] == {"prompt_tokens": 200, "completion_tokens": 20}
    # The local transcript still stitches the whole conversation for the record,
    # ending on the submission.
    assert record["messages"][2]["role"] == "assistant"
    assert "reasoning_content" not in record["messages"][2]
    assert record["messages"][-1]["tool_calls"][0]["function"]["name"] == "submit_answer"
    shown = {
        hit["chunk_id"]
        for message in tool_messages(second)
        for hit in json.loads(message["content"])["results"]
    }
    assert {chunk["chunk_id"] for chunk in record["seen"]} == shown
    assert set(record["seen"][0]) == {"chunk_id", "filename", "chunk_index"}


def test_the_last_round_asks_for_the_submission_by_name() -> None:
    searches = [
        response(tool_calls=[search(f"query {n}", call_id=f"call_{n}")], completion_id=f"cmpl_{n}")
        for n in range(harness.MAX_ROUNDS - 1)
    ]
    insufficient = "The evidence is insufficient."
    client = ScriptedClient([*searches, response(tool_calls=[submit(insufficient)])])
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    # Every request after the first continues the completion before it.
    chained = [request["extra_body"].get("previous_completion_id") for request in client.requests]
    assert chained == [None, *(f"cmpl_{n}" for n in range(harness.MAX_ROUNDS - 1))]
    labels = [request["messages"][-1]["content"] for request in client.requests[1:-1]]
    assert labels == [f"Search round {n} of 6." for n in range(2, harness.MAX_ROUNDS)]
    last = client.requests[-1]
    names = [tool["function"]["name"] for tool in last["tools"]]
    assert names == ["bm25_search", "grep", "submit_answer"]
    assert last["tool_choice"] == harness.SUBMIT_ONLY
    assert last["parallel_tool_calls"] is False
    assert [message["role"] for message in last["messages"]] == ["tool", "user"]
    assert last["messages"][-1] == {"role": "user", "content": harness.LAST_ROUND}
    assert record["answer"] == insufficient
    assert record["rounds"] == harness.MAX_ROUNDS
    assert record["generations"] == harness.MAX_ROUNDS


def test_a_turn_without_tool_calls_goes_on_to_the_next_round() -> None:
    """Should the model answer in prose anyway, the next round asks it again."""
    client = ScriptedClient(
        [
            response(content="EUR 190, I believe.", completion_id="cmpl_a"),
            response(tool_calls=[submit()]),
        ]
    )
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    second = client.requests[1]
    assert second["extra_body"]["previous_completion_id"] == "cmpl_a"
    assert second["messages"] == [{"role": "user", "content": "Search round 2 of 6."}]
    assert second["tool_choice"] == "required"
    assert record["answer"] == ANSWER
    assert record["rounds"] == 2


def test_a_failed_submission_goes_back_as_data() -> None:
    incomplete = tool_call("submit_answer", {"answer": ANSWER}, call_id="call_bad")
    client = ScriptedClient([response(tool_calls=[incomplete]), response(tool_calls=[submit()])])
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    (message,) = tool_messages(client.requests[1])
    assert message["tool_call_id"] == "call_bad"
    assert "chain_of_thought" in json.loads(message["content"])["error"]
    assert record["answer"] == ANSWER
    assert record["rounds"] == 2


def test_applied_context_edits_aggregate_across_the_chain() -> None:
    pruned = {"applied_edits": [{"type": "prune_context", "calls": 1, "cleared_input_tokens": 900}]}
    client = ScriptedClient(
        [
            response(tool_calls=[search("Nordhavn", call_id="call_a")], context_management=pruned),
            response(tool_calls=[submit()], context_management=pruned),
        ]
    )
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    assert record["context_edits"] == pruned["applied_edits"] * 2


def test_a_last_round_without_a_submission_raises() -> None:
    searches = [
        response(tool_calls=[search(f"query {n}", call_id=f"call_{n}")])
        for n in range(harness.MAX_ROUNDS)
    ]
    client = ScriptedClient(searches)
    with pytest.raises(RuntimeError, match="submit_answer"):
        harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
