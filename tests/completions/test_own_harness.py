"""The own-harness example against a scripted model: corpus, tools, loop, plain-text ending."""

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


# --- the loop -----------------------------------------------------------------------


def test_a_fan_out_round_then_a_plain_text_answer() -> None:
    client = ScriptedClient(
        [
            response(
                tool_calls=[
                    search("Nordhavn Modbus TCP", call_id="call_a"),
                    grep("ET-001", call_id="call_b"),
                ],
                reasoning_content="planning",
            ),
            response(content=ANSWER),
        ]
    )
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    first, second = client.requests
    assert first["model"] == "toast-1"
    assert first["tool_choice"] == "auto"
    assert first["parallel_tool_calls"] is True
    assert first["store"] is False
    assert [tool["function"]["name"] for tool in first["tools"]] == ["bm25_search", "grep"]
    assert first["messages"] == [
        {"role": "system", "content": harness.SYSTEM_PROMPT},
        {"role": "user", "content": QUERY},
    ]
    # One tool message per call, in the model's order, then the round label.
    assert "reasoning_content" not in second["messages"][2]
    assert [message["tool_call_id"] for message in tool_messages(second)] == ["call_a", "call_b"]
    grepped = json.loads(tool_messages(second)[1]["content"])
    assert grepped["pattern"] == "ET-001"
    assert second["messages"][-1] == {"role": "user", "content": "Search round 2 of 4."}
    assert record["answer"] == ANSWER
    assert record["forced_final"] is False
    assert record["generations"] == 2
    assert record["usage"] == {"prompt_tokens": 200, "completion_tokens": 20}
    shown = {
        hit["chunk_id"]
        for message in tool_messages(second)
        for hit in json.loads(message["content"])["results"]
    }
    assert {chunk["chunk_id"] for chunk in record["seen"]} == shown
    assert set(record["seen"][0]) == {"chunk_id", "filename", "chunk_index"}


def test_the_round_limit_withholds_the_tools() -> None:
    searches = [
        response(tool_calls=[search(f"query {n}", call_id=f"call_{n}")])
        for n in range(harness.MAX_ROUNDS)
    ]
    client = ScriptedClient([*searches, response(content="The evidence is insufficient.")])
    record = harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
    final = client.requests[-1]
    assert [tool["function"]["name"] for tool in final["tools"]] == ["bm25_search", "grep"]
    assert final["tool_choice"] == "none"
    assert final["parallel_tool_calls"] is False
    labels = [
        message["content"]
        for message in final["messages"]
        if message["role"] == "user" and message["content"].startswith("Search round")
    ]
    assert labels == [f"Search round {n} of 4." for n in range(2, harness.MAX_ROUNDS + 1)]
    assert final["messages"][-1] == {"role": "user", "content": harness.ROUND_LIMIT}
    assert record["answer"] == "The evidence is insufficient."
    assert record["forced_final"] is True
    assert record["generations"] == harness.MAX_ROUNDS + 1


def test_an_empty_final_reply_raises() -> None:
    searches = [
        response(tool_calls=[search(f"query {n}", call_id=f"call_{n}")])
        for n in range(harness.MAX_ROUNDS)
    ]
    client = ScriptedClient([*searches, response(content="  ")])
    with pytest.raises(RuntimeError, match="no answer"):
        harness.run(client, QUERY, corpus=SAMPLE_CORPUS)
