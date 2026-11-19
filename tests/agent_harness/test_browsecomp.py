"""Exercise the example from CLI arguments to its written rollout and metrics."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import mixedbread
import pytest
import replay_scenarios as rs

from agent_harness.testing import response, tool_call
from agent_harness.tools import functions

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "browsecomp"


def test_reranked_batch_keeps_submission_and_reports_failed_queries(tmp_path, monkeypatch):
    labels = ["relevant"]
    queries = [
        {"query_id": key, "query": key, "relevant_ids": labels} for key in ("failed", "completed")
    ]
    query_file = tmp_path / "queries.json"
    query_file.write_text(json.dumps(queries))
    monkeypatch.setattr(
        sys, "argv", ["run.py", str(tmp_path), str(query_file), "--rerank-model", "test-reranker"]
    )
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.setenv("TOAST_ANSWER_MODE", "none")
    monkeypatch.setenv("AGENT_HARNESS_TOKENIZER", "estimate")

    def complete(**request):
        if any(m.get("content") == "USER_QUERY:\nfailed" for m in request["messages"]):
            raise RuntimeError("model offline")
        return response(
            [
                tool_call(
                    "submit", "submit_ranking", rs.ranking_arguments(["c1"], strategy="evidence")
                )
            ],
            response_id="response",
        )

    completions = SimpleNamespace(create=complete)
    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda **_: SimpleNamespace(chat=SimpleNamespace(completions=completions))
        ),
    )

    stores = rs.ScriptedRetrievalClient()
    chunks = [
        {**rs.corpus_chunk(i), "external_id": doc} for i, doc in enumerate(["other", "relevant"])
    ]
    monkeypatch.setattr(stores.stores, "search", lambda **_: rs.ScriptedSearchResults(chunks))
    monkeypatch.setattr(functions, "get_mixedbread_client", lambda **_: stores)

    requests = []

    def handle_rerank(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "test-reranker",
                "top_k": 2,
                "return_input": False,
                "data": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}],
                "usage": {"prompt_tokens": 42, "total_tokens": 42},
            },
        )

    sdk = mixedbread.AsyncMixedbread
    monkeypatch.setattr(
        mixedbread,
        "AsyncMixedbread",
        lambda **kw: sdk(
            **kw, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle_rerank))
        ),
    )
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(EXAMPLE / "run.py"), run_name="__main__")
    assert exit_info.value.code == 1
    failed, completed = json.loads((tmp_path / "client_summary.json").read_text())
    assert "model offline" in failed["error"]
    assert completed["error"] is None
    assert len(requests) == 1
    assert len(requests[0]["input"]) == requests[0]["top_k"] == 2
    assert completed["ranked_docids"] == ["relevant", "other"]
    assert completed["ndcg_at_10"] == completed["recall_at_10"] == 1
    record = json.loads((tmp_path / "rollout-completed.json").read_text())
    assert record["retrieval"]["ranked_ids"] == ["c2", "c1"]
    assert [chunk["chunk_id"] for chunk in record["submission"]["chunks"]] == ["c1"]
    assert completed["submission"] == record["submission"]
    usage = completed["reranking"]
    assert usage == record["retrieval"]["reranking"]
    assert usage["model"] == "test-reranker"
    assert usage["candidate_count"] == 2
    assert usage["input_tokens"] == 42
    assert usage["duration_ms"] >= 0
