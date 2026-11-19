"""Strict top_k asks the model for exactly top_k chunks but never rejects a count.

A rollout can only rank what it retrieved. A submission shorter than top_k is
accepted as it is; a longer one is clipped to its first top_k chunks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import replay_scenarios as rs

import agent_harness
from agent_harness.testing import response, tool_call

QUERY = "Which contract governs the 2019 Nike distribution agreement?"


def _submit_turn(chunk_ids: list[str]) -> Any:
    def build(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [
                tool_call(
                    "call-submit",
                    "submit_ranking",
                    rs.ranking_arguments(chunk_ids, strategy="seeded evidence"),
                )
            ],
            response_id="resp-submit",
        )

    return build


def _run_strict(chunk_ids: list[str], *, top_k: int) -> dict[str, Any]:
    return agent_harness.fast_agentic_search(
        QUERY,
        store_identifiers=[rs.STORE_ID],
        top_k=top_k,
        strict_top_k=True,
        client=rs.ScriptedRetrievalClient(),
        generation_fn=rs.ScriptedGeneration([_submit_turn(chunk_ids)]),
    )


def _submitted_ids(record: dict[str, Any]) -> list[str]:
    return [chunk["chunk_id"] for chunk in record["chunks"]]


def test_a_submission_shorter_than_top_k_is_accepted() -> None:
    record = _run_strict(["c1"], top_k=len(rs.SEEDED_CHUNK_IDS) + 1)

    assert record["forced_ranking"] is False
    assert _submitted_ids(record) == ["c1"]


def test_a_submission_longer_than_top_k_is_clipped() -> None:
    record = _run_strict(list(rs.SEEDED_CHUNK_IDS), top_k=1)

    assert record["forced_ranking"] is False
    assert _submitted_ids(record) == ["c1"]
