"""Deterministic rollout scenarios shared across the record-parity suites.

Each scenario drives an agent rollout with scripted generation and an
in-process retrieval client, covering the loop's edge paths: a clean
multi-round submit, the forced-ranking retry under strict top_k, a round of
tool errors, and a prose-only stop. ``test_record_parity.py`` asserts the
sync and async surfaces produce identical normalized records from these
scripts and that records are deterministic run-to-run; ``test_aio.py`` and
``test_tuning.py`` reuse the fakes.

Determinism rules the scenarios obey: at most one *remote* query-producing
tool call per round (``queries_made`` follows dispatch order, but keeping
rounds single-call also pins handle minting), no token counter installed
(the chars/4 heuristic is deterministic), and normalization strips
wall-clock fields.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import agent_harness
from agent_harness.agents.tool_trace import jsonable
from agent_harness.testing import (
    ScriptedGeneration,
    TurnBuilder,
    response,
    tool_call,
)

STORE_ID = "in-process-store"
FILE_ID = "file-1"

# Wall-clock fields scrubbed before comparison; everything else must be stable.
_VOLATILE_KEYS = frozenset({"started_at", "completed_at", "runtime_s"})

# The bootstrap search runs before any generation and returns corpus chunks 0
# and 1, so these handles are minted first in every scenario. Submissions name
# them explicitly: scraping handles out of the transcript is unsafe because
# error payloads echo invalid handles back in the same JSON shape.
SEEDED_CHUNK_IDS: tuple[str, str] = ("c1", "c2")


def corpus_chunk(index: int) -> dict[str, Any]:
    return {
        "id": f"chunk-{index}",
        "file_id": FILE_ID,
        "store_id": STORE_ID,
        "chunk_index": index,
        "filename": "contract.pdf",
        "text": f"clause {index}: the distribution agreement is governed by New York law",
        "score": 1.0 - index / 100,
        "generated_metadata": {"year": 2019},
    }


class ScriptedSearchResults:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class ScriptedStoreFiles:
    def retrieve(self, *, file_identifier: str, store_identifier: str, return_chunks: Any) -> Any:
        indices = return_chunks if isinstance(return_chunks, list) else [0, 1]
        return {
            "id": file_identifier,
            "store_id": store_identifier,
            "filename": "contract.pdf",
            "chunks": [corpus_chunk(index) for index in indices],
        }

    def list(self, **kwargs: Any) -> Any:
        return {"data": [], "pagination": {}}


class ScriptedStores:
    def __init__(
        self,
        calls: list[tuple[str, dict[str, Any]]],
        *,
        failing_queries: frozenset[str] = frozenset(),
    ) -> None:
        self._calls = calls
        self._failing_queries = failing_queries
        self.files = ScriptedStoreFiles()

    def search(self, **kwargs: Any) -> ScriptedSearchResults:
        self._calls.append(("search", kwargs))
        if kwargs.get("query") in self._failing_queries:
            msg = "provider unavailable"
            raise RuntimeError(msg)
        return ScriptedSearchResults([corpus_chunk(0), corpus_chunk(1)])

    def metadata_facets(self, **kwargs: Any) -> Any:
        self._calls.append(("metadata_facets", kwargs))
        return {"metadata_fields": {"year": {"values": [2019]}}}

    def grep(self, **kwargs: Any) -> Any:
        self._calls.append(("grep", kwargs))
        return {"data": []}

    def list_chunks(self, **kwargs: Any) -> Any:
        self._calls.append(("list_chunks", kwargs))
        return {"data": [corpus_chunk(0)]}


class ScriptedRetrievalClient:
    """In-process retrieval fake with per-scenario failure injection.

    Deliberately has no ``post``: grep is a typed method on the async stores
    seam, so nothing in a rollout should ever reach a raw-request escape hatch.
    """

    def __init__(self, *, failing_queries: frozenset[str] = frozenset()) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.stores = ScriptedStores(self.calls, failing_queries=failing_queries)

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


def ranking_arguments(chunk_ids: Sequence[str], *, strategy: str) -> dict[str, Any]:
    return {
        "ranking_strategy": strategy,
        "chunks": [
            {"chunk_id": chunk_id, "relevance_score": round(0.95 - 0.05 * position, 2)}
            for position, chunk_id in enumerate(chunk_ids)
        ],
    }


@dataclass(frozen=True)
class Scenario:
    name: str
    build_script: Callable[[], list[TurnBuilder]]
    top_k: int | None = None
    strict_top_k: bool = False
    failing_queries: frozenset[str] = field(default_factory=frozenset)

    def client(self) -> ScriptedRetrievalClient:
        return ScriptedRetrievalClient(failing_queries=self.failing_queries)

    def generation(self) -> ScriptedGeneration:
        return ScriptedGeneration(self.build_script())


def _search_turn(response_id: str, query: str, *, input_tokens: int) -> TurnBuilder:
    def build(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [tool_call(f"call-{response_id}", "search_corpus", {"query": query})],
            response_id=response_id,
            input_tokens=input_tokens,
        )

    return build


def _submit_turn(
    response_id: str,
    *,
    strategy: str,
    input_tokens: int,
    chunk_ids: Sequence[str] = SEEDED_CHUNK_IDS,
) -> TurnBuilder:
    def build(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [
                tool_call(
                    f"call-{response_id}",
                    "submit_ranking",
                    ranking_arguments(list(chunk_ids), strategy=strategy),
                )
            ],
            response_id=response_id,
            input_tokens=input_tokens,
        )

    return build


def _submit_after_two_rounds() -> list[TurnBuilder]:
    def grep_turn(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [tool_call("call-r2", "grep", {"pattern": "New York law"})],
            response_id="resp-2",
            input_tokens=140,
        )

    return [
        _search_turn("resp-1", "governing law", input_tokens=120),
        grep_turn,
        _submit_turn(
            "resp-3", strategy="relevance to the governing-law question", input_tokens=160
        ),
    ]


def _forced_ranking_after_budget() -> list[TurnBuilder]:
    rounds = [
        _search_turn(f"resp-{index}", f"angle {index}", input_tokens=100 + index * 10)
        for index in range(1, 5)
    ]

    def invalid_forced(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [
                tool_call(
                    "call-forced-1",
                    "submit_ranking",
                    ranking_arguments(["c99"], strategy="hallucinated handle"),
                )
            ],
            response_id="resp-forced-1",
            input_tokens=180,
        )

    return [
        *rounds,
        invalid_forced,
        _submit_turn("resp-forced-2", strategy="fallback to seeded evidence", input_tokens=190),
    ]


def _tool_error_round() -> list[TurnBuilder]:
    def error_round(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [
                tool_call("call-unknown", "summon_expert", {"topic": "contracts"}),
                tool_call("call-fail", "search_corpus", {"query": "flaky query"}),
                tool_call("call-bad-id", "get_chunks", {"chunk_ids": ["c77"]}),
            ],
            response_id="resp-1",
            input_tokens=130,
        )

    return [
        error_round,
        _submit_turn("resp-2", strategy="survivors of the error round", input_tokens=150),
    ]


def _no_tool_calls_then_forced() -> list[TurnBuilder]:
    def prose_only(messages: list[dict[str, Any]]) -> SimpleNamespace:
        return response(
            [],
            response_id="resp-1",
            content="I already know the answer.",
            input_tokens=110,
        )

    return [
        prose_only,
        _submit_turn("resp-forced-1", strategy="forced after prose stop", input_tokens=170),
    ]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(name="submit_after_two_rounds", build_script=_submit_after_two_rounds),
    Scenario(
        name="forced_ranking_after_budget",
        build_script=_forced_ranking_after_budget,
        top_k=2,
        strict_top_k=True,
    ),
    Scenario(
        name="tool_error_round",
        build_script=_tool_error_round,
        failing_queries=frozenset({"flaky query"}),
    ),
    Scenario(name="no_tool_calls_then_forced", build_script=_no_tool_calls_then_forced),
)

SCENARIOS_BY_NAME: dict[str, Scenario] = {scenario.name: scenario for scenario in SCENARIOS}


def run_sync(scenario: Scenario) -> dict[str, Any]:
    return agent_harness.fast_agentic_search(
        "Which contract governs the 2019 Nike distribution agreement?",
        store_identifiers=[STORE_ID],
        top_k=scenario.top_k,
        strict_top_k=scenario.strict_top_k,
        client=scenario.client(),
        generation_fn=scenario.generation(),
    )


def normalize(value: Any) -> Any:
    """Strip wall-clock fields; everything left must be deterministic."""
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in value.items()
            if key not in _VOLATILE_KEYS and not str(key).endswith("_ms")
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def normalized_record(scenario: Scenario) -> dict[str, Any]:
    return normalize_record(run_sync(scenario))


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """JSON-shape a raw record (pydantic models included) and scrub clocks."""
    return normalize(jsonable(record))
