"""Every rollout entry installs the exact counter and records which one measured it.

The budgets in ``agent_harness.config`` are only as good as the counter behind
``count_text_tokens``: a rollout that starts on the chars/4 heuristic undercounts
JSON-heavy retrieval payloads and can overflow the serving deployment's context
window.
These tests pin that the entry seams install the tokenizer once, that the rollout
record says which counter was used, and that the hard-require knob fails loudly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import config, token_counter
from agent_harness.agents import searcher as searcher_runtime
from agent_harness.execution_policy import run_searcher
from agent_harness.llm import sync_generation_as_async
from agent_harness.search import ToolOutcome

UNRESOLVABLE_MODEL = "definitely-not-a-real-model-xyz-000"


class _WordTokenizer:
    """Stands in for the model's tokenizer: one token per word, unlike chars/4."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _spy_tokenizer_loads(monkeypatch: pytest.MonkeyPatch, tokenizer: Any | None) -> list[str]:
    """Record every checkpoint the harness tries to load, without loading one."""
    loads: list[str] = []
    monkeypatch.setattr(token_counter, "_RESOLVED_MODEL", None)

    def fake_load(model: str) -> Any | None:
        loads.append(model)
        return tokenizer

    monkeypatch.setattr(token_counter, "_load_tokenizer", fake_load)
    return loads


async def _fake_initial_metadata_facets(**kwargs: Any) -> ToolOutcome:
    return ToolOutcome(
        {"type": "INITIAL_METADATA_FACETS", "metadata_fields": {}},
        {"tool": "inspect_metadata"},
    )


def _stub_searcher_io(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub the fast-searcher bootstrap fetches and return the seeded chunk id."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_metadata_facets",
        _fake_initial_metadata_facets,
    )

    async def fake_initial_search_results(*args: Any, **kwargs: Any) -> ToolOutcome:
        del args
        index = kwargs["index"]
        index.add_chunk(
            {
                "store_id": "store-a",
                "file_id": "file-a",
                "chunk_index": 0,
                "text": "sample launch ad",
            }
        )
        captured["chunk_id"] = index.visible_chunk_ids()[0]
        return ToolOutcome(
            {
                "type": "INITIAL_SEARCH_RESULTS",
                "query": "find the launch ads",
                "results": [{"chunk_id": captured["chunk_id"], "text": "sample launch ad"}],
            },
            {"tool": "search_corpus", "new_chunks_added": 1},
        )

    monkeypatch.setattr(
        searcher_runtime,
        "_fetch_initial_search_results",
        fake_initial_search_results,
    )
    return captured


def _searcher_generation_fn(captured: dict[str, Any]) -> Any:
    def generation_fn(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        del messages, kwargs
        return _fake_response(
            [
                _fake_tool_call(
                    "call_submit",
                    "submit_ranking",
                    {
                        "ranking_strategy": "counter installed at entry",
                        "chunks": [{"chunk_id": captured["chunk_id"], "relevance_score": 1.0}],
                    },
                )
            ]
        )

    return generation_fn


def _fake_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _fake_response(tool_calls: list[Any]) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))],
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
    )


async def test_searcher_entry_installs_the_counter_once_across_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads = _spy_tokenizer_loads(monkeypatch, _WordTokenizer())
    captured = _stub_searcher_io(monkeypatch)
    config.set_token_counter(None)
    try:
        for _ in range(2):
            result = await searcher_runtime.run_fast_agentic_search(
                "find the launch ads",
                store_identifiers=["store-a"],
                generation_fn=sync_generation_as_async(_searcher_generation_fn(captured)),
            )
            assert result.ranking is not None
            assert result.to_record()["token_counter_mode"] == "exact"

        # Two rollouts, one load: ensure_token_counter is idempotent per model.
        assert loads == [config.SEARCHER_AGENT_CONFIG["model"]]
        # 100 words, which the chars/4 heuristic would have called 125 tokens.
        assert config.count_text_tokens("word " * 100) == 100
    finally:
        config.set_token_counter(None)


def test_rollout_record_reports_the_heuristic_when_no_counter_installs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _spy_tokenizer_loads(monkeypatch, None)
    captured = _stub_searcher_io(monkeypatch)
    config.set_token_counter(None)
    try:
        record = run_searcher(
            "find the launch ads",
            store_identifiers=["store-a"],
            generation_fn=_searcher_generation_fn(captured),
        )

        assert record["openai"]["metadata"]["token_counter_mode"] == "chars-heuristic"
        assert config.TOKEN_COUNTER is None
    finally:
        config.set_token_counter(None)


@pytest.mark.parametrize(
    ("mode_label", "expected"),
    [(None, "exact"), ("exact-gigatoken", "exact-gigatoken")],
    ids=["plain", "labelled"],
)
def test_rollout_record_reports_the_installed_counter(
    monkeypatch: pytest.MonkeyPatch, mode_label: str | None, expected: str
) -> None:
    """A counter may name itself; anything installed is at least ``exact``."""
    tokenizer = _WordTokenizer()
    if mode_label is not None:
        tokenizer.token_counter_mode = mode_label  # type: ignore[attr-defined]
    _spy_tokenizer_loads(monkeypatch, tokenizer)
    captured = _stub_searcher_io(monkeypatch)
    config.set_token_counter(None)
    try:
        record = run_searcher(
            "find the launch ads",
            store_identifiers=["store-a"],
            generation_fn=_searcher_generation_fn(captured),
        )

        assert record["openai"]["metadata"]["token_counter_mode"] == expected
    finally:
        config.set_token_counter(None)


async def test_require_exact_tokenizer_fails_every_rollout_that_cannot_install_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(token_counter, "_RESOLVED_MODEL", None)
    monkeypatch.delenv("AGENT_HARNESS_TOKENIZER", raising=False)
    monkeypatch.setenv("AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER", "1")
    monkeypatch.setitem(config.SEARCHER_AGENT_CONFIG, "model", UNRESOLVABLE_MODEL)
    captured = _stub_searcher_io(monkeypatch)
    config.set_token_counter(None)
    try:
        for _ in range(2):
            # The second rollout must fail too: the load is not retried, but the
            # requirement is re-checked, so no rollout slips through on the heuristic.
            with pytest.raises(RuntimeError, match=UNRESOLVABLE_MODEL) as failure:
                await searcher_runtime.run_fast_agentic_search(
                    "find the launch ads",
                    store_identifiers=["store-a"],
                    generation_fn=sync_generation_as_async(_searcher_generation_fn(captured)),
                )
            assert "AGENT_HARNESS_TOKENIZER" in str(failure.value)
            assert "Exact token counting is required" in str(failure.value)
    finally:
        config.set_token_counter(None)


async def test_the_requirement_defaults_on_and_opting_out_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact counting is the main path: unset env means required; an explicit
    opt-out degrades to the heuristic instead of raising."""
    monkeypatch.setattr(token_counter, "_RESOLVED_MODEL", None)
    monkeypatch.delenv("AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER", raising=False)
    assert token_counter._require_exact_tokenizer() is True
    monkeypatch.setenv("AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER", "0")
    monkeypatch.setitem(config.SEARCHER_AGENT_CONFIG, "model", UNRESOLVABLE_MODEL)
    captured = _stub_searcher_io(monkeypatch)
    config.set_token_counter(None)
    try:
        result = await searcher_runtime.run_fast_agentic_search(
            "find the launch ads",
            store_identifiers=["store-a"],
            generation_fn=sync_generation_as_async(_searcher_generation_fn(captured)),
        )

        assert result.ranking is not None
        assert result.to_record()["token_counter_mode"] == "chars-heuristic"
    finally:
        config.set_token_counter(None)
