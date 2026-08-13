"""Provider-vs-agent failure classification and rollout failure counting."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from mixedbread import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    UnprocessableEntityError,
)

from agent_harness.errors import (
    AGENT_ERROR_KIND,
    PROVIDER_ERROR_KIND,
    ProviderFailure,
    error_kind,
    is_provider_failure,
    wire_status_error,
)
from agent_harness.execution_policy import count_provider_failures, count_trace_events

_REQUEST = httpx.Request("POST", "https://api.mixedbread.test/v1/stores/search")


def _status_error(cls: type[APIStatusError], status_code: int) -> APIStatusError:
    response = httpx.Response(status_code, request=_REQUEST)
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    "exc",
    [
        _status_error(RateLimitError, 429),
        _status_error(InternalServerError, 500),
        _status_error(APIStatusError, 503),
        _status_error(APIStatusError, 408),
        APITimeoutError(request=_REQUEST),
        APIConnectionError(request=_REQUEST),
    ],
    ids=["rate-limit", "internal-500", "status-503", "timeout-408", "sdk-timeout", "connection"],
)
def test_provider_side_failures_classify_as_provider(exc: BaseException) -> None:
    assert is_provider_failure(exc) is True
    assert error_kind(exc) == PROVIDER_ERROR_KIND


@pytest.mark.parametrize(
    "exc",
    [
        _status_error(UnprocessableEntityError, 422),
        _status_error(NotFoundError, 404),
        _status_error(APIStatusError, 400),
        ValueError("filter_chunks failed for rank_by='year': no numeric values found"),
        RuntimeError("some non-mixedbread failure"),
    ],
    ids=["unprocessable-422", "not-found-404", "bad-request-400", "value-error", "runtime-error"],
)
def test_agent_side_failures_classify_as_agent(exc: BaseException) -> None:
    assert is_provider_failure(exc) is False
    assert error_kind(exc) == AGENT_ERROR_KIND


def test_count_trace_events_counts_provider_errors_separately() -> None:
    tool_trace: list[dict[str, Any]] = [
        {
            "agent": "searcher",
            "name": "search_corpus",
            "status": "error",
            "error": "429 rate limited",
            "error_kind": PROVIDER_ERROR_KIND,
        },
        {
            "agent": "searcher",
            "name": "filter_chunks",
            "status": "error",
            "error": "invalid metadata filter",
            "error_kind": AGENT_ERROR_KIND,
        },
        {"agent": "searcher", "name": "search_corpus", "status": "success"},
    ]

    counts = count_trace_events(tool_trace)

    assert counts["tool_calls"] == 3
    assert counts["errors"] == 2
    assert counts["provider_errors"] == 1


def test_count_provider_failures_includes_bootstrap_queries() -> None:
    result = {
        "tool_trace": [
            {"name": "grep", "status": "error", "error_kind": PROVIDER_ERROR_KIND},
            {"name": "filter_chunks", "status": "error", "error_kind": AGENT_ERROR_KIND},
        ],
        "queries_made": [
            {
                "tool": "inspect_metadata",
                "error": "503 service unavailable",
                "error_kind": PROVIDER_ERROR_KIND,
                "source": "initial_metadata_facets",
            },
            {"tool": "search_corpus", "new_chunks_added": 3},
        ],
    }

    assert count_provider_failures(result) == 2


def test_count_provider_failures_is_zero_without_failures() -> None:
    assert count_provider_failures({}) == 0
    assert (
        count_provider_failures(
            {
                "tool_trace": [{"name": "search_corpus", "status": "success"}],
                "queries_made": [{"tool": "search_corpus", "new_chunks_added": 1}],
            }
        )
        == 0
    )


def test_provider_failure_is_classified_provider_side() -> None:
    """In-process bindings raise ProviderFailure instead of fabricating SDK errors."""
    exc = ProviderFailure("store backend unavailable")
    assert is_provider_failure(exc)
    assert error_kind(exc) == PROVIDER_ERROR_KIND

    class StoreDown(ProviderFailure):
        pass

    assert error_kind(StoreDown("boom")) == PROVIDER_ERROR_KIND


def test_wire_status_error_builds_the_typed_sdk_errors() -> None:
    not_found = wire_status_error(404, "store not found", body={"type": "not_found_error"})
    assert isinstance(not_found, NotFoundError)
    assert not_found.body == {"type": "not_found_error"}
    assert not is_provider_failure(not_found)

    invalid = wire_status_error(422, "bad params", body=None)
    assert isinstance(invalid, UnprocessableEntityError)

    boom = wire_status_error(500, "internal", body=None)
    assert isinstance(boom, InternalServerError)
    assert is_provider_failure(boom)
