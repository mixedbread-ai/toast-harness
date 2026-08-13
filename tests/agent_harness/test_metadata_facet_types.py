"""Tests for metadata facet value typing in inspect_metadata and filter validation."""

from __future__ import annotations

import json

import httpx
import pytest
from mixedbread import UnprocessableEntityError

from agent_harness.agents import searcher as searcher_runtime
from agent_harness.metadata_guard import (
    build_metadata_registry,
    validate_metadata_filter_args,
)
from agent_harness.prompts import initial_metadata_facets_message
from agent_harness.tools import functions
from agent_harness.tools.functions import (
    _collect_metadata_value_types,
    _typed_facet_samples,
    filter_chunks,
)


def test_collect_metadata_value_types_uses_real_payload_types() -> None:
    types: dict[str, set[str]] = {}
    _collect_metadata_value_types(
        {
            "spend": 59163.21,
            "days_active": 90,
            "ad_id": "1456043385917139",
            "is_active": True,
            "nested": {"score": 0.5},
        },
        prefix="",
        types=types,
    )
    assert types["spend"] == {"number"}
    assert types["days_active"] == {"number"}
    assert types["ad_id"] == {"string"}
    assert types["is_active"] == {"boolean"}
    assert types["nested.score"] == {"number"}


def test_typed_facet_samples_coerces_stringified_numbers() -> None:
    samples = _typed_facet_samples({"59163.21": 1, "22775.92": 2}, {"number"})
    assert samples == [
        {"value": 59163.21, "type": "number", "count": 1},
        {"value": 22775.92, "type": "number", "count": 2},
    ]


def test_typed_facet_samples_keeps_digit_strings_for_string_fields() -> None:
    samples = _typed_facet_samples({"1456043385917139": 1}, {"string"})
    assert samples == [{"value": "1456043385917139", "type": "string", "count": 1}]


def test_typed_facet_samples_without_observed_types_omits_type() -> None:
    samples = _typed_facet_samples({"value-a": 3}, None)
    assert samples == [{"value": "value-a", "count": 3}]


def test_typed_facet_samples_coerces_booleans() -> None:
    samples = _typed_facet_samples({"true": 4, "false": 2}, {"boolean"})
    assert samples == [
        {"value": True, "type": "boolean", "count": 4},
        {"value": False, "type": "boolean", "count": 2},
    ]


def test_initial_metadata_facets_message_omits_tool_name() -> None:
    message = initial_metadata_facets_message(
        {
            "type": "INITIAL_METADATA_FACETS",
            "tool": "inspect_metadata",
            "metadata_fields": {},
        }
    )

    payload = json.loads(message["content"].split("\n", 1)[1])

    assert payload["type"] == "INITIAL_METADATA_FACETS"
    assert "tool" not in payload


async def test_inspect_metadata_treats_empty_facets_wrapper_as_no_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stores:
        async def metadata_facets(self, request):
            return {"facets": []}

    class Client:
        stores = Stores()

    async def fake_sampled_metadata_value_types(**kwargs):
        return {}

    monkeypatch.setattr(
        functions, "_sampled_metadata_value_types", fake_sampled_metadata_value_types
    )

    result = await functions.inspect_metadata(
        store_identifiers=["officeqa-pages"],
        client=Client(),
    )

    assert result["metadata_field_count"] == 0
    assert result["metadata_fields"] == {}


def test_registry_accepts_numeric_comparison_for_number_typed_facets() -> None:
    facets = {
        "metadata_fields": {
            "spend": [
                {"value": 59163.21, "type": "number", "count": 1},
                {"value": 22775.92, "type": "number", "count": 1},
            ],
            "ctr": [{"value": 1.27, "type": "number", "count": 1}],
            "fatigue_tier": [{"value": "severe", "type": "string", "count": 7}],
        }
    }
    registry = build_metadata_registry(initial_metadata_facets=facets)

    result = validate_metadata_filter_args(
        {
            "filter_by": [
                {"key": "spend", "operator": "gt", "value": 20000},
                {"key": "ctr", "operator": "lt", "value": 2},
            ]
        },
        registry=registry,
    )

    assert result.invalid == []
    assert result.args["filter_by"] == [
        {"key": "spend", "operator": "gt", "value": 20000},
        {"key": "ctr", "operator": "lt", "value": 2},
    ]


def test_registry_still_rejects_numeric_comparison_for_string_fields() -> None:
    facets = {
        "metadata_fields": {
            "fatigue_tier": [{"value": "severe", "type": "string", "count": 7}],
        }
    }
    registry = build_metadata_registry(initial_metadata_facets=facets)

    result = validate_metadata_filter_args(
        {"filter_by": [{"key": "fatigue_tier", "operator": "gt", "value": 2}]},
        registry=registry,
    )

    assert len(result.invalid) == 1
    assert result.invalid[0]["reason"] == "value type does not match field"


async def test_filter_chunks_falls_back_to_client_side_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {"chunk_index": 0, "metadata": {"ad_name": "low", "spend": 100.0}},
        {"chunk_index": 0, "metadata": {"ad_name": "high", "spend": 900.0}},
        {"chunk_index": 0, "metadata": {"ad_name": "mid", "spend": 500.0}},
    ]
    calls: list[dict] = []

    async def fake_list_chunks_raw(**kwargs):
        calls.append(kwargs)
        if kwargs.get("sort_by") is not None:
            request = httpx.Request("POST", "https://api.mixedbread.com/v1/stores/list-chunks")
            response = httpx.Response(422, request=request)
            raise UnprocessableEntityError(
                "Sort field must be a numeric (int/float) value.",
                response=response,
                body=None,
            )
        return [dict(chunk) for chunk in chunks]

    monkeypatch.setattr(functions, "list_chunks_raw", fake_list_chunks_raw)

    out = await filter_chunks(
        filter_by=[],
        rank_by="spend",
        direction="desc",
        k=2,
        store_identifiers=["store-a"],
    )

    names = [chunk["metadata"]["ad_name"] for chunk in out["results"]]
    assert names == ["high", "mid"]
    assert calls[0]["sort_by"] is not None
    assert calls[1]["sort_by"] is None


async def test_inspect_metadata_names_the_rankable_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stores:
        async def metadata_facets(self, request):
            return {
                "facets": {
                    "spend": {"15059.41": 1, "34200": 1},
                    "days_active": {"29": 1, "34": 2},
                    "ad_id": {"1059616022456296": 1},
                    "fatigue_tier": {"severe": 7, "healthy": 23},
                    "vs_peak_pct": {"-3.25": 1},
                }
            }

    class Client:
        stores = Stores()

    async def fake_sampled_metadata_value_types(**kwargs):
        return {
            "spend": {"number"},
            "days_active": {"number"},
            "ad_id": {"string"},
            "fatigue_tier": {"string"},
            "vs_peak_pct": {"number", "null"},
        }

    monkeypatch.setattr(
        functions, "_sampled_metadata_value_types", fake_sampled_metadata_value_types
    )

    result = await functions.inspect_metadata(store_identifiers=["nike-demo"], client=Client())

    # ad_id is all digits but string-typed in facets — exactly the field agents mis-guess as rank_by.
    assert result["rankable_fields"] == ["days_active", "spend", "vs_peak_pct"]
    assert result["field_types_sampled"] is True


async def test_metadata_failure_fallbacks_carry_the_rankable_keys() -> None:
    class Stores:
        async def metadata_facets(self, request):
            raise ConnectionError("provider down")

    class Client:
        stores = Stores()

    # The note references rankable_fields, so every facets payload shape must define it.
    outcome = await searcher_runtime._fetch_initial_metadata_facets(
        store_identifiers=["nike-demo"], client=Client()
    )
    payload = outcome.payload
    assert payload["rankable_fields"] == []
    assert payload["field_types_sampled"] is False
    assert payload["metadata_fields"] == {}


async def test_inspect_metadata_reports_unsampled_field_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Stores:
        async def metadata_facets(self, request):
            return {"facets": {"spend": {"15059.41": 1}}}

    class Client:
        stores = Stores()

    async def fake_sampled_metadata_value_types(**kwargs):
        return {}

    monkeypatch.setattr(
        functions, "_sampled_metadata_value_types", fake_sampled_metadata_value_types
    )

    result = await functions.inspect_metadata(store_identifiers=["nike-demo"], client=Client())

    # Type sampling can come back empty; the flag distinguishes that from a store with no numeric fields.
    assert result["field_types_sampled"] is False
    assert result["rankable_fields"] == []


def test_initial_metadata_facets_message_keeps_rankable_fields() -> None:
    message = initial_metadata_facets_message(
        {
            "type": "INITIAL_METADATA_FACETS",
            "rankable_fields": ["clicks", "spend"],
            "metadata_fields": {"spend": [{"value": 1.0, "type": "number"}]},
        }
    )

    payload = json.loads(message["content"].split("\n", 1)[1])

    assert payload["rankable_fields"] == ["clicks", "spend"]
    assert "rankable_fields" in payload["metadata_filter_syntax"]["note"]
