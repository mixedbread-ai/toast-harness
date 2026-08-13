"""Golden coverage for the model-visible searcher contract.

``agent_harness.searcher_spec`` exports the prompt, tool, and message-wire
semantics a non-Python host must reproduce byte-for-byte to serve this searcher.
The goldens under ``searcher_spec/fixtures`` are the pinned projection of that
surface; the tests below assert the export still matches the live code that
actually builds the prompts and tools.

Regenerate the golden after an intentional prompt/tool change:

    python -c "
    from pathlib import Path
    from agent_harness.searcher_spec import build_searcher_contract, dumps_searcher_contract
    out = Path('src/agent_harness/searcher_spec/fixtures')
    (out / 'submit_ranking.top5.strict.v1.json').write_text(
        dumps_searcher_contract(
            build_searcher_contract(top_k=5, strict_top_k=True), indent=2
        ) + '\\n',
        encoding='utf-8',
    )
    "

then update the digest literal below with the reported value.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date
from importlib.resources import files

import pytest
from jsonschema import Draft202012Validator, ValidationError

from agent_harness.agents.searcher import _searcher_tools
from agent_harness.agents.shared import tool_message
from agent_harness.config import SEARCHER_MAX_ROUNDS, HarnessTuning, tuning_setting
from agent_harness.searcher_prompts import _runtime_context, fast_searcher_messages
from agent_harness.searcher_spec import (
    build_searcher_contract,
    dumps_searcher_contract,
    load_searcher_contract_schema,
    searcher_contract_digest,
)

SUBMIT_RANKING_GOLDEN_DIGEST = (
    "sha256:6f971e1ef3b7aea865f43d1168cb77d089039c82669dde5dceb473d644bd2e1f"
)
_UTC_DATE = re.compile(r"Current UTC date: \d{4}-\d{2}-\d{2}\.")


def _validator() -> Draft202012Validator:
    schema = load_searcher_contract_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _golden(name: str) -> dict:
    resource = files("agent_harness.searcher_spec").joinpath(f"fixtures/{name}")
    return json.loads(resource.read_text(encoding="utf-8"))


def test_fast_searcher_contract_matches_live_tool_order_and_schema() -> None:
    contract = build_searcher_contract(top_k=5, strict_top_k=True)
    live_tools = _searcher_tools(top_k=5, strict_top_k=True)
    assert contract["tools"] == [tool["function"] for tool in live_tools]
    assert contract["initialization_stages"] == [
        "initial_metadata",
        "initial_search",
        "prompt_assembly",
    ]
    assert contract["policy"]["final_tool_must_be_only_call"] is True
    assert contract["bootstrap"]["initial_metadata"]["arguments"] == {"max_values_per_field": 5}
    assert contract["bootstrap"]["initial_search"]["arguments"]["top_k"] == 5
    assert contract["bootstrap"]["execution"] == {
        "default": "parallel",
        "parallel_group": ["initial_metadata", "initial_search"],
        "join_before": "prompt_assembly",
        "sequential_fallback": True,
    }
    assert contract["bootstrap"]["prompt_assembly"]["message_order"] == [
        "system",
        "initial_metadata",
        "initial_search",
        "user_query",
        "initial_search_media_messages",
    ]


def test_exported_tools_carry_no_provider_specific_envelope() -> None:
    """The tool export is the bare function schema, not a provider tool wrapper.

    Asserted against the live tool builders rather than against literals copied
    out of contract.py, so the check cannot pass by agreeing with itself.
    """
    contract = build_searcher_contract()
    live_tools = _searcher_tools()

    assert all(set(tool) == {"name", "description", "parameters"} for tool in contract["tools"])
    assert all(set(tool) == {"type", "function"} for tool in live_tools)
    assert all(tool["type"] == "function" for tool in live_tools)
    assert contract["tools"] == [tool["function"] for tool in live_tools]


def test_tool_result_wire_names_the_in_package_serialization() -> None:
    """prod emits the chat-completions tool shape from inside agent_harness.

    The contract must say so rather than claim a host adapter owns it, or a host
    runtime will build a second, conflicting serializer.
    """
    contract = build_searcher_contract()
    wire = contract["message_wire"]["tool_result"]

    assert wire["semantic_role"] == "tool_result"
    assert wire["call_id_field"] == "tool_call_id"
    assert "chat-completions" in wire["provider_serialization"]

    emitted = tool_message("call-1", {"ok": True})
    assert set(emitted) == {"role", "tool_call_id", "content"}
    assert emitted["role"] == "tool"
    assert emitted[wire["call_id_field"]] == "call-1"
    assert json.loads(emitted["content"]) == {"ok": True}


def test_searcher_contract_digest_is_deterministic_and_condition_sensitive() -> None:
    first = build_searcher_contract(top_k=5)
    second = build_searcher_contract(top_k=5)
    different = build_searcher_contract(top_k=10)
    assert searcher_contract_digest(first) == searcher_contract_digest(second)
    assert searcher_contract_digest(first) != searcher_contract_digest(different)


def test_searcher_contract_is_language_neutral_and_schema_valid() -> None:
    validator = _validator()
    contracts = [
        build_searcher_contract(top_k=5, strict_top_k=True),
        build_searcher_contract(),
    ]

    for contract in contracts:
        validator.validate(contract)
        encoded = dumps_searcher_contract(contract)
        assert json.loads(encoded) == contract
        assert "NaN" not in encoded
        assert "Infinity" not in encoded


def test_golden_contract_matches_live_python_export_exactly() -> None:
    golden = _golden("submit_ranking.top5.strict.v1.json")
    current = build_searcher_contract(top_k=5, strict_top_k=True)

    _validator().validate(golden)
    assert current == golden
    assert searcher_contract_digest(golden) == SUBMIT_RANKING_GOLDEN_DIGEST


def test_schema_rejects_unknown_contract_fields() -> None:
    contract = deepcopy(build_searcher_contract())
    contract["python_object"] = "not portable"

    with pytest.raises(ValidationError):
        _validator().validate(contract)


def test_contract_contains_no_secret_bearing_fields() -> None:
    contract = build_searcher_contract()
    forbidden = {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "cookies",
        "headers",
        "oauth_token",
    }

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert forbidden.isdisjoint(key.casefold() for key in value)
            for nested in value.values():
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(contract)


def test_portable_message_wire_matches_real_first_provider_context() -> None:
    metadata = {
        "type": "INITIAL_METADATA_FACETS",
        "tool": "inspect_metadata",
        "store_identifiers": ["store-1"],
        "metadata_fields": {
            "year": [
                {"value": 2025, "type": "number", "count": 1},
                {"value": 2026, "type": "number", "count": 1},
            ]
        },
    }
    # Carry real score fields: the initial_search projection is NOT identity, and
    # a fixture without them asserts the rounding rule vacuously.
    search = {
        "type": "INITIAL_SEARCH_RESULTS",
        "query": "portable query",
        "results": [
            {
                "chunk_id": "c1",
                "text": "evidence",
                "score": 0.8412345,
                "search_score": 0.7719,
                "metadata": {"score": 0.005554},
            }
        ],
    }
    contract = build_searcher_contract(top_k=3, strict_top_k=True)
    messages = fast_searcher_messages(
        user_text="portable query",
        initial_metadata_facets=metadata,
        initial_search_results=search,
        top_k=3,
        strict_top_k=True,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "user",
    ]
    assert messages[0]["content"].startswith(contract["prompt"]["system"])
    assert _UTC_DATE.search(messages[0]["content"]) is not None
    metadata_prefix = contract["message_wire"]["initial_metadata"]["prefix"]
    search_prefix = contract["message_wire"]["initial_search"]["prefix"]
    query_prefix = contract["message_wire"]["user_query"]["prefix"]
    assert messages[1]["content"].startswith(metadata_prefix)
    assert messages[3]["content"] == query_prefix + "portable query"

    rendered_search = json.loads(messages[2]["content"][len(search_prefix) :])
    projection = contract["message_wire"]["initial_search"]["projection"]
    assert projection["base"] == "identity"
    assert projection["result_score_fields"] == ["score", "search_score"]
    result = rendered_search["results"][0]
    # 2 significant figures on the result's own score fields ...
    assert result["score"] == 0.84
    assert result["search_score"] == 0.77
    # ... and nested domain metadata named "score" is data, left untouched.
    assert result["metadata"]["score"] == 0.005554
    assert result["chunk_id"] == "c1"
    assert rendered_search["query"] == "portable query"

    rendered_metadata = json.loads(messages[1]["content"][len(metadata_prefix) :])
    assert "tool" not in rendered_metadata
    assert rendered_metadata["metadata_fields"] == {
        "year": {
            "sample_values": [
                {"value": 2025, "type": "number", "count": 1},
                {"value": 2026, "type": "number", "count": 1},
            ]
        }
    }
    assert rendered_metadata["metadata_filter_syntax"]["filter_by"][0]["operator"] == "eq"


def test_exported_system_prompt_rebuilds_byte_exactly_with_the_runtime_context() -> None:
    """``prompt.system`` + ``runtime_context.template`` IS the live system message.

    The live message appends a runtime-context block after the exported prefix;
    pinning the composition with ``as_of`` locks the ENTIRE system prompt, so a
    wording change in the runtime-context block fails here instead of drifting
    unpinned behind a startswith(). additional_instructions still break the
    prefix: they are folded into the task description *before* assembly, so they
    land mid-string, and a host that appends them emits a byte-different prompt.
    """
    contract = build_searcher_contract(top_k=5, strict_top_k=True)
    system = contract["prompt"]["system"]
    suffix = contract["prompt"]["runtime_context"]["template"].format(
        utc_date="2026-01-01", utc_yesterday="2025-12-31"
    )

    plain = fast_searcher_messages(
        user_text="q", top_k=5, strict_top_k=True, as_of=date(2026, 1, 1)
    )
    assert plain[0]["content"] == system + suffix

    extended = fast_searcher_messages(
        user_text="q",
        top_k=5,
        strict_top_k=True,
        additional_instructions="BE TERSE",
    )
    assert not extended[0]["content"].startswith(system)
    assert "ADDITIONAL INSTRUCTIONS:\nBE TERSE" in extended[0]["content"]
    # The marker lands inside the task description, not appended at the end.
    assert extended[0]["content"].index("BE TERSE") < len(system)
    assert "runtime_context.template" in contract["prompt"]["system_validity"]


def test_runtime_context_template_matches_the_live_builder() -> None:
    """The exported template is the live suffix with only the dates abstracted.

    This pins the suffix builder itself; the full composition around it is
    pinned by the byte-equality test above."""
    contract = build_searcher_contract()
    template = contract["prompt"]["runtime_context"]["template"]
    assert _runtime_context(date(2026, 3, 1)) == template.format(
        utc_date="2026-03-01", utc_yesterday="2026-02-28"
    )


def test_policy_max_rounds_tracks_the_tuned_prompt() -> None:
    """policy.max_rounds and the prompt's round text must come from one source.

    The system prompt absorbs a per-rollout HarnessTuning.searcher_max_rounds;
    a policy bound to the import-time constant would let a tuned export promise
    one round budget in prose and another in the machine-readable field, and a
    host would force-submit the searcher early (or late)."""
    for tuning, expected in [
        (None, SEARCHER_MAX_ROUNDS),
        (HarnessTuning(searcher_max_rounds=9), 9),
    ]:
        with tuning_setting(tuning):
            contract = build_searcher_contract(top_k=5, strict_top_k=True)
        assert contract["policy"]["max_rounds"] == expected
        assert f"at most {expected} rounds" in contract["prompt"]["system"]
