"""The sync and async surfaces produce identical, deterministic records.

Every scenario runs once through the sync compatibility surface (adapted
seams, ``run_coroutine_sync``) and once through ``agent_harness.aio`` with
async-wrapped fakes; the normalized records must match exactly, and repeated
runs of one surface must agree with themselves (guarding against clocks or
scheduling races leaking into records).
"""

from __future__ import annotations

import pytest
import replay_scenarios as rs
from test_aio import AsyncScriptedClient, AsyncScriptedGeneration

from agent_harness import aio

QUERY = "Which contract governs the 2019 Nike distribution agreement?"


async def _async_record(scenario: rs.Scenario) -> dict:
    return rs.normalize_record(
        await aio.fast_agentic_search(
            QUERY,
            store_identifiers=[rs.STORE_ID],
            top_k=scenario.top_k,
            strict_top_k=scenario.strict_top_k,
            client=AsyncScriptedClient(scenario.client()),
            generation_fn=AsyncScriptedGeneration(scenario.generation()),
        )
    )


@pytest.mark.parametrize("name", [scenario.name for scenario in rs.SCENARIOS])
async def test_sync_and_async_surfaces_produce_identical_records(name: str) -> None:
    scenario = rs.SCENARIOS_BY_NAME[name]
    assert await _async_record(scenario) == rs.normalized_record(scenario)


@pytest.mark.parametrize("name", [scenario.name for scenario in rs.SCENARIOS])
def test_records_are_deterministic(name: str) -> None:
    scenario = rs.SCENARIOS_BY_NAME[name]
    assert rs.normalized_record(scenario) == rs.normalized_record(scenario)
