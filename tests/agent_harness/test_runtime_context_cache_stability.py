import copy
from datetime import UTC, date, datetime

import pytest

from agent_harness import prompts, searcher_prompts


@pytest.mark.parametrize("module", [prompts, searcher_prompts])
def test_runtime_context_is_stable_within_utc_day(monkeypatch, module) -> None:
    class Morning(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 5, 0, 0, 1, tzinfo=UTC)

    class Evening(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 5, 23, 59, 59, tzinfo=UTC)

    monkeypatch.setattr(module, "datetime", Morning)
    morning = module._runtime_context()
    monkeypatch.setattr(module, "datetime", Evening)
    evening = module._runtime_context()

    assert morning == evening
    assert "2026-08-05" in morning
    assert "yesterday is 2026-08-04" in morning
    assert "UTC time" not in morning


def test_initial_search_scores_are_prompt_only_two_significant_figures() -> None:
    source = {
        "results": [
            {"chunk_id": "c1", "search_score": 0.5554, "nested": {"score": 0.005554}},
            {"chunk_id": "c2", "search_score": 0.0},
        ]
    }
    original = copy.deepcopy(source)

    message = searcher_prompts.initial_search_results_message(source)
    rendered = message["content"]

    assert '"search_score": 0.56' in rendered
    assert '"score": 0.005554' in rendered
    assert '"search_score": 0.0' in rendered
    assert source == original


@pytest.mark.parametrize("module", [prompts, searcher_prompts])
def test_runtime_context_as_of_pins_the_date_over_the_wall_clock(monkeypatch, module) -> None:
    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 8, 5, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(module, "datetime", WallClock)
    pinned = module._runtime_context(date(2024, 3, 1))

    assert "Current UTC date: 2024-03-01" in pinned
    assert "yesterday is 2024-02-29" in pinned
    assert "2026-08-05" not in pinned


def test_message_builders_thread_as_of_into_runtime_context() -> None:
    pinned = date(2024, 3, 1)

    fast_system = searcher_prompts.fast_searcher_messages(user_text="query", as_of=pinned)[0][
        "content"
    ]

    assert "Current UTC date: 2024-03-01" in fast_system
    assert "yesterday is 2024-02-29" in fast_system
