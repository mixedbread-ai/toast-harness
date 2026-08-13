"""Model-authored grep/filter regexes must be interruptible, never wedge a worker."""

from __future__ import annotations

import time

import regex as interruptible_regex

from agent_harness.search import _grep_match_windows
from agent_harness.tools.functions import _evaluate_filter_condition

# The job-722 wedge class: nested quantifiers backtrack catastrophically on a
# long near-matching subject, and stdlib re cannot be interrupted once started.
PATHOLOGICAL_PATTERN = r"(?:.*a){40}$"
SUBJECT = "a" * 39 + "b" * 30


def test_grep_clip_pass_bounds_pathological_patterns() -> None:
    focus = interruptible_regex.compile(PATHOLOGICAL_PATTERN)
    started = time.monotonic()
    result = _grep_match_windows(SUBJECT * 4, window_chars=80, focus=focus)
    elapsed = time.monotonic() - started
    assert result is None
    assert elapsed < 10.0


def test_filter_regex_operator_bounds_pathological_patterns() -> None:
    condition = {"key": "text", "operator": "regex", "value": PATHOLOGICAL_PATTERN}
    started = time.monotonic()
    matched = _evaluate_filter_condition({"text": SUBJECT * 4}, condition)
    elapsed = time.monotonic() - started
    assert matched is False
    assert elapsed < 10.0


def test_grep_clip_pass_still_windows_normal_matches() -> None:
    focus = interruptible_regex.compile("needle", interruptible_regex.IGNORECASE)
    value = ("x" * 300) + "Needle" + ("y" * 300)
    result = _grep_match_windows(value, window_chars=80, focus=focus)
    assert result is not None
    assert "Needle" in result
