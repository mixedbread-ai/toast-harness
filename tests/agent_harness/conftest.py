"""Suite-wide defaults for the harness behavior tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _allow_heuristic_token_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests exercise harness mechanics with fake models; production's
    exact-tokenizer requirement would fail every tokenizer-less rollout."""
    monkeypatch.setenv("AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER", "0")


@pytest.fixture(autouse=True)
def _hermetic_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin a dummy Mixedbread key so the suite behaves the same everywhere.

    resolve_async_retrieval_client validates the key eagerly; without one the
    bootstrap takes its error branch and tests exercise a path production never
    reaches, while a developer's real key flips them back. Fixing the value
    keeps every run on the credentialed path -- no test here talks to the real
    API (clients and raw seams are faked), so the key is never actually used.
    The aliases resolve BEFORE the canonical name, so they are stripped too --
    otherwise an exported MBREAD_API_KEY would out-rank the pin."""
    monkeypatch.delenv("MBREAD_API_KEY", raising=False)
    monkeypatch.delenv("MIXEDBREAD_API_KEY", raising=False)
    monkeypatch.setenv("MXBAI_API_KEY", "test-suite-dummy-key")
