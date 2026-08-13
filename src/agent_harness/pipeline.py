"""Compatibility entry points for the fast-searcher pipeline."""

from __future__ import annotations

from .execution_policy import SearcherExecutionPolicy, run_searcher
from .llm import GenerationFn
from .sync_api import fast_agentic_search

__all__ = [
    "GenerationFn",
    "SearcherExecutionPolicy",
    "fast_agentic_search",
    "run_searcher",
]
