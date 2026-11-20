"""Agentic search helpers: the fast searcher and its tool surface over a Mixedbread store.

Importing this package must stay free of network, auth, and filesystem side
effects. Tracing and telemetry belong to callers at their own entry points.
"""

from .config import HarnessTuning
from .errors import ProviderFailure
from .llm import AsyncGenerationFn, GenerationFn, apply_force_submit
from .pipeline import (
    SearcherExecutionPolicy,
    fast_agentic_search,
    run_searcher,
)
from .retrieval import (
    AsyncRetrievalClient,
    AsyncStoreFiles,
    AsyncStores,
    RetrievalClient,
    SearchResults,
    StoreFiles,
    Stores,
)
from .schemas import AnswerMode
from .sync_api import (
    TOOL_FUNCTIONS,
    filter_chunks,
    get_chunk,
    get_chunks,
    grep,
    overview_search,
    prune_context,
    read_document,
    search_corpus,
    submit_ranking,
)
from .token_counter import ensure_token_counter
from .versions import (
    __version__,
    assert_compatible_versions,
    check_version_compatibility,
    current_version_manifest,
    extract_version_manifest,
)

__all__ = [
    "TOOL_FUNCTIONS",
    "AnswerMode",
    "AsyncGenerationFn",
    "AsyncRetrievalClient",
    "AsyncStoreFiles",
    "AsyncStores",
    "GenerationFn",
    "HarnessTuning",
    "ProviderFailure",
    "RetrievalClient",
    "SearchResults",
    "SearcherExecutionPolicy",
    "StoreFiles",
    "Stores",
    "__version__",
    "apply_force_submit",
    "assert_compatible_versions",
    "check_version_compatibility",
    "current_version_manifest",
    "ensure_token_counter",
    "extract_version_manifest",
    "fast_agentic_search",
    "filter_chunks",
    "get_chunk",
    "get_chunks",
    "grep",
    "overview_search",
    "prune_context",
    "read_document",
    "run_searcher",
    "search_corpus",
    "submit_ranking",
]
