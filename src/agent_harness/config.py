"""Configuration for the fast-searcher pipeline."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, cast

AGENTIC_SEARCH_DEFAULT_K = 10  # final chunk count when strict top-k is requested without a top_k

SEARCH_CORPUS_TOP_K = 5
INITIAL_SEARCH_TOP_K = 5
# Per-chunk token cap (estimated) on the text fields of a single search-result
# chunk payload. Page-sized hits otherwise inflate every searcher turn into a
# 30-60k-token, prefix-cache-hostile prefill. Applies to search-style tools
# (search_corpus / grep / filter_chunks); read_document and get_chunks use the
# higher GET_CHUNKS_CHUNK_TOKEN_LIMIT, and every tool's call total is bounded
# by TOOL_CALL_PAYLOAD_TOKEN_BUDGET below.
SEARCH_CHUNK_TOKEN_LIMIT = 2000
OVERVIEW_SEARCH_TOP_K = 25
SEARCH_OVERFETCH_FACTOR = 5


@dataclass(frozen=True, slots=True)
class HarnessTuning:
    """Per-rollout overrides for knobs that otherwise come from env vars.

    ``None`` fields defer to the environment defaults. Set for one rollout via
    the ``tuning=`` parameter on the entry points (contextvar-scoped, so
    concurrent rollouts in one process each see their own values -- the same
    mechanism as ``media_content``). The env vars remain the deployment-wide
    defaults; this exists because a multi-tenant service cannot vary
    process-global, read-once configuration per request.
    """

    backend_top_k: int | None = None
    tool_choice: str | None = None
    keep_reasoning_history: bool | None = None
    searcher_max_rounds: int | None = None

    def __post_init__(self) -> None:
        if self.backend_top_k is not None and self.backend_top_k < SEARCH_CORPUS_TOP_K:
            msg = f"backend_top_k must be at least SEARCH_CORPUS_TOP_K ({SEARCH_CORPUS_TOP_K})"
            raise ValueError(msg)
        if self.searcher_max_rounds is not None and self.searcher_max_rounds < 1:
            msg = (
                "searcher_max_rounds must be at least 1: the final submit turn is one of the rounds"
            )
            raise ValueError(msg)


_TUNING: ContextVar[HarnessTuning | None] = ContextVar("HARNESS_TUNING", default=None)
_DEFAULT_TUNING = HarnessTuning()


def current_tuning() -> HarnessTuning:
    return _TUNING.get() or _DEFAULT_TUNING


@contextmanager
def tuning_setting(tuning: HarnessTuning | None) -> Iterator[HarnessTuning]:
    """Apply per-rollout tuning for the calling context, then restore it."""
    if tuning is None:
        yield current_tuning()
        return
    token = _TUNING.set(tuning)
    try:
        yield tuning
    finally:
        _TUNING.reset(token)


@functools.lru_cache(maxsize=1)
def _env_corpus_backend_top_k() -> int:
    value = int(
        os.environ.get(
            "AGENT_HARNESS_CORPUS_BACKEND_TOP_K",
            str(SEARCH_CORPUS_TOP_K * SEARCH_OVERFETCH_FACTOR),
        )
    )
    if value < SEARCH_CORPUS_TOP_K:
        msg = (
            "AGENT_HARNESS_CORPUS_BACKEND_TOP_K must be at least "
            f"SEARCH_CORPUS_TOP_K ({SEARCH_CORPUS_TOP_K})"
        )
        raise ValueError(msg)
    return value


def corpus_backend_top_k() -> int:
    """Candidates requested from Mixedbread for each search_corpus call.

    Kept separate from SEARCH_OVERFETCH_FACTOR: that factor also controls grep
    and overview_search, while latency benchmarks need to vary only the
    search_corpus provider request without changing any agent-visible limits.
    A per-rollout ``HarnessTuning.backend_top_k`` wins; otherwise the value is
    read from AGENT_HARNESS_CORPUS_BACKEND_TOP_K on first use and frozen for
    the process, so a bad deployment value is a configuration error raised at
    the first search rather than out of `import agent_harness`.
    """
    override = current_tuning().backend_top_k
    if override is not None:
        return override
    return _env_corpus_backend_top_k()


@functools.lru_cache(maxsize=1)
def search_rerank() -> bool | dict[str, Any]:
    """Server-side rerank option sent with every ``stores.search`` call.

    Parsed from AGENT_HARNESS_SEARCH_RERANK: unset or falsey ("0"/"false"/"no"/
    "off") -> False, today's behavior. A truthy flag ("1"/"true"/"yes"/"on") ->
    True, i.e. the server reranks each search's candidate pool with its default
    model. Any other value is taken as a reranker model name, e.g.
    ``mixedbread-ai/mxbai-rerank-v3-listwise``. Applied in ``search_raw`` — the
    single ``stores.search`` chokepoint — so it covers the initial search,
    search_corpus, and overview_search alike. Read from the environment on
    first use and frozen for the process, like ``_env_corpus_backend_top_k``
    (unlike ``corpus_backend_top_k``, no per-rollout tuning overrides it).
    """
    raw = os.environ.get("AGENT_HARNESS_SEARCH_RERANK", "").strip()
    if not raw or raw.lower() in {"0", "false", "no", "off"}:
        return False
    if raw.lower() in {"1", "true", "yes", "on"}:
        return True
    return {"model": raw}


FILTER_CHUNKS_DEFAULT_K = 10
FILTER_CHUNKS_MAX_K = 30
GREP_DEFAULT_K = 10
# grep always clips each returned chunk to a fixed window of context around
# EVERY occurrence of the matched pattern (merged when they overlap), regardless
# of SEARCH_CHUNK_TOKEN_LIMIT. The matched text is the whole reason grep returned
# the chunk, so we keep ~this many tokens of context on each side-of-match window
# rather than the head or a single first-match window.
GREP_MATCH_WINDOW_TOKENS = 100
FILTER_CHUNK_FILE_SCAN_LIMIT = 200
METADATA_REPRESENTATIVE_VALUES_PER_FIELD = 5
METADATA_INSPECT_OVERFETCH_FACTOR = 4
METADATA_INSPECT_MAX_INTERNAL_VALUES_PER_FIELD = 100
METADATA_TYPE_SAMPLE_TOP_K = 100

# The final submit turn is one of the rounds.
SEARCHER_MAX_ROUNDS = 4


def searcher_max_rounds() -> int:
    """Round budget for one searcher rollout, the final submit turn included.

    A per-rollout ``HarnessTuning.searcher_max_rounds`` wins; otherwise the
    module default applies.
    """
    override = current_tuning().searcher_max_rounds
    if override is not None:
        return override
    return SEARCHER_MAX_ROUNDS


# Per-turn cap on tool calls, prune_context included; depth stays bounded by
# SEARCHER_MAX_ROUNDS. Watch payload_truncated_count after raising: 8 full-size
# search payloads can cross SEARCHER_PRUNE_REMINDER_TOKENS in one round.
MAX_PARALLEL_TOOL_CALLS = 8
# Global cap on search_corpus throughput across ALL rollouts combined (thread-safe
# token bucket in search.py). Unrelated to MAX_PARALLEL_TOOL_CALLS above, which
# is the per-turn tool-call limit.
SEARCH_CORPUS_MAX_QPS = 100

# Token estimate at which a searcher prompt triggers the prune_context reminder:
# the model is asked to include prune_context among its next tool calls -- in
# parallel with searches. Failing to prune (or submit) on a round past this
# trigger is recorded on the iteration summary; context growth is otherwise
# bounded by truncation at SEARCHER_PROMPT_TOKEN_LIMIT and by
# SEARCHER_MAX_ROUNDS.
SEARCHER_PRUNE_REMINDER_TOKENS = 50_000
# Hard ceiling for one searcher prompt: tool payloads returned in a round are
# truncated so the next request stays below this. Must sit under the inference
# deployment's max_model_len with headroom for the chars/token estimate error.
SEARCHER_PROMPT_TOKEN_LIMIT = 100_000
# Aggregate ceiling on the payload of ONE retrieval tool call. Oversized calls
# are spread-truncated in presentation order (more budget to earlier results);
# nothing is deferred. Applies to grep, overview_search, and get_chunks.
TOOL_CALL_PAYLOAD_TOKEN_BUDGET = SEARCHER_PROMPT_TOKEN_LIMIT // 4
# A 5-chunk search payload measures ~18k EXACT tokens (the per-chunk clip's
# chars/4 conversion under-clips dense text), so search_corpus gets headroom
# above the shared budget rather than clipping the default result shape.
SEARCH_CORPUS_PAYLOAD_TOKEN_BUDGET = 30_000
# filter_chunks at its default k=10 measures ~36k exact tokens; its ceiling
# sits above that so the default-shaped call never clips.
FILTER_CHUNKS_PAYLOAD_TOKEN_BUDGET = 40_000
# read_document returns a window of up to 2*READ_DOCUMENT_MAX_WINDOW+1 chunks, so
# its per-call ceiling sits above TOOL_CALL_PAYLOAD_TOKEN_BUDGET. The window is the
# whole reason now: chunks[] is the only text carrier, so nothing is paid for twice.
READ_DOCUMENT_PAYLOAD_TOKEN_BUDGET = 32_000
# Ceiling on the combined tool payloads of ONE round, enforced by the round
# truncation pass; the effective bound is min(this, the remaining context headroom).
# At 96k the headroom is always the tighter bound in practice (the bootstrap
# alone keeps baselines above 4k), so this acts as a backstop. Env-overridable
# for A/B (100000 = headroom-only).
TURN_TOOL_PAYLOAD_TOKEN_BUDGET = int(os.environ.get("AGENT_HARNESS_TURN_PAYLOAD_BUDGET", "96000"))
# Smallest share the spread allocator gives any returned item; floor x max
# items fits every tool's budget (20x512=10.2k, 30x512=15.4k, 41x512=21k).
MIN_ALLOCATION_TOKENS = 512
# Per-chunk ceiling inside the per-call budget, so one page-sized chunk cannot
# consume the whole call. 4x SEARCH_CHUNK_TOKEN_LIMIT keeps the "fuller than
# search" promise while capping any single chunk at about a third of the budget.
GET_CHUNKS_CHUNK_TOKEN_LIMIT = SEARCH_CHUNK_TOKEN_LIMIT * 4
# maxItems on get_chunks.chunk_ids, stated in the tool description: without a
# published bound the model guesses one and has the whole call rejected.
GET_CHUNKS_MAX_CHUNK_IDS = 20
# Window radius cap on read_document.x: the chunk indices travel in the provider
# URL query, so an unbounded window fails as a 414 past a few hundred entries.
READ_DOCUMENT_MAX_WINDOW = 20
TOKEN_ESTIMATE_CHARS_PER_TOKEN = 4
# Counts tokens of serialized text for the context-budget and truncation
# decisions above. None = the chars/4 heuristic, which undercounts JSON-heavy
# retrieval payloads (~3 chars/token) by ~25-35%, so an "80k" prompt can be
# ~100-110k real tokens. Callers that know the policy tokenizer should install
# it via ``set_token_counter`` (read at call time via count_text_tokens) so
# pruning/truncation measures what the model will actually see.
TOKEN_COUNTER: Callable[[str], int] | None = None
# The object behind TOKEN_COUNTER, kept only so token_counter_mode can label it.
_TOKEN_COUNTER_SOURCE: Any | None = None


def count_text_tokens(text: str) -> int:
    """Tokens in ``text`` under ``TOKEN_COUNTER``, else the chars/4 heuristic."""
    counter = TOKEN_COUNTER
    if counter is not None:
        return counter(text)
    return max(1, len(text) // TOKEN_ESTIMATE_CHARS_PER_TOKEN)


def token_counter_mode() -> str:
    """Which counter is measuring the token budgets, for the rollout record.

    ``chars-heuristic`` means no tokenizer is installed and every budget in this
    module is an estimate; ``exact`` (or a counter's own finer label, e.g.
    ``exact-gigatoken``) means the counts are the policy tokenizer's own.
    """
    if TOKEN_COUNTER is None:
        return "chars-heuristic"
    return str(getattr(_TOKEN_COUNTER_SOURCE, "token_counter_mode", "exact"))


def set_token_counter(tokenizer: Any | None) -> None:
    """Install the policy tokenizer as the counter behind ``count_text_tokens``.

    ``tokenizer`` is any object exposing ``encode(text) -> Sequence`` (e.g. a
    Hugging Face tokenizer); pass ``None`` to reset to the chars/4 heuristic.
    Special tokens are excluded so the count reflects the serialized text
    content, matching how messages/payloads are measured for the budget and
    truncation decisions. Without this the heuristic undercounts JSON-heavy
    retrieval payloads and lets pruned/truncated prompts overflow the deployment
    ``max_model_len``. A tokenizer that can count without materializing the ids
    (``count_tokens(text) -> int``) is used through that instead.
    """
    global TOKEN_COUNTER, _TOKEN_COUNTER_SOURCE  # noqa: PLW0603
    _TOKEN_COUNTER_SOURCE = tokenizer
    if tokenizer is None:
        TOKEN_COUNTER = None
        return

    count_tokens = getattr(tokenizer, "count_tokens", None)
    if callable(count_tokens):

        def _count_only(text: str) -> int:
            return max(1, count_tokens(text))

        TOKEN_COUNTER = _count_only
        return

    def _count(text: str) -> int:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            # Tokenizers that do not accept ``add_special_tokens`` (or a
            # non-HF encoder) still expose a plain ``encode``.
            tokens = tokenizer.encode(text)
        return max(1, len(tokens))

    TOKEN_COUNTER = _count


type MediaContentMode = Literal["auto", "always", "never"]
type MediaContentInput = MediaContentMode | bool | None

MEDIA_CONTENT_MODES: tuple[MediaContentMode, ...] = ("auto", "always", "never")
STRICT_TOPK = False

# Per-rollout setting, not process state: callers run rollouts concurrently,
# each with its own media_content. A ContextVar gives every thread (and every
# asyncio task) its own value; contexts that never set it read the default.
# `config.MEDIA_CONTENT` reads as a plain attribute -- see __getattr__ --
# but nothing may assign it: an assignment would create a real module
# attribute that shadows __getattr__ for every thread. Use
# media_content_setting().
_MEDIA_CONTENT: ContextVar[MediaContentMode] = ContextVar("MEDIA_CONTENT", default="auto")


def __getattr__(name: str) -> Any:
    if name == "MEDIA_CONTENT":
        return _MEDIA_CONTENT.get()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def normalize_media_content(media_content: MediaContentInput) -> MediaContentMode:
    """Normalize public media_content values into the internal mode enum."""
    if media_content is None:
        return _MEDIA_CONTENT.get()
    if isinstance(media_content, bool):
        return "always" if media_content else "never"

    value = str(media_content).strip().lower()
    if value in MEDIA_CONTENT_MODES:
        return cast(MediaContentMode, value)
    if value in {"true", "1", "yes", "on"}:
        return "always"
    if value in {"false", "0", "no", "off"}:
        return "never"
    raise ValueError(
        f"media_content must be one of 'auto', 'always', or 'never'; got {media_content!r}"
    )


def include_media_content_for_chunk(
    chunk: Mapping[str, Any],
    *,
    mode: MediaContentInput = None,
) -> bool:
    """Return whether image content for a chunk should be sent to the agent."""
    media_mode = normalize_media_content(mode)
    if media_mode == "never":
        return False
    if media_mode == "always":
        return True
    return not (_has_text_value(chunk.get("ocr_text")) or _has_text_value(chunk.get("summary")))


def _has_text_value(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@contextmanager
def media_content_setting(media_content: MediaContentInput) -> Iterator[MediaContentMode]:
    """Override the media_content mode for the calling context, then restore it."""
    if media_content is None:
        yield _MEDIA_CONTENT.get()
        return
    token = _MEDIA_CONTENT.set(normalize_media_content(media_content))
    try:
        yield _MEDIA_CONTENT.get()
    finally:
        _MEDIA_CONTENT.reset(token)


AGENTIC_GENERATION_MAX_INVALID_RETRIES = 1
AGENTIC_FINAL_SUBMIT_MAX_INVALID_RETRIES = 2

TOOL_ONLY_CORRECTION_MESSAGE = (
    "Your previous response was invalid. Use tools only. Do not include plain text content."
)

FINAL_SUBMIT_CORRECTION_MESSAGE = (
    "Your previous final submission was invalid: {error}. "
    "Call submit_ranking again with only valid schema fields and valid chunk_id handles."
)

FINAL_ANSWER_CORRECTION_MESSAGE = (
    "Your previous response was invalid: {error}. "
    "Reply now with your final answer to the user query as plain text. Do not call any tools."
)

SEARCHER_AGENT_CONFIG: dict[str, Any] = {
    "model": "gpt-5.5",
    "reasoning_effort": "high",
    # Wire-level tool_choice. Defaults to "auto", which leaves decoding
    # unconstrained; "required" changes the sampling distribution on engines
    # that grammar-constrain tool calls (some replace the model's native tool
    # syntax with guided JSON). Overridable so callers can A/B it without a
    # code edit.
    "tool_choice": os.environ.get("AGENT_HARNESS_TOOL_CHOICE", "auto"),
    # Harness-side policy, never sent on the wire: a turn must be answered with
    # tool calls, and a non-tool turn is corrected/retried.
    "require_tool_calls": True,
    "timeout": 5000,
    "num_retries": 3,
    "parallel_tool_calls": True,
}


def _tuned_config(base: dict[str, Any]) -> dict[str, Any]:
    override = current_tuning().tool_choice
    if override is None:
        # Identity matters: with no override, callers receive the module-level
        # config object itself, exactly as before tuning existed.
        return base
    return {**base, "tool_choice": override}


def searcher_agent_config() -> dict[str, Any]:
    """``SEARCHER_AGENT_CONFIG`` with any per-rollout tuning applied."""
    return _tuned_config(SEARCHER_AGENT_CONFIG)
