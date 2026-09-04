"""Install the policy tokenizer as the harness token counter for rollouts.

The searcher's context-budget and payload-truncation logic funnel through
``agent_harness.config.count_text_tokens``. Left at its default it uses a
``chars/4`` heuristic that undercounts JSON-heavy retrieval payloads by ~25-35%,
so an "80k" truncated prompt can tokenize to ~100-110k real tokens and overflow
the inference deployment's ``max_model_len`` ("Prompt length ... exceeds maximum
context length"). Loading the real policy tokenizer once and installing it as the
counter makes those limits measure what the model actually sees. Every rollout
entry point does this through ``ensure_rollout_token_counter``; callers may also
install it up front with ``ensure_token_counter``.

Configuration:

- ``AGENT_HARNESS_TOKENIZER`` -- load this checkpoint instead of ``model``; or
  ``estimate`` to budget on the ``chars/4`` heuristic by choice: no tokenizer is
  looked up and the exactness requirement below is satisfied. For a hosted
  policy whose tokenizer is not available locally.
- ``AGENT_HARNESS_TOKEN_COUNTER_BACKEND`` -- ``gigatoken`` (default) or ``hf``.
  gigatoken must pass an exact token-ID parity check against the HF tokenizer
  before it is installed, and any failure falls back to HF -- the fast path is
  never allowed to change a count.
- ``AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER`` -- on by default: a rollout whose
  tokenizer could not be installed fails rather than budgeting with the
  ``chars/4`` estimate; set to ``0`` to allow the heuristic.

gigatoken reads ``tokenizer.json`` directly, so the checkpoint name is resolved
in this order: a local checkpoint directory, then a hub download through
``huggingface_hub`` (which arrives with transformers; absent it, resolution just
fails), then the HF tokenizer as the counter. A hub identifier is not a path on
disk, so without the download step the default backend would be inert for every
name that is not a local directory.
"""

from __future__ import annotations

import logging
import os
import threading
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import agent_harness.config as harness_config

_LOGGER = logging.getLogger(__name__)
_LOCK = threading.Lock()
# The AGENT_HARNESS_TOKENIZER value that opts into the chars/4 estimate outright.
ESTIMATE_TOKENIZER = "estimate"
# Model name whose tokenizer is installed (or whose load was attempted and
# failed); guards against reloading / re-warning on every rollout.
_RESOLVED_MODEL: str | None = None
# Why the last load failed, for the AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER error.
_LAST_LOAD_ERROR: str | None = None
_GIGATOKEN_VERSION_PIN = ((0, 10), (0, 11))
_GIGATOKEN_PARITY_TEXTS = (
    "",
    "hello world",
    "unicode: héllo → 中文测试 😀 done",
    "repeated whitespace run " + " " * 200,
    '<|im_start|>user\n<tool_response>\n{"hits":[1,2,3]}\n</tool_response>',
    '{"tool":"search_corpus","results":[{"text":"quoted \\"evidence\\""}]}',
)


class _ParityGatedGigatokenCounter:
    """HF-shaped fast counter with permanent, safe demotion on runtime error."""

    token_counter_mode = "exact-gigatoken"

    def __init__(self, fast: Any, hf: Any, count_only: Any = None) -> None:
        self._fast = fast
        self._hf = hf
        self._count_only = count_only

    def count_tokens(self, text: str) -> int:
        """Count through the raw encoder, which never materializes the ids list."""
        if self._count_only is not None:
            try:
                return len(self._count_only(text))
            except Exception as exc:
                _LOGGER.warning(
                    "agent_harness.token_counter: gigatoken count-only path failed (%s); "
                    "counting through the ids list.",
                    exc,
                )
                self._count_only = None
        return len(self.encode(text))

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        if self._fast is not None:
            try:
                return self._fast.encode(text, add_special_tokens=add_special_tokens)
            except Exception as exc:
                _LOGGER.warning(
                    "agent_harness.token_counter: gigatoken counter failed (%s); permanently "
                    "falling back to Hugging Face.",
                    exc,
                )
                self._fast = None
                self._count_only = None
                self.token_counter_mode = "exact-hf"
        return self._hf.encode(text, add_special_tokens=add_special_tokens)


def ensure_rollout_token_counter(model: str | None) -> None:
    """Install ``model``'s tokenizer at a rollout entry point, once per process.

    Every token budget in ``agent_harness.config`` measures through
    ``count_text_tokens``, so a rollout that starts without a tokenizer silently
    budgets JSON-heavy payloads with the chars/4 heuristic and can overflow the
    deployment's ``max_model_len``. By default such a rollout fails here instead
    of running mismeasured; AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER=0 allows it, as
    does choosing the estimate outright with AGENT_HARNESS_TOKENIZER=estimate.
    """
    ensure_token_counter(model)
    if (
        harness_config.TOKEN_COUNTER is not None
        or _estimate_requested()
        or not _require_exact_tokenizer()
    ):
        return
    override = _tokenizer_override()
    reason = _LAST_LOAD_ERROR or "no tokenizer load was attempted"
    msg = (
        "Exact token counting is required but no tokenizer could be installed "
        f"for model {model!r} (AGENT_HARNESS_TOKENIZER={override or 'unset'}): {reason}. "
        "Point AGENT_HARNESS_TOKENIZER at the policy checkpoint, or set "
        "AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER=0 to run on the chars/4 estimate."
    )
    raise RuntimeError(msg)


def _require_exact_tokenizer() -> bool:
    """Whether a rollout must fail rather than budget with the chars/4 estimate.

    Exact counting is the main path: every budget is calibrated in exact tokens,
    so the default is to refuse a mismeasured rollout. Set the variable to a
    falsey value ("0"/"false"/"no"/"off") to explicitly allow the heuristic.
    """
    raw = os.environ.get("AGENT_HARNESS_REQUIRE_EXACT_TOKENIZER", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", ""}


def _tokenizer_override() -> str:
    return os.environ.get("AGENT_HARNESS_TOKENIZER", "").strip()


def _estimate_requested() -> bool:
    """Whether AGENT_HARNESS_TOKENIZER asks for the chars/4 estimate by name."""
    return _tokenizer_override().lower() == ESTIMATE_TOKENIZER


def ensure_token_counter(model: str | None) -> None:
    """Install ``model``'s tokenizer as the harness token counter, once.

    Idempotent per model and safe under concurrency. On any failure to load the
    tokenizer (e.g. a served alias that is not a resolvable checkpoint) the
    harness keeps its char-based estimate; the failure is logged once, not per
    rollout. ``AGENT_HARNESS_TOKENIZER=estimate`` keeps the estimate by choice:
    no load is attempted and nothing is warned.
    """
    global _RESOLVED_MODEL  # noqa: PLW0603
    if not model or model == _RESOLVED_MODEL:
        return
    with _LOCK:
        if model == _RESOLVED_MODEL:
            return
        if _estimate_requested():
            _LOGGER.info(
                "agent_harness.token_counter: AGENT_HARNESS_TOKENIZER=%s; budgeting on the "
                "chars/4 estimate, no tokenizer loaded.",
                ESTIMATE_TOKENIZER,
            )
            _RESOLVED_MODEL = model
            return
        tokenizer = _load_tokenizer(model)
        if tokenizer is not None:
            harness_config.set_token_counter(tokenizer)
            _LOGGER.info(
                "agent_harness.token_counter: installed policy tokenizer for %r as token counter.",
                model,
            )
        # Record the attempt regardless so a failed load is not retried per rollout.
        _RESOLVED_MODEL = model


def _load_tokenizer(model: str) -> object | None:
    global _LAST_LOAD_ERROR  # noqa: PLW0603
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - transformers should be present
        _LOGGER.warning(
            "agent_harness.token_counter: transformers unavailable; keeping char-based token estimate (%s).",
            exc,
        )
        _LAST_LOAD_ERROR = f"transformers unavailable: {exc}"
        return None

    # The served model name may be an alias (e.g. an adapter name) that no
    # tokenizer resolves for; AGENT_HARNESS_TOKENIZER names
    # the base checkpoint to load instead.
    override = _tokenizer_override()
    last_error: Exception | None = None
    for name in [override, model] if override else [model]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
        except Exception:
            # Some tokenizers require custom code; retry once before giving up.
            try:
                tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            except Exception as exc:
                last_error = exc
                continue
        return _maybe_gigatoken_counter(tokenizer, name)
    _LOGGER.warning(
        "agent_harness.token_counter: could not load tokenizer for %r%s; keeping "
        "char-based token estimate (%s).",
        model,
        f" or AGENT_HARNESS_TOKENIZER={override!r}" if override else "",
        last_error,
    )
    _LAST_LOAD_ERROR = str(last_error)
    return None


def _tokenizer_json_bytes(tokenizer_path: str) -> bytes:
    """Read a checkpoint's ``tokenizer.json`` from disk, or fetch it from the hub."""
    local_path = Path(tokenizer_path) / "tokenizer.json"
    if local_path.is_file():
        return local_path.read_bytes()
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    return Path(hf_hub_download(tokenizer_path, "tokenizer.json")).read_bytes()


def _count_only_encoder(native: Any, hf_tokenizer: Any) -> Any | None:
    """gigatoken's raw encoder, which returns ids without building a Python list.

    Counting only ever takes the length, so the ``as_hf()`` wrapper's list costs
    one Python int per token for nothing (~3-4x on a full-size prompt). Gated on
    the same exact-parity rule as the ids path -- every probe's count must equal
    the Hugging Face tokenizer's -- and absent it, counting stays on ``encode``.
    """
    encode = getattr(native, "encode", None)
    if encode is None:
        return None
    try:
        for text in _GIGATOKEN_PARITY_TEXTS:
            if len(encode(text)) != len(hf_tokenizer.encode(text, add_special_tokens=False)):
                raise RuntimeError(f"count parity failed on probe of {len(text)} characters")
    except Exception as exc:
        _LOGGER.warning(
            "agent_harness.token_counter: gigatoken count-only path unavailable (%s); "
            "counting through the ids list.",
            exc,
        )
        return None
    return encode


def _maybe_gigatoken_counter(hf_tokenizer: Any, tokenizer_path: str) -> Any:
    """Use gigatoken unless explicitly disabled; install it only on exact parity."""
    backend = os.getenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", "gigatoken").strip().lower()
    if backend != "gigatoken":
        return hf_tokenizer
    try:
        import gigatoken  # noqa: PLC0415

        version = metadata.version("gigatoken")
        pair = tuple(int(part) for part in version.split(".")[:2])
        if not (_GIGATOKEN_VERSION_PIN[0] <= pair < _GIGATOKEN_VERSION_PIN[1]):
            raise RuntimeError(f"gigatoken {version} is outside >=0.10,<0.11")
        data = _tokenizer_json_bytes(tokenizer_path)
        native = gigatoken.Tokenizer.from_json(data)
        fast = native.as_hf()
        if len(fast) != len(hf_tokenizer):
            raise RuntimeError(f"vocabulary mismatch: gigatoken={len(fast)} HF={len(hf_tokenizer)}")
        for text in _GIGATOKEN_PARITY_TEXTS:
            expected = hf_tokenizer.encode(text, add_special_tokens=False)
            actual = fast.encode(text, add_special_tokens=False)
            if list(actual) != list(expected):
                raise RuntimeError(f"exact-id parity failed on probe of {len(text)} characters")
        _LOGGER.info(
            "agent_harness.token_counter: gigatoken counter parity passed (%d probes, sha256=%s).",
            len(_GIGATOKEN_PARITY_TEXTS),
            sha256(data).hexdigest(),
        )
        return _ParityGatedGigatokenCounter(
            fast,
            hf_tokenizer,
            _count_only_encoder(native, hf_tokenizer),
        )
    except Exception as exc:
        _LOGGER.warning(
            "agent_harness.token_counter: requested gigatoken counter unavailable or failed "
            "exact parity (%s); using Hugging Face.",
            exc,
        )
        return hf_tokenizer
