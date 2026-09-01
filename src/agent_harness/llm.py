"""Provider-neutral generation contract for the fast-searcher loop.

The harness never bundles a model provider. ``GenerationFn`` is the seam: the
caller passes a callable that produces the next assistant message, and the
harness owns everything around it -- response and tool-call normalization, the
Responses API trace, and parsing of the terminal ``submit_ranking`` call (or,
under answer_mode="plain_text", the terminal prose turn).
Callers that talk to a Responses API endpoint can normalize with
``response_to_chat_completion``; callers that talk chat-completions already
return the shape the loop reads.

Three contracts an implementation must honor. It signals a turn it could not
make valid with ``failed_generation_response`` (what ``generation_failed``
reads); it decides whether a turn must be answered with tool calls via
``requires_tool_call``/``validate_required_tool_response`` -- that requirement
is harness-side policy deliberately separate from the wire ``tool_choice``;
and it applies ``apply_force_submit`` (exported from the package) when the
harness passes ``force_submit=True``, since the loop signals the forced turn
but does not rewrite the config itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

from .config import (
    AGENTIC_FINAL_SUBMIT_MAX_INVALID_RETRIES,
    FINAL_ANSWER_CORRECTION_MESSAGE,
    FINAL_SUBMIT_CORRECTION_MESSAGE,
    current_tuning,
)
from .schemas import AnsweredRankedChunkList, RankedChunkList

logger = logging.getLogger(__name__)

RESPONSES_API_TRACE_SCHEMA_VERSION = "openai_responses.v1"


class GenerationFn(Protocol):
    """Sync callable producing the next assistant message for an agent loop.

    The compatibility seam: the public sync entry points accept it and adapt it
    onto the async loops. New async callers implement ``AsyncGenerationFn``.
    """

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any | None: ...


@runtime_checkable
class AsyncGenerationFn(Protocol):
    """Async mirror of ``GenerationFn``: the seam the agent loops await on."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Awaitable[Any | None]: ...


@dataclass(frozen=True, slots=True)
class SyncGenerationAdapter:
    """``AsyncGenerationFn`` over a sync ``GenerationFn``.

    The sync callable runs on a worker thread so a loop awaiting it stays
    responsive and parallel turns still overlap.
    """

    generation_fn: GenerationFn

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any | None:
        return await asyncio.to_thread(
            self.generation_fn,
            messages,
            tools=tools,
            completion_config=dict(completion_config),
            force_submit=force_submit,
            forced_tool_name=forced_tool_name,
        )


def sync_generation_as_async(
    generation_fn: GenerationFn | None,
) -> AsyncGenerationFn | None:
    """Adapt an optional sync generation seam for the async loops."""
    if generation_fn is None:
        return None
    return SyncGenerationAdapter(generation_fn)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Prompt and completion token counts for one model turn."""

    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def of_response(cls, response: Any) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if not usage:
            return cls()
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", 0)
        return cls(int(input_tokens or 0), int(output_tokens or 0))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True)
class ForcedSubmission[SubmissionT]:
    """Outcome of a forced final turn: the parsed submission and the cost of every attempt.

    ``max_input_tokens`` and ``final_input_tokens`` are provider counts, zero
    when the provider reported none; the rollout prefers them over its own
    estimate of the forced prompt.
    """

    submission: SubmissionT | None
    usage: TokenUsage
    reasoning_tokens: int = 0
    max_input_tokens: int = 0
    final_input_tokens: int = 0


def require_generation_fn[GenerationT](
    generation_fn: GenerationT | None,
    *,
    parameter: str = "generation_fn",
) -> GenerationT:
    """Fail at the entry point rather than mid-round when generation is missing."""
    if generation_fn is None:
        msg = (
            f"{parameter} is required: agent_harness generates through an injected callable "
            "and ships no model provider of its own."
        )
        raise ValueError(msg)
    return generation_fn


def generation_failed(response: Any) -> bool:
    return response is None or bool(getattr(response, "failed_generation", False))


def completion_usage(response: Any) -> TokenUsage:
    return TokenUsage.of_response(response)


def completion_reasoning_tokens(response: Any) -> int:
    """Return thinking tokens for one response, or 0 when the API omits them.

    Reasoning tokens are a subset of the output tokens already counted by
    ``completion_usage``; they are tracked separately so thinking-level settings
    (``reasoning_effort`` / ``enable_thinking``) can be measured rather than
    inferred from output totals. Chat-completions reports them under
    ``completion_tokens_details``, the Responses API under
    ``output_tokens_details``.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return 0
    for attr in ("completion_tokens_details", "output_tokens_details"):
        details = getattr(usage, attr, None)
        if details is None:
            continue
        value = getattr(details, "reasoning_tokens", None)
        if value is None and isinstance(details, Mapping):
            value = details.get("reasoning_tokens")
        if value is not None:
            return int(value or 0)
    return 0


def chat_content_from_message_content(content: Any) -> Any:
    """Normalize message content into valid chat-completions content parts.

    Media parts built by the harness carry internal ``chunk_id``/``document_id``
    keys (used for redaction) and may use Responses-style ``input_text``/
    ``input_image`` types; both would be rejected or silently dropped by strict
    OpenAI-compatible chat endpoints, so this strips them at the wire boundary.
    """
    if not isinstance(content, list):
        return content

    parts: list[dict[str, Any]] = []
    for item in content:
        converted = _chat_content_part(item)
        if converted is not None:
            parts.append(converted)
    return parts if parts else ""


def _chat_content_part(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if not isinstance(item, dict):
        return {"type": "text", "text": json.dumps(item, ensure_ascii=False, default=str)}

    item_type = str(item.get("type") or "").strip()
    if item_type in {"input_text", "text"}:
        text = item.get("text")
        if text is None:
            return None
        return {"type": "text", "text": str(text)}

    if item_type in {"input_image", "image_url"}:
        image_url = item.get("image_url")
        detail = item.get("detail")
        if isinstance(image_url, Mapping):
            if detail is None:
                detail = image_url.get("detail")
            image_url = image_url.get("url") or image_url.get("image_url")
        if not (isinstance(image_url, str) and image_url.strip()):
            return None
        source: dict[str, Any] = {"url": image_url.strip()}
        if isinstance(detail, str) and detail.strip():
            source["detail"] = detail.strip()
        return {"type": "image_url", "image_url": source}

    return {"type": "text", "text": json.dumps(item, ensure_ascii=False, default=str)}


def response_to_chat_completion(
    response: Any,
    *,
    request: Mapping[str, Any] | None = None,
) -> Any:
    tool_calls = []
    content_parts: list[str] = []
    full_response_output = list(getattr(response, "output", None) or [])
    response_output = [item for item in full_response_output if not _is_reasoning_output_item(item)]

    for item in response_output:
        item_type = _value(item, "type")
        if item_type == "function_call":
            call_id = str(_value(item, "call_id") or _value(item, "id") or "")
            tool_calls.append(
                SimpleNamespace(
                    id=call_id,
                    type="function",
                    function=SimpleNamespace(
                        name=str(_value(item, "name") or ""),
                        arguments=str(_value(item, "arguments") or "{}"),
                    ),
                )
            )
            continue
        if item_type == "message":
            content_parts.extend(_message_output_text(item))

    content = "\n".join(part for part in content_parts if part).strip() or None
    jsonable_response_output = [_jsonable_response_item(item) for item in response_output]
    jsonable_full_response_output = [_jsonable_response_item(item) for item in full_response_output]
    raw_response = _sanitized_raw_response(response, jsonable_full_response_output)
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        response_output=jsonable_response_output,
        responses_output=jsonable_full_response_output,
    )
    completion = SimpleNamespace(
        id=getattr(response, "id", None),
        choices=[SimpleNamespace(message=message)],
        usage=getattr(response, "usage", None),
        output=response_output,
        raw_response=raw_response,
    )
    if request is not None:
        completion.responses_api = responses_api_turn(
            request=request,
            response=raw_response,
        )
    return completion


def responses_api_turn(
    *,
    request: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one replayable OpenAI Responses API request/response pair."""
    return {
        "request": _jsonable_runtime_object(request),
        "response": _jsonable_runtime_object(response),
    }


def responses_api_trace_payload(turns: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": RESPONSES_API_TRACE_SCHEMA_VERSION,
        "api": "responses",
        "turns": _jsonable_runtime_object(turns),
    }


def extend_responses_api_trace(
    trace: list[dict[str, Any]],
    response: Any,
    **metadata: Any,
) -> None:
    """Append Responses API turns from ``response`` to ``trace`` with metadata."""
    for turn in response_responses_api_turns(response):
        trace_turn = deepcopy(turn)
        cleaned_metadata = {
            str(key): _jsonable_runtime_object(value)
            for key, value in metadata.items()
            if value is not None
        }
        if cleaned_metadata:
            trace_turn["metadata"] = {
                **dict(trace_turn.get("metadata") or {}),
                **cleaned_metadata,
            }
        trace.append(trace_turn)


def response_responses_api_turns(response: Any) -> list[dict[str, Any]]:
    turns = getattr(response, "responses_api_turns", None)
    if isinstance(turns, list):
        return [deepcopy(turn) for turn in turns if isinstance(turn, Mapping)]
    turn = getattr(response, "responses_api", None)
    if isinstance(turn, Mapping):
        return [deepcopy(dict(turn))]
    return []


def _message_output_text(message: Any) -> list[str]:
    parts: list[str] = []
    for content_item in _value(message, "content") or []:
        content_type = _value(content_item, "type")
        if content_type in {"output_text", "text"}:
            text = _value(content_item, "text")
            if text:
                parts.append(str(text))
    return parts


def _is_reasoning_output_item(item: Any) -> bool:
    item_type = str(_value(item, "type") or "").strip().lower()
    return item_type == "reasoning" or item_type.startswith("reasoning_")


def _jsonable_response_item(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    if isinstance(item, dict):
        return deepcopy(item)
    return {}


def _sanitized_raw_response(
    response: Any,
    jsonable_response_output: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_response = getattr(response, "raw_response", None)
    if isinstance(raw_response, dict):
        sanitized = deepcopy(raw_response)
        sanitized["output"] = jsonable_response_output
        return sanitized

    sanitized: dict[str, Any] = {
        "id": getattr(response, "id", None),
        "output": jsonable_response_output,
    }
    usage = getattr(response, "usage", None)
    if usage is not None:
        sanitized["usage"] = _jsonable_runtime_object(usage)
    error = getattr(response, "error", None)
    if error is not None:
        sanitized["error"] = _jsonable_runtime_object(error)
    return sanitized


def _jsonable_runtime_object(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_runtime_object(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_runtime_object(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable_runtime_object(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _value(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _keep_reasoning_history() -> bool:
    """Gate for round-tripping reasoning_content into the message history.

    Off by default; per-rollout ``HarnessTuning.keep_reasoning_history`` wins,
    else KEEP_REASONING_HISTORY=1."""
    override = current_tuning().keep_reasoning_history
    if override is not None:
        return override
    return os.environ.get("KEEP_REASONING_HISTORY", "").strip().lower() in {"1", "true", "yes"}


def response_message_to_dict(message: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"role": "assistant", "content": message.content}
    # When enabled, round-trip the reasoning so the assistant turn re-enters
    # history exactly as sampled. Only enable this together with a chat template
    # that keeps past thinking in the rendered history: under a template that
    # strips it, the re-rendered turn would differ from what the model produced.
    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning and _keep_reasoning_history():
        result["reasoning_content"] = reasoning
    if message.tool_calls:
        result["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return result


def append_response_message(
    messages: list[dict[str, Any]],
    response: Any | None,
) -> bool:
    """Append a model response to conversation history when one is available."""
    choices = list(getattr(response, "choices", None) or [])
    message = getattr(choices[0], "message", None) if choices else None
    if message is None:
        return False
    messages.append(response_message_to_dict(message))
    return True


def append_tool_error_messages(
    messages: list[dict[str, Any]],
    response: Any | None,
    error: str,
) -> None:
    """Close rejected tool calls before the next user correction."""
    choices = list(getattr(response, "choices", None) or [])
    message = getattr(choices[0], "message", None) if choices else None
    for tool_call in response_tool_calls(message):
        tool_call_id = getattr(tool_call, "id", None)
        if isinstance(tool_call_id, str) and tool_call_id:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": _compact_error(error)}),
                }
            )


def response_tool_calls(message: Any) -> list[Any]:
    return list((message.tool_calls if message else None) or [])


def validate_required_tool_response(
    message: Any,
    tool_calls: list[Any],
    completion_config: dict[str, Any],
) -> str | None:
    if not requires_tool_call(completion_config):
        return None
    if not tool_calls:
        return "required tool call missing"
    content = message.content if message else None
    # Prose narration beside tool calls is allowed; only a missing tool call violates.
    if content is not None and not isinstance(content, str):
        return "non-text content is not allowed in agentic tool responses"
    return None


def requires_tool_call(completion_config: Mapping[str, Any]) -> bool:
    """Whether this turn must be answered with tool calls (harness-side policy).

    Deliberately separate from the wire ``tool_choice``: keeping the requirement
    as its own config key lets a caller send ``tool_choice: auto`` (leaving
    decoding unconstrained) while the harness still validates and corrects
    non-tool turns.
    """
    if completion_config.get("require_tool_calls"):
        return True
    return wire_forces_tool_call(completion_config.get("tool_choice"))


def wire_forces_tool_call(tool_choice: Any) -> bool:
    """Whether a wire ``tool_choice`` value constrains decoding to a tool call."""
    if tool_choice == "required":
        return True
    return isinstance(tool_choice, dict) and tool_choice.get("type") == "function"


def apply_force_submit(config: dict[str, Any], forced_tool_name: str) -> dict[str, Any]:
    """Mark ``config`` as a forced-submission turn, in place.

    The requirement is always recorded as harness-side policy. The wire
    ``tool_choice`` is only pinned to the named function when the caller is
    already forcing on the wire (``tool_choice: required``): under the default
    ``auto`` this keeps decoding unconstrained (some engines' forced/named
    paths swap the model's native tool syntax for guided JSON). The textual
    force-submit instruction carries the turn in both cases.
    """
    config["require_tool_calls"] = True
    config["parallel_tool_calls"] = False
    if not wire_forces_tool_call(config.get("tool_choice")):
        return config
    submit_choice = {"type": "function", "function": {"name": forced_tool_name}}
    config["tool_choice"] = submit_choice
    return config


def failed_generation_response(
    response: Any,
    responses_api_turns: list[dict[str, Any]],
    validation_error: str,
) -> Any:
    return SimpleNamespace(
        id=getattr(response, "id", None),
        choices=[],
        usage=getattr(response, "usage", None),
        failed_generation=True,
        validation_error=validation_error,
        responses_api_turns=deepcopy(responses_api_turns),
    )


def parse_ranking(arguments: str, *, require_answer: bool = False) -> RankedChunkList:
    """Parse a ``submit_ranking`` payload.

    ``require_answer`` (answer_mode="submit_ranking") parses the model that
    carries the ``answer`` field and rejects a payload that omits it.
    """
    ranking_model = AnsweredRankedChunkList if require_answer else RankedChunkList
    ranking = ranking_model.model_validate_json(arguments)
    if not ranking.ranking_strategy:
        raise ValueError("submit_ranking missing ranking_strategy")
    if require_answer and not ranking.answer:
        raise ValueError("submit_ranking missing answer")
    return ranking


async def force_ranking(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    completion_config: dict[str, Any],
    validate: Callable[[RankedChunkList], None] | None = None,
    require_answer: bool = False,
    responses_trace: list[dict[str, Any]] | None = None,
    response_trace_metadata: Mapping[str, Any] | None = None,
    generation_fn: AsyncGenerationFn,
    on_invalid_attempt: Callable[[int, str], None] | None = None,
) -> ForcedSubmission[RankedChunkList]:
    """Force a ``submit_ranking`` call, retrying on invalid submissions.

    ``on_invalid_attempt`` is called with ``(attempt, validation_error)`` for each
    rejected submission that is followed by a retry, so callers can record the
    failure on their tool trace. The terminal failure (retries exhausted) is not
    reported here -- callers already trace that as the forced call's own outcome.
    """
    return await _retry_forced_turn(
        messages,
        tools=tools,
        completion_config=completion_config,
        trace_phase="force_submit",
        forcing={"force_submit": True, "forced_tool_name": "submit_ranking"},
        submission_label="ranking",
        parse_response=lambda response: _parse_forced_submission(
            response,
            expected_tool_name="submit_ranking",
            parse_arguments=lambda arguments: parse_ranking(
                arguments, require_answer=require_answer
            ),
            validate=validate,
        ),
        correction_message=_final_submit_correction_message,
        responses_trace=responses_trace,
        response_trace_metadata=response_trace_metadata,
        generation_fn=generation_fn,
        on_invalid_attempt=on_invalid_attempt,
    )


async def force_answer(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    completion_config: dict[str, Any],
    responses_trace: list[dict[str, Any]] | None = None,
    response_trace_metadata: Mapping[str, Any] | None = None,
    generation_fn: AsyncGenerationFn,
    on_invalid_attempt: Callable[[int, str], None] | None = None,
) -> ForcedSubmission[str]:
    """Force a plain-text final answer, retrying on tool-call or empty turns.

    The answer_mode="plain_text" counterpart of ``force_ranking``: there is no
    tool to force, so the turn is an ordinary generation whose response must
    carry text content and no tool calls. See ``force_ranking`` for the
    ``on_invalid_attempt`` contract.
    """
    return await _retry_forced_turn(
        messages,
        tools=tools,
        completion_config=completion_config,
        trace_phase="force_answer",
        forcing={},
        submission_label="answer",
        parse_response=_parse_forced_answer_response,
        correction_message=_final_answer_correction_message,
        responses_trace=responses_trace,
        response_trace_metadata=response_trace_metadata,
        generation_fn=generation_fn,
        on_invalid_attempt=on_invalid_attempt,
    )


@dataclass(slots=True)
class _ForcedTurnLedger:
    """The cost of every attempt of one forced tail, in ``ForcedSubmission`` terms."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    reasoning_tokens: int = 0
    max_input_tokens: int = 0
    final_input_tokens: int = 0

    def record(self, response: Any) -> None:
        turn = TokenUsage.of_response(response)
        self.usage = self.usage + turn
        self.reasoning_tokens += completion_reasoning_tokens(response)
        self.max_input_tokens = max(self.max_input_tokens, turn.input_tokens)
        self.final_input_tokens = turn.input_tokens

    def close[SubmissionT](self, submission: SubmissionT | None) -> ForcedSubmission[SubmissionT]:
        return ForcedSubmission(
            submission,
            self.usage,
            reasoning_tokens=self.reasoning_tokens,
            max_input_tokens=self.max_input_tokens,
            final_input_tokens=self.final_input_tokens,
        )


async def _retry_forced_turn[SubmissionT](
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    completion_config: dict[str, Any],
    trace_phase: str,
    forcing: Mapping[str, Any],
    submission_label: str,
    parse_response: Callable[[Any], tuple[SubmissionT | None, str | None]],
    correction_message: Callable[[str], dict[str, str]],
    responses_trace: list[dict[str, Any]] | None,
    response_trace_metadata: Mapping[str, Any] | None,
    generation_fn: AsyncGenerationFn,
    on_invalid_attempt: Callable[[int, str], None] | None,
) -> ForcedSubmission[SubmissionT]:
    """Retry the episode's last turn until ``parse_response`` accepts it.

    The two forced tails differ on three axes and share everything else, so
    those three are the parameters: ``forcing`` carries the generation kwargs
    that pin the turn to a tool call and is empty when the final turn is prose,
    ``trace_phase`` labels it, and ``parse_response`` decides what a valid
    submission looks like. Rejected turns are closed out the same way either
    way -- ``append_tool_error_messages`` is a no-op on a response that carried
    no tool calls, and answers the stray ones when a prose turn wrongly did.

    Every attempt gets its own copy of ``completion_config``: a generation seam
    that applies ``apply_force_submit`` in place must not leak the forced
    policy into later turns.
    """
    ledger = _ForcedTurnLedger()
    for attempt in range(AGENTIC_FINAL_SUBMIT_MAX_INVALID_RETRIES + 1):
        response = await generation_fn(
            messages,
            tools=tools,
            completion_config=dict(completion_config),
            **forcing,
        )
        if responses_trace is not None:
            extend_responses_api_trace(
                responses_trace,
                response,
                phase=trace_phase,
                **forcing,
                attempt=attempt + 1,
                **dict(response_trace_metadata or {}),
            )
        ledger.record(response)

        submission, validation_error = parse_response(response)
        if validation_error is None:
            return ledger.close(submission)

        logger.warning("Invalid forced %s: %s", submission_label, validation_error)
        append_response_message(messages, response)
        if attempt < AGENTIC_FINAL_SUBMIT_MAX_INVALID_RETRIES:
            append_tool_error_messages(messages, response, validation_error)
            messages.append(correction_message(validation_error))
            if on_invalid_attempt is not None:
                on_invalid_attempt(attempt + 1, validation_error)

    return ledger.close(None)


def _parse_forced_submission[SubmissionT](
    response: Any | None,
    *,
    expected_tool_name: str,
    parse_arguments: Callable[[str], SubmissionT],
    validate: Callable[[SubmissionT], None] | None,
) -> tuple[SubmissionT | None, str | None]:
    """Parse a forced tool turn: exactly one call to the expected tool, valid."""
    tool_call, validation_error = _single_tool_call(response, expected_tool_name)
    if validation_error is not None or tool_call is None:
        return None, validation_error or f"{expected_tool_name} tool call missing"
    try:
        submission = parse_arguments(tool_call.function.arguments)
        if validate is not None:
            validate(submission)
        return submission, None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _parse_forced_answer_response(response: Any | None) -> tuple[str | None, str | None]:
    """Parse a forced prose turn: text content, and no tool calls."""
    choices = list(getattr(response, "choices", None) or [])
    message = getattr(choices[0], "message", None) if choices else None
    if message is None:
        return None, "final answer missing"
    if response_tool_calls(message):
        return None, "tool calls are not allowed on the final answer turn"
    content = getattr(message, "content", None)
    answer = content.strip() if isinstance(content, str) else ""
    if not answer:
        return None, "final answer text missing"
    return answer, None


def _single_tool_call(
    response: Any | None,
    expected_tool_name: str,
) -> tuple[Any | None, str | None]:
    message = response.choices[0].message if response is not None and response.choices else None
    tool_calls = response_tool_calls(message)
    if not tool_calls:
        return None, f"{expected_tool_name} tool call missing"
    if len(tool_calls) != 1:
        return None, f"exactly one {expected_tool_name} tool call required, got {len(tool_calls)}"

    tool_call = tool_calls[0]
    if tool_call.function.name != expected_tool_name:
        return None, f"expected {expected_tool_name} tool call, got {tool_call.function.name}"
    return tool_call, None


def _final_submit_correction_message(error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": FINAL_SUBMIT_CORRECTION_MESSAGE.format(error=_compact_error(error)),
    }


def _final_answer_correction_message(error: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": FINAL_ANSWER_CORRECTION_MESSAGE.format(error=_compact_error(error)),
    }


def _compact_error(error: str, max_length: int = 1200) -> str:
    error = " ".join(str(error).split())
    if len(error) <= max_length:
        return error
    return error[: max_length - 3] + "..."
