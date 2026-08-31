"""The stable async import surface, plus progress events for rollouts.

The loops are async-native, so the awaitable entry points here are re-exports
of the natives -- one import path that survives internal reshuffling. What
this module adds is ``stream_fast_agentic_search``: the injected seams are
wrapped with event-emitting decorators, the rollout runs as a task, and every
seam call surfaces as a start/complete event pair on an ordered queue.

Events are typed dataclasses: consumers ``match`` on the class (or read the
``type`` string every class carries) and get fields, not dict keys. Every
stream ends with exactly one terminal event: ``RolloutCompleted`` with the
entry point's return value, ``RolloutFailed``, or ``RolloutCancelled``;
failures additionally re-raise from the iterator. Every seam ``*Started`` is
followed by a matching ``*Completed`` whose ``ok`` records the outcome, except
when the rollout is cancelled mid-call.

Cancellation is ordinary task cancellation: cancelling the consumer cancels
the rollout task, whose in-flight seam awaits unwind natively -- the loops
re-raise anything that is not a plain ``Exception`` rather than degrading it
into model-visible tool feedback. With ``client=None`` the rollout resolves
the SDK client behind a worker-thread adapter; an in-flight SDK call finishes
on its thread, but the rollout still stops at the next await.

``completion_config`` reaches the wrapped generation fn as a per-call shallow
copy, so a caller applying ``apply_force_submit`` never mutates the
module-level config shared across rollouts.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

from agent_harness.agents.searcher import (
    FastAgenticSearchResult,
    fast_agentic_search,
    run_fast_agentic_search,
)
from agent_harness.config import STRICT_TOPK, HarnessTuning, MediaContentInput
from agent_harness.errors import error_kind
from agent_harness.execution_policy import run_searcher_async as run_searcher
from agent_harness.llm import AsyncGenerationFn
from agent_harness.retrieval import (
    AsyncRetrievalClient,
    AsyncStoreFiles,
    AsyncStores,
    FileListRequest,
    FileRetrieveRequest,
    GrepRequest,
    ListChunksRequest,
    MetadataFacetsRequest,
    SearchRequest,
)
from agent_harness.schemas import AnswerMode

__all__ = [
    "AgentEvent",
    "AsyncGenerationFn",
    "AsyncRetrievalClient",
    "AsyncStoreFiles",
    "AsyncStores",
    "GenerationCompleted",
    "GenerationStarted",
    "RetrievalCompleted",
    "RetrievalStarted",
    "RolloutCancelled",
    "RolloutCompleted",
    "RolloutFailed",
    "RolloutStarted",
    "fast_agentic_search",
    "run_fast_agentic_search",
    "run_searcher",
    "stream_fast_agentic_search",
]


@dataclass(frozen=True, slots=True)
class RolloutStarted:
    """First event of every stream."""

    type: ClassVar[str] = "rollout_started"
    seq: int
    entry: str
    store_identifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationStarted:
    """One model turn is leaving for the policy backend."""

    type: ClassVar[str] = "generation_started"
    seq: int
    turn: int
    message_count: int
    force_submit: bool
    forced_tool_name: str | None


@dataclass(frozen=True, slots=True)
class GenerationCompleted:
    """The matching completion for a ``GenerationStarted``."""

    type: ClassVar[str] = "generation_completed"
    seq: int
    turn: int
    message_count: int
    force_submit: bool
    forced_tool_name: str | None
    ok: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalStarted:
    """One retrieval seam call is executing.

    ``call`` names the seam method (``search``, ``grep``, ``metadata_facets``,
    ``list_chunks``, ``files.retrieve``, ``files.list``); ``query``/``top_k``/
    ``pattern`` are carried when the call has them. Payload bodies and filters
    stay off the event stream.
    """

    type: ClassVar[str] = "retrieval_started"
    seq: int
    call: str
    query: str | None = None
    top_k: int | None = None
    pattern: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalCompleted:
    """The matching completion for a ``RetrievalStarted``."""

    type: ClassVar[str] = "retrieval_completed"
    seq: int
    call: str
    query: str | None = None
    top_k: int | None = None
    pattern: str | None = None
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutCompleted:
    """Final event of every successful stream, carrying the result."""

    type: ClassVar[str] = "rollout_completed"
    seq: int
    result: FastAgenticSearchResult


@dataclass(frozen=True, slots=True)
class RolloutFailed:
    """Final event of a stream whose rollout raised; the error re-raises after it.

    ``error_kind`` is the ``provider``/``agent`` classification, so consumers
    that account rollouts can tell an unusable one from a genuine agent failure.
    """

    type: ClassVar[str] = "rollout_failed"
    seq: int
    error: str
    error_kind: str


@dataclass(frozen=True, slots=True)
class RolloutCancelled:
    """Final event of a stream whose rollout task was cancelled from outside.

    Consumer-initiated cancellation raises ``CancelledError`` in the consumer
    instead, which ends its iteration before any terminal event reaches it.
    """

    type: ClassVar[str] = "rollout_cancelled"
    seq: int


AgentEvent = (
    RolloutStarted
    | GenerationStarted
    | GenerationCompleted
    | RetrievalStarted
    | RetrievalCompleted
    | RolloutCompleted
    | RolloutFailed
    | RolloutCancelled
)


class _EventChannel:
    """Ordered event queue for one rollout, closed with a ``None`` sentinel."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._seq = itertools.count(1)

    def next_seq(self) -> int:
        return next(self._seq)

    def emit(self, event: AgentEvent) -> None:
        self.queue.put_nowait(event)

    def close(self) -> None:
        self.queue.put_nowait(None)


@dataclass(slots=True)
class _EmittingGeneration:
    """``AsyncGenerationFn`` decorator emitting one event pair per model turn."""

    channel: _EventChannel
    fn: AsyncGenerationFn
    _turns: itertools.count = field(default_factory=lambda: itertools.count(1))

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any:
        turn = next(self._turns)
        message_count = len(messages)
        forced_name = forced_tool_name if force_submit else None
        channel = self.channel
        channel.emit(
            GenerationStarted(
                seq=channel.next_seq(),
                turn=turn,
                message_count=message_count,
                force_submit=force_submit,
                forced_tool_name=forced_name,
            )
        )
        try:
            result = await self.fn(
                messages,
                tools=tools,
                completion_config=dict(completion_config),
                force_submit=force_submit,
                forced_tool_name=forced_tool_name,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            channel.emit(
                GenerationCompleted(
                    seq=channel.next_seq(),
                    turn=turn,
                    message_count=message_count,
                    force_submit=force_submit,
                    forced_tool_name=forced_name,
                    ok=False,
                    error=str(exc),
                )
            )
            raise
        channel.emit(
            GenerationCompleted(
                seq=channel.next_seq(),
                turn=turn,
                message_count=message_count,
                force_submit=force_submit,
                forced_tool_name=forced_name,
                ok=True,
            )
        )
        return result


def _retrieval_fields(request: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    query = getattr(request, "query", None)
    if isinstance(query, str):
        fields["query"] = query
    top_k = getattr(request, "top_k", None)
    if isinstance(top_k, int):
        fields["top_k"] = top_k
    pattern = getattr(request, "pattern", None)
    if isinstance(pattern, str):
        fields["pattern"] = pattern
    return fields


@dataclass(slots=True)
class _EmittingCall:
    """Shared emit-around-await behavior for the retrieval decorators."""

    channel: _EventChannel

    async def _run(self, call: str, request: Any, awaitable: Any) -> Any:
        fields = _retrieval_fields(request)
        channel = self.channel
        channel.emit(RetrievalStarted(seq=channel.next_seq(), call=call, **fields))
        try:
            result = await awaitable
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            channel.emit(
                RetrievalCompleted(
                    seq=channel.next_seq(), call=call, ok=False, error=str(exc), **fields
                )
            )
            raise
        channel.emit(RetrievalCompleted(seq=channel.next_seq(), call=call, ok=True, **fields))
        return result


@dataclass(slots=True)
class _EmittingFiles(_EmittingCall):
    files: AsyncStoreFiles

    async def retrieve(self, request: FileRetrieveRequest) -> Any:
        return await self._run("files.retrieve", request, self.files.retrieve(request))

    async def list(self, request: FileListRequest) -> Any:
        return await self._run("files.list", request, self.files.list(request))


@dataclass(slots=True)
class _EmittingStores(_EmittingCall):
    stores: AsyncStores

    async def search(self, request: SearchRequest) -> Any:
        return await self._run("search", request, self.stores.search(request))

    async def grep(self, request: GrepRequest) -> Any:
        return await self._run("grep", request, self.stores.grep(request))

    async def metadata_facets(self, request: MetadataFacetsRequest) -> Any:
        return await self._run("metadata_facets", request, self.stores.metadata_facets(request))

    async def list_chunks(self, request: ListChunksRequest) -> Any:
        return await self._run("list_chunks", request, self.stores.list_chunks(request))

    @property
    def files(self) -> _EmittingFiles:
        return _EmittingFiles(self.channel, self.stores.files)


@dataclass(slots=True)
class _EmittingRetrievalClient(_EmittingCall):
    """``AsyncRetrievalClient`` decorator emitting one event pair per call."""

    client: AsyncRetrievalClient

    @property
    def stores(self) -> _EmittingStores:
        return _EmittingStores(self.channel, self.client.stores)


def _terminal_emitter(channel: _EventChannel) -> Callable[[asyncio.Task[Any]], None]:
    """Build the rollout's done callback: one terminal event, then close.

    It runs on the loop outside the finished task, so cancellation cannot
    swallow the event, and done callbacks fire exactly once. Reading the
    exception here also keeps the loop from reporting it as never retrieved.
    """

    def emit_terminal(task: asyncio.Task[Any]) -> None:
        try:
            if task.cancelled():
                channel.emit(RolloutCancelled(seq=channel.next_seq()))
            elif (exc := task.exception()) is not None:
                channel.emit(
                    RolloutFailed(
                        seq=channel.next_seq(), error=str(exc), error_kind=error_kind(exc)
                    )
                )
        finally:
            # The close sentinel must reach the consumer even when building the
            # terminal event blows up (e.g. an exception whose __str__ raises);
            # a channel that never closes wedges the iterator forever.
            channel.close()

    return emit_terminal


async def _stream(
    make_rollout: Callable[[], Any],
    *,
    channel: _EventChannel,
    label: str,
    store_identifiers: Sequence[str],
) -> AsyncIterator[AgentEvent]:
    channel.emit(
        RolloutStarted(
            seq=channel.next_seq(),
            entry=label,
            store_identifiers=tuple(str(store_id) for store_id in store_identifiers),
        )
    )
    # Created here, on first iteration, so an unconsumed stream never leaks an
    # un-awaited rollout coroutine.
    task = asyncio.create_task(make_rollout(), name=f"agent-harness-aio-{label}")
    task.add_done_callback(_terminal_emitter(channel))
    try:
        while (event := await channel.queue.get()) is not None:
            yield event
        if task.cancelled():
            # RolloutCancelled already went out; re-raising would cancel the consumer.
            return
        result = task.result()
        yield RolloutCompleted(seq=channel.next_seq(), result=result)
    except (GeneratorExit, asyncio.CancelledError):
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        raise
    finally:
        if not task.done():
            task.cancel()


def stream_fast_agentic_search(
    user_text: str,
    *,
    store_identifiers: Sequence[str],
    generation_fn: AsyncGenerationFn,
    client: AsyncRetrievalClient | None = None,
    top_k: int | None = None,
    strict_top_k: bool = STRICT_TOPK,
    api_key: str | None = None,
    api_key_env: str | None = None,
    additional_instructions: str | None = None,
    include_prompt_snapshot: bool = False,
    media_content: MediaContentInput = None,
    tuning: HarnessTuning | None = None,
    as_of: date | None = None,
    answer_mode: AnswerMode = "none",
) -> AsyncIterator[AgentEvent]:
    """Drive the fast searcher, yielding progress events as it runs.

    The stream ends with exactly one terminal event: ``RolloutCompleted`` with
    the ``FastAgenticSearchResult`` under ``.result``, ``RolloutFailed`` whose
    error then re-raises from the iterator, or ``RolloutCancelled`` when the
    rollout task is cancelled from outside. Cancelling the consuming task
    cancels the rollout task and its in-flight seam awaits, raising
    ``CancelledError`` in the consumer before any terminal event reaches it.
    """
    channel = _EventChannel()

    def make_rollout() -> Any:
        return run_fast_agentic_search(
            user_text,
            store_identifiers=store_identifiers,
            top_k=top_k,
            strict_top_k=strict_top_k,
            client=_EmittingRetrievalClient(channel, client) if client is not None else None,
            api_key=api_key,
            api_key_env=api_key_env,
            additional_instructions=additional_instructions,
            include_prompt_snapshot=include_prompt_snapshot,
            media_content=media_content,
            generation_fn=_EmittingGeneration(channel, generation_fn),
            tuning=tuning,
            as_of=as_of,
            answer_mode=answer_mode,
        )

    return _stream(
        make_rollout,
        channel=channel,
        label="fast_agentic_search",
        store_identifiers=store_identifiers,
    )
