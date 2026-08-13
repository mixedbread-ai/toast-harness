"""Thread pooling that carries the submitter's contextvars into workers.

Per-rollout settings live in contextvars (see `config._MEDIA_CONTENT`), and a
plain `ThreadPoolExecutor` worker starts in an empty context, so a fan-out
inside a rollout would silently read defaults instead of that rollout's values.
Every thread that runs harness code on behalf of a rollout must carry the
rollout's context: through this executor, or through ``asyncio.to_thread``,
which copies the caller's context itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

__all__ = ["ContextThreadPoolExecutor", "run_coroutine_sync"]

# How long to wait for the worker to publish its task before giving up on cancelling.
_TASK_HANDOFF_TIMEOUT_S = 2.0


def run_coroutine_sync[ResultT](coro: Coroutine[Any, Any, ResultT]) -> ResultT:
    """Run a harness coroutine to completion from synchronous code.

    Outside a loop this is ``asyncio.run``. Inside one, the coroutine runs on
    its own loop on a worker thread and this call blocks -- the caller chose a
    sync entry point; async callers use the coroutines directly.

    Interrupting that blocked caller (Ctrl-C in a notebook or REPL) cancels the
    rollout task inside the worker loop; otherwise the executor's
    ``shutdown(wait=True)`` would hold the interrupt until the whole rollout
    finished. An in-flight sync seam call (``asyncio.to_thread``) still runs to
    its own completion, so propagation is bounded by one provider round-trip.
    Async callers should use ``agent_harness.aio``, where an interrupt is
    ordinary task cancellation under ``asyncio.Runner``'s SIGINT handling -- a
    shape no blocking facade can reproduce.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    handoff: dict[str, Any] = {}

    def run_on_worker_loop() -> ResultT:
        async def main() -> ResultT:
            handoff["task"] = asyncio.current_task()
            return await coro

        return asyncio.run(main())

    with ContextThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_on_worker_loop)
        try:
            return future.result()
        except BaseException:  # KeyboardInterrupt is the point; nothing else survives the wait.
            if not future.done() and not future.cancel():
                task = handoff.get("task")
                deadline = time.monotonic() + _TASK_HANDOFF_TIMEOUT_S
                while task is None and time.monotonic() < deadline:
                    # Submitted but not yet at its first step: the task exists in a moment.
                    time.sleep(0.005)
                    task = handoff.get("task")
                if task is not None:
                    with contextlib.suppress(RuntimeError):  # the worker loop already closed
                        task.get_loop().call_soon_threadsafe(task.cancel)
            raise


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose tasks run in a copy of the submitting context."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        # copy_context() must be evaluated here, in the submitting thread.
        return super().submit(copy_context().run, fn, *args, **kwargs)
