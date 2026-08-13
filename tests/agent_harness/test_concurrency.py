from __future__ import annotations

import asyncio
import os
import signal
import threading
import time

import pytest

from agent_harness.concurrency import run_coroutine_sync


def test_run_coroutine_sync_interrupt_cancels_rollout_promptly() -> None:
    """Ctrl-C from a caller with a running loop must not wait out the rollout."""
    threads_before = set(threading.enumerate())
    started = threading.Event()
    cancelled = threading.Event()

    async def rollout() -> str:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "never reached"

    def interrupt_once() -> None:
        if not started.wait(timeout=10):
            return
        # The rollout is running, so the caller is blocked in the executor wait.
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGINT)

    async def notebook_cell() -> str:
        return run_coroutine_sync(rollout())

    watchdog = threading.Thread(target=interrupt_once, name="sigint-watchdog")
    loop = asyncio.new_event_loop()
    started_at = time.monotonic()
    try:
        watchdog.start()
        with pytest.raises(KeyboardInterrupt):
            loop.run_until_complete(notebook_cell())
        elapsed = time.monotonic() - started_at
    finally:
        watchdog.join(timeout=10)
        loop.close()

    assert elapsed < 5
    assert cancelled.is_set()
    assert not watchdog.is_alive()
    assert set(threading.enumerate()) - threads_before == set()


def test_run_coroutine_sync_no_loop_path_unchanged() -> None:
    async def value() -> str:
        return "rollout result"

    async def failure() -> str:
        raise ValueError("rollout blew up")

    assert run_coroutine_sync(value()) == "rollout result"
    with pytest.raises(ValueError, match="rollout blew up"):
        run_coroutine_sync(failure())


def test_run_coroutine_sync_worker_path_propagates_rollout_error() -> None:
    """The cancel path stays out of the way when the rollout itself raises."""
    threads_before = set(threading.enumerate())

    async def value() -> str:
        return "rollout result"

    async def failure() -> str:
        raise ValueError("rollout blew up")

    async def notebook_cell() -> None:
        assert run_coroutine_sync(value()) == "rollout result"
        with pytest.raises(ValueError, match="rollout blew up"):
            run_coroutine_sync(failure())

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(notebook_cell())
    finally:
        loop.close()

    assert set(threading.enumerate()) - threads_before == set()
