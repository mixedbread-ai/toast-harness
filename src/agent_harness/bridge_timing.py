"""Opt-in, content-free timing for the live searcher/model bridge.

Enable with ``AGENT_HARNESS_BRIDGE_TIMING=1``.  Events are written as one-line
JSON after the ``BRIDGE_TIMING`` marker so benchmark logs can be aggregated
without recording prompts, tool arguments, results, or credentials.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from typing import Any


def enabled() -> bool:
    return os.getenv("AGENT_HARNESS_BRIDGE_TIMING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def emit(event: str, **fields: Any) -> None:
    if not enabled():
        return
    payload = {
        "event": event,
        "thread": threading.current_thread().name,
        "wall_time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        **{key: value for key, value in fields.items() if value is not None},
    }
    # Host processes often configure module loggers above INFO, so use one
    # explicit, flushed line when this opt-in profiler is enabled.  The payload contains
    # only durations and shape counts, never prompt/tool content or secrets.
    line = "BRIDGE_TIMING " + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    with suppress(OSError):
        print(line, flush=True)
    # A host process may capture worker stdout without forwarding it to its
    # own log.  An explicit file path keeps live profiling observable;
    # opening with append for each short line also makes independent c>1 worker
    # processes safe without a shared Python lock.
    output_path = os.getenv("AGENT_HARNESS_BRIDGE_TIMING_FILE", "").strip()
    if output_path:
        try:
            with open(output_path, "a", encoding="utf-8") as output:
                output.write(line + "\n")
        except OSError:
            # An optional profiler path must not change application behavior.
            pass


def message_shape(messages: list[dict[str, Any]]) -> Mapping[str, int]:
    """Cheap payload-size counters; deliberately does not serialize content."""
    content_chars = 0
    tool_calls = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            content_chars += len(content)
        elif isinstance(content, list):
            content_chars += sum(
                len(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            )
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            tool_calls += len(calls)
    return {
        "message_count": len(messages),
        "message_content_chars": content_chars,
        "historical_tool_call_count": tool_calls,
    }
