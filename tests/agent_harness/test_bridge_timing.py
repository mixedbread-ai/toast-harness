from __future__ import annotations

import json

from agent_harness.bridge_timing import emit


def test_emit_can_append_to_explicit_profiler_file(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "bridge.jsonl"
    monkeypatch.setenv("AGENT_HARNESS_BRIDGE_TIMING", "1")
    monkeypatch.setenv("AGENT_HARNESS_BRIDGE_TIMING_FILE", str(output))

    emit("unit_test", duration_ms=1.25, message_count=3)

    stdout_line = capsys.readouterr().out.strip()
    file_line = output.read_text(encoding="utf-8").strip()
    assert file_line == stdout_line
    marker, raw_payload = file_line.split(" ", 1)
    assert marker == "BRIDGE_TIMING"
    payload = json.loads(raw_payload)
    assert payload["event"] == "unit_test"
    assert payload["duration_ms"] == 1.25
    assert payload["message_count"] == 3


def test_emit_ignores_unwritable_optional_profiler_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_HARNESS_BRIDGE_TIMING", "1")
    monkeypatch.setenv(
        "AGENT_HARNESS_BRIDGE_TIMING_FILE",
        str(tmp_path / "missing-directory" / "bridge.jsonl"),
    )

    emit("unit_test", duration_ms=1.25)


def test_emit_ignores_closed_stdout(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_HARNESS_BRIDGE_TIMING", "1")

    def closed_stdout(*args, **kwargs) -> None:
        del args, kwargs
        raise BrokenPipeError("closed")

    monkeypatch.setattr("builtins.print", closed_stdout)
    emit("unit_test", duration_ms=1.25)
