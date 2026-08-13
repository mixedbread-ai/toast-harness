from __future__ import annotations

import threading

import pytest

from agent_harness import config
from agent_harness.agents.shared import media_messages_for_payload
from agent_harness.concurrency import ContextThreadPoolExecutor
from agent_harness.llm import chat_content_from_message_content
from agent_harness.references import ReferenceRegistry
from agent_harness.search import agent_file_payload, serialize_agent_chunk


def _image_chunk(**overrides: object) -> dict[str, object]:
    chunk: dict[str, object] = {
        "store_id": "store-a",
        "file_id": "file-a",
        "chunk_index": 0,
        "chunk_id": "c1",
        "document_id": "d1",
        "image_url": "https://example.com/image.png",
    }
    chunk.update(overrides)
    return chunk


def test_media_content_setting_normalizes_modes_and_booleans() -> None:
    previous = config.MEDIA_CONTENT

    with config.media_content_setting("always") as mode:
        assert mode == "always"
        assert config.MEDIA_CONTENT == "always"

    with config.media_content_setting(True) as mode:
        assert mode == "always"

    with config.media_content_setting(False) as mode:
        assert mode == "never"

    assert previous == config.MEDIA_CONTENT
    with pytest.raises(ValueError, match="media_content must be one of"):
        config.normalize_media_content("sometimes")  # type: ignore[arg-type]


def test_serialize_agent_chunk_respects_media_content_modes() -> None:
    refs = ReferenceRegistry()

    with config.media_content_setting("auto"):
        assert "image_url" in serialize_agent_chunk(_image_chunk(), refs=refs)
        assert "image_url" not in serialize_agent_chunk(
            _image_chunk(ocr_text="readable text"),
            refs=refs,
        )
        assert "image_url" not in serialize_agent_chunk(
            _image_chunk(summary="visual summary"),
            refs=refs,
        )

    with config.media_content_setting("always"):
        assert "image_url" in serialize_agent_chunk(
            _image_chunk(summary="visual summary"),
            refs=refs,
        )

    with config.media_content_setting("never"):
        assert "image_url" not in serialize_agent_chunk(_image_chunk(), refs=refs)


def test_agent_visible_payloads_hide_backend_store_refs() -> None:
    refs = ReferenceRegistry()
    chunk_payload = serialize_agent_chunk(_image_chunk(), refs=refs)

    assert chunk_payload["chunk_index"] == 0
    assert "file_id" not in chunk_payload
    assert "store_id" not in chunk_payload
    assert agent_file_payload(
        {
            "document_id": "d1",
            "file_id": "file-a",
            "store_id": "store-a",
            "filename": "creative.png",
        }
    ) == {"document_id": "d1", "filename": "creative.png"}


def test_media_messages_for_payload_auto_only_attaches_images_without_text_surrogate() -> None:
    payload = {
        "results": [
            _image_chunk(chunk_id="c1", document_id="d1", image_url="https://example.com/1.png"),
            _image_chunk(
                chunk_id="c2",
                document_id="d2",
                image_url="https://example.com/2.png",
                ocr_text="sale",
            ),
            _image_chunk(
                chunk_id="c3",
                document_id="d3",
                image_url="https://example.com/3.png",
                summary="image summary",
            ),
        ],
    }

    with config.media_content_setting("auto"):
        messages = media_messages_for_payload(payload)

    image_parts = [
        part
        for message in messages
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert image_parts == [
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/1.png"},
            "chunk_id": "c1",
            "document_id": "d1",
        }
    ]

    with config.media_content_setting("never"):
        assert media_messages_for_payload(payload) == []


def test_media_messages_survive_chat_wire_sanitization() -> None:
    payload = {
        "results": [
            _image_chunk(chunk_id="c1", document_id="d1", image_url="https://example.com/1.png"),
        ],
    }

    with config.media_content_setting("always"):
        messages = media_messages_for_payload(payload)

    assert len(messages) == 1
    wire_content = chat_content_from_message_content(messages[0]["content"])
    assert {part["type"] for part in wire_content} == {"text", "image_url"}
    for part in wire_content:
        assert "chunk_id" not in part
        assert "document_id" not in part
    image_parts = [part for part in wire_content if part["type"] == "image_url"]
    assert image_parts == [{"type": "image_url", "image_url": {"url": "https://example.com/1.png"}}]


def test_chat_content_from_message_content_converts_responses_style_parts() -> None:
    content = [
        {"type": "input_text", "text": "label", "chunk_id": "c1", "document_id": "d1"},
        {
            "type": "input_image",
            "image_url": "https://example.com/1.png",
            "detail": "auto",
            "chunk_id": "c1",
            "document_id": "d1",
        },
    ]

    assert chat_content_from_message_content(content) == [
        {"type": "text", "text": "label"},
        {"type": "image_url", "image_url": {"url": "https://example.com/1.png", "detail": "auto"}},
    ]
    assert chat_content_from_message_content("plain text") == "plain text"
    assert chat_content_from_message_content(None) is None


def test_media_content_setting_is_isolated_between_concurrent_rollouts() -> None:
    """Two rollout threads must each read their own mode, not the last writer's."""
    started = threading.Barrier(2)
    observed: dict[str, str] = {}

    def rollout(name: str, mode: config.MediaContentMode) -> None:
        with config.media_content_setting(mode):
            started.wait(timeout=5)  # both threads are inside their setting here
            observed[name] = config.MEDIA_CONTENT

    threads = [
        threading.Thread(target=rollout, args=("a", "always")),
        threading.Thread(target=rollout, args=("b", "never")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert observed == {"a": "always", "b": "never"}
    assert config.MEDIA_CONTENT == "auto"


def test_media_content_setting_reaches_fan_out_worker_threads() -> None:
    """Tool fan-out inside a rollout runs on ContextThreadPoolExecutor, so it inherits."""
    with (
        config.media_content_setting("always"),
        ContextThreadPoolExecutor(max_workers=2) as executor,
    ):
        modes = list(executor.map(lambda _: config.MEDIA_CONTENT, range(2)))

    assert modes == ["always", "always"]
