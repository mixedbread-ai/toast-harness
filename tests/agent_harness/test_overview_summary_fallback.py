"""overview_search must never ship a chunk with no readable content.

The payload drops every raw content field, so if no summary can be derived the
chunk arrives as bare ids. Chunks whose body lives outside ``text`` -- scanned
pages (``ocr_text``), media (``transcription``) -- used to hit exactly that.
"""

from __future__ import annotations

import pytest

from agent_harness.tools import functions as search_provider


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        ({"text": "body text"}, "body text"),
        ({"ocr_text": "scanned page"}, "scanned page"),
        ({"content": "joined body"}, "joined body"),
        ({"context": "surrounding context"}, "surrounding context"),
        ({"transcription": "spoken words"}, "spoken words"),
        # summary-likeness order: context > text > ocr_text > transcription > content
        ({"context": "ctx", "text": "body text", "content": "joined"}, "ctx"),
        ({"text": "body text", "ocr_text": "scanned", "content": "joined"}, "body text"),
        ({"text": "   ", "ocr_text": "scanned page", "content": "joined"}, "scanned page"),
        (
            {"ocr_text": "", "transcription": "spoken words", "content": "joined"},
            "spoken words",
        ),
        ({"context": "   ", "transcription": None, "content": "joined body"}, "joined body"),
        ({"content": "", "context": "surrounding context"}, "surrounding context"),
        # a real store summary always beats any derived text
        ({"summary": "real summary", "text": "body text"}, "real summary"),
    ],
)
async def test_overview_returns_a_readable_preview(monkeypatch, chunk, expected):
    async def search(*args, **kwargs):
        return [chunk]

    monkeypatch.setattr(search_provider, "search_raw", search)
    result = await search_provider.overview_search("query", store_identifiers=["store"])
    assert result["results"][0]["summary"] == expected
    assert result["summaries_found"] == 1
    assert result["top_k"] == 50


def test_fallback_collapses_whitespace_and_caps_length():
    summary = search_provider._text_fallback_summary({"text": "word   " * 2000})
    assert "  " not in summary
    assert len(summary) <= search_provider._OVERVIEW_TEXT_FALLBACK_SUMMARY_MAX_CHARS
