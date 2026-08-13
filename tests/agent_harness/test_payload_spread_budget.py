"""The spread allocator and budget-true clipping behind the per-call payload budgets."""

from __future__ import annotations

from agent_harness import config
from agent_harness.search import (
    _clip_chunk_to_cap,
    _payload_budget_notice,
    _spread_token_budget,
    estimate_payload_tokens,
)

FLOOR = config.MIN_ALLOCATION_TOKENS


class TestSpreadTokenBudget:
    def test_passthrough_when_the_call_fits(self) -> None:
        assert _spread_token_budget([3000, 2000], 25_000, floor=FLOOR) == [3000, 2000]
        assert _spread_token_budget([], 25_000, floor=FLOOR) == []

    def test_caps_sum_within_budget(self) -> None:
        for n in (1, 2, 3, 5, 20, 25, 30):
            caps = _spread_token_budget([9000] * n, 25_000, floor=FLOOR)
            assert sum(caps) <= 25_000
            assert all(cap >= 1 for cap in caps)

    def test_earlier_items_keep_more(self) -> None:
        caps = _spread_token_budget([9000] * 5, 25_000, floor=FLOOR)
        assert caps == sorted(caps, reverse=True)
        assert caps[0] > caps[-1]

    def test_every_item_keeps_at_least_the_floor(self) -> None:
        caps = _spread_token_budget([9000] * 4, 25_000, floor=FLOOR)
        assert min(caps) >= FLOOR

    def test_small_items_keep_full_size_and_donate_slack(self) -> None:
        caps = _spread_token_budget([100, 9000, 9000], 10_000, floor=FLOOR)
        assert caps[0] == 100
        assert caps[1] + caps[2] >= 9_800  # the 100-token item's unused share flows right
        assert caps[1] > caps[2]

    def test_single_oversized_item_gets_the_whole_budget(self) -> None:
        assert _spread_token_budget([9000], 25_000, floor=FLOOR) == [9000]
        assert _spread_token_budget([90_000], 25_000, floor=FLOOR) == [25_000]

    def test_deterministic(self) -> None:
        sizes = [8000, 500, 12_000, 3000, 7500]
        assert _spread_token_budget(sizes, 25_000, floor=FLOOR) == _spread_token_budget(
            sizes, 25_000, floor=FLOOR
        )

    def test_floor_scales_down_when_budget_cannot_cover_floors(self) -> None:
        caps = _spread_token_budget([9000] * 20, 1000, floor=FLOOR)
        assert sum(caps) <= 1000
        assert all(cap >= 1 for cap in caps)

    def test_all_items_at_floor_still_respect_the_budget(self) -> None:
        caps = _spread_token_budget([400, 400], 512, floor=FLOOR)
        assert sum(caps) <= 512
        caps = _spread_token_budget([100, 9000, 9000], 0, floor=FLOOR)
        assert sum(caps) <= len(caps)  # one token per item is the degenerate minimum


class TestClipChunkToCap:
    def _payload(self, **fields: str) -> dict:
        return {"chunk_id": "c1", "document_id": "d1", **fields}

    def test_untouched_when_it_fits(self) -> None:
        payload = self._payload(text="hello")
        assert _clip_chunk_to_cap(payload, 25_000) is False
        assert payload["text"] == "hello"

    def test_proportional_across_fields_with_pristine_totals(self) -> None:
        payload = self._payload(text="t" * 30_000, ocr_text="o" * 30_000)
        assert _clip_chunk_to_cap(payload, 8_000) is True
        # Both fields lose roughly half, unlike the field-order clip's first-field gutting.
        text_kept = len(payload["text"])
        ocr_kept = len(payload["ocr_text"])
        assert 0.7 <= text_kept / ocr_kept <= 1.4
        assert payload["text"].endswith(" of 30000 characters]")
        assert payload["ocr_text"].endswith(" of 30000 characters]")
        assert estimate_payload_tokens(payload) <= 8_000 + 100

    def test_metadata_strings_clipped_and_flagged(self) -> None:
        payload = self._payload(text="short")
        payload["metadata"] = {"notes": "n" * 30_000, "keep": "v"}
        assert _clip_chunk_to_cap(payload, 2_000) is True
        assert len(payload["metadata"]["notes"]) < 30_000
        assert payload["metadata"]["keep"] == "v"
        assert payload["metadata_clipped"] is True

    def test_oversize_non_string_metadata_is_elided(self) -> None:
        payload = self._payload(text="short")
        payload["metadata"] = {"blob": {"rows": ["x" * 100] * 3000}}
        assert _clip_chunk_to_cap(payload, 2_000) is True
        assert payload["metadata"]["blob"]["_truncated"]["original_json_chars"] > 100_000
        assert payload["metadata_clipped"] is True
        assert estimate_payload_tokens(payload) <= 2_000 + 100

    def test_plain_marker_variant(self) -> None:
        payload = self._payload(text="t" * 30_000)
        assert _clip_chunk_to_cap(payload, 2_000, quantified=False) is True
        assert payload["text"].endswith("…[truncated: chunk payload shortened]")

    def test_long_tail_of_small_metadata_values_is_elided(self) -> None:
        payload = self._payload(text="short")
        payload["metadata"] = {f"field_{i}": "v" * 190 for i in range(45)}
        assert _clip_chunk_to_cap(payload, 1_000) is True
        assert estimate_payload_tokens(payload) <= 1_000 + 100
        assert payload["metadata_clipped"] is True

    def test_clip_converges_with_a_cheap_tokenizer(self) -> None:
        class CheapCounter:
            def encode(self, text: str) -> range:
                return range(len(text) // 8)

        config.set_token_counter(CheapCounter())
        try:
            payload = self._payload(text="t" * 100_000)
            assert _clip_chunk_to_cap(payload, 8_000) is True
            assert estimate_payload_tokens(payload) <= 8_000 + 100
        finally:
            config.set_token_counter(None)

    def test_second_call_is_a_no_op(self) -> None:
        payload = self._payload(text="t" * 30_000)
        assert _clip_chunk_to_cap(payload, 2_000) is True
        snapshot = payload["text"]
        assert _clip_chunk_to_cap(payload, 2_000) is False
        assert payload["text"] == snapshot


def test_budget_notice_text() -> None:
    notice = _payload_budget_notice(3, 25_000)
    assert notice == (
        "3 chunks had to be truncated due to this call reaching a 25000-token "
        "payload budget. Please inspect the returned results and proceed with the "
        "context provided. You can call prune_context to free budget, and if "
        "needed you may use some of your calls to make more targeted requests — a "
        "truncated chunk re-requested alone is shown in full."
    )
