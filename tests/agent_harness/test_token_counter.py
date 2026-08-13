"""Tests for the tokenizer-backed token counter used for budgeting/truncation."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_harness import config, token_counter


def test_falls_back_to_char_heuristic_without_counter() -> None:
    config.set_token_counter(None)
    try:
        assert config.count_text_tokens("x" * 400) == 100  # chars / 4
    finally:
        config.set_token_counter(None)


def test_uses_tokenizer_when_installed() -> None:
    class WordTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            assert add_special_tokens is False  # special tokens excluded
            return list(range(len(text.split())))

    config.set_token_counter(WordTokenizer())
    try:
        assert config.count_text_tokens("a b c d e") == 5
        # A JSON-heavy string the chars/4 heuristic would undercount is measured
        # by the real tokenizer instead.
        assert config.count_text_tokens("one two three") == 3
    finally:
        config.set_token_counter(None)


def test_supports_encoders_without_add_special_tokens_kwarg() -> None:
    class CharTokenizer:
        def encode(self, text: str) -> list[str]:
            return list(text)

    config.set_token_counter(CharTokenizer())
    try:
        assert config.count_text_tokens("abcd") == 4
    finally:
        config.set_token_counter(None)


def test_set_token_counter_prefers_a_count_only_api() -> None:
    """A counter that can count without building the ids list is used that way."""

    class CountingTokenizer:
        def __init__(self) -> None:
            self.ids_calls = 0

        def count_tokens(self, text: str) -> int:
            return len(text.split())

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            self.ids_calls += 1
            return list(range(len(text.split())))

    tokenizer = CountingTokenizer()
    config.set_token_counter(tokenizer)
    try:
        assert config.count_text_tokens("a b c") == 3
        assert tokenizer.ids_calls == 0
    finally:
        config.set_token_counter(None)


def test_reset_restores_heuristic() -> None:
    class WordTokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return list(range(len(text.split())))

    config.set_token_counter(WordTokenizer())
    assert config.TOKEN_COUNTER is not None
    config.set_token_counter(None)
    assert config.TOKEN_COUNTER is None


def test_ensure_token_counter_degrades_gracefully_on_bad_model() -> None:
    config.set_token_counter(None)
    # An unresolvable model name must not raise and must leave the heuristic in place.
    token_counter.ensure_token_counter("definitely-not-a-real-model-xyz-000")
    try:
        assert config.TOKEN_COUNTER is None
    finally:
        config.set_token_counter(None)


def test_hf_backend_is_an_explicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    hf = object()
    monkeypatch.setenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", "hf")
    assert token_counter._maybe_gigatoken_counter(hf, "/unused") is hf


def test_unmet_gigatoken_preconditions_fall_back_to_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No local tokenizer.json: the attempt must degrade to HF, never raise.

    This case cannot pin *which* backend the unset env selects -- both defaults
    return the HF tokenizer here. The selection is pinned by the ``default``
    parametrization of test_gigatoken_counter_requires_exact_parity below.
    """
    hf = object()
    monkeypatch.delenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", raising=False)
    assert token_counter._maybe_gigatoken_counter(hf, "/unused") is hf


@pytest.mark.parametrize("backend_env", ["gigatoken", None], ids=["explicit", "default"])
def test_gigatoken_counter_requires_exact_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_env: str | None
) -> None:
    """Parity passes => the fast counter is installed, under both selections.

    The ``default`` case is the only test that fails if the backend default
    reverts to ``hf``: every other path returns the HF tokenizer either way.
    """
    tokenizer_dir = tmp_path
    (tokenizer_dir / "tokenizer.json").write_text("{}")

    class FakeHF:
        def __len__(self) -> int:
            return 256

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            assert add_special_tokens is False
            return list(text.encode())

    class Fast(FakeHF):
        pass

    class FakeSerializedTokenizer:
        @staticmethod
        def as_hf() -> Fast:
            return Fast()

    class FakeTokenizer:
        @classmethod
        def from_json(cls, data: bytes) -> SimpleNamespace:
            assert data == b"{}"
            return FakeSerializedTokenizer()

    if backend_env is None:
        monkeypatch.delenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", raising=False)
    else:
        monkeypatch.setenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", backend_env)
    monkeypatch.setattr(token_counter.metadata, "version", lambda _name: "0.10.0")
    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=FakeTokenizer))
    hf = FakeHF()
    counter = token_counter._maybe_gigatoken_counter(hf, str(tokenizer_dir))

    assert isinstance(counter, token_counter._ParityGatedGigatokenCounter)
    assert counter.encode("exact", add_special_tokens=False) == hf.encode("exact")


def test_hub_identifier_resolves_tokenizer_json_through_the_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The documented usage is a hub id, which is not a path: it must still engage."""
    downloaded = tmp_path / "tokenizer.json"
    downloaded.write_text("{}")
    requests: list[tuple[str, str]] = []

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        requests.append((repo_id, filename))
        return str(downloaded)

    class FakeHF:
        def __len__(self) -> int:
            return 256

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(text.encode())

    class FakeSerializedTokenizer:
        @staticmethod
        def as_hf() -> FakeHF:
            return FakeHF()

    class FakeTokenizer:
        @staticmethod
        def from_json(data: bytes) -> FakeSerializedTokenizer:
            assert data == b"{}"
            return FakeSerializedTokenizer()

    monkeypatch.delenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", raising=False)
    monkeypatch.setattr(token_counter.metadata, "version", lambda _name: "0.10.0")
    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=FakeTokenizer))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    counter = token_counter._maybe_gigatoken_counter(FakeHF(), "Qwen/Qwen3-8B")

    assert isinstance(counter, token_counter._ParityGatedGigatokenCounter)
    assert requests == [("Qwen/Qwen3-8B", "tokenizer.json")]


def test_hub_identifier_without_huggingface_hub_falls_back_to_hf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hub step is a guarded import, not a dependency: absent it, nothing changes."""
    monkeypatch.delenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", raising=False)
    monkeypatch.setattr(token_counter.metadata, "version", lambda _name: "0.10.0")
    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=object()))
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    hf = object()

    assert token_counter._maybe_gigatoken_counter(hf, "Qwen/Qwen3-8B") is hf


def test_gigatoken_parity_failure_falls_back_to_hf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer_dir = tmp_path
    (tokenizer_dir / "tokenizer.json").write_text("{}")

    class FakeHF:
        def __len__(self) -> int:
            return 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [1]

    class BadFast(FakeHF):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [2]

    class FakeSerializedTokenizer:
        @staticmethod
        def as_hf() -> BadFast:
            return BadFast()

    class FakeTokenizer:
        @staticmethod
        def from_json(_data: bytes) -> FakeSerializedTokenizer:
            return FakeSerializedTokenizer()

    monkeypatch.setenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", "gigatoken")
    monkeypatch.setattr(token_counter.metadata, "version", lambda _name: "0.10.0")
    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=FakeTokenizer))
    hf = FakeHF()

    assert token_counter._maybe_gigatoken_counter(hf, str(tokenizer_dir)) is hf


class _FakeHF:
    def __len__(self) -> int:
        return 256

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode())


def _install_fake_gigatoken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, native_encode: Any
) -> list[str]:
    """Wire up a fake gigatoken; the returned list records every ids-path encode."""
    (tmp_path / "tokenizer.json").write_text("{}")
    ids_calls: list[str] = []

    class Fast(_FakeHF):
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            ids_calls.append(text)
            return list(text.encode())

    class FakeNativeTokenizer:
        encode = staticmethod(native_encode)

        @staticmethod
        def as_hf() -> Fast:
            return Fast()

    class FakeTokenizer:
        @staticmethod
        def from_json(data: bytes) -> FakeNativeTokenizer:
            assert data == b"{}"
            return FakeNativeTokenizer()

    monkeypatch.setenv("AGENT_HARNESS_TOKEN_COUNTER_BACKEND", "gigatoken")
    monkeypatch.setattr(token_counter.metadata, "version", lambda _name: "0.10.0")
    monkeypatch.setitem(sys.modules, "gigatoken", SimpleNamespace(Tokenizer=FakeTokenizer))
    return ids_calls


def test_gigatoken_counts_without_materializing_the_ids_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Counting takes only a length, so it reads the raw encoder's ids buffer."""
    hf = _FakeHF()
    ids_calls = _install_fake_gigatoken(
        monkeypatch, tmp_path, native_encode=lambda text: tuple(text.encode())
    )

    counter = token_counter._maybe_gigatoken_counter(hf, str(tmp_path))
    assert isinstance(counter, token_counter._ParityGatedGigatokenCounter)
    after_parity_probes = len(ids_calls)
    for text in token_counter._GIGATOKEN_PARITY_TEXTS:
        assert counter.count_tokens(text) == len(hf.encode(text, add_special_tokens=False))

    assert ids_calls[after_parity_probes:] == []


def test_count_only_path_is_dropped_when_its_counts_disagree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A count that is not the HF count is never installed; the ids path stays."""
    hf = _FakeHF()
    ids_calls = _install_fake_gigatoken(
        monkeypatch, tmp_path, native_encode=lambda text: (*text.encode(), 0)
    )

    counter = token_counter._maybe_gigatoken_counter(hf, str(tmp_path))
    assert isinstance(counter, token_counter._ParityGatedGigatokenCounter)
    after_parity_probes = len(ids_calls)
    for text in token_counter._GIGATOKEN_PARITY_TEXTS:
        assert counter.count_tokens(text) == len(hf.encode(text, add_special_tokens=False))

    assert ids_calls[after_parity_probes:] == list(token_counter._GIGATOKEN_PARITY_TEXTS)


def test_count_only_runtime_failure_falls_back_to_the_ids_path() -> None:
    def broken_encode(text: str) -> tuple[int, ...]:
        raise RuntimeError("broken")

    counter = token_counter._ParityGatedGigatokenCounter(
        _FakeHF(), _FakeHF(), count_only=broken_encode
    )

    assert counter.count_tokens("abc") == 3
    assert counter._count_only is None


def test_gigatoken_runtime_failure_demotes_permanently() -> None:
    class BadFast:
        calls = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            self.calls += 1
            raise RuntimeError("broken")

    class HF:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [len(text)]

    fast = BadFast()
    counter = token_counter._ParityGatedGigatokenCounter(fast, HF())
    assert counter.encode("abc") == [3]
    assert counter.encode("abcd") == [4]
    assert fast.calls == 1
