"""Short, model-visible references for canonical Mixedbread IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .schemas import ChunkKey, DocumentKey, chunk_key_from_parts, document_key_from_parts


@dataclass
class ReferenceRegistry:
    """Maps compact agent references to canonical store/file/chunk IDs.

    The registry is runtime application state. Tool traces should carry only
    ``chunk_id`` and ``document_id`` handles; backend calls resolve those
    handles here before touching Mixedbread.
    """

    chunk_prefix: str = "c"
    document_prefix: str = "d"
    _chunk_ids_by_key: dict[ChunkKey, str] = field(default_factory=dict, init=False)
    _chunk_keys_by_id: dict[str, ChunkKey] = field(default_factory=dict, init=False)
    _document_ids_by_key: dict[DocumentKey, str] = field(default_factory=dict, init=False)
    _document_keys_by_id: dict[str, DocumentKey] = field(default_factory=dict, init=False)
    _lock: Any = field(default_factory=RLock, init=False, repr=False)

    def chunk_id_for_key(self, key: ChunkKey) -> str:
        normalized = chunk_key_from_parts(key[0], key[1], key[2])
        with self._lock:
            existing = self._chunk_ids_by_key.get(normalized)
            if existing is not None:
                return existing

            document_key = document_key_from_parts(normalized[0], normalized[1])
            self._document_id_for_key_unlocked(document_key)
            ref_id = f"{self.chunk_prefix}{len(self._chunk_ids_by_key) + 1}"
            self._chunk_ids_by_key[normalized] = ref_id
            self._chunk_keys_by_id[ref_id] = normalized
            return ref_id

    def document_id_for_key(self, key: DocumentKey) -> str:
        normalized = document_key_from_parts(key[0], key[1])
        with self._lock:
            return self._document_id_for_key_unlocked(normalized)

    def ids_for_chunk_key(self, key: ChunkKey) -> tuple[str, str]:
        normalized = chunk_key_from_parts(key[0], key[1], key[2])
        chunk_id = self.chunk_id_for_key(normalized)
        document_id = self.document_id_for_key((normalized[0], normalized[1]))
        return chunk_id, document_id

    def chunk_key_for_id(self, chunk_id: str) -> ChunkKey:
        ref_id = str(chunk_id).strip()
        with self._lock:
            key = self._chunk_keys_by_id.get(ref_id)
        if key is None:
            raise ValueError(f"Unknown chunk_id: {chunk_id}")
        return key

    def document_key_for_id(self, document_id: str) -> DocumentKey:
        ref_id = str(document_id).strip()
        with self._lock:
            key = self._document_keys_by_id.get(ref_id)
        if key is None:
            raise ValueError(f"Unknown document_id: {document_id}")
        return key

    def snapshot(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the external mapping from short references to canonical IDs."""
        with self._lock:
            chunk_items = list(self._chunk_keys_by_id.items())
            document_items = list(self._document_keys_by_id.items())

        return {
            "chunks": {
                ref_id: {
                    "store_id": key[0],
                    "file_id": key[1],
                    "chunk_index": key[2],
                }
                for ref_id, key in chunk_items
            },
            "documents": {
                ref_id: {
                    "store_id": key[0],
                    "file_id": key[1],
                }
                for ref_id, key in document_items
            },
        }

    def _document_id_for_key_unlocked(self, key: DocumentKey) -> str:
        existing = self._document_ids_by_key.get(key)
        if existing is not None:
            return existing

        ref_id = f"{self.document_prefix}{len(self._document_ids_by_key) + 1}"
        self._document_ids_by_key[key] = ref_id
        self._document_keys_by_id[ref_id] = key
        return ref_id
