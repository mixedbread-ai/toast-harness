"""Binary nDCG@10 and recall@10 for BrowseComp-Plus rankings.

Dedupe by first occurrence, then cut at k; IDCG over min(|relevant|, k).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_ID_KEYS = ("external_id", "corpus_id", "doc_id", "document_id", "docid", "id")


def normalize_docid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.rsplit("/", 1)[-1].removesuffix(".txt")


def chunk_docid(chunk: Mapping[str, Any]) -> str | None:
    payloads = [chunk]
    for key in ("generated_metadata", "metadata", "file", "document"):
        payload = chunk.get(key)
        if isinstance(payload, Mapping):
            payloads.append(payload)
    for payload in payloads:
        for key in _ID_KEYS:
            normalized = normalize_docid(payload.get(key))
            if normalized:
                return normalized
    return None


def extract_ranked_docids(retrieval: Mapping[str, Any]) -> list[str]:
    """Ranked corpus docids, order-preserving and deduped by first occurrence."""
    chunks = retrieval.get("chunks")
    if isinstance(chunks, Sequence) and chunks:
        docids = [chunk_docid(chunk) for chunk in chunks if isinstance(chunk, Mapping)]
        resolved = [docid for docid in docids if docid]
        if resolved:
            return list(dict.fromkeys(resolved))
    ranked = [normalize_docid(item) for item in (retrieval.get("ranked_ids") or [])]
    return list(dict.fromkeys(item for item in ranked if item))


def _dcg(gains: Iterable[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10) -> float | None:
    relevant = {str(item) for item in relevant_ids}
    if not relevant:
        return None
    deduped = list(dict.fromkeys(str(item) for item in ranked_ids))[:k]
    actual = _dcg([1.0 if item in relevant else 0.0 for item in deduped])
    ideal = _dcg([1.0] * min(k, len(relevant)))
    return actual / ideal if ideal else None


def recall_at_k(
    ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 10
) -> float | None:
    relevant = {str(item) for item in relevant_ids}
    if not relevant:
        return None
    deduped = list(dict.fromkeys(str(item) for item in ranked_ids))[:k]
    return len(relevant.intersection(deduped)) / len(relevant)


def score(retrieval: Mapping[str, Any], relevant_ids: Iterable[str], k: int = 10) -> dict[str, Any]:
    ranked = extract_ranked_docids(retrieval)
    return {
        "ranked_docids": ranked,
        "ndcg_at_10": ndcg_at_k(ranked, relevant_ids, k),
        "recall_at_10": recall_at_k(ranked, relevant_ids, k),
    }
