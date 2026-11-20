"""Final listwise ranking of the evidence presented to the searcher."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from mixedbread import AsyncMixedbread
from pydantic import BaseModel

from agent_harness.llm import AsyncGenerationFn
from agent_harness.search import ChunkIndex
from agent_harness.tools.functions import (
    chunk_content_text,
    resolve_mixedbread_api_key,
    resolve_mixedbread_base_url,
)

LISTWISE_MODEL = "mixedbread-ai/mxbai-rerank-v3.1-listwise"


class RerankingUsage(BaseModel):
    """Separate from Toast's generation usage and search-round count."""

    model: str
    candidate_count: int
    input_tokens: int
    duration_ms: float


class Explanation(BaseModel):
    turn: int
    assistant: str


def rerank_document(chunk: Mapping[str, Any]) -> str:
    """Use the benchmark's descriptor fields and full textual chunk content."""
    title = chunk.get("file_title") or chunk.get("filename")
    lines = [f"title: {title}"] if title else []
    for name in ("filename", "external_id"):
        value = chunk.get(name)
        if value and value != title:
            lines.append(f"{name}: {value}")
    if metadata := chunk.get("metadata"):
        lines.append(f"metadata: {json.dumps(metadata, ensure_ascii=False, sort_keys=True)}")
    body = "\n\n".join(
        value.strip()
        for name in ("text", "context", "ocr_text", "transcription", "summary")
        if isinstance(value := chunk.get(name), str) and value.strip()
    )
    if title and body.startswith(str(title)):
        body = body[len(str(title)) :].lstrip()
    return "\n".join(lines) + "\n\n" + body


@dataclass
class SearchTrajectory:
    """Capture requests before context pruning can remove their evidence."""

    index: ChunkIndex
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    explanations: list[Explanation] = field(default_factory=list)
    turns: int = 0

    def observe_request(self, messages: list[dict[str, Any]]) -> None:
        # Only evidence returned by the harness can introduce a candidate.
        known = set(self.index.visible_chunk_ids())
        for message in messages:
            text = message.get("content")
            if not isinstance(text, str):
                continue
            if message.get("role") != "tool" and not (
                message.get("role") == "user" and text.startswith("INITIAL_SEARCH_RESULTS:\n")
            ):
                continue
            for handle in re.findall(r'"chunk_id":\s*"(c\d+)"', text):
                if handle not in known:
                    continue
                key = self.index.refs.chunk_key_for_id(handle)
                if (chunk := self.index.get(key)) is not None:
                    previous = self.candidates.get(handle)
                    if previous is None or len(chunk_content_text(chunk)) > len(
                        chunk_content_text(previous)
                    ):
                        self.candidates[handle] = deepcopy(chunk)

    def contextual_query(self, query: str) -> str:
        referenced = set(re.findall(r"\bc\d+\b", "\n".join(e.assistant for e in self.explanations)))
        references = [
            {
                "handle": handle,
                "input_index_zero_based": position,
                "filename": chunk.get("filename"),
                "external_id": chunk.get("external_id"),
            }
            for position, (handle, chunk) in enumerate(self.candidates.items())
            if handle in referenced
        ]
        history = "\n".join(
            json.dumps(explanation.model_dump(), ensure_ascii=False)
            for explanation in self.explanations
        )
        return (
            f"Original request:\n{query}\n\n"
            "Search history from the agent that retrieved these candidates follows. "
            "Use it as context when assessing relevance to the original request. "
            "The agent's statements may be tentative or mistaken; assess them against the candidate evidence.\n"
            "Candidate reference map (the input order is unchanged):\n"
            + json.dumps(references, ensure_ascii=False)
            + "\nChronological search history:\n"
            + history
        )

    async def rerank(
        self, query: str, *, top_k: int, api_key: str | None, api_key_env: str | None
    ) -> tuple[list[dict[str, Any]], RerankingUsage | None]:
        if not self.candidates:
            return [], None
        started = time.perf_counter()
        documents = [rerank_document(chunk) for chunk in self.candidates.values()]
        async with AsyncMixedbread(
            api_key=resolve_mixedbread_api_key(api_key=api_key, api_key_env=api_key_env),
            base_url=resolve_mixedbread_base_url(),
        ) as client:
            result = await client.rerank(
                model=LISTWISE_MODEL,
                query=self.contextual_query(query),
                input=documents,
                top_k=len(documents),
            )
        if any(not math.isfinite(item.score) for item in result.data):
            raise ValueError("Rerank scores must be finite")
        positions = [item.index for item in result.data]
        if sorted(positions) != list(range(len(self.candidates))):
            raise ValueError("Reranker must score every candidate exactly once")
        ordered = sorted(result.data, key=lambda item: (-item.score, item.index))
        candidates = list(self.candidates.items())
        chunks = []
        for item in ordered[:top_k]:
            handle, chunk = candidates[item.index]
            key = self.index.refs.chunk_key_for_id(handle)
            chunks.append(
                {
                    **chunk,
                    "chunk_id": handle,
                    "document_id": self.index.refs.document_id_for_key(key[:2]),
                    "relevance_score": item.score,
                }
            )
        usage = RerankingUsage(
            model=result.model,
            candidate_count=len(candidates),
            input_tokens=result.usage.prompt_tokens,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        return chunks, usage


@dataclass
class TrajectoryGeneration:
    """Observe the generation seam, including forced submission and reasoning."""

    inner: AsyncGenerationFn
    trajectory: SearchTrajectory

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        completion_config: dict[str, Any],
        force_submit: bool = False,
        forced_tool_name: str = "submit_ranking",
    ) -> Any | None:
        self.trajectory.observe_request(messages)
        self.trajectory.turns += 1
        response = await self.inner(
            messages,
            tools=tools,
            completion_config=completion_config,
            force_submit=force_submit,
            forced_tool_name=forced_tool_name,
        )
        if getattr(response, "choices", None):
            message = response.choices[0].message
            for text in (
                getattr(message, "reasoning_content", None),
                getattr(message, "content", None),
            ):
                if isinstance(text, str) and text.strip():
                    self.trajectory.explanations.append(
                        Explanation(turn=self.trajectory.turns, assistant=text)
                    )
        return response
