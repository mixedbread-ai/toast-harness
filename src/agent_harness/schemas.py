"""Pydantic schemas for the two-agent deep search pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_harness.config import (
    FILTER_CHUNKS_DEFAULT_K,
    FILTER_CHUNKS_MAX_K,
    GET_CHUNKS_MAX_CHUNK_IDS,
    READ_DOCUMENT_MAX_WINDOW,
)

ChunkKey = tuple[str, str, int]
DocumentKey = tuple[str, str]
MetadataFilterValue = str | int | float | bool | None | list[str | int | float | bool | None]
MetadataHintSource = Literal["inspect_metadata", "result_metadata"]
MetadataHintUsage = Literal["soft_hint", "hard_filter"]

FILTER_OPERATORS = (
    "eq",
    "not_eq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "like",
    "starts_with",
    "not_like",
    "regex",
)
FILTER_MODES = ("all", "any", "none")
AGENTIC_SEARCH_FILTER_OPERATORS = (
    "eq",
    "not_eq",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "like",
    "starts_with",
    "regex",
)

StoreChunkGrepTarget = Literal["text", "generated"]


class RankedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    relevance_score: float = Field(ge=0.0, le=1.0)

    @field_validator("chunk_id")
    @classmethod
    def strip_chunk_id(cls, chunk_id: str) -> str:
        chunk_id = chunk_id.strip()
        if not chunk_id:
            raise ValueError("chunk_id must not be empty")
        return chunk_id


def _dedupe_ranked_chunks(chunks: list[RankedChunk]) -> list[RankedChunk]:
    seen: set[str] = set()
    deduped: list[RankedChunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append(chunk)
    return deduped


class RankedChunkList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking_strategy: str | None = None
    chunks: list[RankedChunk] = Field(min_length=0)

    @field_validator("ranking_strategy")
    @classmethod
    def strip_ranking_strategy(cls, ranking_strategy: str | None) -> str | None:
        if ranking_strategy is None:
            return None
        ranking_strategy = ranking_strategy.strip()
        return ranking_strategy or None

    @field_validator("chunks")
    @classmethod
    def dedupe_duplicate_chunks(cls, chunks: list[RankedChunk]) -> list[RankedChunk]:
        return _dedupe_ranked_chunks(chunks)


class MetadataFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    value: MetadataFilterValue

    @field_validator("key")
    @classmethod
    def strip_key(cls, key: str) -> str:
        key = key.strip()
        if not key:
            raise ValueError("metadata filter key must not be empty")
        return key

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, operator: str) -> str:
        operator = operator.strip()
        if operator not in FILTER_OPERATORS:
            raise ValueError("operator must be one of: " + ", ".join(FILTER_OPERATORS))
        return operator


class MetadataFilterExpression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    all: list[MetadataFilter] | None = None
    any: list[MetadataFilter] | None = None
    none: list[MetadataFilter] | None = None

    @model_validator(mode="after")
    def require_clause_group(self) -> MetadataFilterExpression:
        if not (self.all or self.any or self.none):
            raise ValueError("metadata_filter must include all, any, or none")
        return self


class InspectMetadataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facets: list[str] | None = None
    max_values_per_field: int = Field(default=8, ge=1, le=20)
    metadata_filter: MetadataFilterExpression | None = None

    @field_validator("facets")
    @classmethod
    def strip_facets(cls, facets: list[str] | None) -> list[str] | None:
        if facets is None:
            return None
        cleaned = [facet.strip() for facet in facets if facet.strip()]
        return cleaned or None


class AgenticMetadataFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    value: Any

    @field_validator("key")
    @classmethod
    def strip_key(cls, key: str) -> str:
        key = key.strip()
        if not key:
            raise ValueError("metadata filter key must not be empty")
        return key

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, operator: str) -> str:
        operator = operator.strip()
        if operator not in AGENTIC_SEARCH_FILTER_OPERATORS:
            raise ValueError(
                "operator must be one of: " + ", ".join(AGENTIC_SEARCH_FILTER_OPERATORS)
            )
        return operator


class MetadataFacetHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    value: MetadataFilterValue
    source: MetadataHintSource
    usage: MetadataHintUsage = "soft_hint"
    evidence: str | None = None

    @field_validator("field")
    @classmethod
    def strip_field(cls, field: str) -> str:
        field = field.strip()
        if not field:
            raise ValueError("metadata hint field must not be empty")
        return field

    @field_validator("evidence")
    @classmethod
    def strip_evidence(cls, evidence: str | None) -> str | None:
        if evidence is None:
            return None
        evidence = evidence.strip()
        return evidence or None


class SearchCorpusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    filter_by: list[AgenticMetadataFilter] = Field(default_factory=list)
    metadata_filters: list[AgenticMetadataFilter] = Field(default_factory=list, exclude=True)
    metadata_filter: MetadataFilterExpression | None = None
    filter_mode: Literal["all", "any"] = "all"

    @field_validator("query")
    @classmethod
    def strip_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query

    @model_validator(mode="after")
    def normalize_filter_alias(self) -> SearchCorpusArgs:
        if not self.filter_by and self.metadata_filters:
            self.filter_by = self.metadata_filters
        return self


class FilterChunksArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter_by: list[AgenticMetadataFilter] = Field(default_factory=list)
    metadata_filters: list[AgenticMetadataFilter] = Field(default_factory=list, exclude=True)
    filter_mode: Literal["all", "any"] = "all"
    rank_by: str | None = Field(default=None, min_length=1)
    direction: Literal["asc", "desc"] = "desc"
    k: int = Field(default=FILTER_CHUNKS_DEFAULT_K, ge=1, le=FILTER_CHUNKS_MAX_K)

    @field_validator("rank_by")
    @classmethod
    def strip_rank_by(cls, rank_by: str | None) -> str | None:
        if rank_by is None:
            return None
        rank_by = rank_by.strip()
        return rank_by or None

    @model_validator(mode="after")
    def normalize_filter_alias(self) -> FilterChunksArgs:
        if not self.filter_by and self.metadata_filters:
            self.filter_by = self.metadata_filters
        return self


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(min_length=1, max_length=1024)
    targets: list[StoreChunkGrepTarget] = Field(default=["text", "generated"])
    case_sensitive: bool = False
    filter_by: list[AgenticMetadataFilter] = Field(default_factory=list)
    metadata_filters: list[AgenticMetadataFilter] = Field(default_factory=list, exclude=True)
    filter_mode: Literal["all", "any"] = "all"

    @field_validator("pattern")
    @classmethod
    def strip_pattern(cls, pattern: str) -> str:
        pattern = pattern.strip()
        if not pattern:
            raise ValueError("pattern must not be empty")
        return pattern

    @field_validator("targets")
    @classmethod
    def validate_targets(cls, targets: list[StoreChunkGrepTarget]) -> list[StoreChunkGrepTarget]:
        deduped: list[StoreChunkGrepTarget] = []
        for target in targets:
            if target not in deduped:
                deduped.append(target)
        if not deduped:
            raise ValueError("targets must include at least one of: text, generated")
        return deduped

    @model_validator(mode="after")
    def normalize_filter_alias(self) -> GrepArgs:
        if not self.filter_by and self.metadata_filters:
            self.filter_by = self.metadata_filters
        return self


class FilterMetadataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata_filters: list[MetadataFilter] = Field(default_factory=list)
    metadata_filter: MetadataFilterExpression | None = None
    filter_mode: str = "all"
    limit: int = Field(default=20, ge=1, le=100)

    @model_validator(mode="after")
    def require_filter(self) -> FilterMetadataArgs:
        if not self.metadata_filters and self.metadata_filter is None:
            raise ValueError("filter_metadata requires metadata_filters or metadata_filter")
        return self

    @field_validator("filter_mode")
    @classmethod
    def validate_filter_mode(cls, filter_mode: str) -> str:
        filter_mode = str(filter_mode or "all").strip()
        if filter_mode not in FILTER_MODES:
            raise ValueError("filter_mode must be one of: " + ", ".join(FILTER_MODES))
        return filter_mode


class RankMetadataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sort_key: str = Field(min_length=1)
    sort_order: str = "desc"
    metadata_filters: list[MetadataFilter] = Field(default_factory=list)
    metadata_filter: MetadataFilterExpression | None = None
    filter_mode: str = "all"
    limit: int = Field(default=10, ge=1, le=50)
    fetch_limit: int = Field(default=100, ge=1, le=100)
    include_chunks: bool = True

    @field_validator("sort_key")
    @classmethod
    def strip_sort_key(cls, sort_key: str) -> str:
        sort_key = sort_key.strip()
        if not sort_key:
            raise ValueError("sort_key must not be empty")
        return sort_key

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, sort_order: str) -> str:
        sort_order = str(sort_order or "desc").strip().lower()
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be asc or desc")
        return sort_order

    @field_validator("filter_mode")
    @classmethod
    def validate_filter_mode(cls, filter_mode: str) -> str:
        filter_mode = str(filter_mode or "all").strip()
        if filter_mode not in FILTER_MODES:
            raise ValueError("filter_mode must be one of: " + ", ".join(FILTER_MODES))
        return filter_mode


class DistinctMetadataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distinct_key: str = Field(min_length=1)
    metadata_filters: list[MetadataFilter] = Field(default_factory=list)
    metadata_filter: MetadataFilterExpression | None = None
    filter_mode: str = "all"
    examples_per_value: int = Field(default=1, ge=1, le=5)
    fetch_limit: int = Field(default=100, ge=1, le=100)
    include_chunks: bool = True
    sort_key: str | None = None
    sort_order: str = "desc"

    @field_validator("distinct_key")
    @classmethod
    def strip_distinct_key(cls, distinct_key: str) -> str:
        distinct_key = distinct_key.strip()
        if not distinct_key:
            raise ValueError("distinct_key must not be empty")
        return distinct_key

    @field_validator("sort_key")
    @classmethod
    def strip_optional_sort_key(cls, sort_key: str | None) -> str | None:
        if sort_key is None:
            return None
        sort_key = sort_key.strip()
        return sort_key or None

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, sort_order: str) -> str:
        sort_order = str(sort_order or "desc").strip().lower()
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order must be asc or desc")
        return sort_order

    @field_validator("filter_mode")
    @classmethod
    def validate_filter_mode(cls, filter_mode: str) -> str:
        filter_mode = str(filter_mode or "all").strip()
        if filter_mode not in FILTER_MODES:
            raise ValueError("filter_mode must be one of: " + ", ".join(FILTER_MODES))
        return filter_mode


class OverviewSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def strip_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        return query


class ReadDocumentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    x: int = Field(default=1, ge=0, le=READ_DOCUMENT_MAX_WINDOW)

    @field_validator("document_id")
    @classmethod
    def strip_document_id(cls, document_id: str) -> str:
        document_id = document_id.strip()
        if not document_id:
            raise ValueError("document_id must not be empty")
        return document_id

    @field_validator("chunk_id")
    @classmethod
    def strip_chunk_id(cls, chunk_id: str) -> str:
        chunk_id = chunk_id.strip()
        if not chunk_id:
            raise ValueError("chunk_id must not be empty")
        return chunk_id


class GetChunksArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[str] = Field(min_length=1, max_length=GET_CHUNKS_MAX_CHUNK_IDS)

    @field_validator("chunk_ids")
    @classmethod
    def strip_chunk_ids(cls, chunk_ids: list[str]) -> list[str]:
        stripped_ids = [chunk_id.strip() for chunk_id in chunk_ids]
        if any(not chunk_id for chunk_id in stripped_ids):
            raise ValueError("chunk_ids must not contain empty values")
        return list(dict.fromkeys(stripped_ids))


class ChunkIdentifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)

    @field_validator("chunk_id")
    @classmethod
    def strip_chunk_id(cls, chunk_id: str) -> str:
        chunk_id = chunk_id.strip()
        if not chunk_id:
            raise ValueError("chunk_id must not be empty")
        return chunk_id


class DocumentIdentifier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)

    @field_validator("document_id")
    @classmethod
    def strip_document_id(cls, document_id: str) -> str:
        document_id = document_id.strip()
        if not document_id:
            raise ValueError("document_id must not be empty")
        return document_id


class PruneContextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_ids: list[str] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)

    @field_validator("chunk_ids", "document_ids")
    @classmethod
    def strip_ids(cls, values: list[str]) -> list[str]:
        return [str(value).strip() for value in values if str(value).strip()]

    @model_validator(mode="after")
    def reject_noop_prune(self) -> PruneContextArgs:
        if not self.chunk_ids and not self.document_ids:
            raise ValueError("prune_context requires at least one chunk_id or document_id")
        return self


class AgentChunkPayload(BaseModel):
    """Compact chunk payload sent back to agents."""

    chunk_id: str
    document_id: str
    chunk_index: int
    filename: str | None = None
    external_id: str | None = None
    file_title: str | None = None
    mime_type: str | None = None
    search_score: float | None = None
    text: str | None = None
    context: str | None = None
    ocr_text: str | None = None
    transcription: str | None = None
    summary: str | None = None
    metadata: Any | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: dict[str, Any],
        *,
        chunk_id: str,
        document_id: str,
    ) -> AgentChunkPayload:
        summary = chunk.get("summary")
        summary = summary.strip() or None if isinstance(summary, str) else None

        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            chunk_index=int(chunk.get("chunk_index", 0) or 0),
            filename=chunk.get("filename"),
            external_id=chunk.get("external_id"),
            file_title=chunk.get("file_title"),
            mime_type=chunk.get("mime_type") or chunk.get("type"),
            search_score=(
                round(float(chunk.get("search_score", 0.0) or 0.0), 4)
                if chunk.get("search_score") is not None
                else None
            ),
            text=chunk.get("text"),
            context=chunk.get("context"),
            ocr_text=chunk.get("ocr_text"),
            transcription=chunk.get("transcription"),
            summary=summary,
            metadata=chunk.get("metadata"),
        )


def chunk_key_from_parts(store_id: str, file_id: str, chunk_index: int) -> ChunkKey:
    return (str(store_id), str(file_id), int(chunk_index or 0))


def document_key_from_parts(store_id: str, file_id: str) -> DocumentKey:
    return (str(store_id), str(file_id))
