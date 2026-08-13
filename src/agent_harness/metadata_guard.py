"""Validation helpers for metadata hints and filters."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .schemas import MetadataFacetHint
from .search import ChunkIndex

NUMERIC_METADATA_FIELDS = {
    "chunk_index",
    "congress",
    "file_size",
    "num_lines",
    "start_line",
    "word_count",
}

EXACT_FILTER_OPERATORS = {"eq", "not_eq", "in", "not_in"}
COMPARISON_FILTER_OPERATORS = {"gt", "gte", "lt", "lte"}
TEXT_FILTER_OPERATORS = {"like", "not_like", "starts_with", "regex"}
FILTER_GROUPS = ("all", "any", "none")


@dataclass(slots=True)
class MetadataValueRecord:
    value: Any
    sources: set[str] = field(default_factory=set)
    initial: bool = False


@dataclass(slots=True)
class MetadataFieldRecord:
    values: dict[str, MetadataValueRecord] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    value_types: set[str] = field(default_factory=set)


@dataclass(slots=True)
class MetadataRegistry:
    fields: dict[str, MetadataFieldRecord] = field(default_factory=dict)

    def add_value(
        self,
        field_name: str,
        value: Any,
        *,
        source: str,
        initial: bool = False,
    ) -> None:
        field_name = str(field_name or "").strip()
        if not field_name:
            return
        value = _typed_facet_value(field_name, value)
        record = self.fields.setdefault(field_name, MetadataFieldRecord())
        record.sources.add(source)
        record.value_types.add(_value_type(value))
        value_key = _value_key(value)
        value_record = record.values.get(value_key)
        if value_record is None:
            record.values[value_key] = MetadataValueRecord(
                value=value,
                sources={source},
                initial=initial,
            )
            return
        value_record.sources.add(source)
        value_record.initial = value_record.initial or initial

    def knows_field(self, field_name: str) -> bool:
        return str(field_name or "").strip() in self.fields

    def field_record(self, field_name: str) -> MetadataFieldRecord | None:
        return self.fields.get(str(field_name or "").strip())

    def known_value(
        self,
        field_name: str,
        value: Any,
        *,
        require_exact: bool = True,
    ) -> tuple[bool, Any, bool, MetadataValueRecord | None]:
        field_name = str(field_name or "").strip()
        field_record = self.fields.get(field_name)
        if field_record is None:
            return False, value, False, None

        exact_record = field_record.values.get(_value_key(value))
        if exact_record is not None:
            return True, value, False, exact_record

        for candidate in _coercion_candidates(field_name, value, field_record):
            candidate_record = field_record.values.get(_value_key(candidate))
            if candidate_record is not None:
                return True, candidate, candidate != value, candidate_record

        if require_exact:
            return False, value, False, None

        comparable = _coerce_to_known_type(field_name, value, field_record)
        if comparable is _MISSING:
            return False, value, False, None
        return True, comparable, comparable != value, None


@dataclass(slots=True)
class HintValidation:
    hints: list[MetadataFacetHint]
    dropped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def dropped_metadata_hint_count(self) -> int:
        return len(self.dropped)

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "dropped_metadata_hint_count": self.dropped_metadata_hint_count,
            "dropped_metadata_hints": self.dropped,
        }


@dataclass(slots=True)
class FilterValidation:
    args: dict[str, Any]
    invalid: list[dict[str, Any]] = field(default_factory=list)
    coerced: list[dict[str, Any]] = field(default_factory=list)

    @property
    def invalid_metadata_filter_count(self) -> int:
        return len(self.invalid)

    @property
    def coerced_metadata_filter_count(self) -> int:
        return len(self.coerced)

    @property
    def valid(self) -> bool:
        return not self.invalid

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "invalid_metadata_filter_count": self.invalid_metadata_filter_count,
            "coerced_metadata_filter_count": self.coerced_metadata_filter_count,
            "invalid_metadata_filters": self.invalid,
            "coerced_metadata_filters": self.coerced,
        }


def filter_new_metadata_facet_values(
    result: Mapping[str, Any],
    *,
    registry: MetadataRegistry,
    max_values_per_field: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return inspect_metadata output with already-known facet values removed."""
    try:
        max_values = int(max_values_per_field)
    except (TypeError, ValueError):
        max_values = 8
    max_values = max(1, min(max_values, 20))

    payload = dict(result)
    fields = payload.get("metadata_fields")
    if not isinstance(fields, Mapping):
        return payload, {
            "metadata_facet_known_value_count": 0,
            "metadata_facet_new_value_count": 0,
            "metadata_facet_raw_value_count": 0,
            "metadata_facet_fields_with_new_values": 0,
            "metadata_facet_fields_without_new_values": 0,
        }

    filtered_fields: dict[str, list[dict[str, Any]]] = {}
    known_value_count = 0
    new_value_count = 0
    raw_value_count = 0
    fields_without_new_values = 0

    for raw_field_name, values in fields.items():
        field_name = str(raw_field_name)
        samples: list[dict[str, Any]] = []
        seen_values: set[str] = set()
        for value, sample in _iter_facet_value_samples(values):
            value_key = _value_key(value)
            if value_key in seen_values:
                continue
            seen_values.add(value_key)
            raw_value_count += 1
            known, _, _, _ = registry.known_value(
                field_name,
                value,
                require_exact=True,
            )
            if known:
                known_value_count += 1
                continue
            samples.append(sample)
            new_value_count += 1
            if len(samples) >= max_values:
                break
        if samples:
            filtered_fields[field_name] = samples
        else:
            fields_without_new_values += 1

    stats = {
        "metadata_facet_known_value_count": known_value_count,
        "metadata_facet_new_value_count": new_value_count,
        "metadata_facet_raw_value_count": raw_value_count,
        "metadata_facet_fields_with_new_values": len(filtered_fields),
        "metadata_facet_fields_without_new_values": fields_without_new_values,
    }
    payload["metadata_fields"] = filtered_fields
    payload["metadata_field_count"] = len(filtered_fields)
    payload["max_values_per_field"] = max_values
    payload["metadata_value_deduplication"] = stats
    return payload, stats


class _Missing:
    pass


_MISSING = _Missing()


def build_metadata_registry(
    *,
    initial_metadata_facets: Mapping[str, Any] | None = None,
    additional_metadata_facets: Sequence[Mapping[str, Any]] | None = None,
    index: ChunkIndex | None = None,
) -> MetadataRegistry:
    registry = MetadataRegistry()
    _add_facet_result(
        registry,
        initial_metadata_facets,
        source="initial_metadata",
        initial=True,
    )
    for result in additional_metadata_facets or []:
        _add_facet_result(
            registry,
            result,
            source="inspect_metadata",
            initial=False,
        )
    if index is not None:
        for chunk in index.top_scored():
            _add_metadata_payload(
                registry,
                chunk.get("metadata"),
                source="result_metadata",
            )
            _add_metadata_payload(
                registry,
                chunk.get("generated_metadata"),
                source="result_metadata",
            )
    return registry


def validate_metadata_hints(
    hints: Sequence[Any],
    *,
    registry: MetadataRegistry,
) -> HintValidation:
    accepted: list[MetadataFacetHint] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for hint in hints:
        if not isinstance(hint, MetadataFacetHint):
            dropped.append(
                {
                    "value": hint,
                    "reason": "not a typed metadata hint",
                }
            )
            continue
        field_name = hint.field
        values = hint.value if isinstance(hint.value, list) else [hint.value]
        if not registry.knows_field(field_name):
            dropped.append(_hint_drop(hint, "unknown field"))
            continue

        valid_values: list[Any] = []
        initial_covered = False
        invalid_value = False
        for value in values:
            valid, normalized, _, record = registry.known_value(
                field_name,
                value,
                require_exact=True,
            )
            if not valid or record is None:
                invalid_value = True
                break
            if record.initial:
                initial_covered = True
                break
            valid_values.append(normalized)
        if invalid_value:
            dropped.append(_hint_drop(hint, "unverified value"))
            continue
        if initial_covered:
            dropped.append(_hint_drop(hint, "already covered by INITIAL_METADATA_FACETS"))
            continue

        normalized_value: Any = valid_values
        if not isinstance(hint.value, list):
            normalized_value = valid_values[0]
        dedupe_key = (field_name, _value_key(normalized_value))
        if dedupe_key in seen:
            dropped.append(_hint_drop(hint, "duplicate hint"))
            continue
        seen.add(dedupe_key)
        accepted.append(hint.model_copy(update={"value": normalized_value}))
    return HintValidation(hints=accepted, dropped=dropped)


def validate_metadata_filter_args(
    args: Mapping[str, Any],
    *,
    registry: MetadataRegistry,
) -> FilterValidation:
    normalized_args = dict(args)
    invalid: list[dict[str, Any]] = []
    coerced: list[dict[str, Any]] = []

    filter_by = args.get("filter_by") or args.get("metadata_filters") or []
    if filter_by:
        normalized_filters = []
        for condition in filter_by:
            normalized = _validate_condition(
                condition,
                registry=registry,
                invalid=invalid,
                coerced=coerced,
            )
            if normalized is not None:
                normalized_filters.append(normalized)
        normalized_args["filter_by"] = normalized_filters
        normalized_args.pop("metadata_filters", None)

    metadata_filter = args.get("metadata_filter")
    if metadata_filter:
        normalized_expression = _validate_expression(
            metadata_filter,
            registry=registry,
            invalid=invalid,
            coerced=coerced,
        )
        normalized_args["metadata_filter"] = normalized_expression

    return FilterValidation(args=normalized_args, invalid=invalid, coerced=coerced)


def zero_result_filtered_search_count(metadata: Mapping[str, Any]) -> int:
    has_filter = bool(
        metadata.get("metadata_filter")
        or metadata.get("metadata_filters")
        or metadata.get("filter_by")
    )
    candidate_count = metadata.get("candidate_count")
    if not has_filter or not isinstance(candidate_count, int):
        return 0
    return 1 if candidate_count == 0 else 0


def _add_facet_result(
    registry: MetadataRegistry,
    result: Mapping[str, Any] | None,
    *,
    source: str,
    initial: bool,
) -> None:
    if not isinstance(result, Mapping):
        return
    fields = result.get("metadata_fields")
    if not isinstance(fields, Mapping):
        return
    for field_name, values in fields.items():
        for value in _iter_facet_values(values):
            registry.add_value(
                str(field_name),
                value,
                source=source,
                initial=initial,
            )


def _add_metadata_payload(
    registry: MetadataRegistry,
    payload: Any,
    *,
    source: str,
    prefix: str = "",
) -> None:
    if not isinstance(payload, Mapping):
        return
    for key, value in payload.items():
        field_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _add_metadata_payload(registry, value, source=source, prefix=field_name)
            continue
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, Mapping):
                    registry.add_value(field_name, item, source=source)
            continue
        registry.add_value(field_name, value, source=source)


def _iter_facet_values(values: Any) -> Iterable[Any]:
    if isinstance(values, Mapping):
        for key in values:
            yield key
        return
    if isinstance(values, list):
        for item in values:
            if isinstance(item, Mapping):
                if "value" in item:
                    yield item["value"]
                elif "key" in item:
                    yield item["key"]
                elif "name" in item:
                    yield item["name"]
                else:
                    for key in item:
                        yield key
            else:
                yield item
        return
    if values not in (None, ""):
        yield values


def _iter_facet_value_samples(values: Any) -> Iterable[tuple[Any, dict[str, Any]]]:
    if isinstance(values, Mapping):
        for key, metadata in values.items():
            yield key, _facet_value_sample(key, metadata)
        return
    if isinstance(values, list):
        for item in values:
            if isinstance(item, Mapping):
                if "value" in item:
                    value = item["value"]
                elif "key" in item:
                    value = item["key"]
                elif "name" in item:
                    value = item["name"]
                else:
                    for key in item:
                        value = key
                        break
                    else:
                        continue
                yield value, _facet_value_sample(value, item)
            else:
                yield item, {"value": item}
        return
    if values not in (None, ""):
        yield values, {"value": values}


def _facet_value_sample(value: Any, metadata: Any) -> dict[str, Any]:
    sample: dict[str, Any] = {"value": value}
    if isinstance(metadata, (int, float)):
        sample["count"] = metadata
        return sample
    if isinstance(metadata, Mapping):
        type_name = metadata.get("type")
        if isinstance(type_name, str) and type_name:
            sample["type"] = type_name
        for count_key in ("count", "doc_count", "frequency"):
            count = metadata.get(count_key)
            if isinstance(count, (int, float)):
                sample["count"] = count
                break
    return sample


def _validate_expression(
    expression: Any,
    *,
    registry: MetadataRegistry,
    invalid: list[dict[str, Any]],
    coerced: list[dict[str, Any]],
) -> Any:
    if hasattr(expression, "model_dump"):
        expression = expression.model_dump(mode="json", exclude_none=True)
    if not isinstance(expression, Mapping):
        invalid.append({"filter": expression, "reason": "metadata_filter is not an object"})
        return expression

    normalized: dict[str, Any] = {}
    for group in FILTER_GROUPS:
        raw_conditions = expression.get(group)
        if not raw_conditions:
            continue
        if not isinstance(raw_conditions, Sequence) or isinstance(raw_conditions, (str, bytes)):
            invalid.append(
                {"filter": {group: raw_conditions}, "reason": "filter group is not a list"}
            )
            continue
        normalized_group = []
        for raw_condition in raw_conditions:
            nested = isinstance(raw_condition, Mapping) and any(
                key in raw_condition for key in FILTER_GROUPS
            )
            if nested:
                normalized_group.append(
                    _validate_expression(
                        raw_condition,
                        registry=registry,
                        invalid=invalid,
                        coerced=coerced,
                    )
                )
                continue
            condition = _validate_condition(
                raw_condition,
                registry=registry,
                invalid=invalid,
                coerced=coerced,
            )
            if condition is not None:
                normalized_group.append(condition)
        if normalized_group:
            normalized[group] = normalized_group
    return normalized


def _validate_condition(
    condition: Any,
    *,
    registry: MetadataRegistry,
    invalid: list[dict[str, Any]],
    coerced: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if hasattr(condition, "model_dump"):
        condition = condition.model_dump(mode="json", exclude_none=True)
    if not isinstance(condition, Mapping):
        invalid.append({"filter": condition, "reason": "filter condition is not an object"})
        return None

    field_name = str(condition.get("key") or "").strip()
    operator = str(condition.get("operator") or "").strip()
    value = condition.get("value")
    if not field_name:
        invalid.append({"filter": dict(condition), "reason": "missing metadata field"})
        return None
    if not registry.knows_field(field_name):
        invalid.append({"filter": dict(condition), "reason": "unknown metadata field"})
        return None

    normalized = dict(condition)
    normalized["key"] = field_name
    normalized["operator"] = operator

    if operator in {"in", "not_in"}:
        if not isinstance(value, list):
            invalid.append({"filter": dict(condition), "reason": "in/not_in value must be a list"})
            return None
        normalized_values = []
        for item in value:
            valid, normalized_item, was_coerced, _ = registry.known_value(
                field_name,
                item,
                require_exact=True,
            )
            if not valid:
                allowed, normalized_item, was_coerced = _allow_unverified_exact_value(
                    field_name,
                    item,
                    registry,
                )
                if not allowed:
                    invalid.append(
                        {
                            "filter": dict(condition),
                            "reason": "unverified metadata value",
                            "invalid_value": item,
                        }
                    )
                    return None
            if was_coerced:
                coerced.append(
                    {
                        "field": field_name,
                        "from": item,
                        "to": normalized_item,
                    }
                )
            normalized_values.append(normalized_item)
        normalized["value"] = normalized_values
        return normalized

    if operator in {"eq", "not_eq"}:
        valid, normalized_value, was_coerced, _ = registry.known_value(
            field_name,
            value,
            require_exact=True,
        )
        if not valid:
            valid, normalized_value, was_coerced = _allow_unverified_exact_value(
                field_name,
                value,
                registry,
            )
            if not valid:
                invalid.append({"filter": dict(condition), "reason": "unverified metadata value"})
                return None
        if was_coerced:
            coerced.append({"field": field_name, "from": value, "to": normalized_value})
        normalized["value"] = normalized_value
        return normalized

    if operator in COMPARISON_FILTER_OPERATORS:
        valid, normalized_value, was_coerced, _ = registry.known_value(
            field_name,
            value,
            require_exact=False,
        )
        if not valid:
            invalid.append({"filter": dict(condition), "reason": "value type does not match field"})
            return None
        if was_coerced:
            coerced.append({"field": field_name, "from": value, "to": normalized_value})
        normalized["value"] = normalized_value
        return normalized

    if operator in TEXT_FILTER_OPERATORS:
        field_record = registry.field_record(field_name)
        if (
            field_record is None
            or "str" not in field_record.value_types
            or not isinstance(value, str)
        ):
            invalid.append(
                {
                    "filter": dict(condition),
                    "reason": "text operator requires a string field and value",
                }
            )
            return None
        normalized["value"] = value
        return normalized

    invalid.append({"filter": dict(condition), "reason": "unsupported metadata operator"})
    return None


def _hint_drop(hint: MetadataFacetHint, reason: str) -> dict[str, Any]:
    return {
        "field": hint.field,
        "value": hint.value,
        "source": hint.source,
        "usage": hint.usage,
        "reason": reason,
    }


def _allow_unverified_exact_value(
    field_name: str,
    value: Any,
    registry: MetadataRegistry,
) -> tuple[bool, Any, bool]:
    field_record = registry.field_record(field_name)
    if field_record is None:
        return False, value, False
    if not (_field_is_identifier_like(field_name) or _field_is_numeric(field_name)):
        return False, value, False
    normalized = _coerce_to_known_type(field_name, value, field_record)
    if normalized is _MISSING:
        return False, value, False
    return True, normalized, normalized != value


def _field_is_identifier_like(field_name: str) -> bool:
    leaf = str(field_name or "").strip().split(".")[-1].casefold()
    return (
        leaf == "id" or leaf.endswith("_id") or leaf.endswith("-id") or leaf.endswith("identifier")
    )


def _coercion_candidates(
    field_name: str,
    value: Any,
    field_record: MetadataFieldRecord,
) -> list[Any]:
    candidates: list[Any] = []
    coerced = _coerce_to_known_type(field_name, value, field_record)
    if coerced is not _MISSING:
        candidates.append(coerced)
    if "str" in field_record.value_types and not isinstance(value, str):
        candidates.append(str(value))
    return candidates


def _coerce_to_known_type(
    field_name: str,
    value: Any,
    field_record: MetadataFieldRecord,
) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if "int" in field_record.value_types:
            try:
                return int(stripped)
            except ValueError:
                pass
        if "float" in field_record.value_types:
            try:
                return float(stripped)
            except ValueError:
                pass
        if "bool" in field_record.value_types:
            lowered = stripped.casefold()
            if lowered in {"true", "false"}:
                return lowered == "true"
    if isinstance(value, (int, float)) and (
        _field_is_numeric(field_name)
        or "int" in field_record.value_types
        or "float" in field_record.value_types
    ):
        return value
    if "str" in field_record.value_types and isinstance(value, str):
        return value
    return _MISSING


def _typed_facet_value(field_name: str, value: Any) -> Any:
    if isinstance(value, str) and _field_is_numeric(field_name):
        stripped = value.strip()
        try:
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return value
    return value


def _field_is_numeric(field_name: str) -> bool:
    return str(field_name or "").strip().split(".")[-1] in NUMERIC_METADATA_FIELDS


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "null"
    return type(value).__name__


def _value_key(value: Any) -> str:
    return f"{_value_type(value)}:{json.dumps(value, sort_keys=True, default=str)}"
