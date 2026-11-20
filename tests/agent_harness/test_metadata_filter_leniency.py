from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_harness.schemas import FilterMetadataArgs, InspectMetadataArgs
from agent_harness.tools.functions import build_mixedbread_filters


@pytest.mark.parametrize("expression", [{}, {"all": []}, {"all": None, "none": []}])
def test_empty_metadata_filter_parses_as_no_filter(expression: dict) -> None:
    parsed = InspectMetadataArgs.model_validate({"metadata_filter": expression})
    assert parsed.metadata_filter is not None and parsed.metadata_filter.is_empty()
    assert build_mixedbread_filters(None, None, metadata_filter=parsed.metadata_filter) is None


def test_filter_metadata_still_requires_a_real_filter() -> None:
    with pytest.raises(ValidationError, match="requires metadata_filters or metadata_filter"):
        FilterMetadataArgs.model_validate({"metadata_filter": {"all": []}})
