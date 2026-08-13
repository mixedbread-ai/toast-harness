"""Classify tool-call failures as provider-side (Mixedbread) or agent-caused."""

from __future__ import annotations

from typing import Any

import httpx
from mixedbread import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

PROVIDER_ERROR_KIND = "provider"
AGENT_ERROR_KIND = "agent"

_PROVIDER_STATUS_CODES = frozenset({408, 429})

_STATUS_ERRORS: dict[int, type[APIStatusError]] = {
    error.status_code: error
    for error in (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        ConflictError,
        UnprocessableEntityError,
        RateLimitError,
    )
}


def wire_status_error(status_code: int, message: str, *, body: Any | None = None) -> APIStatusError:
    """Build the SDK error a real client raises for this wire response.

    The seam's error contract is typed by the SDK: the loops classify provider
    failures by these exception types, catch ``UnprocessableEntityError`` for
    the sort and chunk-index fallbacks, and surface the wire
    ``{type, code, message}`` body as model-visible feedback. An in-process
    binding maps its service failures through this helper instead of
    fabricating SDK internals itself.
    """
    if status_code >= 500:
        error_type: type[APIStatusError] = InternalServerError
    else:
        error_type = _STATUS_ERRORS.get(status_code, APIStatusError)
    response = httpx.Response(
        status_code, request=httpx.Request("POST", "http://in-process.invalid")
    )
    return error_type(message, response=response, body=body)


class ProviderFailure(Exception):
    """A retrieval-side infrastructure failure, raised by in-process bindings.

    The SDK signals provider failures with its own exception types, which this
    module classifies by status code. An in-process ``AsyncRetrievalClient``
    has no HTTP response to wrap and should not fabricate one: raising this
    type marks the failure provider-side (outside the agent's control) exactly
    like an SDK 5xx. Anything else an implementation raises stays agent-caused
    and is fed back to the model as tool feedback.
    """


def is_provider_failure(exc: BaseException) -> bool:
    """Return True when a failure came from the Mixedbread service itself.

    Provider-side failures (rate limits, timeouts, 5xx, connection errors) are
    outside the agent's control and are marked as such on the rollout record.
    Request errors such as 400/404/422 stay agent-caused: they are feedback to
    the agent.
    """
    if isinstance(exc, ProviderFailure):
        return True
    if isinstance(exc, APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, APIStatusError):  # includes RateLimitError, InternalServerError
        status = getattr(exc, "status_code", None)
        return status in _PROVIDER_STATUS_CODES or (isinstance(status, int) and status >= 500)
    return False


def error_kind(exc: BaseException) -> str:
    return PROVIDER_ERROR_KIND if is_provider_failure(exc) else AGENT_ERROR_KIND
