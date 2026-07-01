"""The API's error contract.

A caller should get the same *shape* back whether a request succeeds or fails, so
that error handling on the other side is simple and a failure is never an
unparseable surprise. This module defines that error shape and the single function
that maps an exception to an ``(HTTP status, body)`` pair. The web layer (Phase 7)
installs thin handlers that defer to :func:`classify_error`; keeping the logic here,
free of the framework, means the contract can be tested without a server and stays
identical across every endpoint.

Two principles drive the design:

* **Stable, minimal shape.** Every error body is ``{"error": {"type", "message"}}``
  with an optional ``detail`` for the one case where structured field information
  genuinely helps the caller (request validation). ``type`` is a short stable
  string a client can branch on; ``message`` is a human-readable sentence.

* **Never leak internals to the caller; always leak them to the logs.** A raw
  provider exception, a stack trace, or an internal path must not cross the wire —
  it is noise at best and an information leak at worst. Unexpected failures collapse
  to a generic 500 with a safe message, while the real cause is logged server-side
  where an operator can see it. The mapping is deliberately small and closed: known
  failure modes get specific handling; everything else is a 500.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from shl_recommender.llm.client import LLMError


class ErrorType(str, Enum):
    """Stable, machine-branchable error categories returned to the caller."""

    VALIDATION = "validation_error"
    UPSTREAM_MODEL = "upstream_model_error"
    INTERNAL = "internal_error"


class ErrorBody(BaseModel):
    """The inner error object. ``detail`` is populated only when it helps."""

    type: ErrorType
    message: str
    detail: list[dict] | None = None


class ErrorResponse(BaseModel):
    """The full error envelope returned on any non-success response."""

    error: ErrorBody = Field(...)

    @classmethod
    def of(
        cls, error_type: ErrorType, message: str, detail: list[dict] | None = None
    ) -> "ErrorResponse":
        return cls(error=ErrorBody(type=error_type, message=message, detail=detail))


# Human-facing messages. Kept as constants so wording is consistent and reviewed in
# one place rather than scattered across handlers.
_VALIDATION_MESSAGE = "The request body did not match the expected schema."
_UPSTREAM_MESSAGE = (
    "The language model is temporarily unavailable. The request could not be "
    "completed; please retry."
)
_INTERNAL_MESSAGE = "An internal error occurred while handling the request."


def classify_error(exc: Exception) -> tuple[int, ErrorResponse]:
    """Map an exception to an ``(HTTP status, ErrorResponse)`` pair.

    The mapping is closed and ordered from most specific to least. Request
    validation is handled by the framework's own validation exception (translated
    by the web layer, which knows that type) and is not seen here; this function
    covers the failures that originate inside the service.

    * :class:`LLMError` -> ``502``. The model is an upstream dependency; a failure
      there is not the service's fault and is a distinct, retryable condition, so it
      gets its own status and type rather than being hidden inside a generic 500.
    * anything else -> ``500`` with a generic message. The specific cause is *not*
      placed in the response; the caller of this function is expected to log ``exc``
      with its traceback first.
    """
    if isinstance(exc, LLMError):
        return 502, ErrorResponse.of(ErrorType.UPSTREAM_MODEL, _UPSTREAM_MESSAGE)

    return 500, ErrorResponse.of(ErrorType.INTERNAL, _INTERNAL_MESSAGE)


def validation_error(detail: list[dict]) -> tuple[int, ErrorResponse]:
    """Build the ``422`` body for a request that failed schema validation.

    ``detail`` is the framework's structured list of field errors, passed through
    so a caller can see *which* fields were wrong. This is the one case where
    echoing structured detail is helpful and safe — it describes the caller's own
    input, not our internals.
    """
    return 422, ErrorResponse.of(ErrorType.VALIDATION, _VALIDATION_MESSAGE, detail=detail)
