"""Tests for the API error contract.

Pin the two things that make the contract trustworthy: the mapping from exception
to status/type is exactly as documented, and internal failures never leak their
cause into the response body. The validation helper is checked to echo field detail
(the one case where detail is safe and useful).
"""

from __future__ import annotations

from shl_recommender.api.errors import (
    ErrorResponse,
    ErrorType,
    classify_error,
    validation_error,
)
from shl_recommender.llm.client import LLMError


def test_llm_error_maps_to_502_upstream():
    status, body = classify_error(LLMError("gemini timed out: secret-key-in-here"))
    assert status == 502
    assert body.error.type is ErrorType.UPSTREAM_MODEL
    # The raw provider message (which could contain sensitive detail) must not leak.
    assert "secret-key-in-here" not in body.error.message


def test_unknown_error_maps_to_generic_500():
    status, body = classify_error(RuntimeError("internal path /home/app/secret"))
    assert status == 500
    assert body.error.type is ErrorType.INTERNAL
    assert "/home/app/secret" not in body.error.message
    assert body.error.detail is None


def test_validation_error_echoes_field_detail():
    detail = [{"loc": ["body", "messages"], "msg": "field required", "type": "missing"}]
    status, body = validation_error(detail)
    assert status == 422
    assert body.error.type is ErrorType.VALIDATION
    assert body.error.detail == detail


def test_error_response_serialises_with_stable_shape():
    body = ErrorResponse.of(ErrorType.INTERNAL, "x")
    payload = body.model_dump()
    assert payload == {"error": {"type": "internal_error", "message": "x", "detail": None}}
