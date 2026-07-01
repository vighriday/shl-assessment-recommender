"""Tests for the LLM understanding layer and the JSON-coercion in its schema."""

from __future__ import annotations

from shl_recommender.api.schemas import Message
from shl_recommender.conversation.state import Purpose
from shl_recommender.llm.client import LLMError
from shl_recommender.llm.understanding import Understanding, extract_understanding


class FakeLLM:
    def __init__(self, payload=None, *, fail=False, bad=False):
        self._payload = payload or {}
        self._fail = fail

    def complete(self, messages, *, temperature=0.2):
        return "x"

    def complete_json(self, messages, *, schema=None):
        if self._fail:
            raise LLMError("boom")
        return self._payload


def test_valid_payload_is_parsed():
    llm = FakeLLM({
        "role": "nurse",
        "seniority": "senior",
        "purpose": "screening",
        "languages": ["Spanish"],
    })
    result = extract_understanding([Message(role="user", content="hi")], llm)
    assert result.role == "nurse"
    assert result.purpose is Purpose.SCREENING
    assert result.languages == ("Spanish",)


def test_model_failure_returns_empty_understanding():
    result = extract_understanding([Message(role="user", content="hi")], FakeLLM(fail=True))
    assert result == Understanding()


def test_unknown_purpose_is_coerced_to_unknown():
    result = extract_understanding(
        [Message(role="user", content="hi")], FakeLLM({"purpose": "banana"})
    )
    assert result.purpose is Purpose.UNKNOWN


def test_scalar_skill_is_coerced_to_tuple():
    result = extract_understanding(
        [Message(role="user", content="hi")], FakeLLM({"must_have_skills": "Java"})
    )
    assert result.must_have_skills == ("Java",)


def test_blank_list_entries_are_dropped():
    result = extract_understanding(
        [Message(role="user", content="hi")],
        FakeLLM({"optional_skills": ["", "  ", "SQL"]}),
    )
    assert result.optional_skills == ("SQL",)


def test_garbage_payload_does_not_crash():
    # A payload with wrong types for fields should yield an empty understanding,
    # not an exception.
    result = extract_understanding(
        [Message(role="user", content="hi")], FakeLLM({"role": {"unexpected": "object"}})
    )
    assert result == Understanding()
