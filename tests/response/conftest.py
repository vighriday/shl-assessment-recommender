"""Shared fakes for response-assembly tests.

A fake LLM client lets the reply and engine tests run offline and deterministically,
and lets each test choose whether the model "works" (returns canned text/JSON) or
"fails" (raises ``LLMError``) so both the happy path and the fallback path are
covered. The fake satisfies the :class:`LLMClient` protocol structurally.
"""

from __future__ import annotations

import pytest

from shl_recommender.llm.client import LLMError


class FakeLLM:
    """Stands in for the model.

    ``text`` is returned from ``complete`` (the reply prose); ``json`` from
    ``complete_json`` (the understanding). Set ``fail=True`` to make every call
    raise, exercising the deterministic fallbacks.
    """

    def __init__(self, *, text: str = "model-written reply", json: dict | None = None, fail: bool = False):
        self._text = text
        self._json = json or {}
        self._fail = fail
        self.complete_calls = 0
        self.json_calls = 0

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        self.complete_calls += 1
        if self._fail:
            raise LLMError("model down")
        return self._text

    def complete_json(self, messages, *, schema=None) -> dict:
        self.json_calls += 1
        if self._fail:
            raise LLMError("model down")
        return self._json


@pytest.fixture
def working_llm() -> FakeLLM:
    return FakeLLM(text="Here's a tailored set of assessments for that role.")


@pytest.fixture
def failing_llm() -> FakeLLM:
    return FakeLLM(fail=True)
