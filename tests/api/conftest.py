"""Fixtures for HTTP-level API tests.

Builds the real application via ``create_app`` — the same wiring the deployed
service uses — but with one substitution that keeps the tests fast, offline, and
deterministic:

* the language model is a fake, so no network call is made and each test can choose
  whether the model "works" or "fails".

Retrieval is lexical-only, so nothing here loads a model or reaches the network.

The app is built against the real catalog (it is small and loads quickly), so these
tests exercise the true request path, not a mock of it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shl_recommender.api.app import create_app
from shl_recommender.config import Settings
from shl_recommender.llm.client import LLMError


class FakeLLM:
    """A model stand-in for the API tests. See ``tests/response/conftest`` for the
    same idea; duplicated here so the API test package is self-contained."""

    def __init__(self, *, text: str = "Here are some suitable assessments.", json: dict | None = None, fail: bool = False):
        self._text = text
        self._json = json or {}
        self._fail = fail

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        if self._fail:
            raise LLMError("model down")
        return self._text

    def complete_json(self, messages, *, schema=None) -> dict:
        if self._fail:
            raise LLMError("model down")
        return self._json


def _config() -> Settings:
    return Settings()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A TestClient over an app wired with a working (but trivial) fake model."""
    app = create_app(config=_config(), llm_client=FakeLLM(json={"ready_to_recommend": False}))
    return TestClient(app)


def make_client(llm: FakeLLM) -> TestClient:
    """Build a client with a specific fake model, for tests that need model control."""
    return TestClient(create_app(config=_config(), llm_client=llm))
