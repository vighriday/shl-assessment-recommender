"""Tests for the LLM client wrapper.

The provider call itself is not exercised (no network); these cover the parts we
own — JSON parsing, error wrapping, and the protocol surface — by driving the
private call hook through a subclass.
"""

from __future__ import annotations

import sys
import types

import pytest

from shl_recommender.llm.client import LiteLLMClient, LLMClient, LLMError


class StubClient(LiteLLMClient):
    """Overrides the provider call so we can test the JSON/error handling around it."""

    def __init__(self, raw: str = "", *, raise_error: bool = False):
        super().__init__()
        self._raw = raw
        self._raise = raise_error

    def _call(self, messages, *, temperature, json_mode):
        if self._raise:
            raise LLMError("provider down")
        return self._raw


def test_complete_json_parses_object():
    client = StubClient('{"role": "nurse", "purpose": "screening"}')
    assert client.complete_json([{"role": "user", "content": "x"}]) == {
        "role": "nurse",
        "purpose": "screening",
    }


def test_complete_json_rejects_non_object():
    client = StubClient('["not", "an", "object"]')
    with pytest.raises(LLMError, match="expected a JSON object"):
        client.complete_json([{"role": "user", "content": "x"}])


def test_complete_json_rejects_invalid_json():
    client = StubClient("not json at all")
    with pytest.raises(LLMError, match="valid JSON"):
        client.complete_json([{"role": "user", "content": "x"}])


def test_provider_error_is_wrapped():
    client = StubClient(raise_error=True)
    with pytest.raises(LLMError):
        client.complete_json([{"role": "user", "content": "x"}])


def test_litellm_client_satisfies_protocol():
    assert isinstance(LiteLLMClient(), LLMClient)


class _RateLimitError(Exception):
    """Stand-in for litellm.RateLimitError."""


def _fake_litellm(monkeypatch, calls):
    """Install a fake ``litellm`` module whose ``completion`` pops behaviours from
    ``calls`` — each entry is either an Exception to raise or a string to return —
    and records the ``api_key`` seen on every call in ``seen_keys``."""
    seen_keys: list = []

    def completion(**kwargs):
        seen_keys.append(kwargs.get("api_key"))
        behaviour = calls.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return {"choices": [{"message": {"content": behaviour}}]}

    module = types.ModuleType("litellm")
    module.completion = completion
    module.RateLimitError = _RateLimitError
    monkeypatch.setitem(sys.modules, "litellm", module)
    return seen_keys


def test_rate_limit_fails_over_to_secondary_key(monkeypatch):
    # Primary call is rate-limited; the client retries the same call with the
    # configured fallback key and returns its result.
    seen = _fake_litellm(monkeypatch, [_RateLimitError("429 quota"), "ok"])
    client = LiteLLMClient(api_key_fallback="second-key")
    assert client.complete([{"role": "user", "content": "x"}]) == "ok"
    # First attempt used the primary path (no explicit key), the retry used the fallback.
    assert seen == [None, "second-key"]


def test_rate_limit_without_fallback_raises(monkeypatch):
    # No fallback configured: a rate-limit surfaces as a normal LLMError, no retry.
    # Force the settings default to None so a locally-configured fallback key does not
    # leak into this case.
    from shl_recommender.llm import client as client_module

    monkeypatch.setattr(client_module.settings, "llm_api_key_fallback", None)
    monkeypatch.setattr(client_module.settings, "llm_fallback_model", None)
    seen = _fake_litellm(monkeypatch, [_RateLimitError("429 quota")])
    client = LiteLLMClient()
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "x"}])
    assert seen == [None]


def test_missing_key_also_fails_over(monkeypatch):
    # An auth/missing-key failure on the primary is recoverable by the secondary too.
    seen = _fake_litellm(
        monkeypatch, [ValueError("Missing Gemini API key"), "recovered"]
    )
    client = LiteLLMClient(api_key_fallback="second-key")
    assert client.complete([{"role": "user", "content": "x"}]) == "recovered"
    assert seen == [None, "second-key"]


def test_non_rate_limit_error_does_not_fail_over(monkeypatch):
    # A generic provider error is not retried on the second key: it raises at once and
    # the fallback is never attempted (only one call made).
    seen = _fake_litellm(monkeypatch, [RuntimeError("bad request")])
    client = LiteLLMClient(api_key_fallback="second-key")
    with pytest.raises(LLMError):
        client.complete([{"role": "user", "content": "x"}])
    assert seen == [None]


def _fake_litellm_models(monkeypatch, calls):
    """Like ``_fake_litellm`` but records the ``model`` seen on each call, so a
    cross-provider failover (a different model) can be asserted."""
    seen_models: list = []

    def completion(**kwargs):
        seen_models.append(kwargs.get("model"))
        behaviour = calls.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return {"choices": [{"message": {"content": behaviour}}]}

    module = types.ModuleType("litellm")
    module.completion = completion
    module.RateLimitError = _RateLimitError
    monkeypatch.setitem(sys.modules, "litellm", module)
    return seen_models


def test_cross_provider_failover_after_both_keys(monkeypatch):
    # Primary model rate-limited on both its keys, then the different-provider fallback
    # model succeeds. The chain is: primary model -> primary model (2nd key) -> fallback
    # model. The last attempt uses the fallback model name.
    seen = _fake_litellm_models(
        monkeypatch,
        [_RateLimitError("429"), _RateLimitError("429"), "from groq"],
    )
    client = LiteLLMClient(
        model="gemini/gemini-2.5-flash",
        api_key_fallback="second-key",
        fallback_model="groq/llama-3.3-70b-versatile",
    )
    assert client.complete([{"role": "user", "content": "x"}]) == "from groq"
    assert seen == [
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-flash",
        "groq/llama-3.3-70b-versatile",
    ]


def test_error_message_redacts_a_leaked_key(monkeypatch):
    # Defensive: if a provider ever echoed the key inside its error text, the LLMError
    # we raise (and therefore log) must not contain it. No fallback, so the primary
    # error surfaces directly.
    from shl_recommender.llm import client as client_module

    monkeypatch.setattr(client_module.settings, "llm_api_key_fallback", None)
    monkeypatch.setattr(client_module.settings, "llm_fallback_model", None)
    leaked = "gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _fake_litellm(monkeypatch, [RuntimeError(f"401 invalid key {leaked}")])
    client = LiteLLMClient()
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "x"}])
    assert leaked not in str(excinfo.value)
    assert "[REDACTED]" in str(excinfo.value)


def test_cross_provider_not_tried_when_primary_succeeds(monkeypatch):
    # If the primary model works, neither the fallback key nor the fallback provider is
    # touched — only one call is made.
    seen = _fake_litellm_models(monkeypatch, ["ok from gemini"])
    client = LiteLLMClient(
        model="gemini/gemini-2.5-flash",
        api_key_fallback="second-key",
        fallback_model="groq/llama-3.3-70b-versatile",
    )
    assert client.complete([{"role": "user", "content": "x"}]) == "ok from gemini"
    assert seen == ["gemini/gemini-2.5-flash"]
