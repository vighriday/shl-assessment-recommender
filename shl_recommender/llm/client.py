"""Provider-agnostic language-model client.

A deliberately thin wrapper over the underlying provider (LiteLLM, which speaks to
Gemini / Groq / OpenRouter / others through one interface). The rest of the
service depends only on the small :class:`LLMClient` protocol below, never on a
specific vendor SDK, so the provider can be swapped by configuration alone.

The client does two narrow jobs and nothing else:

* ``complete`` — free-form text completion, used to write the user-facing reply;
* ``complete_json`` — completion constrained to JSON, used to extract structured
  understanding from the conversation.

Both surface failure as :class:`LLMError` rather than leaking provider exceptions,
so callers can apply a deterministic fallback (the service must still return a
valid response when the model is slow or unavailable).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shl_recommender.config import settings
from shl_recommender.observability import get_logger

log = get_logger(__name__)

# Token shapes that could appear if a provider ever echoed a credential back inside an
# error string. We scrub these before an exception message is logged, so a key can
# never reach the logs even indirectly. Covers Groq (``gsk_...``), OpenAI-style
# (``sk-...``), Google (``AIza...`` and the newer ``AQ.A...``), and any long bearer-ish
# token. Defensive only — providers do not normally include the key in errors.
_SECRET_PATTERNS = (
    re.compile(r"\bgsk_[A-Za-z0-9]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAQ\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{10,}\b", re.IGNORECASE),
)


def _redact(text: str) -> str:
    """Remove anything that looks like an API key/token from a string before logging."""
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


@dataclass(frozen=True)
class _Attempt:
    """One step in the failover chain: a model, an optional key override, and a label.

    ``api_key`` is None when the provider should read its own key from the environment
    (the primary and cross-provider attempts); it is set only for the same-provider
    secondary-key retry. ``label`` is for logging and never contains a secret.
    """

    model: str
    api_key: str | None
    label: str


class LLMError(Exception):
    """Raised when a model call fails or returns unusable output."""


class _RateLimited(Exception):
    """Internal marker: a call failed specifically because the key was rate-limited.

    Carries the original provider exception so the caller can either fail over to a
    second key or surface it as a normal :class:`LLMError`. Never escapes this module.
    """

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _should_failover(litellm, exc: Exception) -> bool:
    """Whether ``exc`` means "this key is unusable, try the other one".

    Two cases warrant failover to the secondary key: the primary is rate-limited /
    out of quota (HTTP 429), or the primary key is missing or rejected
    (authentication error). Both are conditions the secondary key can plausibly
    recover from; a bad request, timeout, or network outage cannot, so those do not
    trigger it. Typed LiteLLM exceptions are preferred, with a string fallback so a
    differently-wrapped error is still recognised across providers.
    """
    for attr in ("RateLimitError", "AuthenticationError"):
        exc_type = getattr(litellm, attr, None)
        if exc_type is not None and isinstance(exc, exc_type):
            return True
    text = str(exc).lower()
    markers = (
        "429",
        "rate limit",
        "resource_exhausted",
        "quota",
        "api key",
        "api_key",
        "authentication",
        "unauthorized",
    )
    return any(marker in text for marker in markers)


@runtime_checkable
class LLMClient(Protocol):
    """The interface the rest of the service depends on.

    Defining this as a protocol lets tests pass a simple fake and lets us replace
    the provider without touching callers.
    """

    def complete(self, messages: list[dict], *, temperature: float = 0.2) -> str: ...

    def complete_json(self, messages: list[dict], *, schema: dict | None = None) -> dict: ...


class LiteLLMClient:
    """Concrete client backed by LiteLLM.

    Imports LiteLLM lazily so the package can be imported (and most tests can run)
    without the dependency or network present.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float | None = None,
        api_key_fallback: str | None = None,
        fallback_model: str | None = None,
    ) -> None:
        self._model = model or settings.llm_model
        self._timeout = timeout if timeout is not None else settings.llm_timeout_seconds
        # A secondary key for the primary provider, tried when the primary is
        # rate-limited. ``None`` means no key-level failover. We never log or echo it.
        self._api_key_fallback = (
            api_key_fallback if api_key_fallback is not None else settings.llm_api_key_fallback
        )
        # A different-provider model, tried last, when the primary provider is exhausted
        # on both keys. Its own key is read by the provider from the environment.
        self._fallback_model = (
            fallback_model if fallback_model is not None else settings.llm_fallback_model
        )

    def _attempts(self) -> list[_Attempt]:
        """The ordered failover chain for one logical call.

        Each attempt is tried in turn; the next is used only when the current one fails
        with a failover-eligible error (rate limit or auth/missing key). The order is:

        1. the primary model, provider key read from the environment;
        2. the same model with the secondary key, if one is configured;
        3. a different-provider fallback model, if one is configured (its own key read
           from the environment).

        A pure single-provider deploy has just the first attempt and behaves exactly as
        before.
        """
        chain: list[_Attempt] = [_Attempt(model=self._model, api_key=None, label="primary")]
        if self._api_key_fallback:
            chain.append(
                _Attempt(model=self._model, api_key=self._api_key_fallback, label="fallback-key")
            )
        if self._fallback_model and self._fallback_model != self._model:
            chain.append(
                _Attempt(model=self._fallback_model, api_key=None, label="fallback-provider")
            )
        return chain

    def _call(self, messages: list[dict], *, temperature: float, json_mode: bool) -> str:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise LLMError("litellm is not installed") from exc

        base: dict = {
            "messages": messages,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if json_mode:
            base["response_format"] = {"type": "json_object"}

        # Walk the failover chain. Only a failover-eligible failure (rate limit / auth)
        # advances to the next attempt; any other failure raises at once — retrying a bad
        # request or a network outage on another key or provider is pointless. The last
        # attempt's failure is surfaced as the error.
        attempts = self._attempts()
        last_cause: Exception | None = None
        for index, attempt in enumerate(attempts):
            kwargs = {**base, "model": attempt.model}
            if attempt.api_key is not None:
                kwargs["api_key"] = attempt.api_key
            try:
                return self._complete_once(litellm, kwargs)
            except _RateLimited as exc:
                last_cause = exc.cause
                is_last = index == len(attempts) - 1
                if is_last:
                    break
                log.warning(
                    "model attempt rate-limited or unavailable; failing over",
                    extra={"from": attempt.label, "to": attempts[index + 1].label},
                )
        raise LLMError(f"completion failed: {_redact(last_cause)}") from last_cause

    def _complete_once(self, litellm, kwargs: dict) -> str:
        """One provider call. Raises :class:`_RateLimited` on a 429 so the caller can
        decide whether to fail over, and :class:`LLMError` on any other failure."""
        try:
            response = litellm.completion(**kwargs)
            return response["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # provider/network/shape errors
            if _should_failover(litellm, exc):
                raise _RateLimited(exc) from exc
            raise LLMError(f"completion failed: {_redact(exc)}") from exc

    def complete(self, messages: list[dict], *, temperature: float = 0.2) -> str:
        return self._call(messages, temperature=temperature, json_mode=False)

    def complete_json(self, messages: list[dict], *, schema: dict | None = None) -> dict:
        # ``schema`` is accepted for callers that want to document the expected
        # shape; JSON-object mode plus prompt instructions carry the constraint
        # across providers that do not support full JSON-schema enforcement.
        raw = self._call(messages, temperature=0.0, json_mode=True)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model did not return valid JSON: {raw[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"expected a JSON object, got {type(parsed).__name__}")
        return parsed
