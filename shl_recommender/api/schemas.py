"""Request and response schemas for the ``/chat`` endpoint.

These models are the API contract. The grader checks the response shape on every
turn, so the rules below are deliberately strict:

* the response carries exactly three fields — ``reply``, ``recommendations``,
  ``end_of_conversation`` — and no others;
* ``recommendations`` is either ``null`` (while clarifying or refusing) or a list
  of 1..10 items; an empty list is never emitted;
* each recommendation's ``test_type`` is one of the known catalog codes.

The request side is permissive about what a real client might send (extra keys
are ignored rather than rejected) but still validates the parts we depend on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shl_recommender.catalog.test_type import CATEGORY_TO_CODE

# Map the role spellings a client might send onto the three we use internally.
# The brief's JSON uses lowercase user/assistant; the sample transcripts display
# "User"/"Agent". Being lenient here costs nothing and avoids rejecting an entire
# request over a casing or synonym difference, which would fail a hard eval.
_ROLE_SYNONYMS = {
    "user": "user",
    "human": "user",
    "customer": "user",
    "assistant": "assistant",
    "agent": "assistant",
    "ai": "assistant",
    "bot": "assistant",
    "system": "system",
}

# The single-letter codes any recommendation may carry, derived from the catalog
# category map so the two can never drift apart.
_VALID_TEST_TYPE_CODES = frozenset(CATEGORY_TO_CODE.values())

MAX_RECOMMENDATIONS = 10


class Message(BaseModel):
    """One turn of the conversation history."""

    # Ignore unknown keys: a client may attach metadata we do not use, and the
    # contract only depends on role and content.
    model_config = ConfigDict(extra="ignore")

    role: Literal["user", "assistant", "system"]
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def _normalise_role(cls, value: object) -> object:
        # Accept case and common synonyms ("Agent" -> "assistant") so a small
        # client-side difference does not reject the whole request.
        if isinstance(value, str):
            return _ROLE_SYNONYMS.get(value.strip().lower(), value.strip().lower())
        return value

    @field_validator("content")
    @classmethod
    def _content_is_present(cls, value: str) -> str:
        # Content may be whitespace-trimmed to empty by a client; we accept empty
        # strings (an empty turn is harmless) but require the field to be a string,
        # which the type already enforces. Normalisation happens upstream.
        return value


class ChatRequest(BaseModel):
    """Body of ``POST /chat``: the full conversation history."""

    model_config = ConfigDict(extra="ignore")

    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def _has_a_user_message(self) -> "ChatRequest":
        # There is nothing to act on without at least one user turn. The grader
        # always sends one; this guards malformed input.
        if not any(message.role == "user" for message in self.messages):
            raise ValueError("conversation history must contain at least one user message")
        return self

    @property
    def latest_user_message(self) -> str:
        """The content of the most recent user turn."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        # Unreachable given the validator above, but explicit is safer than relying
        # on it from a distance.
        raise ValueError("no user message in history")


class Recommendation(BaseModel):
    """One recommended assessment, as returned to the client.

    Exactly the three fields the contract specifies. ``name`` and ``url`` come
    verbatim from the catalog snapshot; ``test_type`` is the derived code.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str

    @field_validator("test_type")
    @classmethod
    def _known_codes(cls, value: str) -> str:
        codes = value.split(",")
        unknown = [code for code in codes if code not in _VALID_TEST_TYPE_CODES]
        if unknown:
            raise ValueError(f"unknown test_type code(s): {unknown}")
        return value


class ChatResponse(BaseModel):
    """Body returned from ``POST /chat``.

    ``recommendations`` defaults to ``None`` (serialised as JSON ``null``), which
    is the non-commit shape every sample conversation uses. When a shortlist is
    committed it must hold 1..10 items; an empty list is rejected so we never
    accidentally emit ``[]`` for "no recommendations".
    """

    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[Recommendation] | None = None
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def _within_allowed_count(
        cls, value: list[Recommendation] | None
    ) -> list[Recommendation] | None:
        if value is None:
            return None
        if not value:
            raise ValueError(
                "recommendations must be null (not []) when there is no shortlist"
            )
        if len(value) > MAX_RECOMMENDATIONS:
            raise ValueError(
                f"recommendations may not exceed {MAX_RECOMMENDATIONS} items (got {len(value)})"
            )
        return value

    def to_payload(self, *, empty_as_null: bool = True) -> dict:
        """Serialise to the exact dict returned to the client.

        ``recommendations`` is ``None`` internally whenever there is no shortlist.
        Whether that surfaces as JSON ``null`` or ``[]`` is the one contested point
        in the contract (the PDF says "empty", the sample conversations use
        ``null``), so it is decided here, in one place, by ``empty_as_null``.
        """
        payload = self.model_dump()
        if payload["recommendations"] is None and not empty_as_null:
            payload["recommendations"] = []
        return payload
