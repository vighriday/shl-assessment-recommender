"""The internal view of a conversation.

The API is stateless: every request carries the whole history and the server
stores nothing. So on each turn we rebuild this ``ConversationState`` from the
messages and drive the turn off it. This module defines only the *shape* of that
state; the logic that fills it from messages lives in the extractor (built next),
which keeps the data contract stable and independently testable.

Design notes that the shape encodes:

* Single current values, not append-only lists, for things a user can change
  (seniority, purpose, ...). When the user corrects themselves the extractor
  overwrites the field, so "the latest correction wins" falls out naturally.
* The four task behaviours plus refusal are modelled as one ``Mode`` enum so the
  policy engine has a single thing to switch on.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field


class Mode(str, enum.Enum):
    """What the agent should do on this turn.

    Exactly one applies per turn. ``CLARIFY`` and ``REFUSE`` return no shortlist;
    ``RECOMMEND`` and ``REFINE`` commit one; ``COMPARE`` answers from catalog facts
    and usually returns no shortlist.
    """

    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"


class Purpose(str, enum.Enum):
    """Why the assessment is being used. Drives ranking, not just wording."""

    SELECTION = "selection"
    SCREENING = "screening"
    DEVELOPMENT = "development"
    UNKNOWN = "unknown"


class ConversationState(BaseModel):
    """Everything we need to decide and answer the current turn."""

    model_config = ConfigDict(frozen=True)

    # --- What the user is hiring for ----------------------------------------
    role: str | None = Field(default=None, description="Target role or candidate population.")
    seniority: str | None = Field(default=None, description="Seniority level if stated.")
    years_experience: str | None = Field(default=None, description="Experience hint, as given.")
    domain: str | None = Field(default=None, description="Industry or domain if stated.")
    purpose: Purpose = Field(default=Purpose.UNKNOWN, description="Selection / screening / development.")

    must_have_skills: tuple[str, ...] = Field(default=(), description="Explicitly required skills.")
    optional_skills: tuple[str, ...] = Field(default=(), description="Nice-to-have skills.")
    languages: tuple[str, ...] = Field(default=(), description="Required assessment languages.")
    test_type_preferences: tuple[str, ...] = Field(
        default=(), description="Categories the user asked for, e.g. 'personality'."
    )

    # The raw user query for this turn, normalised. Retrieval reads this directly.
    query_text: str = Field(default="", description="Latest user message, normalised.")

    # Model's readiness judgement (see Understanding). None means "no opinion" and
    # the policy falls back to the structural minimum-context rule.
    ready_to_recommend: bool | None = Field(
        default=None, description="Model view: specific enough to recommend well?"
    )
    suggested_question: str | None = Field(
        default=None, description="Model's single clarifying question, if not ready."
    )

    # --- Turn intent --------------------------------------------------------
    is_comparison: bool = Field(default=False, description="User asked to compare products.")
    comparison_targets: tuple[str, ...] = Field(
        default=(), description="Named products to compare, e.g. ('OPQ', 'GSA')."
    )
    wants_addition: bool = Field(default=False, description="User asked to add to the shortlist.")
    wants_removal: bool = Field(default=False, description="User asked to drop from the shortlist.")
    is_off_topic: bool = Field(default=False, description="Out-of-scope (legal/general-hiring) ask present.")
    is_prompt_injection: bool = Field(default=False, description="Attempt to override instructions present.")
    user_confirmed: bool = Field(default=False, description="User signalled the shortlist is accepted.")

    # --- Progress -----------------------------------------------------------
    clarifications_asked: int = Field(default=0, ge=0, description="Agent questions asked so far.")
    has_prior_recommendations: bool = Field(
        default=False, description="A shortlist was already offered earlier in the conversation."
    )

    def has_minimum_context(self) -> bool:
        """Whether there is enough to commit to a shortlist.

        Mirrors the architecture's minimum-viable-context rule: a target (role,
        population, or a substantive query) plus at least one differentiator
        (seniority, domain, purpose, skills, languages, or a requested category).
        """
        has_target = bool(self.role) or len(self.query_text) >= 40
        has_differentiator = any(
            (
                self.seniority,
                self.domain,
                self.purpose is not Purpose.UNKNOWN,
                self.must_have_skills,
                self.optional_skills,
                self.languages,
                self.test_type_preferences,
            )
        )
        return has_target and has_differentiator
