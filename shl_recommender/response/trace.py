"""An opt-in, human-readable trace of one turn's reasoning.

The graded API returns only the three contract fields. But everything that *decided*
those fields — the state read from the conversation, the mode the policy chose and
why, the candidates retrieval scored, whether the model or a fallback wrote the reply
— is computed on every turn and otherwise only visible in the server logs. This module
packages that reasoning into a plain, serialisable object so a tester can request it
with ``?debug=1`` and see the whole turn end to end, without any of it leaking into
the normal response.

Two rules hold this to a safe shape:

* **It never contains a secret.** No API keys, no raw credentials — only the
  understanding, the decision, and the scores. The failover record is a count and a
  boolean, never a key value.
* **It is strictly additive.** The trace is built from the same objects the turn
  already produced; assembling it cannot change the contract fields, and when it is
  not requested it is not built at all.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from shl_recommender.conversation.policy import PolicyDecision
from shl_recommender.conversation.state import ConversationState
from shl_recommender.retrieval.types import ScoredItem


class ScoredCandidate(BaseModel):
    """One retrieval candidate and the final rank score it received."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entity_id: str
    test_type: str
    score: float


class StateView(BaseModel):
    """The understanding the turn was decided on, flattened for reading.

    Mirrors the decision-relevant fields of :class:`ConversationState`. It is a view,
    not the state itself, so the trace shape stays stable if the internal state grows.
    """

    model_config = ConfigDict(extra="forbid")

    role: str | None
    seniority: str | None
    domain: str | None
    purpose: str
    must_have_skills: list[str]
    optional_skills: list[str]
    languages: list[str]
    test_type_preferences: list[str]
    query_text: str
    # The one model-advised hinge: was the request judged specific enough to
    # recommend? ``None`` means the model gave no opinion and the structural rule
    # decided (see the policy's readiness fallback).
    ready_to_recommend: bool | None
    suggested_question: str | None
    # Deterministic signals that can override the readiness hinge.
    is_comparison: bool
    comparison_targets: list[str]
    wants_addition: bool
    wants_removal: bool
    is_off_topic: bool
    is_prompt_injection: bool
    user_confirmed: bool
    clarifications_asked: int
    has_prior_recommendations: bool


class TurnTrace(BaseModel):
    """The full, opt-in reasoning record for one turn.

    Returned under ``_trace`` on the ``/chat`` response only when ``?debug=1`` is set.
    Absent from the normal contract.
    """

    model_config = ConfigDict(extra="forbid")

    state: StateView
    mode: str
    reason: str
    commits_shortlist: bool
    end_of_conversation: bool
    # Up to the top-K in-scope candidates the ranker produced, with scores, so the
    # shortlist can be explained (why these, in this order). Empty on turns that do
    # not retrieve (clarify, refuse, compare).
    retrieval: list[ScoredCandidate]
    # Whether the model wrote the reply or a deterministic fallback did, and whether a
    # secondary-key failover happened on this turn. Booleans/counts only — no keys.
    reply_from_model: bool
    comparison_facts_resolved: bool


def _state_view(state: ConversationState) -> StateView:
    return StateView(
        role=state.role,
        seniority=state.seniority,
        domain=state.domain,
        purpose=state.purpose.value,
        must_have_skills=list(state.must_have_skills),
        optional_skills=list(state.optional_skills),
        languages=list(state.languages),
        test_type_preferences=list(state.test_type_preferences),
        query_text=state.query_text,
        ready_to_recommend=state.ready_to_recommend,
        suggested_question=state.suggested_question,
        is_comparison=state.is_comparison,
        comparison_targets=list(state.comparison_targets),
        wants_addition=state.wants_addition,
        wants_removal=state.wants_removal,
        is_off_topic=state.is_off_topic,
        is_prompt_injection=state.is_prompt_injection,
        user_confirmed=state.user_confirmed,
        clarifications_asked=state.clarifications_asked,
        has_prior_recommendations=state.has_prior_recommendations,
    )


def build_trace(
    state: ConversationState,
    decision: PolicyDecision,
    scored: list[ScoredItem],
    *,
    reply_from_model: bool,
    comparison_facts_resolved: bool,
) -> TurnTrace:
    """Assemble the trace from the objects the turn already produced."""
    return TurnTrace(
        state=_state_view(state),
        mode=decision.mode.value,
        reason=decision.reason,
        commits_shortlist=decision.commits_shortlist,
        end_of_conversation=decision.end_of_conversation,
        retrieval=[
            ScoredCandidate(
                name=s.item.name,
                entity_id=s.item.entity_id,
                test_type=s.item.test_type,
                score=round(s.score, 4),
            )
            for s in scored
        ],
        reply_from_model=reply_from_model,
        comparison_facts_resolved=comparison_facts_resolved,
    )
