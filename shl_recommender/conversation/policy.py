"""The policy engine: decide what each turn needs.

Given the reconstructed :class:`ConversationState`, this chooses the turn's
:class:`Mode` and the surrounding decisions (whether to commit or keep a
shortlist, whether the conversation ends). The reasoning behind the ordering is
written up in ``docs/policy_design.md``; in short it is a precedence of intents
that models how a careful consultant prioritises what a conversation needs:

1. handle unsafe/out-of-scope asks first, but refuse surgically;
2. honour the user's explicit intent (confirm, compare, edit) over inference;
3. commit once there is enough context, with an adaptive clarify budget;
4. let the user own closure.

The engine is pure and deterministic: same state in, same decision out. It does
not call the model or touch the catalog, which makes the behaviour easy to test
and to defend.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from shl_recommender.config import settings
from shl_recommender.conversation.state import ConversationState, Mode


class PolicyDecision(BaseModel):
    """The outcome of the policy for one turn."""

    model_config = ConfigDict(frozen=True)

    mode: Mode
    # Whether this turn should produce/keep a committed shortlist. CLARIFY and
    # REFUSE do not; RECOMMEND and REFINE do. COMPARE does not commit a new
    # shortlist but an existing one persists, which the response layer handles.
    commits_shortlist: bool
    end_of_conversation: bool
    # Short machine-readable reason, for logging and for the reply layer to know
    # why it is in this mode (e.g. distinguishing a refusal cause).
    reason: str


def _budget_remaining(state: ConversationState) -> bool:
    """Whether there is room to ask another clarifying question.

    Bounded two ways: by the question budget (how many we are willing to ask
    before a first shortlist) and by the turn cap (we must leave room to actually
    deliver a shortlist before the conversation is cut off).
    """
    under_question_budget = state.clarifications_asked < settings.max_clarifying_questions
    # Each clarification consumes a user+assistant pair. Keep at least one
    # exchange in hand to deliver the shortlist.
    asked_pairs = state.clarifications_asked
    under_turn_cap = (asked_pairs + 1) * 2 < settings.turn_cap
    return under_question_budget and under_turn_cap


def _ready_to_recommend(state: ConversationState) -> bool:
    """Whether the request is specific enough to commit to a shortlist now.

    Prefers the model's readiness judgement, which captures nuance rules cannot:
    a request can name many skills yet be too broad to be precise, or name few yet
    be perfectly clear. When the model gave no opinion (it failed, or was not
    asked), fall back to the structural minimum-context rule so the decision is
    always defined and the model is never a hard dependency.
    """
    if state.ready_to_recommend is not None:
        return state.ready_to_recommend
    return state.has_minimum_context()


def decide(state: ConversationState) -> PolicyDecision:
    """Choose the mode and decisions for the current turn."""

    # 1. Safety first, but surgical. A refusal never discards an existing
    #    shortlist; the response layer may re-show it alongside the refusal.
    if state.is_prompt_injection:
        return PolicyDecision(
            mode=Mode.REFUSE,
            commits_shortlist=False,
            end_of_conversation=False,
            reason="prompt_injection",
        )
    if state.is_off_topic:
        return PolicyDecision(
            mode=Mode.REFUSE,
            commits_shortlist=False,
            end_of_conversation=False,
            reason="off_topic",
        )

    # 2. The user accepts and there is something to accept -> close. If the closing
    #    turn *also* edits the list ("drop the OPQ, final list: ..."), it is a final
    #    edit-and-accept: still end, but commit a freshly refined shortlist that
    #    honours the edit rather than re-showing the prior one. The distinct reason
    #    tells the response layer to retrieve instead of recovering the old list.
    if state.user_confirmed and state.has_prior_recommendations:
        edits_on_close = state.wants_addition or state.wants_removal
        return PolicyDecision(
            mode=Mode.RECOMMEND if not edits_on_close else Mode.REFINE,
            commits_shortlist=True,
            end_of_conversation=True,
            reason="confirmed_with_edit" if edits_on_close else "user_confirmed",
        )

    # 3. Explicit comparison intent. Answered from catalog facts; no new commit.
    if state.is_comparison:
        return PolicyDecision(
            mode=Mode.COMPARE,
            commits_shortlist=False,
            end_of_conversation=False,
            reason="comparison_requested",
        )

    # 4. A shortlist exists and the user edits or questions it -> refine. Once a
    #    shortlist is on the table, an add/drop or a follow-up that carries real
    #    requirements is a refinement, not a restart.
    if state.has_prior_recommendations and (
        state.wants_addition or state.wants_removal or state.has_minimum_context()
    ):
        return PolicyDecision(
            mode=Mode.REFINE,
            commits_shortlist=True,
            end_of_conversation=False,
            reason="refinement",
        )

    # 5. Specific enough to commit, and nothing committed yet -> first shortlist.
    if _ready_to_recommend(state):
        return PolicyDecision(
            mode=Mode.RECOMMEND,
            commits_shortlist=True,
            end_of_conversation=False,
            reason="sufficient_context",
        )

    # 6. Not specific enough yet. Ask one useful question if there is room.
    if _budget_remaining(state):
        return PolicyDecision(
            mode=Mode.CLARIFY,
            commits_shortlist=False,
            end_of_conversation=False,
            reason="insufficient_context",
        )

    # 7. Out of room to clarify: commit to the best shortlist we can rather than
    #    spending the last turn on a question we cannot follow up on.
    return PolicyDecision(
        mode=Mode.RECOMMEND,
        commits_shortlist=True,
        end_of_conversation=False,
        reason="budget_exhausted_commit",
    )
