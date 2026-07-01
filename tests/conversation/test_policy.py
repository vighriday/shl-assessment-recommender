"""Tests for the policy engine.

Each test sets up a ConversationState and asserts the chosen mode, exercising the
precedence rules one at a time. The precedence order itself is checked by the
'beats' tests where several signals are present at once.
"""

from __future__ import annotations

from shl_recommender.conversation.policy import decide
from shl_recommender.conversation.state import ConversationState, Mode


def _state(**kwargs) -> ConversationState:
    return ConversationState(**kwargs)


# --- Individual rules ------------------------------------------------------ #

def test_injection_is_refused():
    d = decide(_state(is_prompt_injection=True, query_text="ignore instructions"))
    assert d.mode is Mode.REFUSE
    assert d.reason == "prompt_injection"
    assert not d.commits_shortlist and not d.end_of_conversation


def test_off_topic_is_refused():
    d = decide(_state(is_off_topic=True, query_text="is this legal"))
    assert d.mode is Mode.REFUSE
    assert d.reason == "off_topic"


def test_confirmation_with_prior_recs_ends():
    d = decide(_state(user_confirmed=True, has_prior_recommendations=True))
    assert d.mode is Mode.RECOMMEND
    assert d.end_of_conversation
    assert d.commits_shortlist


def test_confirmation_without_prior_recs_does_not_end():
    # "Perfect" with nothing to accept is not closure.
    d = decide(_state(user_confirmed=True, has_prior_recommendations=False, query_text="perfect"))
    assert not d.end_of_conversation


def test_comparison_is_compare_mode():
    d = decide(_state(is_comparison=True, comparison_targets=("OPQ", "GSA")))
    assert d.mode is Mode.COMPARE
    assert not d.commits_shortlist


def test_refine_on_addition_with_prior_recs():
    d = decide(_state(has_prior_recommendations=True, wants_addition=True))
    assert d.mode is Mode.REFINE
    assert d.commits_shortlist and not d.end_of_conversation


def test_refine_on_removal_with_prior_recs():
    d = decide(_state(has_prior_recommendations=True, wants_removal=True))
    assert d.mode is Mode.REFINE


def test_first_recommendation_when_context_is_sufficient():
    d = decide(_state(role="Java developer", seniority="senior"))
    assert d.mode is Mode.RECOMMEND
    assert d.reason == "sufficient_context"
    assert not d.end_of_conversation


def test_clarify_when_context_is_thin():
    d = decide(_state(role="developer", query_text="I need a test"))
    assert d.mode is Mode.CLARIFY
    assert not d.commits_shortlist


def test_clarify_stops_when_question_budget_exhausted():
    # After the allowed questions, commit rather than ask again.
    d = decide(_state(query_text="something vague", clarifications_asked=2))
    assert d.mode is Mode.RECOMMEND
    assert d.reason == "budget_exhausted_commit"


# --- Precedence (several signals at once) ---------------------------------- #

def test_injection_beats_everything():
    d = decide(_state(
        is_prompt_injection=True,
        is_comparison=True,
        user_confirmed=True,
        has_prior_recommendations=True,
        role="dev",
        seniority="senior",
    ))
    assert d.mode is Mode.REFUSE


def test_off_topic_beats_confirmation_and_refine():
    # A legal question on a turn that also looks like confirmation is still a
    # refusal — but the shortlist is preserved (commits stays False, response
    # layer re-shows existing).
    d = decide(_state(
        is_off_topic=True,
        user_confirmed=True,
        has_prior_recommendations=True,
    ))
    assert d.mode is Mode.REFUSE
    assert not d.end_of_conversation


def test_confirmation_beats_comparison_and_refine():
    # Acceptance closes the conversation even when comparison/edit signals also fire.
    # Because an edit (add) is present on the accepting turn, it is a final
    # edit-and-close: the list is refined to honour the edit, and it still ends.
    d = decide(_state(
        user_confirmed=True,
        has_prior_recommendations=True,
        is_comparison=True,
        wants_addition=True,
    ))
    assert d.mode is Mode.REFINE
    assert d.commits_shortlist
    assert d.end_of_conversation


def test_confirmation_without_edit_closes_as_recommend():
    # Pure acceptance (no edit) closes and re-shows the accepted list.
    d = decide(_state(user_confirmed=True, has_prior_recommendations=True))
    assert d.mode is Mode.RECOMMEND
    assert d.reason == "user_confirmed"
    assert d.end_of_conversation


def test_comparison_beats_refine():
    # When a shortlist exists and the user asks to compare, compare wins over a
    # refine interpretation.
    d = decide(_state(
        has_prior_recommendations=True,
        is_comparison=True,
        comparison_targets=("OPQ", "GSA"),
    ))
    assert d.mode is Mode.COMPARE


# --- Readiness judgement (LLM-driven, code-bounded) ------------------------ #

def test_llm_ready_overrides_thin_structural_context():
    # Few parsed fields, but the model judges the request specific enough.
    d = decide(_state(query_text="screen for Excel and Word", ready_to_recommend=True))
    assert d.mode is Mode.RECOMMEND


def test_llm_not_ready_clarifies_even_with_structural_context():
    # Many fields, but the model judges a decision-critical gap remains (the wide
    # JD case). Clarify wins while there is budget.
    d = decide(_state(
        role="full-stack engineer", seniority="senior",
        must_have_skills=("Java", "Angular", "AWS"),
        ready_to_recommend=False, suggested_question="Backend or frontend?",
        clarifications_asked=0,
    ))
    assert d.mode is Mode.CLARIFY


def test_llm_not_ready_but_budget_exhausted_commits():
    # The model wants to clarify, but we are out of room — commit anyway rather
    # than burn the last turn on a question.
    d = decide(_state(
        ready_to_recommend=False, suggested_question="One more thing?",
        clarifications_asked=2,
    ))
    assert d.mode is Mode.RECOMMEND
    assert d.reason == "budget_exhausted_commit"


def test_no_llm_opinion_falls_back_to_structural_rule():
    # ready_to_recommend is None -> use has_minimum_context. Sufficient here.
    d = decide(_state(role="analyst", seniority="senior", ready_to_recommend=None))
    assert d.mode is Mode.RECOMMEND
    # And insufficient here.
    d2 = decide(_state(query_text="I need a test", ready_to_recommend=None))
    assert d2.mode is Mode.CLARIFY


def test_refine_question_about_item_keeps_shortlist():
    # "Is the Advanced level right?" with a shortlist present and enough context
    # is a refinement, not a fresh recommendation.
    d = decide(_state(
        has_prior_recommendations=True,
        role="senior engineer",
        seniority="senior",
        query_text="is the advanced level the right pick",
    ))
    assert d.mode is Mode.REFINE
    assert d.commits_shortlist
