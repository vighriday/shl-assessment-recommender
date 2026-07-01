"""Tests for the ConversationState shape and its minimum-context rule."""

from __future__ import annotations

import pytest

from shl_recommender.conversation.state import ConversationState, Mode, Purpose


def test_defaults_are_empty_and_safe():
    state = ConversationState()
    assert state.role is None
    assert state.purpose is Purpose.UNKNOWN
    assert state.must_have_skills == ()
    assert state.clarifications_asked == 0
    assert state.has_minimum_context() is False


def test_state_is_immutable():
    state = ConversationState()
    with pytest.raises(Exception):
        state.role = "developer"


def test_role_plus_seniority_is_enough_context():
    state = ConversationState(role="Java developer", seniority="mid-level")
    assert state.has_minimum_context() is True


def test_role_alone_is_not_enough_context():
    # A target with no differentiator should still be clarified.
    state = ConversationState(role="developer")
    assert state.has_minimum_context() is False


def test_long_query_counts_as_a_target():
    # A pasted job description has no parsed role yet but is substantive enough
    # to act on when paired with a differentiator.
    jd = "We are hiring a backend engineer who collaborates with stakeholders daily"
    state = ConversationState(query_text=jd, must_have_skills=("python",))
    assert len(jd) >= 40
    assert state.has_minimum_context() is True


def test_requested_category_is_a_differentiator():
    state = ConversationState(role="manager", test_type_preferences=("personality",))
    assert state.has_minimum_context() is True


def test_language_is_a_differentiator():
    state = ConversationState(role="support agent", languages=("English (USA)",))
    assert state.has_minimum_context() is True


@pytest.mark.parametrize("mode", list(Mode))
def test_all_modes_have_string_values(mode):
    assert isinstance(mode.value, str)


def test_mode_values_are_stable():
    # These strings are referenced across the policy engine and logs; pin them.
    assert {m.value for m in Mode} == {"clarify", "recommend", "refine", "compare", "refuse"}
