"""Tests for the /chat request and response schemas.

These lock the API contract: the response must be exactly the three required
fields, recommendations must be null or 1..10 items (never []), test_type codes
must be valid, and the null-vs-[] switch must work.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from shl_recommender.api.schemas import (
    MAX_RECOMMENDATIONS,
    ChatRequest,
    ChatResponse,
    Message,
    Recommendation,
)


def _rec(name="OPQ32r", url="https://www.shl.com/x/", test_type="P") -> Recommendation:
    return Recommendation(name=name, url=url, test_type=test_type)


# --- Request --------------------------------------------------------------- #

def test_request_parses_full_history():
    req = ChatRequest(
        messages=[
            {"role": "user", "content": "Hiring a Java developer"},
            {"role": "assistant", "content": "What seniority?"},
            {"role": "user", "content": "Mid-level"},
        ]
    )
    assert req.latest_user_message == "Mid-level"


def test_request_ignores_unknown_message_keys():
    # A client may attach metadata; we depend only on role and content.
    req = ChatRequest(messages=[{"role": "user", "content": "hi", "timestamp": 123}])
    assert req.messages[0].content == "hi"


def test_request_requires_at_least_one_message():
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_request_requires_a_user_message():
    with pytest.raises(ValidationError, match="at least one user message"):
        ChatRequest(messages=[{"role": "assistant", "content": "hello"}])


def test_request_rejects_unknown_role():
    with pytest.raises(ValidationError):
        Message(role="moderator", content="x")


@pytest.mark.parametrize(
    "raw_role,expected",
    [
        ("User", "user"),
        ("USER", "user"),
        ("Assistant", "assistant"),
        ("Agent", "assistant"),
        ("agent", "assistant"),
        ("Human", "user"),
        ("  user  ", "user"),
    ],
)
def test_role_casing_and_synonyms_are_normalised(raw_role, expected):
    # A casing or synonym difference must not reject the request — that would fail
    # a hard eval over nothing.
    assert Message(role=raw_role, content="x").role == expected


def test_latest_user_message_picks_the_most_recent():
    req = ChatRequest(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    )
    assert req.latest_user_message == "second"


# --- Response -------------------------------------------------------------- #

def test_clarify_turn_serialises_recommendations_as_null():
    resp = ChatResponse(reply="What seniority?")
    payload = json.loads(resp.model_dump_json())
    assert payload == {
        "reply": "What seniority?",
        "recommendations": None,
        "end_of_conversation": False,
    }


def test_response_has_exactly_three_fields():
    resp = ChatResponse(reply="x")
    assert set(resp.model_dump().keys()) == {"reply", "recommendations", "end_of_conversation"}


def test_response_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ChatResponse(reply="x", note="should not be here")


def test_committed_shortlist_is_allowed():
    resp = ChatResponse(reply="here", recommendations=[_rec()])
    assert len(resp.recommendations) == 1


def test_empty_list_is_rejected():
    # "No shortlist" must be null, not [].
    with pytest.raises(ValidationError, match="null"):
        ChatResponse(reply="x", recommendations=[])


def test_more_than_ten_is_rejected():
    too_many = [_rec(name=str(i)) for i in range(MAX_RECOMMENDATIONS + 1)]
    with pytest.raises(ValidationError, match="exceed"):
        ChatResponse(reply="x", recommendations=too_many)


def test_exactly_ten_is_allowed():
    ten = [_rec(name=str(i)) for i in range(MAX_RECOMMENDATIONS)]
    assert len(ChatResponse(reply="x", recommendations=ten).recommendations) == 10


def test_recommendation_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Recommendation(name="x", url="https://www.shl.com/x/", test_type="K", score=0.9)


def test_recommendation_rejects_unknown_test_type():
    with pytest.raises(ValidationError, match="unknown test_type"):
        Recommendation(name="x", url="https://www.shl.com/x/", test_type="Z")


def test_recommendation_accepts_multi_code_test_type():
    assert _rec(test_type="K,S").test_type == "K,S"


def test_recommendation_rejects_partially_unknown_multi_code():
    with pytest.raises(ValidationError):
        _rec(test_type="K,Z")


# --- null vs [] switch ----------------------------------------------------- #

def test_to_payload_defaults_to_null():
    resp = ChatResponse(reply="x")
    assert resp.to_payload()["recommendations"] is None


def test_to_payload_can_emit_empty_list():
    resp = ChatResponse(reply="x")
    assert resp.to_payload(empty_as_null=False)["recommendations"] == []


def test_to_payload_keeps_committed_list_regardless_of_flag():
    resp = ChatResponse(reply="x", recommendations=[_rec()])
    assert resp.to_payload(empty_as_null=False)["recommendations"][0]["name"] == "OPQ32r"
