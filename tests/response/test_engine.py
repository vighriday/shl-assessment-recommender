"""End-to-end tests for the response engine.

These drive a full turn — history in, ``ChatResponse`` out — with a fake model and a
small real catalog, so the wiring between every phase is exercised without a network
or a server. The focus is the contract at the seams: which modes commit a shortlist,
that non-commit modes return ``null`` (not []), that ``end_of_conversation`` follows
the policy, and that a model outage never breaks the turn.
"""

from __future__ import annotations

import pytest

from shl_recommender.api.schemas import ChatResponse, Message
from shl_recommender.catalog.models import CatalogItem
from shl_recommender.response.engine import ResponseEngine
from shl_recommender.retrieval.ranker import LexicalRanker

from tests.response.conftest import FakeLLM


def _item(entity_id: str, name: str, code: str = "K") -> CatalogItem:
    return CatalogItem(
        entity_id=entity_id,
        name=name,
        url=f"https://www.shl.com/products/product-catalog/view/{entity_id}/",
        description=f"{name} assessment for screening candidates",
        keys=("Knowledge & Skills",),
        test_type=code,
        search_text=f"{name} assessment screening".lower(),
    )


@pytest.fixture(scope="module")
def catalog() -> list[CatalogItem]:
    # A handful of items is enough to exercise retrieval and assembly.
    return [
        _item("1", "Java Programming"),
        _item("2", "Python Programming"),
        _item("3", "SQL Database"),
        _item("720", "OPQ32r", code="P"),
        _item("3971", "Verify G+", code="A"),
    ]


def _engine(catalog, llm: FakeLLM) -> ResponseEngine:
    retriever = LexicalRanker(catalog)
    return ResponseEngine(retriever, llm)


def _msgs(*turns: tuple[str, str]) -> list[Message]:
    return [Message(role=role, content=content) for role, content in turns]


def test_recommend_turn_commits_a_valid_shortlist(catalog):
    # Model reports a clear, ready request naming a skill in the catalog.
    llm = FakeLLM(
        text="Here are the assessments I'd recommend.",
        json={"role": "developer", "must_have_skills": ["Java"], "ready_to_recommend": True},
    )
    engine = _engine(catalog, llm)
    response = engine.respond(_msgs(("user", "I'm hiring a Java developer, screen for Java skills.")))

    assert isinstance(response, ChatResponse)
    assert response.recommendations is not None
    assert 1 <= len(response.recommendations) <= 10
    assert response.reply
    assert response.end_of_conversation is False
    # Payload serialises to exactly the contract shape.
    payload = response.to_payload()
    assert set(payload) == {"reply", "recommendations", "end_of_conversation"}


def test_vague_opener_clarifies_with_no_shortlist(catalog):
    # Nothing extractable, not ready -> clarify. Model supplies the question.
    llm = FakeLLM(
        json={"ready_to_recommend": False, "clarifying_question": "What role are you hiring for?"}
    )
    engine = _engine(catalog, llm)
    response = engine.respond(_msgs(("user", "I need some assessments.")))

    assert response.recommendations is None  # null, not []
    assert response.reply == "What role are you hiring for?"
    assert response.end_of_conversation is False


def test_off_topic_is_refused_with_no_shortlist(catalog):
    # An off-topic ask is caught by deterministic signals; no model JSON needed.
    llm = FakeLLM()
    engine = _engine(catalog, llm)
    response = engine.respond(_msgs(("user", "Can you give me legal advice about firing someone?")))

    assert response.recommendations is None
    assert response.reply
    assert response.end_of_conversation is False


def test_confirmation_after_a_shortlist_ends_the_conversation(catalog):
    llm = FakeLLM(text="Great — finalising those for you.")
    engine = _engine(catalog, llm)
    # Prior assistant turn carries a catalog URL, so the state knows a shortlist
    # was already offered; the user then accepts.
    response = engine.respond(
        _msgs(
            ("user", "Hiring a Java developer."),
            (
                "assistant",
                "I'd suggest https://www.shl.com/products/product-catalog/view/1/ for that.",
            ),
            ("user", "Perfect, let's go with that."),
        )
    )
    assert response.end_of_conversation is True
    assert response.recommendations is not None  # the confirmed shortlist is committed


def test_turn_succeeds_even_when_model_is_down(catalog):
    # The whole model is failing: understanding falls back to empty, reply falls
    # back to a template, and the turn still returns a valid response.
    engine = _engine(catalog, FakeLLM(fail=True))
    response = engine.respond(_msgs(("user", "I'm hiring a Java developer, screen for Java.")))
    assert isinstance(response, ChatResponse)
    assert response.reply  # non-empty fallback
    # Serialises cleanly regardless of the model outage.
    assert set(response.to_payload()) == {"reply", "recommendations", "end_of_conversation"}


# --- D6: comparison grounded in catalog facts -------------------------------

def test_comparison_targets_resolve_to_catalog_facts(catalog):
    from shl_recommender.conversation.policy import PolicyDecision
    from shl_recommender.conversation.state import ConversationState, Mode

    engine = _engine(catalog, FakeLLM())
    decision = PolicyDecision(
        mode=Mode.COMPARE, commits_shortlist=False, end_of_conversation=False,
        reason="comparison_requested",
    )
    state = ConversationState(comparison_targets=("Java Programming", "OPQ32r"))
    facts = engine._comparison_facts(decision, state)
    assert facts is not None
    # Both named products are present, with their real test_type from the catalog.
    assert "Java Programming" in facts
    assert "OPQ32r" in facts
    assert "test_type P" in facts  # OPQ32r's code


def test_comparison_facts_none_when_targets_unresolvable(catalog):
    from shl_recommender.conversation.policy import PolicyDecision
    from shl_recommender.conversation.state import ConversationState, Mode

    engine = _engine(catalog, FakeLLM())
    decision = PolicyDecision(
        mode=Mode.COMPARE, commits_shortlist=False, end_of_conversation=False,
        reason="comparison_requested",
    )
    # A target that matches no catalog product resolves to nothing -> None, so the
    # reply keeps to safe framing rather than an empty facts block.
    state = ConversationState(comparison_targets=("Nonexistent Widget XYZ",))
    assert engine._comparison_facts(decision, state) is None


def test_comparison_facts_only_on_compare_turn(catalog):
    from shl_recommender.conversation.policy import PolicyDecision
    from shl_recommender.conversation.state import ConversationState, Mode

    engine = _engine(catalog, FakeLLM())
    # A non-compare decision must not compute comparison facts even if targets exist.
    decision = PolicyDecision(
        mode=Mode.RECOMMEND, commits_shortlist=True, end_of_conversation=False, reason="x",
    )
    state = ConversationState(comparison_targets=("OPQ32r",))
    assert engine._comparison_facts(decision, state) is None


# --- Opt-in turn trace -------------------------------------------------------


def test_respond_does_not_build_a_trace(catalog):
    # The normal path returns no trace: it is strictly opt-in.
    llm = FakeLLM(
        text="Here you go.",
        json={"role": "developer", "must_have_skills": ["Java"], "ready_to_recommend": True},
    )
    engine = _engine(catalog, llm)
    response, trace = engine.respond_with_trace(
        _msgs(("user", "Hiring a Java developer, screen Java.")), trace=False
    )
    assert isinstance(response, ChatResponse)
    assert trace is None


def test_trace_reports_decision_and_scored_candidates(catalog):
    # A ready recommend turn: the trace exposes the mode, the reason, the readiness
    # the decision used, and the ranked candidates with scores.
    llm = FakeLLM(
        text="Here are the assessments I'd recommend.",
        json={"role": "developer", "must_have_skills": ["Java"], "ready_to_recommend": True},
    )
    engine = _engine(catalog, llm)
    response, trace = engine.respond_with_trace(
        _msgs(("user", "Hiring a Java developer, screen Java skills."))
    )

    assert trace is not None
    assert trace.mode == "recommend"
    assert trace.commits_shortlist is True
    assert trace.state.ready_to_recommend is True
    assert trace.reply_from_model is True
    # The shortlist is explainable: candidates are present, scored, and ordered high
    # to low, and their count matches what was returned.
    assert trace.retrieval
    scores = [c.score for c in trace.retrieval]
    assert scores == sorted(scores, reverse=True)
    assert len(trace.retrieval) >= len(response.recommendations or [])


def test_trace_marks_fallback_when_model_is_down(catalog):
    # With the model unavailable, the reply is a deterministic fallback and the trace
    # says so — reply_from_model is False.
    llm = FakeLLM(fail=True)
    engine = _engine(catalog, llm)
    _, trace = engine.respond_with_trace(_msgs(("user", "I need some assessments.")))

    assert trace is not None
    assert trace.reply_from_model is False


def test_trace_never_carries_recommendations_on_a_clarify_turn(catalog):
    # A clarify turn does not retrieve, so the trace's candidate list is empty and the
    # contract still shows null recommendations.
    llm = FakeLLM(
        json={"ready_to_recommend": False, "clarifying_question": "What role?"}
    )
    engine = _engine(catalog, llm)
    response, trace = engine.respond_with_trace(_msgs(("user", "I need help.")))

    assert response.recommendations is None
    assert trace is not None
    assert trace.mode == "clarify"
    assert trace.retrieval == []


def test_debug_flag_adds_trace_without_changing_the_contract(catalog):
    # Through the HTTP layer: ?debug=1 attaches _trace; the three contract fields are
    # byte-for-byte identical with and without it.
    from starlette.testclient import TestClient

    from shl_recommender.api.app import create_app
    from shl_recommender.config import Settings

    config = Settings()
    llm = FakeLLM(
        text="Here are the assessments I'd recommend.",
        json={"role": "developer", "must_have_skills": ["Java"], "ready_to_recommend": True},
    )
    client = TestClient(create_app(config=config, llm_client=llm))
    body = {"messages": [{"role": "user", "content": "Hiring a Java developer, screen Java."}]}

    clean = client.post("/chat", json=body).json()
    debug = client.post("/chat?debug=1", json=body).json()

    assert set(clean) == {"reply", "recommendations", "end_of_conversation"}
    assert set(debug) == {"reply", "recommendations", "end_of_conversation", "_trace"}
    # The contract portion is unchanged by asking for the trace.
    assert {k: debug[k] for k in clean} == clean
    assert debug["_trace"]["mode"] == "recommend"
