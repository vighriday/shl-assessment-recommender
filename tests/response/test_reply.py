"""Tests for reply generation.

Two properties matter most: the reply is never empty (every mode has a working
fallback), and when the model fails the fallback is used rather than an error
surfacing. CLARIFY is special-cased to reuse the understanding step's question
instead of a second model round-trip, so that is checked explicitly.
"""

from __future__ import annotations

from shl_recommender.conversation.policy import PolicyDecision
from shl_recommender.conversation.state import ConversationState, Mode
from shl_recommender.response.reply import ReplyWriter, _FALLBACK


def _decision(mode: Mode, *, reason: str = "x", end: bool = False, commits: bool = False) -> PolicyDecision:
    return PolicyDecision(
        mode=mode, commits_shortlist=commits, end_of_conversation=end, reason=reason
    )


def test_clarify_uses_the_models_suggested_question(working_llm):
    writer = ReplyWriter(working_llm)
    state = ConversationState(suggested_question="What seniority is the role?")
    reply = writer.write(_decision(Mode.CLARIFY), state, messages=[])
    assert reply == "What seniority is the role?"
    # No model call needed for clarify — the question already exists.
    assert working_llm.complete_calls == 0


def test_clarify_falls_back_when_no_question(working_llm):
    writer = ReplyWriter(working_llm)
    reply = writer.write(_decision(Mode.CLARIFY), ConversationState(), messages=[])
    assert reply == _FALLBACK[Mode.CLARIFY]


def test_recommend_uses_model_text_when_available(working_llm):
    writer = ReplyWriter(working_llm)
    reply = writer.write(
        _decision(Mode.RECOMMEND, reason="sufficient_context", commits=True),
        ConversationState(role="analyst"),
        messages=[],
        recommendation_count=5,
    )
    assert reply == "Here's a tailored set of assessments for that role."
    assert working_llm.complete_calls == 1


def test_recommend_falls_back_when_model_fails(failing_llm):
    writer = ReplyWriter(failing_llm)
    reply = writer.write(
        _decision(Mode.RECOMMEND, reason="sufficient_context", commits=True),
        ConversationState(role="analyst"),
        messages=[],
        recommendation_count=5,
    )
    assert reply == _FALLBACK[Mode.RECOMMEND]


def test_every_mode_has_a_nonempty_fallback(failing_llm):
    # With the model failing, every mode must still yield its fallback string.
    writer = ReplyWriter(failing_llm)
    for mode in (Mode.RECOMMEND, Mode.REFINE, Mode.COMPARE, Mode.REFUSE):
        reply = writer.write(_decision(mode), ConversationState(), messages=[])
        assert reply == _FALLBACK[mode]
        assert reply  # non-empty


def test_refusal_cause_shapes_the_instruction_but_still_falls_back(failing_llm):
    # Injection vs off-topic take different instructions; with the model down both
    # resolve to the refuse fallback, and neither raises.
    writer = ReplyWriter(failing_llm)
    injection = writer.write(
        _decision(Mode.REFUSE, reason="prompt_injection"), ConversationState(), messages=[]
    )
    off_topic = writer.write(
        _decision(Mode.REFUSE, reason="off_topic"), ConversationState(), messages=[]
    )
    assert injection == _FALLBACK[Mode.REFUSE]
    assert off_topic == _FALLBACK[Mode.REFUSE]


def test_model_returning_blank_is_treated_as_failure(working_llm):
    # A model that returns whitespace must not produce an empty reply.
    from tests.response.conftest import FakeLLM

    writer = ReplyWriter(FakeLLM(text="   "))
    reply = writer.write(_decision(Mode.COMPARE), ConversationState(), messages=[])
    assert reply == _FALLBACK[Mode.COMPARE]


def test_comparison_facts_are_passed_into_the_prompt():
    # When catalog facts are supplied, the compare instruction must include them so
    # the model grounds its answer. We capture the system prompt the writer sends.
    from tests.response.conftest import FakeLLM

    captured = {}

    class CapturingLLM(FakeLLM):
        def complete(self, messages, *, temperature: float = 0.2) -> str:
            captured["system"] = messages[0]["content"]
            return "Grounded comparison."

    writer = ReplyWriter(CapturingLLM())
    facts = "- OPQ32r (test_type P, 25 minutes): measures workplace behaviour."
    reply = writer.write(
        _decision(Mode.COMPARE),
        ConversationState(comparison_targets=("OPQ32r", "DSI")),
        messages=[],
        comparison_facts=facts,
    )
    assert reply == "Grounded comparison."
    assert facts in captured["system"]  # the real facts reached the model
    assert "do not invent" in captured["system"].lower()
