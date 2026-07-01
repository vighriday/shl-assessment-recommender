"""Verify the policy reproduces the sample conversations' decisions.

This is a check, not the design target: the policy is built from principles
(docs/policy_design.md), and these tests confirm those principles produce the
behaviour the ten samples demonstrate. Each case is the user side of a real
trace, turn by turn, with the expected mode and end flag at each step.

State is reconstructed exactly as in production (deterministic signals + catalog
vocabulary), with a scripted understanding double standing in for the model so
the test is offline and stable. The understanding values mirror what the model
would reasonably extract from each turn — enough for the context gate to behave
as it does in the trace.
"""

from __future__ import annotations

import pytest

from shl_recommender.api.schemas import Message
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.conversation.extractor import reconstruct_state
from shl_recommender.conversation.policy import decide
from shl_recommender.conversation.state import Mode
from shl_recommender.config import settings


@pytest.fixture(scope="module")
def vocab():
    return build_vocabulary(load_catalog(settings.raw_catalog_path))


class ScriptedLLM:
    """Returns a preset understanding per turn, keyed by how many user turns exist.

    The script lets each trace supply the understanding the model would produce at
    each point, so the policy is exercised on realistic state without a network.
    """

    def __init__(self, by_user_turn: dict[int, dict]):
        self._by_user_turn = by_user_turn

    def complete(self, messages, **kwargs):
        return "reply"

    def complete_json(self, messages, **kwargs):
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        return self._by_user_turn.get(user_turns, {})


def _run(turns: list[tuple[str, str]], understanding_script: dict[int, dict], vocab):
    """Replay a conversation, returning the policy decision at each user turn.

    ``turns`` is the full (role, content) sequence. We evaluate the policy at each
    point where the latest message is from the user, which is where the agent must
    decide.
    """
    decisions = []
    history: list[Message] = []
    llm = ScriptedLLM(understanding_script)
    for role, content in turns:
        history.append(Message(role=role, content=content))
        if role == "user":
            state = reconstruct_state(history, llm, vocabulary=vocab)
            decisions.append(decide(state))
    return decisions


# A compact reply that contains a catalog URL, so the next turn sees
# has_prior_recommendations=True (production detects prior shortlists this way).
_REC_REPLY = "Here: <https://www.shl.com/products/product-catalog/view/opq/>"


def test_c1_vague_leadership_clarifies_twice_then_commits(vocab):
    turns = [
        ("user", "We need a solution for senior leadership."),
        ("assistant", "Who is this meant for?"),
        ("user", "CXOs, director-level; people with more than 15 years of experience."),
        ("assistant", "Newly created position, or developmental feedback?"),
        ("user", "Selection — comparing candidates against a leadership benchmark."),
        (_REC_REPLY and "assistant", _REC_REPLY),
        ("user", "Perfect, that's what we need."),
    ]
    script = {
        1: {"role": "senior leadership", "ready_to_recommend": False,
            "clarifying_question": "Who is this for?"},
        2: {"role": "senior leadership / CXO", "seniority": "executive",
            "ready_to_recommend": False, "clarifying_question": "Selection or development?"},
        3: {"role": "senior leadership / CXO", "seniority": "executive", "purpose": "selection",
            "ready_to_recommend": True},
        4: {"role": "senior leadership / CXO", "seniority": "executive", "purpose": "selection",
            "ready_to_recommend": True},
    }
    d = _run(turns, script, vocab)
    assert d[0].mode is Mode.CLARIFY
    assert d[1].mode is Mode.CLARIFY
    assert d[2].mode is Mode.RECOMMEND and not d[2].end_of_conversation
    assert d[3].mode is Mode.RECOMMEND and d[3].end_of_conversation  # confirmation


def test_c4_specific_opener_commits_on_first_turn(vocab):
    turns = [
        ("user", "Hiring graduate financial analysts — final-year students, no experience. "
                 "We need numerical reasoning and a finance knowledge test."),
    ]
    script = {1: {"role": "graduate financial analyst", "seniority": "graduate",
                  "must_have_skills": ["numerical reasoning", "finance"],
                  "ready_to_recommend": True}}
    d = _run(turns, script, vocab)
    assert d[0].mode is Mode.RECOMMEND and not d[0].end_of_conversation


def test_c3_screening_clarifies_then_commits_then_compares(vocab):
    turns = [
        ("user", "We're screening 500 entry-level contact centre agents. Inbound calls. What should we use?"),
        ("assistant", "What language are the calls in?"),
        ("user", "English."),
        ("assistant", "SVAR has four English variants. Which fits?"),
        ("user", "US."),
        ("assistant", _REC_REPLY),
        ("user", "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?"),
    ]
    script = {
        1: {"role": "contact centre agent", "purpose": "screening",
            "ready_to_recommend": False, "clarifying_question": "What language are the calls in?"},
        2: {"role": "contact centre agent", "purpose": "screening", "languages": ["English"],
            "ready_to_recommend": False, "clarifying_question": "Which English accent variant?"},
        3: {"role": "contact centre agent", "purpose": "screening", "languages": ["English (USA)"],
            "ready_to_recommend": True},
    }
    d = _run(turns, script, vocab)
    assert d[0].mode is Mode.CLARIFY
    assert d[1].mode is Mode.CLARIFY
    assert d[2].mode is Mode.RECOMMEND
    assert d[3].mode is Mode.COMPARE


def test_c7_clarify_commit_then_refuse_legal(vocab):
    turns = [
        ("user", "Bilingual healthcare admin in South Texas, assessed in Spanish. HIPAA critical. What works?"),
        ("assistant", "Which fits your candidate pool — hybrid or Spanish-only?"),
        ("user", "Functionally bilingual. Go with the hybrid."),
        ("assistant", _REC_REPLY),
        ("user", "Are we legally required under HIPAA to test all staff who touch patient records?"),
    ]
    script = {
        1: {"role": "healthcare admin", "languages": ["Spanish"], "domain": "healthcare",
            "ready_to_recommend": False,
            "clarifying_question": "Are candidates bilingual enough for English knowledge tests?"},
        2: {"role": "healthcare admin", "languages": ["Spanish"], "domain": "healthcare",
            "ready_to_recommend": True},
    }
    d = _run(turns, script, vocab)
    assert d[0].mode is Mode.CLARIFY
    assert d[1].mode is Mode.RECOMMEND
    assert d[2].mode is Mode.REFUSE and not d[2].end_of_conversation  # partial refusal


def test_c9_jd_clarifies_twice_commits_then_refines(vocab):
    turns = [
        ("user", "Here's the JD for a Senior Full-Stack Engineer, 5+ years Java, Spring, REST, "
                 "Angular, SQL, AWS, Docker. Owns microservice delivery, mentors. Recommend a battery?"),
        ("assistant", "Backend-leaning, frontend-heavy, or balanced?"),
        ("user", "Backend-leaning. Java and Spring primary; SQL constant; Angular occasional."),
        ("assistant", "Senior IC or tech lead?"),
        ("user", "Senior IC. Leads design on their services but doesn't manage engineers."),
        ("assistant", _REC_REPLY),
        ("user", "Add AWS and Docker. Drop REST."),
    ]
    script = {
        1: {"role": "full-stack engineer", "seniority": "senior",
            "must_have_skills": ["Java", "Spring", "SQL", "Angular", "AWS", "Docker"],
            "ready_to_recommend": False,
            "clarifying_question": "Backend-leaning, frontend-heavy, or balanced?"},
        2: {"role": "backend engineer", "seniority": "senior",
            "must_have_skills": ["Java", "Spring", "SQL"],
            "ready_to_recommend": False, "clarifying_question": "Senior IC or tech lead?"},
        3: {"role": "backend engineer", "seniority": "senior IC",
            "must_have_skills": ["Java", "Spring", "SQL"], "ready_to_recommend": True},
    }
    d = _run(turns, script, vocab)
    assert d[0].mode is Mode.CLARIFY
    assert d[1].mode is Mode.CLARIFY
    assert d[2].mode is Mode.RECOMMEND
    assert d[3].mode is Mode.REFINE  # add AWS/Docker, drop REST


def test_c10_commit_then_refine_drop_and_end(vocab):
    turns = [
        ("user", "Graduate management trainee scheme. Full battery — cognitive, personality, SJT. All graduates."),
        ("assistant", _REC_REPLY),
        ("user", "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."),
    ]
    script = {
        1: {"role": "graduate management trainee", "seniority": "graduate",
            "test_type_preferences": ["cognitive", "personality", "situational judgement"],
            "ready_to_recommend": True},
        2: {"role": "graduate management trainee", "seniority": "graduate"},
    }
    d = _run(turns, script, vocab)
    assert d[0].mode is Mode.RECOMMEND
    # The final turn drops OPQ *and* accepts ("Final list: ...") — the trace marks it
    # end_of_conversation: true, so it is a final edit-and-close: a refined shortlist
    # honouring the drop, and the conversation ends.
    assert d[1].mode is Mode.REFINE  # a fresh refined list, not a re-show
    assert d[1].end_of_conversation is True
    assert d[1].reason == "confirmed_with_edit"
