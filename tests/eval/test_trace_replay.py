"""Deterministic replay of the sample conversations through the engine.

This replays each trace's *actual user turns* through a real ``ResponseEngine`` and
checks that the behaviour matches the trace, turn by turn — the closest thing to the
grader we can run offline. The language model is faked so the suite is deterministic
and needs no key; the fake supplies the understanding the model *would* extract (the
per-trace hints already used by the recall scoreboard) and a readiness signal derived
from whether the trace itself had committed a shortlist by that turn. Everything else
— the deterministic signals (comparison, legal, injection), the policy, retrieval, and
assembly — is the real code path.

We assert the behaviours that must hold regardless of the exact wording, not the exact
mode label on every turn (which depends on model nuance we are faking):

* the final turn ends the conversation and carries a shortlist;
* a pure comparison turn does not commit a new shortlist (recommendations null);
* every turn returns the valid contract shape and never errors;
* the committed shortlist recalls the trace's gold items well.

The live counterpart (`scripts/replay_traces.py`) runs the same replay against the
real model for a pre-submission check.
"""

from __future__ import annotations

import glob
import os

import pytest

from shl_recommender.api.schemas import Message
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.config import settings
from shl_recommender.response.engine import ResponseEngine
from shl_recommender.retrieval.ranker import LexicalRanker

from scripts.measure_recall import _HINTS
from scripts.trace_utils import all_gold_urls, messages_up_to, parse_turns


def _as_messages(raw: list[dict]) -> list[Message]:
    """The engine consumes Message objects (as the API delivers), not raw dicts."""
    return [Message(role=m["role"], content=m["content"]) for m in raw]

# Comparison turns in the traces, keyed by (trace, turn index). On these the user
# asks to compare products rather than for a new shortlist; the engine should not
# commit one. Sourced by reading the traces (C5 T2, C6 T2).
_COMPARISON_TURNS = {("C5", 2), ("C6", 2)}


class _HintLLM:
    """A fake model that returns a trace's extracted understanding.

    ``ready`` is the readiness verdict for the turn being replayed; it stands in for
    the model's judgement so the policy's clarify-vs-commit gate behaves as it would
    once the conversation is specific enough.
    """

    def __init__(self, hints: dict, *, ready: bool):
        self._json = {**hints, "ready_to_recommend": ready}

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        return "Here are some suitable assessments."

    def complete_json(self, messages, *, schema=None) -> dict:
        return dict(self._json)


def _trace_paths() -> list[str]:
    return sorted(
        glob.glob(str(settings.project_root / "data" / "traces" / "*.md")),
        key=lambda p: (len(p), p),
    )


def _trace_id(path: str) -> str:
    return os.path.basename(path).replace(".md", "")


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(settings.raw_catalog_path)


def _engine(catalog, hints: dict, *, ready: bool) -> ResponseEngine:
    # Lexical + signals carry the behaviour.
    retriever = LexicalRanker(catalog)
    from shl_recommender.catalog.vocabulary import build_vocabulary

    return ResponseEngine(
        retriever, _HintLLM(hints, ready=ready), catalog=catalog,
        vocabulary=build_vocabulary(catalog),
    )


@pytest.mark.parametrize("path", _trace_paths(), ids=_trace_id)
def test_trace_replays_with_expected_behaviour(path, catalog):
    trace = _trace_id(path)
    turns = parse_turns(path)
    hints = _HINTS.get(trace, {})

    for i, turn in enumerate(turns):
        messages = _as_messages(messages_up_to(turns, i))
        # Fake the model's readiness as "ready" once the trace itself is showing (or
        # has shown) a shortlist by this turn — i.e. the point the conversation became
        # specific enough. Before that, "not ready" so the engine clarifies.
        ready = any(t.shows_recommendations for t in turns[: i + 1])
        engine = _engine(catalog, hints, ready=ready)

        response = engine.respond(messages)

        # 1. Always a valid contract, never an error.
        payload = response.to_payload()
        assert set(payload) == {"reply", "recommendations", "end_of_conversation"}
        assert response.reply

        # 2. A pure comparison turn must not commit a new shortlist.
        if (trace, turn.index) in _COMPARISON_TURNS:
            assert response.recommendations is None, (
                f"{trace} T{turn.index}: comparison turn should not commit a shortlist"
            )

        # 3. The final turn ends the conversation and carries the confirmed shortlist.
        if i == len(turns) - 1:
            assert turn.end_of_conversation, "sanity: last trace turn ends the conversation"
            assert response.end_of_conversation is True, (
                f"{trace}: final turn should set end_of_conversation"
            )
            assert response.recommendations is not None, (
                f"{trace}: final turn should carry the confirmed shortlist"
            )


@pytest.mark.parametrize("path", _trace_paths(), ids=_trace_id)
def test_final_shortlist_recalls_gold(path, catalog):
    """When the conversation commits its final shortlist, it should recall the gold.

    Uses the whole-conversation state (all the user's words plus the trace hints),
    which is what the agent holds at commit time, and checks recall against the gold
    at a floor the lexical retriever plus signals meet. The dedicated Recall@10 floor
    test measures the tuned number; this guards the *replayed* path.
    """
    trace = _trace_id(path)
    gold = {u.rstrip("/").lower() for u in all_gold_urls(path)}
    assert gold, "every scored trace has gold URLs"

    from shl_recommender.conversation.state import ConversationState
    from scripts.measure_recall import _user_text

    retriever = LexicalRanker(catalog)
    state = ConversationState(query_text=_user_text(path), **_HINTS.get(trace, {}))
    found = {s.item.url.rstrip("/").lower() for s in retriever.retrieve(state, top_k=10)}

    recall = len(gold & found) / len(gold)
    # Lexical floor: lower than the tuned mean, set to catch a real
    # regression in the replayed path without being flaky.
    assert recall >= 0.4, f"{trace}: replayed recall {recall:.2f} below floor"
