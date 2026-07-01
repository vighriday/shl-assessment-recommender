"""Tests for the Gradio chat UI adapter.

The UI is a thin layer over the same ``ResponseEngine`` the API uses, but it has one
job that is easy to get wrong and catastrophic when it is: turning Gradio's chat
``history`` into engine messages. Gradio does not pass a single fixed shape — depending
on the client and version, a message's ``content`` is a plain string (API client) or a
*list of parts* (``[{"text": ..., "type": "text"}]``) from the browser. If the adapter
only understands one shape, it silently drops the whole history, the clarification
counter reads zero, and the agent loops asking the same question forever.

This module pins ``_to_messages`` against every shape seen in the wild, and drives the
exact multi-turn "grind" through the engine to prove the browser-format history is
preserved and the agent commits instead of looping.
"""

from __future__ import annotations

from shl_recommender.api.ui import _extract_text, _to_messages
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.config import settings
from shl_recommender.response.engine import ResponseEngine
from shl_recommender.retrieval.ranker import LexicalRanker


def _browser_turn(role: str, text: str) -> dict:
    """A history entry in the browser's rich-content shape (Gradio messages format)."""
    return {"role": role, "metadata": None, "content": [{"text": text, "type": "text"}], "options": None}


def _api_turn(role: str, text: str) -> dict:
    """A history entry in the plain-string shape (Gradio API client)."""
    return {"role": role, "content": text}


# --- _extract_text handles every content shape --------------------------------


def test_extract_text_from_plain_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_from_parts_list():
    # The shape the browser actually sends.
    assert _extract_text([{"text": "hello", "type": "text"}]) == "hello"


def test_extract_text_from_single_part_dict():
    assert _extract_text({"text": "hello"}) == "hello"


def test_extract_text_from_multiple_parts():
    assert _extract_text([{"text": "a", "type": "text"}, {"text": "b", "type": "text"}]) == "a b"


def test_extract_text_from_unusable_content_is_empty():
    assert _extract_text(None) == ""
    assert _extract_text([{"type": "image", "url": "x"}]) == ""
    assert _extract_text(123) == ""


# --- _to_messages preserves history in every format ---------------------------


def test_to_messages_browser_rich_content_preserves_history():
    # THE regression: browser content is a parts-list, not a string. Every prior turn
    # must survive, or the agent loops. Two exchanges + a new message -> 5 messages.
    history = [
        _browser_turn("user", "I need an assessment."),
        _browser_turn("assistant", "What role?"),
        _browser_turn("user", "senior Java developer"),
        _browser_turn("assistant", "What skills?"),
    ]
    msgs = _to_messages(history, "Core")
    assert [m.role for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert [m.content for m in msgs] == [
        "I need an assessment.", "What role?", "senior Java developer", "What skills?", "Core",
    ]


def test_to_messages_plain_string_content_preserves_history():
    history = [
        _api_turn("user", "I need an assessment."),
        _api_turn("assistant", "What role?"),
    ]
    msgs = _to_messages(history, "developer")
    assert [m.role for m in msgs] == ["user", "assistant", "user"]


def test_to_messages_legacy_tuples_format_preserves_history():
    history = [["I need an assessment.", "What role?"], ["senior Java developer", "What skills?"]]
    msgs = _to_messages(history, "Core")
    assert len(msgs) == 5
    assert msgs[0].content == "I need an assessment."
    assert msgs[-1].content == "Core"


def test_to_messages_empty_history_is_just_the_message():
    assert [m.content for m in _to_messages([], "hi")] == ["hi"]
    assert [m.content for m in _to_messages(None, "hi")] == ["hi"]


# --- End to end: the browser-format grind must commit, not loop ---------------


def _engine():
    catalog = load_catalog(settings.raw_catalog_path)
    return ResponseEngine(
        LexicalRanker(catalog),
        _NotReadyLLM(),
        catalog=catalog,
        vocabulary=build_vocabulary(catalog),
    )


class _NotReadyLLM:
    """A model that always says 'not ready' — the worst case for looping. The code's
    clarify budget must still force a commit."""

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        return "What else can you tell me?"

    def complete_json(self, messages, *, schema=None) -> dict:
        return {"ready_to_recommend": False, "clarifying_question": "What role?"}


def test_browser_format_grind_commits_and_does_not_loop():
    # Reproduce the exact UI conversation with browser-shaped history at each turn, and
    # assert the agent commits a shortlist within the budget rather than looping.
    engine = _engine()
    history: list = []
    committed_turn = None
    for i, user_msg in enumerate(["I need an assessment.", "senior Java developer", "Core", "Java"], start=1):
        msgs = _to_messages(history, user_msg)
        response = engine.respond(msgs)
        if response.recommendations is not None:
            committed_turn = i
            break
        # Append the turn in the browser's rich-content shape, as Gradio would.
        history.append(_browser_turn("user", user_msg))
        history.append(_browser_turn("assistant", response.reply))
    assert committed_turn is not None, "the UI grind looped — history was not preserved"
    assert committed_turn <= 3
