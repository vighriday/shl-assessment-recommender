"""A human-facing chat UI for the recommender, mounted onto the API.

The HTTP API (``/health``, ``/chat``) is what the grader uses; this module adds a
simple chat interface so a person can talk to the agent the way a hiring manager
would — typing naturally, seeing the shortlist build up, refining it — without
hand-writing JSON. It is a thin presentation layer: it calls the same
:class:`ResponseEngine` in process (no extra network hop, identical behaviour) and
formats the structured response for reading.

Built with Gradio's ``ChatInterface`` and mounted at the root path ``/`` by the app
factory, so one process on one port serves both the API and the UI (the JSON endpoint
index lives at ``/info``).
"""

from __future__ import annotations

import gradio as gr

from shl_recommender.api.schemas import Message
from shl_recommender.response.engine import ResponseEngine

_INTRO = (
    "**SHL Assessment Recommender.** Describe who you're hiring for and I'll suggest "
    "assessments from the SHL catalog. I'll ask a question if the request is broad, "
    "refine the list as you add constraints, and compare products on request.\n\n"
    "_Try: “Hiring a senior Java developer; screen Java and SQL.”_"
)

_EXAMPLES = [
    "I need an assessment.",
    "Hiring a senior Java developer; screen Java and SQL.",
    "We're hiring graduate analysts — cognitive and personality.",
    "What's the difference between OPQ and the OPQ MQ Sales Report?",
]


def _to_messages(history: list, message: str) -> list[Message]:
    """Turn Gradio's chat history plus the new turn into engine Message objects.

    This must be **format-tolerant**. Depending on the Gradio version and the
    ``ChatInterface`` configuration, ``history`` arrives in one of two shapes:

    * the *messages* format — a list of ``{"role": ..., "content": ...}`` dicts;
    * the older *tuples* format — a list of ``[user_text, assistant_text]`` pairs.

    Getting this wrong is not cosmetic: if the history shape is not recognised, every
    prior turn is silently dropped, the engine sees only the current message, the
    clarification counter reads zero, and the agent asks the same question forever (the
    clarify loop). So we handle both shapes explicitly and skip only genuinely empty
    entries. The new user ``message`` is always appended last.
    """
    messages: list[Message] = []
    for turn in history or []:
        if isinstance(turn, dict):
            # Messages format: {"role", "content"}. In current Gradio the browser sends
            # ``content`` as a *list of parts* — ``[{"text": ..., "type": "text"}]`` —
            # not a plain string (the API client sends a plain string). We must handle
            # both, or every prior turn is dropped and the agent loops.
            role = turn.get("role")
            text = _extract_text(turn.get("content"))
            if role in ("user", "assistant") and text:
                messages.append(Message(role=role, content=text))
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            # Legacy tuples format: [user_text, assistant_text]; either side may be None.
            user_text, assistant_text = turn
            if isinstance(user_text, str) and user_text.strip():
                messages.append(Message(role="user", content=user_text))
            if isinstance(assistant_text, str) and assistant_text.strip():
                messages.append(Message(role="assistant", content=assistant_text))
    messages.append(Message(role="user", content=message))
    return messages


def _extract_text(content: object) -> str:
    """Pull plain text out of a Gradio message ``content``, whatever its shape.

    Handles the three forms seen in the wild:

    * a plain string (what the API client / older Gradio send);
    * a list of content parts — ``[{"text": ..., "type": "text"}, ...]`` — which is
      what the current Gradio browser client sends;
    * a single part dict — ``{"text": ...}``.

    Anything else (an image part, ``None``) contributes no text. Returns a stripped
    string, empty if there is nothing usable.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text")
        return text.strip() if isinstance(text, str) else ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts).strip()
    return ""


def _format(response) -> str:
    """Render a ChatResponse as readable Markdown: the reply plus any shortlist."""
    parts = [response.reply]
    if response.recommendations:
        parts.append("")  # blank line before the list
        for i, rec in enumerate(response.recommendations, start=1):
            parts.append(f"{i}. **[{rec.name}]({rec.url})** — `{rec.test_type}`")
    if response.end_of_conversation:
        parts.append("\n_— conversation complete —_")
    return "\n".join(parts)


def build_ui(engine: ResponseEngine) -> gr.Blocks:
    """Build the chat UI bound to a live engine."""

    def respond(message: str, history: list) -> str:
        if not message or not message.strip():
            return "Please type what you're hiring for."
        response = engine.respond(_to_messages(history, message))
        return _format(response)

    chat = gr.ChatInterface(
        fn=respond,
        title="SHL Assessment Recommender",
        description=_INTRO,
        examples=_EXAMPLES,
    )
    return chat
