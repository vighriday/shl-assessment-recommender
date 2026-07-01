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

    Gradio passes history as a list of ``{"role", "content"}`` dicts (messages
    format). We map those onto our schema and append the new user message.
    """
    messages: list[Message] = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append(Message(role=role, content=content))
    messages.append(Message(role="user", content=message))
    return messages


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
