"""LLM-based understanding of the hiring need.

The deterministic layer handles clear signals (comparison, refusal, confirmation).
This layer handles the open-ended part — reading a role, seniority, skills,
purpose and language out of natural language and pasted job descriptions, which
is too varied for rules.

It returns a typed :class:`Understanding`. If the model is unavailable or returns
something unusable, it returns an empty ``Understanding`` and the turn proceeds on
deterministic signals alone; understanding is an enhancement, never a hard
dependency.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from shl_recommender.conversation.state import Purpose
from shl_recommender.llm.client import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Asks the model for exactly the fields we need, as a flat JSON object, with
# explicit "leave it null/empty if not stated" instructions so it does not invent
# constraints the user never gave.
_SYSTEM_PROMPT = """\
You extract structured hiring requirements from a conversation between a user and
an assessment-recommendation assistant. Read the whole conversation, but weight
the most recent user messages most heavily, because the user may correct earlier
statements and the latest statement wins.

Return ONLY a JSON object with these fields:
- role: the job title or candidate population, or null if not stated
- seniority: e.g. "graduate", "mid-level", "senior", "executive", or null
- years_experience: any experience hint as written, or null
- domain: industry or domain (e.g. "manufacturing", "healthcare"), or null
- purpose: one of "selection", "screening", "development", or "unknown"
- must_have_skills: array of explicitly required skills (may be empty)
- optional_skills: array of nice-to-have skills (may be empty)
- languages: array of required assessment languages (may be empty)
- test_type_preferences: array of assessment categories the user explicitly asked
  for, e.g. "personality", "cognitive", "simulation" (may be empty)
- ready_to_recommend: true if the request is specific enough to assemble a
  focused, well-targeted shortlist now; false if there is exactly one
  decision-critical thing that would materially change which assessments fit
- clarifying_question: if ready_to_recommend is false, the single most useful
  question to ask (one sentence); otherwise null

Judging readiness — be genuinely helpful, which usually means asking ONE good
question before a broad first request rather than guessing:
- A bare job title, even with a seniority ("senior Java developer", "graduate
  analyst"), is NOT ready on its own. You know the role but not what the hire will
  actually own, which skills to prioritise, or what the screen is for — set
  ready_to_recommend to false and ask the single most useful question.
- It IS ready when the user has given a real differentiator themselves: specific
  tools/skills to screen on ("must test Excel and SQL"), a named assessment
  category ("cognitive and personality"), or a clear stated purpose/context that
  pins down what to measure.
- A request can name many skills yet still be too broad (a wide full-stack role
  where you do not yet know the focus); prefer a question there too.
Ask only one question, and only when it would materially change which assessments
fit. Do not ask about details the user is unlikely to have a preference on.

Do not re-ask what the user has already answered, and read their tone:
- Base the question on what is still genuinely missing given the WHOLE conversation.
  If the user already told you the role and seniority, do not ask the role again in
  other words — that reads as not listening. Ask about a different, still-open thing
  (the skills to prioritise, the purpose, the format), or recommend.
- If the user pushes back, repeats themselves, or signals they have nothing more to
  add ("as I said", "that's all I have", "just give me options"), stop asking: set
  ready_to_recommend to true and proceed with what you have. One more vague question
  is worse than a reasonable shortlist they can refine.
- If the user asks what you need to know, that is not a request to recommend yet, but
  do not dodge it: make clarifying_question the single most useful concrete thing you
  need (name it plainly), not another broad "tell me more".

Do not infer constraints that are not present — this includes purpose. If the user
did not say why they are assessing (selection vs screening vs development), leave
purpose as "unknown"; do not guess "selection". If the user said nothing about a
field, use null or an empty array. Output the JSON object and nothing else.
"""

_VALID_PURPOSES = {p.value for p in Purpose}


class Understanding(BaseModel):
    """Structured requirements extracted by the model. All fields optional.

    Beyond the requirement fields, the model also reports a *readiness* judgement:
    whether the request is specific enough to recommend well, or whether one
    decision-critical question would materially sharpen the shortlist. This is the
    judgement rules handle poorly — a brief can name many skills yet still be too
    broad to be precise (a wide full-stack JD), or name few yet be perfectly clear
    (the exact tools to screen on). The policy uses this judgement but stays in
    control of the question budget and turn cap, and falls back to a structural
    rule when the model is unavailable.
    """

    model_config = ConfigDict(frozen=True)

    role: str | None = None
    seniority: str | None = None
    years_experience: str | None = None
    domain: str | None = None
    purpose: Purpose = Purpose.UNKNOWN
    must_have_skills: tuple[str, ...] = ()
    optional_skills: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    test_type_preferences: tuple[str, ...] = ()

    # Readiness judgement. ``ready_to_recommend`` is None when the model did not
    # give an opinion (e.g. it was not asked, or it failed), which the policy
    # treats as "fall back to the structural rule".
    ready_to_recommend: bool | None = None
    clarifying_question: str | None = None

    @field_validator("purpose", mode="before")
    @classmethod
    def _coerce_purpose(cls, value: object) -> object:
        # Accept an unknown/garbled purpose gracefully rather than failing the
        # whole extraction.
        if isinstance(value, str) and value.lower() in _VALID_PURPOSES:
            return value.lower()
        return Purpose.UNKNOWN

    @field_validator("ready_to_recommend", mode="before")
    @classmethod
    def _coerce_ready(cls, value: object) -> object:
        # Accept bool, or a stringy "true"/"false"; anything else means "no opinion".
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return None

    @field_validator("clarifying_question", mode="before")
    @classmethod
    def _clean_question(cls, value: object) -> object:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    @field_validator(
        "must_have_skills", "optional_skills", "languages", "test_type_preferences",
        mode="before",
    )
    @classmethod
    def _clean_list(cls, value: object) -> tuple[str, ...]:
        if not value:
            return ()
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(item).strip() for item in value if str(item).strip())


def _to_chat(messages: list) -> list[dict]:
    """Render the request history into provider-style chat messages."""
    rendered: list[dict] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", "")
        if role in ("user", "assistant", "system") and content:
            rendered.append({"role": role, "content": content})
    return rendered


def extract_understanding(messages: list, client: LLMClient) -> Understanding:
    """Extract structured requirements from the conversation.

    Never raises: any failure (model unavailable, bad JSON, schema mismatch) is
    logged and yields an empty :class:`Understanding`, so the caller can fall back
    to deterministic signals.
    """
    chat = [{"role": "system", "content": _SYSTEM_PROMPT}, *_to_chat(messages)]
    try:
        data = client.complete_json(chat)
    except LLMError as exc:
        logger.warning("understanding extraction failed, continuing without it: %s", exc)
        return Understanding()

    try:
        return Understanding.model_validate(data)
    except ValidationError as exc:
        logger.warning("understanding payload did not validate, ignoring it: %s", exc)
        return Understanding()
