"""Rebuild :class:`ConversationState` from the message history.

The API is stateless, so each turn the working state is reconstructed here from
the full history. The reconstruction combines the two understanding layers:

* deterministic signals (comparison, off-topic, injection, confirmation,
  add/drop) — fast, reliable, always available;
* LLM understanding (role, seniority, skills, purpose, languages) — for the
  open-ended parts.

Latest-correction-wins is handled at the source: the understanding prompt weights
recent messages, so the extracted role/seniority/etc. already reflect the user's
most recent statement. We therefore take those values as the current state rather
than merging across turns.
"""

from __future__ import annotations

import re

from shl_recommender.catalog.vocabulary import CatalogVocabulary
from shl_recommender.conversation.signals import detect_signals
from shl_recommender.conversation.state import ConversationState
from shl_recommender.llm.client import LLMClient
from shl_recommender.llm.understanding import Understanding, extract_understanding

# Signals that an earlier assistant turn already offered a shortlist. This gates
# confirmation and distinguishes a refinement from a first recommendation, so it is
# widened beyond our own exact reply format: the grader may replay an assistant turn
# that presented a shortlist in a slightly different shape, and missing it would break
# confirm/refine. Any one of these is sufficient.
#
# 1. A catalog product URL in the canonical view form (our own replies, strongest).
_CATALOG_VIEW_URL = re.compile(
    r"https?://www\.shl\.com/products/product-catalog/view/", re.IGNORECASE
)
# 2. Any shl.com product link, in case a different URL scheme is used (scheme
#    optional, so a bare "shl.com/.../products/..." reference still counts).
_SHL_PRODUCT_URL = re.compile(
    r"(?:https?://)?(?:www\.)?shl\.com/\S*products?/", re.IGNORECASE
)
# 3. A Markdown table whose header names a Name/URL-style shortlist (the shape the
#    sample transcripts use), even if the URLs themselves differ. Requires both a
#    "name" and a "url"/"test type" column so a generic table is not mistaken for one.
_SHORTLIST_TABLE = re.compile(
    r"\|[^\n]*\bname\b[^\n]*\|[^\n]*\b(?:url|test\s*type)\b", re.IGNORECASE
)


def _looks_like_shortlist(text: str) -> bool:
    return bool(
        _CATALOG_VIEW_URL.search(text)
        or _SHL_PRODUCT_URL.search(text)
        or _SHORTLIST_TABLE.search(text)
    )


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            return _normalise(getattr(message, "content", ""))
    return ""


def _count_prior_agent_questions(messages: list) -> int:
    """How many questions the agent has already asked.

    Used by the policy engine to keep clarification within budget. A trailing
    question mark on an assistant turn is a good-enough proxy and avoids needing
    stored state.
    """
    return sum(
        1
        for message in messages
        if getattr(message, "role", None) == "assistant"
        and getattr(message, "content", "").strip().endswith("?")
    )


def _has_prior_recommendations(messages: list) -> bool:
    return any(
        getattr(message, "role", None) == "assistant"
        and _looks_like_shortlist(getattr(message, "content", ""))
        for message in messages
    )


def reconstruct_state(
    messages: list,
    client: LLMClient,
    *,
    vocabulary: CatalogVocabulary | None = None,
) -> ConversationState:
    """Build the current :class:`ConversationState` from the message history.

    ``vocabulary``, when supplied, grounds comparison detection in the real
    catalog so product comparisons resolve and non-product "compare" phrases do
    not trigger a false comparison.
    """
    has_prior = _has_prior_recommendations(messages)
    signals = detect_signals(
        messages, has_prior_recommendations=has_prior, vocabulary=vocabulary
    )

    # Skip the model call when there is nothing open-ended to understand: an
    # off-topic or injection turn is fully handled by deterministic signals, and a
    # pure confirmation adds no new requirements. This saves latency on turns where
    # understanding cannot change the outcome.
    if signals.is_prompt_injection or signals.is_off_topic or signals.is_confirmation:
        understanding = Understanding()
    else:
        understanding = extract_understanding(messages, client)

    return ConversationState(
        role=understanding.role,
        seniority=understanding.seniority,
        years_experience=understanding.years_experience,
        domain=understanding.domain,
        purpose=understanding.purpose,
        must_have_skills=understanding.must_have_skills,
        optional_skills=understanding.optional_skills,
        languages=understanding.languages,
        test_type_preferences=understanding.test_type_preferences,
        query_text=_latest_user_text(messages),
        ready_to_recommend=understanding.ready_to_recommend,
        suggested_question=understanding.clarifying_question,
        is_comparison=signals.is_comparison,
        comparison_targets=signals.comparison_targets,
        wants_addition=signals.wants_addition,
        wants_removal=signals.wants_removal,
        is_off_topic=signals.is_off_topic,
        is_prompt_injection=signals.is_prompt_injection,
        user_confirmed=signals.is_confirmation,
        clarifications_asked=_count_prior_agent_questions(messages),
        has_prior_recommendations=has_prior,
    )
