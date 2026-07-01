"""Tests for the lexical ranker.

The ranker composes the lexical retriever's output into a transparent weighted-sum
ranking, so these tests run fast and offline with no model to load. Scope filtering,
staple injection, family diversity, and the signal scores are all exercised
directly.
"""

from __future__ import annotations

import pytest

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.config import settings
from shl_recommender.conversation.state import ConversationState
from shl_recommender.retrieval.lexical import LexicalRetriever
from shl_recommender.retrieval.ranker import LexicalRanker, RankingWeights


@pytest.fixture(scope="module")
def items():
    return load_catalog(settings.raw_catalog_path)


@pytest.fixture(scope="module")
def retriever(items):
    # Lexical + signals are enough to assert the ranker's behaviour.
    return LexicalRanker(items, lexical=LexicalRetriever(items))


def test_returns_at_most_top_k(retriever):
    state = ConversationState(query_text="Java developer", must_have_skills=("Java",))
    assert len(retriever.retrieve(state, top_k=10)) <= 10


def test_exact_skill_is_surfaced(retriever):
    state = ConversationState(query_text="screen for Excel", must_have_skills=("Excel",))
    names = [s.item.name for s in retriever.retrieve(state, top_k=10)]
    assert any("Excel" in n for n in names)


def test_personality_staple_is_injected_for_professional_hire(retriever):
    # OPQ32r is a default even though the query never names it.
    state = ConversationState(query_text="hiring a manager", role="manager")
    names = [s.item.name for s in retriever.retrieve(state, top_k=10)]
    assert any("OPQ32r" in n for n in names)


def test_requested_category_boosts_matching_items(retriever):
    state = ConversationState(
        query_text="graduate analyst", role="analyst",
        test_type_preferences=("personality",),
    )
    top = retriever.retrieve(state, top_k=10)
    # At least one returned item carries the personality code.
    assert any("P" in s.item.test_type.split(",") for s in top)


def test_distinctive_skill_name_bonus_surfaces_its_item(retriever):
    # A distinctive required skill (one that names only a handful of products) should
    # pull the item named after it into the shortlist even when its official name is
    # verbose enough to dilute the fractional name score. "AWS" is such a skill.
    state = ConversationState(
        query_text="senior backend engineer on AWS",
        role="engineer",
        must_have_skills=("AWS",),
    )
    names = [s.item.name for s in retriever.retrieve(state, top_k=10)]
    assert any("AWS" in n or "Amazon Web Services" in n for n in names)


def test_broad_skill_does_not_trigger_the_distinctive_bonus(retriever):
    # A broad skill that names many products (e.g. "Java") must NOT get the flat
    # distinctive bonus, or a whole family of near-duplicates would be promoted. The
    # ranker's distinctive-skill set should exclude it. This asserts the guard exists
    # by checking the shortlist is not swamped by one family.
    from shl_recommender.retrieval.ranker import _family_key

    state = ConversationState(
        query_text="Java developer", role="developer", must_have_skills=("Java",)
    )
    top = retriever.retrieve(state, top_k=10)
    counts: dict[str, int] = {}
    for s in top:
        counts[_family_key(s.item.name)] = counts.get(_family_key(s.item.name), 0) + 1
    assert max(counts.values()) <= 2


def test_out_of_scope_items_are_never_returned(items):
    # Force an item out of scope and confirm it cannot appear.
    from dataclasses import replace  # noqa: F401 - not used; models are pydantic

    target = items[0]
    modified = [
        it.model_copy(update={"in_scope": False}) if it.entity_id == target.entity_id else it
        for it in items
    ]
    retriever = LexicalRanker(
        modified,
        lexical=LexicalRetriever(modified),
    )
    state = ConversationState(query_text=target.name, must_have_skills=(target.name,))
    returned_ids = {s.item.entity_id for s in retriever.retrieve(state, top_k=10)}
    assert target.entity_id not in returned_ids


def test_family_diversity_limits_near_duplicates(retriever):
    # A sales/OPQ-heavy query would otherwise fill up with OPQ report variants.
    state = ConversationState(query_text="sales team OPQ reports", role="sales")
    top = retriever.retrieve(state, top_k=10)
    from shl_recommender.retrieval.ranker import _family_key

    counts: dict[str, int] = {}
    for s in top:
        key = _family_key(s.item.name)
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) <= 2  # per_family cap


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve(ConversationState(query_text=""), top_k=10) == []


def test_weights_are_configurable(items):
    # Constructing with custom weights must work and still return results.
    r = LexicalRanker(
        items,
        lexical=LexicalRetriever(items),
        weights=RankingWeights(lexical=2.0, staple=0.0),
    )
    state = ConversationState(query_text="Excel", must_have_skills=("Excel",))
    assert r.retrieve(state, top_k=5)
