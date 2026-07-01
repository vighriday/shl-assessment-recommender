"""Tests for the lexical retriever."""

from __future__ import annotations

import pytest

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.config import settings
from shl_recommender.retrieval.lexical import LexicalRetriever


@pytest.fixture(scope="module")
def retriever():
    return LexicalRetriever(load_catalog(settings.raw_catalog_path))


def test_exact_skill_query_finds_its_item(retriever):
    top = retriever.search("Excel", top_k=5)
    names = [s.item.name for s in top]
    assert any("Excel" in n for n in names)


def test_product_code_query_finds_item(retriever):
    # Character n-grams are what make an exact code retrievable.
    top = retriever.search("OPQ32r", top_k=5)
    assert any("OPQ32r" in s.item.name for s in top)


def test_results_are_sorted_by_score(retriever):
    top = retriever.search("Java developer", top_k=10)
    scores = [s.score for s in top]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(retriever):
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_top_k_is_respected(retriever):
    assert len(retriever.search("assessment", top_k=3)) <= 3


def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError):
        LexicalRetriever([])
