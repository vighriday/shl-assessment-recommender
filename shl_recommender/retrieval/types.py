"""Shared retrieval types."""

from __future__ import annotations

from dataclasses import dataclass

from shl_recommender.catalog.models import CatalogItem


@dataclass(frozen=True)
class ScoredItem:
    """A catalog item paired with a score from a retriever or the ranker."""

    item: CatalogItem
    score: float
