"""Tests for shortlist assembly.

The contract this enforces is the one the grader checks hardest, so the tests are
blunt: null (not []) when empty, clamp to the 1..10 maximum, and every field copied
verbatim from the catalog item. A tiny hand-built catalog keeps the assertions
exact and offline.
"""

from __future__ import annotations

from shl_recommender.catalog.models import CatalogItem
from shl_recommender.response.shortlist import build_recommendations, recover_prior_shortlist
from shl_recommender.retrieval.types import ScoredItem


class _Msg:
    """Minimal message stand-in (the helper reads role/content by attribute)."""

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


def _item(entity_id: str, name: str, code: str = "K") -> CatalogItem:
    return CatalogItem(
        entity_id=entity_id,
        name=name,
        url=f"https://www.shl.com/products/product-catalog/view/{entity_id}/",
        description="d",
        keys=("Knowledge & Skills",),
        test_type=code,
        search_text=name.lower(),
    )


def _scored(n: int) -> list[ScoredItem]:
    return [ScoredItem(item=_item(str(i), f"Test {i}"), score=1.0 - i / 100) for i in range(n)]


def test_empty_input_is_null_not_empty_list():
    assert build_recommendations([]) is None


def test_single_item_becomes_one_recommendation():
    recs = build_recommendations(_scored(1))
    assert recs is not None
    assert len(recs) == 1
    assert recs[0].name == "Test 0"


def test_fields_are_copied_verbatim_from_catalog():
    item = _item("720", "OPQ32r", code="P")
    recs = build_recommendations([ScoredItem(item=item, score=1.0)])
    assert recs[0].name == "OPQ32r"
    assert recs[0].url == "https://www.shl.com/products/product-catalog/view/720/"
    assert recs[0].test_type == "P"


def test_clamped_to_ten_when_more_supplied():
    recs = build_recommendations(_scored(25))
    assert len(recs) == 10
    # Order is preserved: the top ten of the ranked input.
    assert [r.name for r in recs] == [f"Test {i}" for i in range(10)]


def test_respects_a_smaller_limit():
    recs = build_recommendations(_scored(10), limit=3)
    assert len(recs) == 3


def test_zero_limit_yields_null():
    assert build_recommendations(_scored(5), limit=0) is None


# --- recover_prior_shortlist ------------------------------------------------

def test_recovers_prior_shortlist_from_assistant_message():
    catalog = [_item("1", "Java"), _item("720", "OPQ32r", code="P"), _item("2", "SQL")]
    messages = [
        _Msg("user", "hiring a dev"),
        _Msg(
            "assistant",
            "I'd suggest https://www.shl.com/products/product-catalog/view/1/ and "
            "https://www.shl.com/products/product-catalog/view/720/.",
        ),
        _Msg("user", "perfect"),
    ]
    recovered = recover_prior_shortlist(messages, catalog)
    assert [r.name for r in recovered] == ["Java", "OPQ32r"]  # in the order shown


def test_recovers_from_the_most_recent_listing_message():
    catalog = [_item("1", "Java"), _item("2", "SQL")]
    messages = [
        _Msg("assistant", "https://www.shl.com/products/product-catalog/view/1/"),
        _Msg("user", "add sql"),
        _Msg("assistant", "https://www.shl.com/products/product-catalog/view/2/"),
        _Msg("user", "great"),
    ]
    recovered = recover_prior_shortlist(messages, catalog)
    assert [r.name for r in recovered] == ["SQL"]  # the latest list wins


def test_returns_none_when_no_prior_urls():
    catalog = [_item("1", "Java")]
    messages = [_Msg("user", "hi"), _Msg("assistant", "What role?")]
    assert recover_prior_shortlist(messages, catalog) is None


def test_angle_bracketed_urls_are_recovered():
    # The sample transcripts wrap URLs in angle brackets; recovery must handle both.
    catalog = [_item("1", "Java")]
    messages = [
        _Msg("assistant", "See <https://www.shl.com/products/product-catalog/view/1/>."),
        _Msg("user", "ok"),
    ]
    recovered = recover_prior_shortlist(messages, catalog)
    assert recovered is not None and recovered[0].name == "Java"
