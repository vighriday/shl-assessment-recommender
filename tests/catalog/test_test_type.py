"""Tests for test_type derivation."""

from __future__ import annotations

import pytest

from shl_recommender.catalog.test_type import (
    CATEGORY_TO_CODE,
    UnknownCategoryError,
    derive_test_type,
)


def test_single_category_maps_to_single_code():
    assert derive_test_type(["Knowledge & Skills"]) == "K"
    assert derive_test_type(["Personality & Behavior"]) == "P"


def test_multi_category_joins_codes_in_given_order():
    assert derive_test_type(["Knowledge & Skills", "Simulations"]) == "K,S"
    # Order is preserved, not sorted: the catalog and traces use both orders.
    assert derive_test_type(["Personality & Behavior", "Competencies"]) == "P,C"
    assert derive_test_type(["Competencies", "Personality & Behavior"]) == "C,P"


def test_every_known_category_has_a_single_letter_code():
    for category, code in CATEGORY_TO_CODE.items():
        assert derive_test_type([category]) == code
        assert len(code) == 1 and code.isalpha()


def test_empty_keys_is_an_error():
    with pytest.raises(ValueError):
        derive_test_type([])


def test_unknown_category_fails_loudly():
    with pytest.raises(UnknownCategoryError):
        derive_test_type(["Telepathy & Clairvoyance"])


def test_override_takes_precedence(monkeypatch):
    monkeypatch.setattr(
        "shl_recommender.catalog.test_type.ORDER_OVERRIDES",
        {"999": "S,K"},
    )
    # Without the override this would derive "K,S"; the override wins.
    assert derive_test_type(["Knowledge & Skills", "Simulations"], entity_id="999") == "S,K"
