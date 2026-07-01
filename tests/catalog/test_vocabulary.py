"""Tests for the catalog-derived vocabulary used to ground signal detection."""

from __future__ import annotations

import pytest

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.config import settings


@pytest.fixture(scope="module")
def vocab():
    return build_vocabulary(load_catalog(settings.raw_catalog_path))


def test_known_codes_are_recognised(vocab):
    for code in ["OPQ", "SVAR", "DSI", "MFS"]:
        assert vocab.mentions_product(code), code


def test_full_product_name_is_recognised(vocab):
    assert vocab.mentions_product("the OPQ32r instrument")


def test_non_product_words_are_not_recognised(vocab):
    for word in ["notes", "options", "prices", "my team", "results"]:
        assert not vocab.mentions_product(word), word


def test_noise_tokens_are_excluded(vocab):
    # "New" and bare locale tokens are not products.
    assert "new" not in vocab.codes
    assert "us" not in vocab.codes


def test_empty_fragment_is_safe(vocab):
    assert not vocab.mentions_product("")
