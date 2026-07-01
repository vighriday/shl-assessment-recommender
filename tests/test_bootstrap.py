"""Tests for startup composition and fail-fast validation.

The happy path builds against the real catalog and asserts the context is wired and
reports serving health. The failure paths prove the *fail-fast* promise: a hard
invariant violation (empty catalog, invalid ``test_type``, unlinkable item) raises
:class:`StartupError` at boot rather than being discovered on a later request.
Retrieval is lexical-only, so these run offline and fast with no model to load.
"""

from __future__ import annotations

import pytest

from shl_recommender.bootstrap import AppContext, StartupError, _validate_catalog, bootstrap
from shl_recommender.config import Settings


@pytest.fixture(scope="module")
def offline_config() -> Settings:
    # Default settings: retrieval is lexical-only, so bootstrap loads no model
    # during the test run.
    return Settings()


def test_bootstrap_builds_and_validates(offline_config):
    ctx = bootstrap(offline_config)
    assert isinstance(ctx, AppContext)
    assert len(ctx.catalog) > 0
    # The health probe reports on the components just built, and the service is
    # serving (degraded is fine — e.g. no model key configured in the test env; the
    # language model is a soft dependency and must not by itself fail startup).
    report = ctx.health.check()
    assert report.is_serving


def test_bootstrap_retriever_is_usable(offline_config):
    from shl_recommender.conversation.state import ConversationState

    ctx = bootstrap(offline_config)
    results = ctx.retriever.retrieve(
        ConversationState(query_text="Excel", must_have_skills=("Excel",)), top_k=5
    )
    assert results  # a wired retriever returns candidates


def test_empty_catalog_fails_fast():
    with pytest.raises(StartupError, match="empty"):
        _validate_catalog([])


def test_invalid_test_type_fails_fast(offline_config):
    from shl_recommender.catalog.loader import load_catalog

    items = load_catalog(offline_config.raw_catalog_path)
    # Corrupt one item's test_type to a code that is not in the known set.
    broken = items[0].model_copy(update={"test_type": "Z"})
    with pytest.raises(StartupError, match="invalid test_type"):
        _validate_catalog([broken, *items[1:]])


def test_unlinkable_item_fails_fast(offline_config):
    from shl_recommender.catalog.loader import load_catalog

    items = load_catalog(offline_config.raw_catalog_path)
    broken = items[0].model_copy(update={"url": ""})
    with pytest.raises(StartupError, match="URL"):
        _validate_catalog([broken, *items[1:]])
