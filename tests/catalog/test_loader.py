"""Tests for the catalog loader.

Two kinds of tests:

* invariants verified against the real provided catalog (these are the
  guarantees the rest of the service relies on);
* synthetic cases that exercise each malformed-input path the loader guards,
  written inline so they do not depend on the real file.
"""

from __future__ import annotations

import json

import pytest

from shl_recommender.catalog.loader import (
    CatalogLoadError,
    load_catalog,
)
from shl_recommender.catalog.models import CatalogItem
from shl_recommender.catalog.test_type import CATEGORY_TO_CODE
from shl_recommender.config import settings


# --------------------------------------------------------------------------- #
# Invariants against the real catalog
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def catalog() -> list[CatalogItem]:
    return load_catalog(settings.raw_catalog_path)


def test_loads_every_item(catalog):
    # The provided export has 377 items; loading must not silently drop any.
    assert len(catalog) == 377


def test_entity_ids_are_unique(catalog):
    ids = [item.entity_id for item in catalog]
    assert len(ids) == len(set(ids))


def test_core_fields_are_always_present(catalog):
    for item in catalog:
        assert item.entity_id
        assert item.name
        assert item.url
        assert item.test_type
        assert item.keys


def test_names_and_descriptions_have_no_embedded_newlines(catalog):
    # The raw export embeds newlines in some names/descriptions; normalisation
    # must remove them so display and search text are clean.
    for item in catalog:
        assert "\n" not in item.name and "\t" not in item.name
        assert "\n" not in item.description and "\t" not in item.description
        assert item.name == item.name.strip()


def test_microsoft_excel_365_name_is_corrected(catalog):
    # Item 4207's raw name is "Microsoft \n    365 (New)": the scrape dropped the
    # word "Excel". The corrected name (confirmed by the URL slug, the
    # description, and sample conversation C8) must be restored, not just
    # whitespace-collapsed to the wrong "Microsoft 365 (New)".
    item = next(i for i in catalog if i.entity_id == "4207")
    assert item.name == "Microsoft Excel 365 (New)"


def test_every_url_comes_from_shl(catalog):
    # URL provenance is a hard requirement: every recommended URL must be a real
    # catalog URL. The loader never constructs URLs, so this should always hold.
    for item in catalog:
        assert item.url.startswith("https://www.shl.com/")


def test_test_type_codes_are_valid(catalog):
    valid_codes = set(CATEGORY_TO_CODE.values())
    for item in catalog:
        codes = item.test_type.split(",")
        assert codes, f"{item.entity_id} has empty test_type"
        for code in codes:
            assert code in valid_codes, f"{item.entity_id} produced unknown code {code!r}"


def test_test_type_matches_keys_order(catalog):
    def expected(item):
        return ",".join(CATEGORY_TO_CODE[k] for k in item.keys)

    for item in catalog:
        assert item.test_type == expected(item)


def test_search_text_includes_name_and_description(catalog):
    sample = catalog[0]
    assert sample.name.split()[0].lower() in sample.search_text.lower()


def test_all_items_in_scope_by_default(catalog):
    # With no out-of-scope ids configured, every item is recommendable.
    assert all(item.in_scope for item in catalog)


def test_items_are_immutable(catalog):
    with pytest.raises(Exception):
        catalog[0].name = "changed"


# --------------------------------------------------------------------------- #
# Synthetic malformed-input cases
# --------------------------------------------------------------------------- #

def _write(tmp_path, payload, *, raw=False):
    path = tmp_path / "catalog.json"
    path.write_text(payload if raw else json.dumps(payload), encoding="utf-8")
    return path


def _valid_raw_item(**overrides):
    item = {
        "entity_id": "1",
        "name": "Example Test",
        "link": "https://www.shl.com/products/product-catalog/view/example/",
        "description": "An example assessment.",
        "keys": ["Knowledge & Skills"],
        "job_levels": ["Graduate"],
        "languages": ["English (USA)"],
        "duration": "30 minutes",
        "adaptive": "no",
    }
    item.update(overrides)
    return item


def test_embedded_newline_in_value_is_tolerated(tmp_path):
    # A literal newline inside a string value (the real export's quirk) must not
    # break loading, and must be normalised away.
    raw = (
        '[{"entity_id": "1", "name": "Multi\n    Line", '
        '"link": "https://www.shl.com/x/", "description": "d", '
        '"keys": ["Simulations"]}]'
    )
    items = load_catalog(_write(tmp_path, raw, raw=True))
    assert items[0].name == "Multi Line"


def test_missing_required_field_raises(tmp_path):
    bad = _valid_raw_item()
    del bad["link"]
    with pytest.raises(CatalogLoadError, match="missing fields"):
        load_catalog(_write(tmp_path, [bad]))


def test_empty_keys_raises(tmp_path):
    with pytest.raises(CatalogLoadError, match="no keys"):
        load_catalog(_write(tmp_path, [_valid_raw_item(keys=[])]))


def test_duplicate_entity_id_raises(tmp_path):
    a = _valid_raw_item(entity_id="dup")
    b = _valid_raw_item(entity_id="dup", name="Other")
    with pytest.raises(CatalogLoadError, match="duplicate entity_id"):
        load_catalog(_write(tmp_path, [a, b]))


def test_top_level_must_be_array(tmp_path):
    with pytest.raises(CatalogLoadError, match="JSON array"):
        load_catalog(_write(tmp_path, {"not": "a list"}))


def test_item_must_be_object(tmp_path):
    with pytest.raises(CatalogLoadError, match="not an object"):
        load_catalog(_write(tmp_path, ["just a string"]))


def test_empty_catalog_raises(tmp_path):
    with pytest.raises(CatalogLoadError, match="empty"):
        load_catalog(_write(tmp_path, []))


def test_missing_file_raises(tmp_path):
    with pytest.raises(CatalogLoadError, match="could not read"):
        load_catalog(tmp_path / "does_not_exist.json")


def test_optional_fields_default_when_absent(tmp_path):
    minimal = {
        "entity_id": "1",
        "name": "Minimal",
        "link": "https://www.shl.com/x/",
        "description": "d",
        "keys": ["Competencies"],
    }
    item = load_catalog(_write(tmp_path, [minimal]))[0]
    assert item.job_levels == ()
    assert item.languages == ()
    assert item.duration == ""
    assert item.adaptive is False
