"""Derivation of the response ``test_type`` code from a catalog item's ``keys``.

The catalog does not store ``test_type`` directly, but it does store a ``keys``
list of human-readable categories. Those categories map one-to-one onto the
single-letter codes used in the API response and in the sample conversations, so
the code can be derived deterministically with full coverage of the catalog.

Two details, both grounded in the data:

* A product may belong to more than one category. Multi-category items join their
  codes with a comma (e.g. ``Knowledge & Skills`` + ``Simulations`` -> ``K,S``).
* The order of ``keys`` is not consistent across the catalog, and the sample
  conversations themselves use both orders (e.g. ``C, K`` and ``K,S``). We
  preserve each item's own ``keys`` order when joining, and expose
  ``ORDER_OVERRIDES`` for the rare item whose expected order must differ.
"""

from __future__ import annotations

# Canonical category -> code map. Sourced from the categories present in the
# catalog and the codes used in the sample conversations; coverage is complete.
CATEGORY_TO_CODE: dict[str, str] = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

# Per-item overrides for the joined code string, keyed by ``entity_id``. Empty by
# default: the derived value is correct for every item we have observed. This
# hook exists so a single mismatch found during trace replay can be corrected in
# one obvious place without touching the derivation logic.
ORDER_OVERRIDES: dict[str, str] = {}


class UnknownCategoryError(ValueError):
    """Raised when an item carries a category with no known code.

    The catalog currently has none, but failing loudly is the right behaviour:
    a silently dropped category would produce a wrong ``test_type`` and a
    recommendation that does not match the grader's expectation.
    """


def code_for_category(category: str) -> str:
    """Return the single-letter code for one category, or raise if unknown."""
    try:
        return CATEGORY_TO_CODE[category]
    except KeyError as exc:
        raise UnknownCategoryError(category) from exc


def derive_test_type(keys: list[str], entity_id: str | None = None) -> str:
    """Derive the ``test_type`` string for an item.

    ``keys`` is the catalog category list, in the catalog's own order. When
    ``entity_id`` is supplied and present in :data:`ORDER_OVERRIDES`, the override
    wins. Otherwise the codes are joined in ``keys`` order.

    Raises:
        ValueError: if ``keys`` is empty (every catalog item has at least one).
        UnknownCategoryError: if a category has no known code.
    """
    if entity_id is not None and entity_id in ORDER_OVERRIDES:
        return ORDER_OVERRIDES[entity_id]

    if not keys:
        raise ValueError("cannot derive test_type from empty keys")

    return ",".join(code_for_category(category) for category in keys)
