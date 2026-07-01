"""Load the raw catalog export into normalised :class:`CatalogItem` records.

This is the one place that knows about the quirks of the provided JSON. Its job
is to turn the messy export into clean, validated records so nothing downstream
has to think about embedded newlines, empty optional fields, or the difference
between a parsed list and its raw string twin.

What it handles, all observed in the actual data:

* The file is not strict JSON — at least one product name contains a literal
  newline inside the quoted string — so it is read with ``strict=False``.
* ``name`` and ``description`` may contain embedded newlines/tabs; these are
  collapsed to single spaces.
* ``job_levels``, ``languages`` and ``duration`` are sometimes empty.
* ``test_type`` is derived from ``keys`` (see :mod:`shl_recommender.catalog.test_type`).
* Items are tagged ``in_scope``; see :data:`OUT_OF_SCOPE_IDS`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import CatalogItem
from .test_type import derive_test_type

# Fields the loader requires on every raw item. The export currently includes all
# of them on all items, but we check rather than assume so a malformed export
# fails with a clear message instead of a stray ``KeyError`` later.
_REQUIRED_RAW_FIELDS = ("entity_id", "name", "link", "description", "keys")

# Items to mark out of scope (Pre-packaged Job Solutions). Empty by design: the
# provided catalog is SHL's export of Individual Test Solutions, so every item is
# treated as recommendable. The seven role-bundled "...Solution" items were
# reviewed and deliberately kept in scope; if a holdout signal later shows they
# should be excluded, their ids go here and nothing else changes.
OUT_OF_SCOPE_IDS: frozenset[str] = frozenset()

# Corrected display names for items whose raw name is damaged in the export.
# Keyed by entity_id. Each correction is backed by independent evidence so it is
# not a guess.
#
# 4207: the raw name is "Microsoft \n    365 (New)" — the scrape dropped the word
#       "Excel". The true name is confirmed three ways: the product URL slug is
#       "microsoft-excel-365-new", the description begins "The Microsoft Excel 365
#       simulation...", and the sample conversation (C8) lists it as
#       "Microsoft Excel 365 (New)". Matching that name matters if the grader
#       compares recommendations by name rather than URL.
NAME_OVERRIDES: dict[str, str] = {
    "4207": "Microsoft Excel 365 (New)",
}

_WHITESPACE_RUN = re.compile(r"\s+")


class CatalogLoadError(Exception):
    """Raised when the raw catalog cannot be read or is structurally invalid."""


def _normalise_whitespace(text: str) -> str:
    """Collapse internal runs of whitespace to single spaces and trim the ends."""
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce a raw list field into a tuple of trimmed, non-empty strings.

    Defensive about the element type: the catalog stores these as lists of
    strings, but anything non-string is stringified rather than crashing, and
    blank entries are dropped.
    """
    if not value:
        return ()
    if not isinstance(value, list):
        raise CatalogLoadError(f"expected a list, got {type(value).__name__}: {value!r}")
    cleaned = (_normalise_whitespace(str(item)) for item in value)
    return tuple(item for item in cleaned if item)


def _build_search_text(name: str, description: str, keys: tuple[str, ...],
                       job_levels: tuple[str, ...]) -> str:
    """Assemble the text the retrieval layer searches over.

    Name and description carry the meaning; the category and job-level labels are
    appended so a role- or category-led query ("personality", "graduate") has
    something to match even when those words are absent from the description.
    """
    parts = [name, description, " ".join(keys), " ".join(job_levels)]
    return _normalise_whitespace(" ".join(part for part in parts if part))


def _to_item(raw: dict, *, index: int) -> CatalogItem:
    """Validate and normalise a single raw catalog entry."""
    missing = [field for field in _REQUIRED_RAW_FIELDS if field not in raw]
    if missing:
        raise CatalogLoadError(f"item at index {index} is missing fields: {missing}")

    entity_id = _normalise_whitespace(str(raw["entity_id"]))
    if not entity_id:
        raise CatalogLoadError(f"item at index {index} has an empty entity_id")

    name = NAME_OVERRIDES.get(entity_id, _normalise_whitespace(str(raw["name"])))
    url = str(raw["link"]).strip()
    description = _normalise_whitespace(str(raw["description"]))
    keys = _as_str_tuple(raw["keys"])
    if not keys:
        raise CatalogLoadError(f"item {entity_id} has no keys; cannot derive test_type")

    job_levels = _as_str_tuple(raw.get("job_levels"))
    languages = _as_str_tuple(raw.get("languages"))
    duration = _normalise_whitespace(str(raw.get("duration") or ""))
    adaptive = str(raw.get("adaptive", "")).strip().lower() == "yes"

    return CatalogItem(
        entity_id=entity_id,
        name=name,
        url=url,
        description=description,
        keys=keys,
        test_type=derive_test_type(list(keys), entity_id=entity_id),
        job_levels=job_levels,
        languages=languages,
        duration=duration,
        adaptive=adaptive,
        in_scope=entity_id not in OUT_OF_SCOPE_IDS,
        search_text=_build_search_text(name, description, keys, job_levels),
    )


def load_raw_catalog(path: Path) -> list[dict]:
    """Read and parse the raw catalog JSON.

    Uses ``strict=False`` because the export contains literal control characters
    (embedded newlines) inside string values, which strict JSON rejects.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogLoadError(f"could not read catalog at {path}: {exc}") from exc

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        raise CatalogLoadError(f"catalog at {path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise CatalogLoadError(f"expected a JSON array at top level, got {type(data).__name__}")
    return data


def load_catalog(path: Path) -> list[CatalogItem]:
    """Load the catalog and return validated, normalised items.

    Guarantees on the returned list:

    * every item has a non-empty ``entity_id``, ``name``, ``url`` and ``test_type``;
    * ``entity_id`` values are unique;
    * every ``url`` is taken verbatim from the export (never constructed).

    Raises:
        CatalogLoadError: if the file cannot be read, is not a JSON array, an item
            is structurally invalid, or two items share an ``entity_id``.
    """
    raw_items = load_raw_catalog(path)

    items: list[CatalogItem] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise CatalogLoadError(f"item at index {index} is not an object: {raw!r}")
        item = _to_item(raw, index=index)
        if item.entity_id in seen_ids:
            raise CatalogLoadError(f"duplicate entity_id: {item.entity_id}")
        seen_ids.add(item.entity_id)
        items.append(item)

    if not items:
        raise CatalogLoadError("catalog is empty")
    return items
