"""Build the catalog snapshot from the raw export.

Offline build step. Run after the raw catalog changes:

    python -m scripts.build_snapshot

It loads and normalises the raw export, writes the versioned snapshot, and prints
an audit summary so the result can be sanity-checked by eye before it is used.
"""

from __future__ import annotations

import collections

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.snapshot import save_snapshot
from shl_recommender.config import settings


def main() -> None:
    items = load_catalog(settings.raw_catalog_path)
    save_snapshot(items, settings.snapshot_path)

    in_scope = sum(1 for item in items if item.in_scope)
    code_counts = collections.Counter(
        code for item in items for code in item.test_type.split(",")
    )
    multi_category = sum(1 for item in items if "," in item.test_type)

    print(f"Loaded {len(items)} items from {settings.raw_catalog_path}")
    print(f"Wrote snapshot to {settings.snapshot_path}")
    print(f"  in scope:        {in_scope}/{len(items)}")
    print(f"  multi-category:  {multi_category}")
    print("  test_type codes: " + ", ".join(
        f"{code}={count}" for code, count in sorted(code_counts.items())
    ))


if __name__ == "__main__":
    main()
