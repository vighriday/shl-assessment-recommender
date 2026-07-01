"""Coverage guard: every assessment the sample conversations recommend must exist
in our catalog.

This is the data-side ceiling on recall. If a gold item is not in the catalog we
can never recommend it, so this test fails loudly if the catalog and the traces
ever drift apart. It parses the recommendation tables in the trace Markdown and
checks each product URL against the loaded catalog.
"""

from __future__ import annotations

import glob
import os
import re

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.config import settings

_URL_RE = re.compile(r"https?://www\.shl\.com[^\s>|)]+")


def _normalise_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _gold_rows() -> list[tuple[str, str]]:
    """Return (trace_file, product_url) for every recommendation row in the traces."""
    rows: list[tuple[str, str]] = []
    trace_dir = settings.project_root / "data" / "traces"
    for path in sorted(glob.glob(str(trace_dir / "*.md"))):
        base = os.path.basename(path)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.count("|") < 6:
                    continue
                match = _URL_RE.search(line)
                if match:
                    rows.append((base, match.group(0)))
    return rows


def test_traces_were_found():
    # Guard against silently passing because no traces were parsed.
    assert _gold_rows(), "no gold recommendation rows parsed from traces"


def test_every_gold_url_is_in_the_catalog():
    catalog_urls = {_normalise_url(item.url) for item in load_catalog(settings.raw_catalog_path)}
    missing = sorted(
        {(base, url) for base, url in _gold_rows() if _normalise_url(url) not in catalog_urls}
    )
    assert not missing, f"gold items missing from catalog: {missing}"
