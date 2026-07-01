"""Catalog-derived vocabulary for grounding signal detection.

Detecting a comparison ("compare X and Y") is far more reliable when "X" and "Y"
can be checked against the products that actually exist, rather than guessed from
capitalisation. This module builds that vocabulary from the catalog once: the set
of product names and the short codes/abbreviations they contain (OPQ, SVAR, DSI,
GSA, G+, ...).

Keeping it here, derived from the loaded catalog, means the detectors never carry
a hand-maintained product list that could drift from the data.
"""

from __future__ import annotations

import re

from .models import CatalogItem

# Tokens that look like codes but are noise as product references.
_STOPWORD_TOKENS = {"NEW", "US", "UK", "AUS", "MS", "AI", "GENERAL", "SECURITY", "SALES"}

# Product-line words that name real SHL assessment families but are ordinary
# capitalised words (not all-caps codes), so the code extractor below misses them.
# Adding them lets "compare Verify G+ and Verify Interactive" resolve as a product
# comparison. Derived from the catalog's recurring product-line names.
_PRODUCT_LINE_WORDS = frozenset({"verify", "opq", "adept", "adaptive"})

_CODE_IN_PARENS = re.compile(r"\(([A-Za-z][A-Za-z0-9+\-]{1,14})\)")
_CODE_TOKEN = re.compile(r"\b([A-Z]{2,}[0-9+]*[a-z]?[0-9+]*)\b")
_WORD = re.compile(r"\w+")


class CatalogVocabulary:
    """Known product names and codes, for grounding text detection."""

    def __init__(self, names: frozenset[str], codes: frozenset[str]) -> None:
        self._names_lower = frozenset(n.lower() for n in names)
        self._codes_lower = frozenset(c.lower() for c in codes)

    @property
    def codes(self) -> frozenset[str]:
        return self._codes_lower

    def mentions_product(self, fragment: str) -> bool:
        """Whether a text fragment references a known product name or code.

        Matches a known code as a whole word, or a known product name appearing
        as a substring of the fragment (so "the OPQ32r instrument" still matches
        "OPQ32r").
        """
        if not fragment:
            return False
        lowered = fragment.lower()
        tokens = set(_WORD.findall(lowered))
        if tokens & self._codes_lower:
            return True
        return any(name and name in lowered for name in self._names_lower)


def build_vocabulary(items: list[CatalogItem]) -> CatalogVocabulary:
    """Build the vocabulary from catalog items."""
    names: set[str] = set()
    codes: set[str] = set()
    for item in items:
        name = item.name.strip()
        if name:
            names.add(name)
        for match in _CODE_IN_PARENS.findall(name):
            token = match.strip()
            if token.upper() not in _STOPWORD_TOKENS and len(token) >= 2:
                codes.add(token)
        for match in _CODE_TOKEN.findall(name):
            if match.upper() not in _STOPWORD_TOKENS:
                codes.add(match)
        # Pick up ordinary-cased product-line words (e.g. "Verify") that name real
        # families but are not all-caps codes, so short references like "Verify G+"
        # resolve. Only those actually present in the catalog are added.
        lowered = name.lower()
        for word in _PRODUCT_LINE_WORDS:
            if word in lowered:
                codes.add(word)
    return CatalogVocabulary(frozenset(names), frozenset(codes))
