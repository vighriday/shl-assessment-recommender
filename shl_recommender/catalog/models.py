"""Internal catalog record.

``CatalogItem`` is the normalised, validated shape that the rest of the service
consumes. The raw export is messy (embedded newlines, empty optional fields,
fields that carry no signal); everything downstream should depend on this clean
record instead of the raw JSON, so the messiness is dealt with exactly once in
the loader.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CatalogItem(BaseModel):
    """One Individual Test Solution from the SHL catalog, normalised."""

    model_config = ConfigDict(frozen=True)

    entity_id: str = Field(description="Stable catalog id; unique across the catalog.")
    name: str = Field(description="Display name, whitespace-normalised.")
    url: str = Field(description="Canonical product URL, taken verbatim from the catalog.")
    description: str = Field(description="Product description, whitespace-normalised.")

    keys: tuple[str, ...] = Field(description="Catalog categories, in catalog order.")
    test_type: str = Field(description="Response code(s) derived from keys, e.g. 'K' or 'K,S'.")

    job_levels: tuple[str, ...] = Field(default=(), description="Targeted job levels, may be empty.")
    languages: tuple[str, ...] = Field(default=(), description="Available languages, may be empty.")
    duration: str = Field(default="", description="Duration as given (e.g. '30 minutes', 'Variable').")
    adaptive: bool = Field(default=False, description="Whether the assessment is adaptive.")

    # Scope marker for the 'Individual Test Solutions only' rule. Defaults to True;
    # the loader sets it explicitly per item so the decision lives with the data.
    in_scope: bool = Field(default=True, description="Whether the item may be recommended.")

    # The text the retrieval layer searches over. Built once in the loader from the
    # fields above so retrieval never has to re-derive or re-clean anything.
    search_text: str = Field(description="Concatenated, normalised text used for retrieval.")
