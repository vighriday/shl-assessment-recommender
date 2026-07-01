"""Deterministic tests for the metamorphic law logic (no real model).

These do not test the *agent's* judgement — that needs the real model and lives in
``scripts/adversarial.py``. They test that the **law-checking logic is correct**: that a
law catches a violation when one exists and passes when the agent behaves. To do that we
drive the probe with a controllable fake understanding model whose readiness we dictate,
so we can construct both a well-behaved agent and a deliberately broken one and confirm
the laws react correctly.
"""

from __future__ import annotations

import pytest

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.config import settings
from shl_recommender.eval.harness import probe_decision
from shl_recommender.eval.metamorphic import (
    law_comparison_suppresses_commit,
    law_confirmation_needs_prior,
    law_determinism,
    law_enrichment_monotonicity,
    law_refusal_dominance,
)


@pytest.fixture(scope="module")
def vocabulary():
    return build_vocabulary(load_catalog(settings.raw_catalog_path))


class WellBehavedLLM:
    """A fake understanding model that behaves correctly: it is ready only when the
    request contains a real differentiator (a skill or a category keyword)."""

    _DIFFERENTIATORS = ("sql", "java and", "personality", "numerical", "cognitive",
                        "simulation", "excel", "situational", "spoken-english", ".net", "python and")

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        return "reply"

    def complete_json(self, messages, *, schema=None) -> dict:
        text = " ".join(m["content"] for m in messages if m.get("role") == "user").lower()
        ready = any(d in text for d in self._DIFFERENTIATORS)
        return {"ready_to_recommend": ready, "role": "role" if text else None}


class BrokenMonotonicLLM(WellBehavedLLM):
    """A deliberately broken model: it is ready for the bare role but NOT once skills are
    added — the exact monotonicity violation the harness must catch."""

    def complete_json(self, messages, *, schema=None) -> dict:
        text = " ".join(m["content"] for m in messages if m.get("role") == "user").lower()
        # Inverted: ready when short/vague, not ready when enriched with skills.
        ready = "sql" not in text and "personality" not in text and "numerical" not in text
        return {"ready_to_recommend": ready, "role": "role"}


def _probe_with(client, vocabulary):
    def probe(text, *, with_prior_shortlist=False):
        return probe_decision(text, client, vocabulary=vocabulary, with_prior_shortlist=with_prior_shortlist)
    return probe


# --- The laws pass for a well-behaved agent ---------------------------------------

def test_wellbehaved_agent_passes_all_relevant_laws(vocabulary):
    probe = _probe_with(WellBehavedLLM(), vocabulary)
    assert law_enrichment_monotonicity(probe) == []
    assert law_refusal_dominance(probe) == []
    assert law_confirmation_needs_prior(probe) == []
    assert law_determinism(probe) == []


# --- The laws CATCH a broken agent ------------------------------------------------

def test_monotonicity_law_catches_the_java_bug(vocabulary):
    # This is exactly the "senior Java developer" class of bug: ready when vague,
    # not-ready once skills are added. The law must flag it.
    probe = _probe_with(BrokenMonotonicLLM(), vocabulary)
    violations = law_enrichment_monotonicity(probe)
    assert violations, "monotonicity law should catch a base-ready / enriched-not-ready agent"
    assert all(v.law == "enrichment_monotonicity" for v in violations)


def test_refusal_dominance_holds_on_real_signals(vocabulary):
    # Refusal is deterministic (regex signals), so it holds regardless of the fake
    # understanding model — a legit sentence must not buy an injection a pass.
    probe = _probe_with(WellBehavedLLM(), vocabulary)
    assert law_refusal_dominance(probe) == []


def test_comparison_law_holds(vocabulary):
    probe = _probe_with(WellBehavedLLM(), vocabulary)
    # Comparison is deterministic too; a comparison turn must be detected and must not
    # commit a new shortlist.
    assert law_comparison_suppresses_commit(probe) == []


def test_confirmation_without_prior_does_not_end(vocabulary):
    probe = _probe_with(WellBehavedLLM(), vocabulary)
    assert law_confirmation_needs_prior(probe) == []
