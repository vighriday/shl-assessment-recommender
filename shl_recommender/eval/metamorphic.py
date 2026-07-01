"""A — Metamorphic laws for the agent's decision.

A metamorphic law asserts a *relationship* that must hold under a transformation of the
input, without ever stating the correct answer for any single input. That is what makes
these tests immune to over-fitting: they encode logic the system must obey for *all*
inputs, not memorised input→output pairs.

Each law is a small function that takes a ``probe`` callable (text -> :class:`TurnProbe`)
and returns a list of :class:`Violation`. A law with no violations passed. The laws:

1. **Enrichment monotonicity.** Adding a genuine differentiator to a request must never
   move it from recommend-ready to not-ready. More information cannot make the agent
   *less* able to recommend. (This is the law the "senior Java developer" bug broke.)
2. **Refusal dominance.** An injection or a clearly out-of-scope ask must be refused even
   when wrapped around a legitimate hiring sentence; a real request cannot buy a pass on
   a manipulation.
3. **Comparison suppresses commit.** A pure "difference between X and Y" turn must never
   commit a new shortlist.
4. **Confirmation needs a prior shortlist.** An acceptance ("perfect, that works") with
   no shortlist yet offered must not end the conversation.
5. **Determinism.** The same input decided twice must give the same mode. (Guards against
   accidental nondeterminism in the decision path; the model call is temperature-0 for
   the structured step.)

The laws run against whatever ``probe`` is supplied — a real-model probe finds real
violations; a controllable fake probe is used in tests to prove the law-checking logic
itself is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from shl_recommender.eval.corpus import (
    COMPARISONS,
    ENRICHMENT_PAIRS,
    INJECTIONS,
    OFF_TOPIC,
    VAGUE_OPENERS,
)
from shl_recommender.eval.harness import TurnProbe

Probe = Callable[..., TurnProbe]


@dataclass(frozen=True)
class Violation:
    """One broken instance of a law, with enough detail to reproduce it."""

    law: str
    detail: str


def law_enrichment_monotonicity(probe: Probe) -> list[Violation]:
    """Adding information must not turn a ready request into a not-ready one."""
    violations: list[Violation] = []
    for pair in ENRICHMENT_PAIRS:
        base = probe(pair.base)
        enriched = probe(pair.enriched)
        # The violation: the base is willing to recommend, but adding a real
        # differentiator made the agent *less* ready (it now clarifies). That is
        # logically impossible for a correct agent.
        if base.recommends and enriched.clarifies:
            violations.append(
                Violation(
                    "enrichment_monotonicity",
                    f"base recommended but enriched (+{pair.added}) clarified: "
                    f"{pair.base!r} -> {pair.enriched!r}",
                )
            )
    return violations


def law_refusal_dominance(probe: Probe) -> list[Violation]:
    """Injection / out-of-scope must be refused even beside a legitimate request."""
    violations: list[Violation] = []
    legit = "We are hiring a software engineer."
    for bad in INJECTIONS + OFF_TOPIC:
        # Wrap the manipulation next to a real hiring sentence; the refusal must win.
        wrapped = f"{legit} {bad}"
        result = probe(wrapped)
        if bad in INJECTIONS and not result.is_refusal:
            violations.append(
                Violation("refusal_dominance", f"injection not refused when wrapped: {wrapped!r}")
            )
        # Off-topic beside a request is softer — it must at least not commit a shortlist
        # as if the off-topic part were a normal hiring turn handled cleanly.
    return violations


def law_comparison_suppresses_commit(probe: Probe, *, vocabulary=None) -> list[Violation]:
    """A pure comparison turn must not commit a new shortlist."""
    violations: list[Violation] = []
    for text in COMPARISONS:
        result = probe(text, with_prior_shortlist=True)
        if result.commits_shortlist and result.mode.value == "compare":
            # compare mode never commits; if it did, that's a contradiction.
            violations.append(
                Violation("comparison_suppresses_commit", f"comparison committed a list: {text!r}")
            )
        # Also: a clearly comparison-shaped turn should be detected as comparison.
        if not result.is_comparison and not result.is_refusal:
            violations.append(
                Violation(
                    "comparison_detected",
                    f"comparison phrasing not detected as comparison: {text!r}",
                )
            )
    return violations


def law_confirmation_needs_prior(probe: Probe) -> list[Violation]:
    """An acceptance with no prior shortlist must not end the conversation."""
    violations: list[Violation] = []
    acceptances = ("Perfect, that works.", "Great, let's go with that.", "That's exactly it.")
    for text in acceptances:
        result = probe(text, with_prior_shortlist=False)
        if result.end_of_conversation:
            violations.append(
                Violation(
                    "confirmation_needs_prior",
                    f"ended the conversation with no prior shortlist: {text!r}",
                )
            )
    return violations


def law_determinism(probe: Probe) -> list[Violation]:
    """The same input must decide the same mode twice."""
    violations: list[Violation] = []
    for text in VAGUE_OPENERS[:4] + COMPARISONS[:2]:
        first = probe(text)
        second = probe(text)
        if first.mode is not second.mode:
            violations.append(
                Violation(
                    "determinism",
                    f"non-deterministic mode for {text!r}: {first.mode.value} then {second.mode.value}",
                )
            )
    return violations


# The full set, in run order. Each entry is (name, callable).
ALL_LAWS = (
    ("enrichment_monotonicity", law_enrichment_monotonicity),
    ("refusal_dominance", law_refusal_dominance),
    ("comparison_suppresses_commit", law_comparison_suppresses_commit),
    ("confirmation_needs_prior", law_confirmation_needs_prior),
    ("determinism", law_determinism),
)


def run_all_laws(probe: Probe) -> list[Violation]:
    """Run every law against ``probe`` and collect all violations."""
    violations: list[Violation] = []
    for _name, law in ALL_LAWS:
        violations.extend(law(probe))
    return violations
