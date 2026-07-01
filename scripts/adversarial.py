"""Run the adversarial harness (A metamorphic laws + B judge) against the real model.

This is the pre-submission check for the one fuzzy part of the system — the model's
clarify-vs-recommend judgement. It is a script, not a CI test, because it calls the real
model many times (it needs a key, costs quota, and is non-deterministic).

    python -m scripts.adversarial                 # laws + judge, paced for a free tier
    python -m scripts.adversarial --laws-only      # just the metamorphic laws (A)
    python -m scripts.adversarial --delay 8        # seconds between model calls

Output:
- **A:** every metamorphic law and whether it held, with the offending input on failure.
  A law failing is a real, non-overfit bug (a broken logical relationship).
- **B:** the judge's agreement rate on the vague and specific seeds, plus the specific
  disagreements for a human to read. This is a measured quality signal, not a pass/fail.
"""

from __future__ import annotations

import sys
import time

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.config import settings
from shl_recommender.eval.corpus import SPECIFIC_REQUESTS, VAGUE_OPENERS
from shl_recommender.eval.harness import probe_decision
from shl_recommender.eval.judge import action_of, judge_batch
from shl_recommender.eval.metamorphic import ALL_LAWS
from shl_recommender.llm.client import LiteLLMClient

_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_DELAY = 0.0


def _make_probe(client, vocabulary):
    """A probe that paces model calls to stay under a rate-limited free tier."""

    def probe(text, *, with_prior_shortlist=False):
        if _DELAY:
            time.sleep(_DELAY)
        return probe_decision(
            text, client, vocabulary=vocabulary, with_prior_shortlist=with_prior_shortlist
        )

    return probe


def _run_laws(probe, vocabulary) -> int:
    print(f"\n{'=' * 70}\nA — Metamorphic laws\n{'=' * 70}")
    total = 0
    for name, law in ALL_LAWS:
        # comparison law takes an extra vocabulary kwarg; call uniformly.
        violations = law(probe)
        total += len(violations)
        if violations:
            print(f"  {_RED}FAIL{_RESET} {name} ({len(violations)})")
            for v in violations:
                print(f"       {_DIM}{v.detail}{_RESET}")
        else:
            print(f"  {_GREEN}ok{_RESET}   {name}")
    return total


def _run_judge(probe, client) -> None:
    print(f"\n{'=' * 70}\nB — LLM-as-judge (clarify-vs-recommend agreement)\n{'=' * 70}")
    pairs = []
    for text in VAGUE_OPENERS + SPECIFIC_REQUESTS:
        decision = probe(text)
        action = action_of(decision.mode.value)
        if action is not None:  # only judge clarify/recommend turns
            pairs.append((text, action))

    report = judge_batch(pairs, client)
    print(f"\n  agreement rate: {report.agreement_rate:.0%}  ({len(pairs)} judged)")
    if report.disagreements:
        print("  disagreements (judge would have chosen differently):")
        for v in report.disagreements:
            print(f"    {_RED}-{_RESET} {_DIM}{v.text!r} → agent {v.agent_action}, "
                  f"judge {v.expected}: {v.why}{_RESET}")


def main(argv: list[str]) -> None:
    global _DELAY
    args = list(argv)
    laws_only = "--laws-only" in args
    if "--laws-only" in args:
        args.remove("--laws-only")
    if "--delay" in args:
        i = args.index("--delay")
        _DELAY = float(args[i + 1])
        del args[i : i + 2]

    items = load_catalog(settings.raw_catalog_path)
    vocabulary = build_vocabulary(items)
    client = LiteLLMClient()
    probe = _make_probe(client, vocabulary)

    total_violations = _run_laws(probe, vocabulary)
    if not laws_only:
        _run_judge(probe, client)

    print()
    if total_violations:
        print(f"{_RED}A: {total_violations} metamorphic violation(s) — real bugs to fix.{_RESET}")
    else:
        print(f"{_GREEN}A: all metamorphic laws held.{_RESET}")


if __name__ == "__main__":
    main(sys.argv[1:])
