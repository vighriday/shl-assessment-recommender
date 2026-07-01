"""Deterministic tests for the LLM-as-judge plumbing (no real model).

These verify the judge's aggregation and mapping logic with a fake grader, not the
quality of any real judgement (that needs the real model and lives in
``scripts/adversarial.py``). We confirm the agreement rate is computed correctly,
disagreements are surfaced, a judge failure degrades safely, and the mode→action
mapping only judges the clarify-vs-recommend hinge.
"""

from __future__ import annotations

from shl_recommender.eval.judge import action_of, judge_batch, judge_one
from shl_recommender.llm.client import LLMError


class FakeJudge:
    """Grades by a rule we control, so the aggregation can be asserted exactly."""

    def __init__(self, *, reasonable: bool = True, fail: bool = False):
        self._reasonable = reasonable
        self._fail = fail

    def complete(self, messages, *, temperature: float = 0.2) -> str:
        return ""

    def complete_json(self, messages, *, schema=None) -> dict:
        if self._fail:
            raise LLMError("judge down")
        return {"reasonable": self._reasonable, "expected": "ask", "why": "test"}


def test_action_mapping_only_covers_the_hinge():
    assert action_of("clarify") == "ask"
    assert action_of("recommend") == "recommend"
    # Refuse / compare / refine are governed by deterministic signals, not judged.
    assert action_of("refuse") is None
    assert action_of("compare") is None
    assert action_of("refine") is None


def test_agreement_rate_all_reasonable():
    report = judge_batch([("a", "ask"), ("b", "recommend")], FakeJudge(reasonable=True))
    assert report.agreement_rate == 1.0
    assert report.disagreements == ()


def test_agreement_rate_with_disagreement():
    # One reasonable, one not: build a judge that flips per call via two fakes.
    verdicts = (
        judge_one("a", "recommend", FakeJudge(reasonable=True)),
        judge_one("b", "recommend", FakeJudge(reasonable=False)),
    )
    from shl_recommender.eval.judge import JudgeReport

    report = JudgeReport(verdicts=verdicts)
    assert report.agreement_rate == 0.5
    assert len(report.disagreements) == 1
    assert report.disagreements[0].text == "b"


def test_judge_failure_is_safe():
    # A judge outage must not raise; it yields a flagged, neutral-reasonable verdict.
    verdict = judge_one("x", "ask", FakeJudge(fail=True))
    assert verdict.reasonable is True
    assert "unavailable" in verdict.why


def test_empty_batch_is_zero_not_error():
    assert judge_batch([], FakeJudge()).agreement_rate == 0.0
