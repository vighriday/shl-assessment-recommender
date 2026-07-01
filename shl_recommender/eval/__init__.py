"""Adversarial evaluation harness for the agent's judgement.

The deterministic parts of the system (retrieval, the policy ladder, the response
contract) are covered by the fast unit and trace tests. The one genuinely fuzzy
input — the model's clarify-vs-recommend readiness judgement — cannot be unit-tested
against fixed answers without hard-coding them, so it is tested two ways here:

* **A — metamorphic laws** (:mod:`shl_recommender.eval.metamorphic`): properties that
  must hold for *any* input under a known transformation (e.g. adding information can
  never make a request *less* ready). These find whole classes of bug without
  asserting a single hand-picked answer, so they cannot overfit.
* **B — LLM-as-judge** (:mod:`shl_recommender.eval.judge`): an *independent* model
  call grades whether the clarify-vs-recommend decision was reasonable, and we report
  the agreement rate — a measured quality signal, not a hard-coded expectation.

The two compose: metamorphic laws prove the system obeys its logic; the judge measures
whether the judgement is *good* within that logic. Neither hard-codes the right answer
for a made-up prompt.
"""

from .harness import TurnProbe, probe_decision
from .judge import JudgeReport, JudgeVerdict, action_of, judge_batch
from .metamorphic import Violation, run_all_laws

__all__ = [
    "action_of",
    "JudgeReport",
    "JudgeVerdict",
    "judge_batch",
    "probe_decision",
    "run_all_laws",
    "TurnProbe",
    "Violation",
]
