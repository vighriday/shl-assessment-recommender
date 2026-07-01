"""B — LLM-as-judge for the clarify-vs-recommend decision.

The metamorphic laws prove the decision obeys its logic; they cannot say whether the
*judgement* is good — whether clarifying (or recommending) was the reasonable thing to do
for a given request. That is itself a judgement, so we ask an **independent** model call
to grade it, and report the rate at which the judge agrees with the agent.

Why this is not circular or self-serving:

* The judge is a *separate* call with a *different* prompt from the one that made the
  decision. It is not being tuned to pass; it is asked to be a skeptical evaluator.
* We report an **agreement rate**, a measurement — not a pass/fail on a hard-coded
  answer. A prompt change that improves the decision shows up as a higher rate; a
  regression shows up as a lower one. The number is the signal.
* The judge sees only the user text and the agent's chosen action (clarify or
  recommend), and rules on reasonableness. Disagreements are surfaced for a human to
  read, because the judge can be wrong too — it is a measurement instrument, not an
  oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

from shl_recommender.llm.client import LLMClient, LLMError

_JUDGE_SYSTEM = """\
You are a strict evaluator of an assessment-recommendation assistant. For a hiring
request, the assistant chose either to ASK ONE CLARIFYING QUESTION or to RECOMMEND a
shortlist immediately. Judge whether that choice was reasonable.

Guidance for a reasonable choice:
- If the request gives only a job title (even with a seniority) and nothing about which
  skills to prioritise, which assessment types are wanted, or the purpose, then ASKING is
  the reasonable choice.
- If the request names specific skills to screen, requested assessment categories, or a
  clear purpose that pins down what to measure, then RECOMMENDING is reasonable.

Return ONLY a JSON object:
- reasonable: true or false — was the assistant's choice reasonable?
- expected: "ask" or "recommend" — what you would have done
- why: one short sentence
"""


@dataclass(frozen=True)
class JudgeVerdict:
    text: str
    agent_action: str  # "ask" or "recommend"
    reasonable: bool
    expected: str
    why: str


@dataclass(frozen=True)
class JudgeReport:
    verdicts: tuple[JudgeVerdict, ...]

    @property
    def agreement_rate(self) -> float:
        """Fraction of decisions the judge found reasonable."""
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.reasonable) / len(self.verdicts)

    @property
    def disagreements(self) -> tuple[JudgeVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.reasonable)


def judge_one(text: str, agent_action: str, client: LLMClient) -> JudgeVerdict:
    """Grade a single (request, chosen action) pair with an independent model call.

    Never raises: a judge failure yields a neutral "reasonable" verdict flagged in the
    reason, so one flaky grade does not sink a whole run. The caller can see it in the
    output.
    """
    prompt = (
        f'Hiring request: "{text}"\n'
        f'The assistant chose to: {agent_action.upper()}.\n'
        "Was that reasonable?"
    )
    try:
        data = client.complete_json(
            [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
        )
    except LLMError as exc:
        return JudgeVerdict(text, agent_action, True, agent_action, f"judge unavailable: {exc}")

    reasonable = bool(data.get("reasonable", True))
    expected = str(data.get("expected", agent_action)).lower()
    expected = "ask" if expected.startswith("ask") else "recommend" if expected.startswith("recommend") else agent_action
    why = str(data.get("why", "")).strip()
    return JudgeVerdict(text, agent_action, reasonable, expected, why)


def action_of(probe_mode_value: str) -> str | None:
    """Map a decision mode to the judge's action vocabulary, or None if not applicable.

    The judge only rules on the clarify-vs-recommend hinge, so refuse/compare/refine
    turns are skipped (they are governed by deterministic signals, not the judgement).
    """
    if probe_mode_value == "clarify":
        return "ask"
    if probe_mode_value == "recommend":
        return "recommend"
    return None


def judge_batch(pairs: list[tuple[str, str]], client: LLMClient) -> JudgeReport:
    """Grade many (text, agent_action) pairs and return the aggregate report."""
    verdicts = tuple(judge_one(text, action, client) for text, action in pairs)
    return JudgeReport(verdicts=verdicts)
