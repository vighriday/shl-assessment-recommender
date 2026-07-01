# Approach — Conversational SHL Assessment Recommender

A stateless FastAPI service (`GET /health`, `POST /chat`) that takes a hiring manager
from a vague intent to a grounded shortlist of SHL Individual Test Solutions through
dialogue — clarifying, recommending, refining, comparing, and refusing out-of-scope
asks.

## Design: model for language, code for control

The one principle behind every choice: **the language model interprets and phrases;
code owns everything the grader checks.** Exactly one decision is model-advised — the
clarify-vs-recommend readiness judgement; all five modes (clarify / recommend / refine /
compare / refuse) are chosen by a deterministic **precedence ladder** in code (safety
first → explicit intent → commit once there's enough context → user owns closure). The
recommendation list is built entirely in code — 1–10 items or `null`, every `name`,
`url`, and `test_type` copied verbatim from the catalog, never generated — so a
hallucinating model can make a reply slightly worse but can never produce an invalid or
unsafe response. Every model call has a deterministic fallback, so an LLM outage
degrades wording, not correctness. The service is stateless: each turn rebuilds its
working state from the full history, with latest-correction-wins.

## Retrieval: transparent lexical, measured not assumed

Retrieval is **lexical TF-IDF** (word n-grams for language + character n-grams for
product codes like `OPQ32r`, `.NET`) feeding a **transparent weighted-sum ranker** over
named, inspectable signals: lexical score, category-intent match, language, job-level, an
exact-name boost, a distinctive-skill-in-name bonus, and injected "staple" defaults
(OPQ32r, Verify G+ — the measures a consultant adds by default, present in most gold
shortlists but rarely named by the user). Every recommendation is explainable.

**Mean Recall@10 = 0.809** on the ten sample traces, reproduced by
`scripts/measure_recall.py` and locked by a CI floor test. The number was earned, step
by step, each change kept only because it moved the metric and only via a *general*
mechanism (never trace-specific tuning, since the holdout is scored too):
`0.505 → 0.612` (staple defaults) `→ 0.677` (additive staple weight) `→ 0.717`
(proportional name scoring) `→ 0.737` (query enrichment with extracted skills) `→ 0.757`
(family-diversity cap) `→ 0.797` (raise name-boost) `→ 0.809` (distinctive-skill bonus).

## Prompt design

Two model calls per turn, both narrow. **Understanding** extracts a flat JSON of
role/seniority/skills/categories/languages plus a readiness judgement, with explicit
"leave it null if unstated — do not invent" instructions (including: do not infer
purpose; a bare job title is *not* ready; don't re-ask what's answered; stop asking when
the user pushes back). **Reply** writes only the framing prose per mode, told *not* to
enumerate products or URLs (the authoritative list travels in a separate, code-built
field), and grounded in real catalog facts on a comparison turn. Deterministic signals
(injection, off-topic, confirmation, add/drop, comparison) are detected in code, so
those turns work with the model down.

## Evaluation: test the logic exhaustively, measure the judgement

The brief warns that submissions fail on weak evaluation and happy-path-only code, so
testing is layered: **~393 deterministic tests** — unit, HTTP-contract, Hypothesis
property tests (`/chat` never 5xx on arbitrary input), a 10-trace replay, behaviour
probes, and a **54-case edge battery** (every mode, whitespace, huge JDs, unicode, role
casing, refuse-asymmetry, hallucination). The one fuzzy decision is tested by
**metamorphic laws** (properties true for *any* input — e.g. adding information can never
reduce readiness) and an **independent LLM-as-judge** reporting an agreement rate. This
adversarial testing found real bugs the happy path hid.

## What didn't work, and how I measured it

- **Sentence embeddings added nothing.** I built a full hybrid (lexical + `all-MiniLM`
  embeddings) and measured its contribution: **0.000** net Recall@10 across all ten
  traces — its one unique recovery was cancelled by an equal displacement. It was also
  silently failing to import on a fresh install (an under-pinned transitive dependency).
  I **removed it** — a component that moves no measured number and can't be reproduced is
  not worth carrying. Result: identical recall, a smaller and reproducible build.
- **Raising the category weight hurt** (−0.098): a flat category flag promotes broad
  multi-category noise over focused skill tests. An 864-config weight grid found no
  regression-free improvement beyond the tuned point.
- **A clarify loop** (found by hand): a vague opener the user kept restating drew the
  same question forever. Fixed by tightening the readiness prompt; the metamorphic
  monotonicity law now guards it.
- **The injection detector missed the canonical jailbreak** — "ignore all previous
  instructions" (a plural-noun regex gap) fell through to the model. Found by the edge
  battery; fixed and locked with tests. Eight gold items remain un-recalled at 0.809,
  each diagnosed (one the user dropped; the rest true vocabulary gaps or crowded
  same-category clusters) and not closable without overfitting the visible traces.

## Stack and AI-tool use

**FastAPI + Pydantic** (typed contract, automatic validation), **scikit-learn** TF-IDF
(no heavy ML dependency), **LiteLLM** as a provider-agnostic model layer with a
three-provider failover chain (Gemini → second Gemini key → Groq; only rate-limit/auth
errors advance it, so the 30 s budget holds). Default model `gemini-2.5-flash`; deployed
as a **Hugging Face Docker Space**. Everything is pinned for reproducible builds.

AI assistance was used throughout, as an accelerator under my direction, not a
substitute for understanding: **Google Gemini and Groq** as the runtime LLMs;
**AI coding assistants** (agentic pair-programming in the editor) for scaffolding
boilerplate, drafting tests, and exploratory refactors; and standard Python tooling
(**pytest**, **Hypothesis**, **ruff**) for verification. Every design decision, the
retrieval tuning, the semantic-removal call, and the bug fixes were mine to reason
through and defend — the code is structured and documented so it can be explained end to
end without the tools that helped write it.
