# SHL Assessment Recommender — Build Plan & Ledger

## 0. What this document is

`ARCHITECTURE_DECISIONS.md` records **what** we built and why each design choice
was made. This document records **how the build is run**: the method we work to,
and a live ledger of every phase — what we *planned* to do in it, what was
*actually* delivered, and how it was proven before we moved on.

It is deliberately a **ledger**, not a plan we wrote once and left behind. Before a
new phase begins, the previous phases are marked done here with the evidence that
closed them. That keeps two promises at the same time:

* to a reader — nothing is claimed as finished that was not verified, and the gap
  between intention and delivery is visible rather than hidden;
* to ourselves — the ledger is the checklist that carries the work forward, so each
  phase starts from a known-good, recorded state instead of a vague memory.

It is written plainly on purpose. If a sentence sounds impressive but cannot be
re-explained in ordinary words, that is a bug in this document.

---

## 1. Why the method matters here

This project was built with AI assistance, which the brief permits. That raises a
fair question for anyone reviewing it: *was AI used as an engineering tool by
someone who owns the result, or did the tool produce a project the author cannot
account for?*

The honest answer is in how the work was run, so it is worth stating the standard
we held to. The difference we care about:

* **Prompted software** is the output of asking a model for code and keeping what
  compiles. The model makes the decisions; when it breaks, nobody can say why it
  was built that way.
* **AI-native engineering** treats the model as a fast, fallible collaborator inside
  a disciplined process. The human sets the contract, decides the architecture, and
  owns the acceptance criteria. The model drafts, explores, and stress-tests against
  those criteria. Nothing ships that is not measured or tested, and every non-obvious
  choice is recorded so it can be defended without the tool that helped write it.

The measure of the second thing is not how much the AI wrote. It is whether, with
the AI taken away, the human still owns the result completely. This document, the
architecture record, and the six design notes under `docs/` are that account.

---

## 2. The principles the build obeys

These are the rules the whole build follows. Each shows up as a concrete practice
or an artifact later, and most are enforced by a test.

1. **Specify before building.** The architecture was written and owned *before*
   code, and corrected when the real data disproved an early guess. A written
   contract is the thing you hold the model to; without it every generation drifts.

2. **Use the model for language, use code for control.** The model reads the messy
   request and writes the reply's prose. **Code** owns everything the grader checks:
   the response schema, the recommendation list, every URL (copied from the catalog,
   never generated), the `test_type` codes (derived mechanically), the question
   budget, the turn cap. A model that hallucinates can make a reply slightly worse;
   it can never make the output invalid.

3. **Measure, don't vibe.** Retrieval quality is a number — mean Recall@10 —
   improved from `0.505` to `0.757` through general mechanisms, each step recorded
   with its reason, and locked by a regression floor test.

4. **Validate the AI's claims against ground truth.** Nothing the model asserted
   about the data was trusted on its word. This caught real traps: the catalog does
   not parse with strict JSON, one product name arrived corrupted, the role casing
   in the samples differs from the JSON contract.

5. **Use the samples to validate, never to memorise.** The ten sample conversations
   are a graded-behaviour reference, not a training set. Every component was built
   from principles first, checked against the traces, and tuned no further than the
   point where gains would only come from overfitting the visible ten.

6. **Degrade, don't collapse.** The one runtime dependency that can fail is the
   language model. Rule-decided turns need no model call; semantic retrieval falls
   back to lexical-only. Every degradation is logged and surfaced in health; there
   are no silent fallbacks.

7. **Leave the reasoning on the record.** Every non-obvious choice, rejected
   alternative, and deliberately out-of-scope idea is written down.

---

## 3. The build ledger

The work runs as a sequence of self-contained phases. Each produces something
verifiable before the next begins — no scaffolding left half-finished, no phase
starting on top of an unproven one. This ordering is deliberate: a problem is caught
in the phase that introduced it, not three phases downstream where the cause is hard
to find.

### Status at a glance

| # | Phase | Status | Proven by |
|---|-------|--------|-----------|
| 0 | Specification | ✅ Done | Architecture record read and owned; corrected against real data. |
| 1 | Data foundation | ✅ Done | Catalog loader, `test_type` derivation, snapshot, vocabulary — with tests. |
| 2 | Response contract | ✅ Done | Pydantic schemas; role-synonym, strict-output, validation tests. |
| 3 | Conversation model | ✅ Done | State + deterministic signal detection; adversarial and negative-case tests. |
| 4 | Understanding | ✅ Done | LLM extraction with a strict fallback; graceful-degradation tests. |
| 5 | Policy & retrieval | ✅ Done | Precedence engine + transparent ranker; floor test. (Retrieval later simplified to lexical-only; see 9.5.) |
| 5.5 | Operational hardening | ✅ Done | Logging, boot validation, health, build stamp, error contract — 28 tests. |
| 6 | Response assembly | ✅ Done | Engine wires state→policy→retrieval→shortlist→reply; 22 tests, confirmation-close validated against C1. |
| 7 | Serving (`/health`, `/chat`) | ✅ Done | FastAPI app mounts the engine + operational primitives; HTTP + property tests. |
| 8 | Evaluation harness | ✅ Done | Offline trace-replay (all 10) + behaviour probes + live replay script; surfaced and fixed a curly-quote signal bug. |
| 8.5 | Adversarial testing (A+B) | ✅ Done | Metamorphic laws + LLM-as-judge for the fuzzy judgement; found and fixed 4 detector gaps; judge agreement 100%. |
| 9 | Deploy | ✅ Done | Live on Hugging Face Spaces; health green, `/chat` returns correct recommendations; real model validated. |
| 9.5 | Pre-submission hardening | ✅ Done | Full audit vs the brief. Removed the dead semantic stage (0 measured recall, under-pinned) → lexical-only; raised Recall@10 0.757→**0.809**; fixed `/health` to exact `{"status":"ok"}`; fixed the injection detector's plural gap + the general-advice gap (found by an exhaustive edge battery); added log key-redaction, coherence + hallucination tests, and 54 edge-case tests → ~393 total, all green. |
| 10 | Approach document | ⏳ Next | *(pending)* the two-page write-up the brief asks for. |

Everything above Phase 10 is complete and tested. The detail for each closed phase
follows.

### Phase-by-phase record

Each entry states what the phase was *meant* to deliver, what was *actually* built
(including anything the plan did not foresee), and the evidence that closed it.

**Phase 0 — Specification.** *Planned:* a design owned before any code.
*Delivered:* `ARCHITECTURE_DECISIONS.md`, rewritten against the real catalog and
the ten real conversations, with every corrected assumption marked. *Closed by:*
the author reading it end to end and being able to defend each decision.

**Phase 1 — Data foundation.** *Planned:* load the catalog into clean records.
*Delivered:* a loader that handles the export's real quirks (non-strict JSON,
embedded newlines, a corrected damaged name); mechanical `test_type` derivation with
full catalog coverage; a versioned offline snapshot; a catalog vocabulary for signal
grounding. *Closed by:* `tests/catalog/` including a 100%-coverage `test_type` test
and a gold-URL-presence check. *Not foreseen at plan time:* the strict-JSON trap and
the corrupted name — both found by inspecting the data, both documented in
`docs/catalog_data_notes.md`.

**Phase 2 — Response contract.** *Planned:* strict request/response models.
*Delivered:* Pydantic schemas that are lenient on input (role casing and synonyms
normalised) and strict on output (exactly the contract's fields, validated
`test_type`). *Closed by:* `tests/api/test_schemas.py`. *Not foreseen:* the
`User`/`Agent` vs lowercase mismatch between the displayed samples and the JSON
contract — a hard-eval risk, handled by a synonym map.

**Phase 3 — Conversation model.** *Planned:* represent the conversation and detect
the signals that route a turn. *Delivered:* an immutable state model and a
deterministic signal detector for comparison, refusal, injection, and confirmation,
grounded in the real catalog vocabulary. *Closed by:* `tests/conversation/`,
including adversarial phrasings and the negative cases that prove the safety guards
hold.

**Phase 4 — Understanding.** *Planned:* pull structured meaning from a messy
request. *Delivered:* an LLM extraction step returning role, skills, categories, and
a readiness judgement, with a strict deterministic fallback and no model call on
turns rules already decide. *Closed by:* `tests/llm/` using a fake client and
covering the degradation paths.

**Phase 5 — Policy & retrieval.** *Planned:* decide the turn's action and produce a
ranked shortlist. *Delivered:* a precedence-ordered policy engine whose
clarify-vs-commit gate is advised by the model but bounded by code, and a hybrid
lexical + semantic retriever with a transparent, inspectable ranker. *Closed by:*
policy tests validated against all ten traces, plus a Recall@10 scoreboard
(`0.505 → 0.757`) and a regression floor test. *Not foreseen:* that a pure
structural gate recommends too early on four traces — which is exactly what the
model-advised readiness judgement fixed.

**Phase 5.5 — Operational hardening.** *Planned:* originally folded into serving,
pulled forward deliberately so the service is safe *before* the app is assembled.
*Delivered, framework-free so it is testable without a server:*

* **Structured logging** (`observability/logging.py`) — JSON or console, promotes
  `extra=` context onto the line, so every degradation is visible. The design rule:
  log every degradation, never log a secret.
* **Startup validation / fail-fast** (`bootstrap.py`) — the single composition point
  loads the catalog and asserts the hard invariants (non-empty, every `test_type`
  valid, every item linkable) *before the first request*. A hard failure refuses to
  start; a soft one (no embedding model) degrades and is logged. Moving failure to
  the front door keeps it out of the middle of an evaluation.
* **Meaningful health** (`observability/health.py`) — reports real per-component
  readiness under a hard/soft weighting (catalog hard; semantic and language model
  soft), so the status is honest: `ok`, `degraded`, or `unhealthy`, never a green
  light over a broken service. It never *calls* the model — readiness is
  configuration, not a paid ping.
* **Build/version stamp** (`observability/build_info.py`) — resolves version and git
  commit defensively across installed / source / deploy-host runs, so a running
  instance can be tied back to a build. Reported in health.
* **Request/error contract** (`api/errors.py`) — one stable error shape and a closed
  exception→status map; internal causes go to the logs, never onto the wire
  (model failure → 502, validation → 422, anything else → a generic 500).

*Closed by:* 28 tests across `tests/observability/`, `tests/api/test_errors.py`, and
`tests/test_bootstrap.py`, including the fail-fast paths and the "internals never
leak" checks. Full suite green with zero regressions. *Why pulled forward:* these
are the artifacts most easily forgotten at deploy time; building them now means
Phase 7 only has to *mount* them, and the service is defensible before it is served.

**Phase 6 — Response assembly.** *Planned:* turn a conversation into the API
response. *Delivered:* a `ResponseEngine` that runs one turn end to end —
reconstruct state → decide policy → retrieve (only when committing) → build the
shortlist in code → write the reply with the model → assemble a validated
`ChatResponse`. The division of labour is made concrete here: `shortlist.py` builds
the 1..10 (or null, never []) list with every field copied verbatim from the
catalog, and `reply.py` lets the model phrase the reply per mode with a
deterministic fallback for *every* mode, so a model outage degrades wording, never
correctness. *Closed by:* 22 tests across `tests/response/`, including the
model-down path (the whole turn still returns a valid response) and the
contract-shape checks. *Not foreseen at plan time:* on a **confirmation close** the
sample conversations re-show the *same* shortlist they already offered, but
retrieving on a bare "yes" (which carries no requirements) cannot reproduce it.
Rather than weaken the model-skip optimisation, we added `recover_prior_shortlist`,
which — since the service is stateless — recovers the exact items from the catalog
URLs in the prior assistant turn. Validated against the real C1 trace: it recovers
the three OPQ items, in order, with the correct `test_type`.

**Phase 7 — Serving.** *Planned:* expose the two endpoints and mount the
operational primitives built in 5.5. *Delivered:* a thin FastAPI app
(`api/app.py`) with a `create_app` factory that runs `bootstrap` (so a bad catalog
fails at process start), constructs the provider-agnostic model client and the
response engine, and serves:

* `GET /health` — the honest health report, 200 while serving (`ok`/`degraded`) and
  503 when `unhealthy`, with the build stamp;
* `POST /chat` — validates the request, runs the turn through the engine, and
  serialises it with the configured null-vs-[] behaviour; any failure inside the
  turn is mapped to the stable error envelope, never a leaked stack trace.

The web layer is deliberately thin — every real decision was already made and tested
below it, so this file only knows requests, responses, and status codes; nothing
about hiring is decided here. *Closed by:* 10 HTTP-level tests through a `TestClient`
(both endpoints, exact contract shape, the 422/valid-turn/refusal paths, role-casing
tolerance, and a model-down turn that still returns a valid 200) plus 3
Hypothesis property tests asserting that for *arbitrary* histories `/chat` never
returns a 5xx and the response is always either a valid three-field 200 or a shaped
4xx — the edge-case robustness the brief rewards, proven rather than asserted.

*Fix found during this phase:* the embedding model was silently failing to load on
the dev machine because a machine-level `HF_HOME` pointed at a drive absent here, so
the whole system had been running lexical-only without saying so beyond the health
report. The model loader now points the Hugging Face cache at our own project-local
directory before importing it, so loading is deterministic across machines and the
deploy host. With that, semantic retrieval loads, the hybrid is genuinely live
(confirmed: "safety-critical plant operators" → Safety & Dependability items via
meaning alone), and the Recall@10 floor test now runs for real instead of skipping.

**Phase 8 — Evaluation harness.** *Planned:* a way to check the whole system against
the sample conversations and the brief's edge cases, not just in units. *Delivered:*
three things sharing one trace parser (`scripts/trace_utils.py`):

* an **offline trace-replay test** (`tests/eval/test_trace_replay.py`) that runs
  every one of the ten traces' real user turns through the engine with the model
  faked to the understanding it would extract, asserting the behaviours that must
  hold — the final turn ends and carries a shortlist, a comparison turn commits no
  new list, every turn returns the valid contract, and the committed shortlist
  recalls the gold;
* a **behaviour-probe suite** (`tests/eval/test_behavior_probes.py`) that runs the
  brief's edge-case list (its section 9) through the engine as behaviour — injection,
  a legal question, general hiring advice, a no-exact-match request, whitespace, and
  a model-down refusal;
* a **live replay script** (`scripts/replay_traces.py`) that runs the same replay
  against the real model for a pre-submission check — a script, not a test, because
  it needs a key and is non-deterministic.

*What the harness caught — the reason it exists.* Replaying the traces immediately
found two real bugs the unit tests had missed, both of which would have cost scored
behaviour:

1. **Curly-apostrophe blindness.** The real transcripts use typographic apostrophes
   (`’`), but every signal pattern matched only the ASCII `'`. So "we'll", "that's",
   "don't" — and therefore confirmation, add, drop, and injection detection — silently
   failed on the exact text a user pasting from a document would send. Fixed at the
   single normalisation point (`_clean`) by folding typographic quotes and dashes to
   ASCII, which repairs all detectors at once.
2. **Confirmation-detector gaps.** Natural closings the traces actually use — "that's
   good", "that covers it", "locking it in", "final list:" — were not recognised, so
   the agent would not have ended the conversation. Widened the confirmation pattern
   to cover them.

*A design refinement the harness forced.* C10's final turn both drops an item and
accepts ("Drop the OPQ. Final list: ..."), and the trace marks it as ending. The
policy now recognises this **edit-and-close**: when an accepting turn also carries an
add/drop, it ends the conversation but commits a *freshly refined* shortlist that
honours the edit, instead of re-showing the prior list. *Closed by:* 32 tests (20
replay across the ten traces, 11 probes, and the updated policy tests), full suite
green with zero skips.

**Phase 8.5 — Adversarial testing of the fuzzy judgement.** *Why it exists:* a real
use surfaced a bug the trace suite missed — "senior Java developer" was recommended when
it should have clarified. That is the one place a rule cannot decide (the model's
clarify-vs-recommend judgement), and hand-testing it does not scale and would overfit.
*Delivered* (`shl_recommender/eval/`, run by `scripts/adversarial.py`): two evaluators
that need no hard-coded answers.

* **A — metamorphic laws.** Properties that must hold for *any* input under a
  transformation: adding information can never make a request *less* ready
  (monotonicity); an injection beside a legitimate sentence is still refused; a
  comparison never commits a new list; an acceptance with no prior shortlist never ends
  the conversation; the same input decides the same mode twice. These test *logic*, not
  answers, so they cannot overfit — and running them immediately found and fixed four
  detector gaps the ten traces never exercised: "compare X and Y", "X or Y — which
  fits", "disregard the catalog", and "reveal your full prompt".
* **B — LLM-as-judge.** An *independent* model call grades whether each
  clarify-vs-recommend decision was reasonable and reports an agreement rate — a measured
  quality signal, not a hard-coded expectation. On the real model the laws all held and
  the judge agreed with the agent on 18 of 18 decisions.

*Closed by:* deterministic tests that drive the laws with a controllable fake (proving
each law catches a broken agent and passes a good one) so CI stays fast and key-free,
plus the real-model run as the pre-submission check. *The principle it embodies:* the one
component that cannot be made deterministic is tested by logic (metamorphic) and
measurement (judge), never by memorising answers. *Root-cause fix:* the readiness prompt
was tightened so a bare job title is not treated as ready, and purpose is no longer
inferred when unstated.

**Phase 9 — Deploy.** *Planned:* a live public endpoint the grader can reach.
*Delivered:* the service runs as a Hugging Face Docker Space at
`https://hriday29-shl-assessment-recommender.hf.space`. A `Dockerfile` (non-root user,
port 7860, embedding model pre-downloaded at build) is built server-side by HF; the
Gemini key is a Space secret, never in the repo. *Verified live:* `/health` reports
`ok` (377 catalog items, semantic model loaded, model configured); `POST /chat` returns
the valid contract with correct recommendations (a Java hire returns Java 8, SQL, Java
Frameworks plus the OPQ32r/Verify G+ staples). *A decision made with evidence:* we
measured that semantic retrieval adds zero Recall@10 on all ten traces, so the service
would fit a tiny lexical-only host with no measured loss — but we deployed to Spaces
(16 GB free) with the hybrid on, because the RAM is free there and it keeps the full
capability. *One honest caveat recorded:* the free-tier Gemini quota (~5 req/min) can
throttle the live model under bursts, at which point the service degrades cleanly to
valid, code-owned responses; a billed key or a different provider removes it via one
env var. *Model note:* the default was set to `gemini/gemini-2.5-flash` (current, with a
more generous free-tier limit than the 3.5 tier). *Also added at deploy:* a
human-facing **chat UI** (a Gradio `ChatInterface`) mounted at `/ui` on the same app,
so a person can test the agent as a user from the same URL — it calls the same engine
in process and is fault-isolated (if Gradio is unavailable the API is unaffected); and
a friendly `GET /` index so the bare URL is self-describing rather than a 404.

---

## 4. What the method has bought so far

Concrete outcomes, because the claim is only worth making if it shows in the result:

* **Traps caught early.** The strict-JSON break, the corrupted product name, and the
  role-casing mismatch were each found by checking the AI's claims against the data.
  Any one, missed, would have failed the eval.
* **Safe under its own uncertainty.** Code owns every contract and the model only
  advises, so the most non-deterministic component in the stack cannot produce an
  invalid response, exceed the recommendation cap, or blow the turn budget.
* **Honest operation by default.** Startup fails loud on broken data, health tells
  the truth about degradation, and errors never leak internals — the boring
  properties a service is actually judged on in production.
* **A deliverable the author can defend.** Every decision has a recorded reason and a
  test, so the project can be explained end to end without the tool that built it.

Through the pre-submission hardening pass (Phase 9.5) the same discipline held: a
full re-audit against the brief removed a subsystem that moved no measured number
(semantic retrieval), raised Recall@10 to 0.809 only through general mechanisms, and
an exhaustive edge-case battery caught real bugs the happy path hid — including a
security gap where the canonical "ignore all previous instructions" jailbreak was not
being refused. Only the two-page approach document (Phase 10) remains.
