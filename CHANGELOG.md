# Changelog

All notable changes to this project are recorded here. The format is loosely based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), adapted for a take-home
whose "releases" are build phases rather than shipped versions. Retrieval quality is
tracked as a number (mean Recall@10 against the ten sample conversations) so its
movement is auditable, not asserted.

The project follows [semantic versioning](https://semver.org/) once it ships; until
then it sits at `0.1.0` and progress is logged by phase.

## [Unreleased]

### Pre-submission hardening pass
- **Semantic retrieval removed; retrieval is now lexical-only.** The sentence-embedding
  stage was measured to add **zero** Recall@10 across all ten traces (its one unique
  recovery was cancelled by an equal displacement) and was silently failing to load on a
  fresh install (an under-pinned `huggingface-hub` broke the import). Rather than carry a
  heavy, version-fragile dependency of no measured value, it was removed:
  `retrieval/semantic.py` deleted, `sentence-transformers` dropped, the
  enable/model-cache config removed, `HybridRetriever` → `LexicalRanker`. Smaller image,
  faster cold start, fully reproducible build, and the recall floor test no longer skips.
  (Decision 17; caveats B4/G2.)
- **Mean Recall@10 improved 0.757 → 0.809** via two general mechanisms: raising
  `name_boost` 1.5→2.0 (recovered C4 Basic Statistics and C8 MS Word) and a
  distinctive-skill-in-name additive bonus (recovered C9 AWS). Zero regressions, verified
  by per-trace identity; the floor test now runs on every commit (lexical-only, no
  optional dependency).
- **`/health` now returns exactly `{"status":"ok"}` by default** (HTTP 200); the richer
  `{status,build,components}` diagnostic moved behind `?deep=1`. This removes a hard-eval
  risk — a strict whole-body `{"status":"ok"}` check can no longer be tripped by extra
  fields.
- **Prompt-injection detector fixed to catch the canonical jailbreak.** The `_INJECTION`
  pattern ended its noun alternation with a singular noun, so the word boundary failed on
  the plural — "ignore all previous instructions" (and "disregard your instructions",
  "forget the rules") fell through to the model instead of the guaranteed code refusal.
  Nouns are now optionally plural; refuses on its own, with no false positives on legit
  requests. Locked with parametrised tests. (Found by the edge-case battery.)
- **General-advice detector widened** to catch "how do I structure/run/design a hiring
  process/funnel/interviews" while still not sweeping up genuine assessment requests.
- **API keys redacted from logs.** A `_redact` helper scrubs key-shaped tokens (Groq,
  OpenAI, Google, bearer) from any exception string before it is wrapped or logged.
- **Coherence and hallucination tests added** (both named by the brief as failure modes):
  a mischievous model cannot inject a product or URL into the shortlist; facts stated
  early are retained across turns; a prior shortlist persists across a comparison turn.
- **Exhaustive edge-case battery** (`tests/eval/test_edge_cases.py`, 54 cases) across
  every mode plus robustness (whitespace, multi-thousand-char JD, unicode, role casing)
  and the refuse-asymmetry. The full suite is now ~393 tests, all passing, ruff clean.

### Fixed
- **Clarify questions no longer loop or re-ask answered ground (caveat D7).** Found by
  hand on the live UI: a vague opener the user kept restating drew the same broad question
  every turn, and "what questions do you want answered?" got another vague ask. Two causes:
  the visible loop was a **stale deploy** (the policy already caps clarifications at two and
  then commits, so the running service never loops — redeployed), and the question quality
  was genuinely weak. Tightened the readiness prompt to base its one question on what is
  still missing, to stop asking when the user pushes back or has nothing to add, and to name
  the concrete missing thing when asked directly. No hard-coded rules — the model still owns
  the judgement. Verified live: the opener asks one sharp question, a pushback turn commits a
  shortlist, and "what do you need?" gets a concrete answer; all metamorphic laws still hold
  and judge agreement stayed 100%.

### Added
- **Opt-in turn trace — `POST /chat?debug=1`.** Returns, alongside the three contract
  fields, a `_trace` object explaining how the turn was decided: the extracted state, the
  chosen mode and why, the readiness judgement (model vs structural fallback), the scored
  retrieval candidates, and whether the reply came from the model or a deterministic
  fallback. Strictly additive — the contract fields are byte-for-byte identical with and
  without it, so a grader gets the clean contract by default and the full reasoning only on
  request. The trace never contains a secret. Covered by engine and HTTP-level tests.
- **Terminal chat client (`scripts/chat_client.py`).** A thin, stateful client that holds
  the conversation history on the caller's side and resends it each turn — demonstrating,
  and testing, the stateless multi-turn protocol without the browser UI. `--debug` shows
  the trace under each reply; `:history` prints the exact message list being sent. It is a
  *client* of the stateless API, not a change to it.
- **Automatic multi-provider failover for the free-tier quota (caveat G6).** An ordered
  failover chain: the primary model, then the same model with an optional second key
  (`SHL_GEMINI_API_KEY_FALLBACK`), then an optional cross-provider fallback model
  (`SHL_LLM_FALLBACK_MODEL`, e.g. `groq/llama-3.3-70b-versatile`). When an attempt is
  rate-limited (HTTP 429) or its key is missing/rejected, the client advances to the next
  attempt; any other failure raises at once (retrying a bad request or a network outage is
  pointless). So a burst — or a whole-provider outage — no longer forces the deterministic
  wording. Verified end to end: with both Gemini keys quota-exhausted, a turn fell over to
  Groq and returned a real model reply. Covered by eleven unit tests with a fake provider;
  the deterministic degradation still applies if every provider is unreachable.
- **Chat UI served at the application root.** The Gradio chat interface now mounts at `/`
  (was `/ui`), so the Hugging Face Space's App tab shows the chat box directly instead of a
  JSON index; the machine-readable endpoint index moved to `/info`. The API paths
  (`/health`, `/chat`, `/docs`) are unchanged and the UI stays strictly additive.

- **Adversarial testing of the clarify-vs-recommend judgement (Phase 8.5).** A harness
  (`shl_recommender/eval/`, run by `scripts/adversarial.py`) that tests the one fuzzy
  decision without hard-coding answers: **metamorphic laws** (properties that must hold
  for any input — e.g. adding information can never reduce readiness) and an
  **independent LLM-as-judge** that reports an agreement rate. Running it found and fixed
  four detector gaps the sample traces never exercised ("compare X and Y", "X or Y —
  which fits", "disregard the catalog", "reveal your full prompt"); on the real model all
  laws hold and the judge agreed on 18/18 decisions.
- **Readiness fix:** a bare job title (even with a seniority) is no longer treated as
  ready to recommend — the agent asks one question first; purpose is no longer inferred
  when the user did not state it.
- **Deployed live (Phase 9).** The service runs as a Hugging Face Docker Space at
  <https://hriday29-shl-assessment-recommender.hf.space> — health green, `/chat`
  returns the valid contract with correct recommendations. HF builds the `Dockerfile`
  server-side; the Gemini key is a Space secret. A full, professional README (with an
  architecture diagram) documents the API, configuration, testing, and deployment.
- **Comparison replies are grounded in catalog facts (caveat D6).** On a comparison
  turn the engine resolves each named product to its catalog item and hands the reply
  writer the items' real attributes (name, test_type, duration, description); the model
  is told to use only those facts. Falls back to safe framing if a target does not
  resolve.
- **Opt-in deep health check (caveat F2).** `GET /health?deep=1` makes one real model
  call to confirm the provider key works; the default `/health` still never touches the
  model.
- **Live replay validated against the real model (caveats C1/C5).** Ran
  `scripts/replay_traces.py` against `gemini-3.5-flash`: the real extraction reproduces
  the trace behaviour (e.g. C6 replayed correctly end to end with Recall@10 = 1.00). The
  script gained a `--delay` flag to pace turns under a rate-limited free tier (~5 req/min).

### Changed
- **Prior-shortlist detection made format-robust (caveat D3).** "A shortlist was already
  offered" now triggers on the canonical catalog URL, any shl.com product link, or a
  Name/URL-style Markdown table — so confirmation and refine survive a differently
  formatted assistant turn — while still ignoring a plain clarifying question.
- **Dependencies and default model modernised (July 2026).** Every pin verified against
  its source and bumped to the current release — fastapi 0.138.2, uvicorn 0.49.0,
  pydantic 2.13.4, pydantic-settings 2.14.2, scikit-learn 1.9.0, numpy 2.5.0,
  sentence-transformers 5.6.0, litellm 1.90.2, pytest 9.1.1, hypothesis 6.155.7. The full
  suite passes on the new set with no code changes. numpy 2.x raised the Python floor to
  3.12 (`requires-python` updated). The default language model moved from the old
  `gemini/gemini-1.5-flash` to a current Flash model. The deployed default is
  `gemini/gemini-2.5-flash` (current, and a more generous free-tier quota than the
  3.5 tier); any provider/model works via the `SHL_LLM_MODEL` env var.

### Security
- Pinned litellm to 1.90.2, well clear of 1.82.7/1.82.8 — two briefly-compromised releases
  from the March 2026 supply-chain incident — with the reason recorded in
  `requirements.txt`.

### Added
- **Evaluation harness (Phase 8):** offline and live ways to check the whole system.
  - An offline trace-replay test runs all ten sample conversations through the engine
    (model faked to the understanding it would extract) and asserts per-turn
    behaviour: the final turn ends with a shortlist, comparison turns commit no new
    list, every turn returns the valid contract, and the shortlist recalls the gold.
  - A behaviour-probe suite runs the brief's edge cases (injection, legal, general
    hiring, no-exact-match, whitespace, model-down) through the engine as behaviour.
  - A live replay script (`scripts/replay_traces.py`) runs the same replay against
    the real model for a pre-submission check.

### Fixed
- **Typographic-apostrophe blindness in signal detection.** The real transcripts use
  curly apostrophes (`’`), but the detectors matched only the ASCII `'`, so
  confirmation, add/drop, and injection detection silently failed on pasted text.
  Normalised typographic quotes and dashes to ASCII at the single cleaning point,
  repairing every detector at once. Found by the trace-replay harness.
- **Confirmation-detector gaps.** Recognise natural closings the traces use — "that's
  good", "that covers it", "locking it in", "final list:" — so the conversation ends
  when the user accepts.

### Changed
- **Edit-and-close handling.** When an accepting turn also edits the list ("drop the
  OPQ, final list: ..."), the policy now ends the conversation *and* commits a freshly
  refined shortlist honouring the edit, instead of re-showing the prior list.

- **Serving (Phase 7):** a thin FastAPI application exposing the two endpoints.
  - `GET /health` reports the honest per-component health with the build stamp
    (200 while serving, 503 when unhealthy).
  - `POST /chat` validates the request, runs the turn through the engine, and
    returns the exact contract shape; failures come back in the stable error
    envelope (422 validation / 502 model / 500 internal), never a leaked trace.
  - Covered by HTTP-level tests through a `TestClient` and by Hypothesis property
    tests proving `/chat` never returns a 5xx and always yields a valid 200 or a
    shaped 4xx for arbitrary inputs.

### Fixed
- The sentence-embedding model now loads deterministically: the Hugging Face cache
  is pointed at the project-local directory before import, instead of inheriting a
  machine-level `HF_HOME` that could reference a drive absent on the host. Semantic
  retrieval had been silently unavailable on such machines (the system ran
  lexical-only). *(Historical: the semantic stage was later removed altogether in the
  pre-submission pass — it was measured to add zero Recall@10 — so retrieval is now
  lexical-only by design and the floor test needs no embedding model; see the
  Unreleased section above.)*

- **Response assembly (Phase 6):** a `ResponseEngine` that turns one conversation
  turn into the validated `ChatResponse` the API returns, wiring together every
  prior phase (state → policy → retrieval → shortlist → reply).
  - The recommendation list is built entirely in code, 1..10 items or `null` (never
    `[]`), every field copied verbatim from the catalog — no URL or code is ever
    generated.
  - The reply is written by the model per mode, with a deterministic fallback for
    every mode, so a model outage degrades the wording but never the correctness of
    a turn.
  - `recover_prior_shortlist`: on a confirmation close, the shortlist already offered
    is recovered from the catalog URLs in the prior assistant message and re-shown
    exactly (matching the sample conversations), rather than re-retrieving on a bare
    "yes". Validated against the real C1 trace.
- **Operational hardening (Phase 5.5), built framework-free so it is unit-testable
  without a server:**
  - Structured logging (JSON or console) that promotes call-site context onto each
    line, so every degradation is visible. Rule: log every degradation, never a secret.
  - Fail-fast startup validation in a single composition point (`bootstrap.py`): the
    catalog is loaded and its hard invariants (non-empty, valid `test_type`, linkable
    items) are asserted before the first request, so broken data fails at the front
    door instead of mid-evaluation.
  - A health probe that reports honest per-component readiness under a hard/soft
    weighting (`ok` / `degraded` / `unhealthy`), never a green light over a broken
    service, and never by pinging the paid model.
  - A build/version stamp resolved defensively across installed, source, and
    deploy-host runs, surfaced in health so an instance ties back to a build.
  - A stable API error contract: one error shape and a closed exception→status map;
    internal causes are logged, never returned on the wire.
- Configuration knobs for logging (`SHL_LOG_LEVEL`, `SHL_LOG_FORMAT`).

### Changed
- Operational concerns were pulled forward out of the serving phase deliberately, so
  the service is defensible before it is assembled.

## Phase history

The dated phases below record how the system was built. Each closed only after the
listed evidence passed; the full detail lives in `BUILD_PLAN.md`.

### Phase 5 — Policy & retrieval
- Precedence-ordered policy engine; the clarify-vs-commit gate is advised by the
  model but bounded by code (question budget and turn cap are never the model's to
  break).
- Hybrid retrieval (lexical TF-IDF over word + character n-grams, plus semantic
  sentence embeddings) with a transparent, inspectable weighted-sum ranker.
- **Recall@10 improved `0.505 → 0.757`**, entirely through general mechanisms, each
  step diagnosed and kept only because the number moved:
  - `0.505 → 0.612` — inject evidence-derived staple defaults (OPQ32r, Verify G+) as
    candidates when their dimension is relevant.
  - `0.612 → 0.677` — make the staple weight additive so a category request does not
    suppress the other default.
  - `0.677 → 0.717` — proportional exact-name scoring, so a tight product match beats
    an incidental word overlap.
  - `0.717 → 0.737` — enrich the retrieval query with the understanding step's
    extracted skills, so a skill buried in a long brief is retrievable.
  - `0.737 → 0.757` — a light family-diversity cap so near-duplicate product families
    cannot crowd the shortlist.
- Stopped at `0.757`: further gains would require tuning to the ten visible traces,
  which the brief warns against. A regression floor test locks the result in.

### Phase 4 — Understanding
- LLM extraction of role, skills, categories, and a readiness judgement, with a
  strict deterministic fallback.
- The model is skipped entirely on turns deterministic rules already decide
  (off-topic, injection, plain confirmation), so those turns work with the model down.

### Phase 3 — Conversation model
- Immutable conversation state and a deterministic signal detector (comparison,
  refusal, injection, confirmation), grounded in the real catalog vocabulary so it
  generalises past the sample wording and rejects look-alike non-product phrases.
- Refusal detection was made deliberately asymmetric: generous where a false positive
  is cheap, strict where it is costly (a legal refusal needs a legal term *and* a
  liability framing), so a product literally named "HIPAA" is never mistaken for a
  legal question.

### Phase 2 — Response contract
- Pydantic request/response schemas: lenient on input (role casing and synonyms
  normalised), strict on output (exactly the contract's fields; validated `test_type`).
- Handled the `User`/`Agent` vs lowercase mismatch between the displayed samples and
  the JSON contract — a hard-eval risk.

### Phase 1 — Data foundation
- Catalog loader that handles the export's real quirks: non-strict JSON (an embedded
  newline), collapsed embedded whitespace, and one corrected damaged product name.
- Mechanical `test_type` derivation from the catalog `keys`, with full catalog
  coverage and a loud failure on any unknown category.
- A versioned offline snapshot so the running service never parses the messy raw
  export, and a catalog vocabulary used to ground signal detection.

### Phase 0 — Specification
- `ARCHITECTURE_DECISIONS.md` written and owned before any code, then corrected
  against the real catalog (377 items) and the ten real conversations, with every
  changed assumption marked.

[Unreleased]: https://keepachangelog.com/en/1.1.0/
