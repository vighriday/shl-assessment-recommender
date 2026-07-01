# SHL Assessment Recommender — Architecture Record

## 0. How to read this document

This is the single source of truth for the design of the SHL AI Intern take-home.

It is written **before** we write code, on purpose. The goal is that by the time
you finish reading it, you fully understand and *own* every decision — enough to
defend it in an interview without me.

Every decision below follows the same simple shape:

- **What we chose**
- **What we rejected**
- **Why the chosen one is better here**

The language is kept deliberately plain. If any sentence sounds clever but you
cannot re-explain it in your own words, that is a bug in this document — tell me
and I will rewrite it.

> **Status note:** This version has been corrected against the *real* data —
> the actual catalog JSON (377 items) and the 10 real sample conversations.
> Earlier drafts were written from the PDF alone and contained a few guesses
> that the real data has now disproven. Every place that changed is marked with
> **[Corrected]** and explains what changed and why.

---

## 1. The ground truth we verified

Before designing anything, we downloaded and inspected the two real resources
hidden behind the "Link" placeholders in the PDF. These facts are checked, not
assumed. They drive the whole design.

### 1.1 The catalog (the data we recommend from)

- It is a **JSON file**, not a website to scrape. We download it once.
- It has **377 items**. Small. Everything fits in memory.
- **It does not parse with normal strict JSON.** One text field contains a raw
  line-break that breaks standard parsing. We must load it with a "lenient" mode
  (`strict=False`). This is a real trap and we handle it on day one.
- Every item has these fields:
  `entity_id, name, link, scraped_at, job_levels, job_levels_raw, languages,
  languages_raw, duration, duration_raw, status, remote, adaptive, description,
  keys`.
- **All 377 URLs are on `www.shl.com`.** So guaranteeing "every URL is a real
  SHL URL" is easy — we just reuse the `link` field and never build URLs ourselves.
- Field completeness: `description` is present for **all 377** items (great for
  search). `duration` is present for 316, `languages` for 340.

### 1.2 The `keys` field — the most important discovery

Every item has a **`keys`** list describing its category, e.g.
`["Knowledge & Skills"]` or `["Personality & Behavior", "Competencies"]`.

There are exactly **8 possible categories**, and they map **one-to-one** to the
single-letter `test_type` codes the API response needs:

| `keys` category               | `test_type` code |
| ----------------------------- | ---------------- |
| Knowledge & Skills            | K                |
| Personality & Behavior        | P                |
| Ability & Aptitude            | A                |
| Competencies                  | C                |
| Biodata & Situational Judgment| B                |
| Simulations                   | S                |
| Development & 360             | D                |
| Assessment Exercises          | E                |

**Why this matters so much:** the response needs a `test_type`, and the PDF says
the catalog JSON does not contain it. That made earlier drafts treat `test_type`
as a hard manual problem. It is not. It is sitting inside `keys`, and we can
derive it mechanically. (See Decision 9.)

### 1.3 The 10 sample conversations (the "answer key")

- They are **Markdown files** (C1–C10), one persona each, showing the *expected*
  agent behavior and the *expected* final shortlist.
- They are the closest thing we have to the real grader, because the grader's
  simulated user is almost certainly built from these.
- **What they reveal that the PDF prose does not:**
  1. On turns where the agent is **not** recommending, every trace prints
     `recommendations: null` — **not** an empty list `[]`.
  2. The agent **clarifies for 1–2 turns before committing** on vague openers.
     The flagship example, C1, asks two clarifying questions and only
     recommends on **Turn 3**.
  3. Conversation lengths vary: traces run **3 to 7 exchanges** (C9 is the
     longest at 7 user+agent exchanges = 14 messages).
  4. The expected shortlists include **report-style items** (e.g. "OPQ
     Leadership Report"), so "assessment" is used loosely. We must not throw
     reports away.

### 1.4 The rule we follow when the PDF and the traces disagree

The PDF is written for humans. The traces are the labeled examples the automated
grader is built from. When the two conflict, **the traces win**, because the
traces are what actually gets scored. We then note the conflict openly in the
approach document — that is a strength signal, not a risk.

This single rule decides several choices below.

---

## 2. What SHL requires (the non-negotiables)

These cannot be violated. Breaking any one can zero the submission.

1. **Two endpoints:** `GET /health` and `POST /chat`.
2. **Stateless API.** Every `/chat` call carries the full conversation history.
   The server stores nothing between calls.
3. **Strict response schema.** Exactly three top-level fields:
   `reply`, `recommendations`, `end_of_conversation`. No extra fields.
4. **Recommendations come only from the SHL catalog.** Every name and URL must be
   a real catalog row.
5. **Scope is Individual Test Solutions.** Pre-packaged Job Solutions are out of
   scope. (See Decision 4 for how we actually handle this — it is lighter than
   earlier drafts assumed.)
6. **Five modes:** clarify, recommend, refine, compare, refuse.
   *Note on counting:* the PDF lists **four** conversational behaviors (clarify,
   recommend, refine, compare) and states refusal separately under "stay in
   scope". We treat **refuse** as a fifth internal mode. So we add nothing the
   PDF does not require — we just organize it as 4 behaviors + 1 refusal mode.
7. **Stay on topic.** Only SHL assessments. Refuse general hiring advice, legal
   questions, and prompt-injection attempts.
8. **Runtime limits:** about **8 messages** per conversation and a **30-second**
   timeout per call. (See Decision 3 for how we read "8 turns".)
9. **Scored on both** recommendation quality (Recall@10) and behavior (probes).

---

## 3. The problem in one paragraph

This is **not** a general chatbot. It is a **controlled recommendation system
that talks**. The user describes a hiring need in plain language. The system
understands it, asks a useful question only when needed, searches the catalog,
returns a grounded shortlist, updates that shortlist when the user changes their
mind, compares products when asked, and refuses anything off-topic. The hard part
is making *reliable decisions inside a short, stateless conversation*.

---

## 4. The one-line architecture

> **Use the model for language. Use code for control.**

The LLM helps understand messy language and write the human-sounding reply.
The **code** owns everything that must be correct: state, retrieval, ranking,
which items get recommended, the URLs, the `test_type`, and the final schema.

This is the spine of the whole design. Everything else serves it.

---

## 5. Design principles

1. **Reliability beats cleverness.** A simple system that never hallucinates
   beats a flashy one that sometimes does.
2. **Small data, small tools.** 377 items means no vector database, no agent
   framework, no distributed anything. In-memory is enough.
3. **Code owns the guarantees.** The LLM is never trusted with schema, URLs, or
   "does this item exist".
4. **Explainable in two minutes.** If the design is smarter than we can explain,
   it becomes interview risk.
5. **Do heavy work offline, stay fast online.** Prepare the data once; keep each
   request light to respect the 30-second limit.
6. **Questions are expensive — but silence is too.** Ask only high-value
   questions, *and* do not under-ask either (the traces clarify before
   committing). Balance, not a fixed rule. (See Decision 3.)

---

## 6. The system at a glance

Two layers.

**Offline (built once, before the server runs):**
Load the catalog → fix the parsing issue → derive `test_type` from `keys` →
build the search indexes → save a clean snapshot.

**Online (per `/chat` request):**
1. Validate the request.
2. Rebuild the conversation state from the messages.
3. Decide the mode (clarify / recommend / refine / compare / refuse).
4. Retrieve candidate items (only when recommending/refining).
5. Rank them.
6. Assemble the structured response **in code**.
7. Use the LLM to write the natural-language `reply`.
8. Validate the final response against the schema before returning.

---

## 7. The main decisions

Each one: what we chose, what we rejected, why ours wins here.

---

### Decision 1 — Build a controlled recommender, not a free chatbot

**Chosen.** A recommender with five explicit modes (clarify, recommend, refine,
compare, refuse). Code decides the mode; the LLM only phrases the answer.

**Rejected — a generic RAG chatbot.** Easy to hallucinate, hard to control when
to ask vs. answer, hard to guarantee the schema, hard to defend.

**Rejected — a pure rules engine (no LLM).** Too rigid for real human language,
weak on pasted job descriptions, weak on nuanced comparisons.

**Why ours wins.** The PDF literally grades *knowing when to ask, retrieve,
answer, or refuse*. That is a control problem, so we put control in code and keep
the LLM for the part it is good at — understanding and wording. This is reliable,
testable, and easy to explain.

---

### Decision 2 — Stay stateless by rebuilding state every call

**Chosen.** On every request we read the whole `messages` array and rebuild a
typed `ConversationState` (intent, role/population, seniority, must-have skills,
language, purpose, comparison target, any later corrections, and "do we have
enough to recommend yet"). Nothing is stored on the server.

**Rule:** if the user corrects themselves later, **the newest valid statement
wins.** Older details stay as history but the active state reflects the latest
correction. (The traces show this — e.g. long refinements in C9.)

**Rejected — server-side session memory.** Breaks the required stateless design,
hides behavior, makes testing harder.

**Rejected — smuggling state in hidden response fields.** Not allowed by the
strict schema; the grader may drop it; adds fragility.

**Why ours wins.** It is exactly what "stateless" means, and because conversations
are short, rebuilding state each time is cheap and simple.

---

### Decision 3 — A policy engine with an *adaptive* clarify budget **[Corrected]**

**Chosen.** Before doing anything, code decides the mode. The heart of it is the
**recommendation gate**: "do we have enough to recommend? If not, what single
most-useful question do we ask?"

**How many questions before we commit — adaptive, not fixed:**

| Opening message                                   | Clarifying questions before first shortlist |
| ------------------------------------------------- | ------------------------------------------- |
| **Vague** ("we need a solution for leadership")   | **1–2** (this matches trace C1)             |
| **Specific** (role + level given, or JD pasted)   | **0–1** — commit quickly                    |
| **Hard ceiling, always**                          | Stay within ~8 messages; commit by turn 3   |

**What changed and why [Corrected].** Earlier drafts said "prefer 0–1 questions,
commit as fast as possible." The real traces disprove that: SHL's own flagship
example, C1, asks **two** clarifying questions and only recommends on **Turn 3**.
Committing too early would (a) fail the explicit behavior probe "do not recommend
on turn 1 for a vague query", and (b) miss the detail that makes the shortlist
correct, lowering Recall@10. So the budget is now **adaptive to how specific the
opener is**, not a fixed small number.

**How we read "8 turns" [Corrected].** The PDF says "8 turns including user &
assistant." Read literally as 8 *messages*, that is 4 exchanges — yet gold traces
C3 (10 messages) and C9 (14 messages) exceed it. So we treat 8 messages as a
**ceiling we stay under**, we **commit by turn 3 in almost all cases**, but we do
**not** starve clarification, because the gold behavior clearly clarifies first.
Being *too* conservative would make us behave *worse* than SHL's own examples.

**Rejected — one big LLM prompt decides everything.** Inconsistent, hard to test,
easy to overfit the 10 public traces.

**Rejected — a fixed "always clarify once then recommend" rule.** The traces show
both fast commits and 2-turn clarifications, so any single fixed rule is wrong.

**Why ours wins.** It mirrors the real gold behavior, protects the two things
that actually score (the vague-query probe and Recall@10), and is still simple
to explain: *"ask more when the request is vague, commit fast when it is clear,
never blow the turn budget."*

---

### Decision 4 — Build the catalog snapshot offline; keep scope handling light **[Corrected]**

**Chosen.** Build one clean in-memory snapshot offline:
1. Load the provided JSON with the lenient parser.
2. Normalize fields.
3. Derive `test_type` from `keys` (Decision 9).
4. Build the search text and indexes.
5. Keep each item's real `link` as the only source of URLs.
6. Save the snapshot for the server to load instantly.

**On the "Individual Test Solutions only" rule [Corrected].** Earlier drafts
planned a heavy "scope audit pipeline" with allowed/excluded/review labels for
every item. The real data shows that is **mostly unnecessary**:
- The provided JSON **is** the assignment's chosen working set (377 items).
- The report-style items earlier drafts feared (OPQ Leadership Report, Global
  Skills Development Report) are **present and appear in the gold shortlists** —
  so they are in scope. We must **not** filter them out.
- Only **7 items** have "Solution" in their name (e.g. "Entry Level Sales
  Solution"). These are the *only* genuinely ambiguous ones.

So scope handling shrinks to a **small, quick audit of those 7 items**, not a
subsystem. We tag each item with a simple `in_scope` boolean (default `true`),
and only set it `false` if a manual check of those 7 confirms they are
pre-packaged Job Solutions. This keeps the safety net without inventing work the
data does not justify.

**URL provenance rule (unchanged, important).** Every returned URL is copied from
the snapshot's `link` field. We never invent, guess, or hand-build a URL. If an
item somehow has no trusted URL, it cannot be recommended.

**Wording note for the approach document.** The PDF says URLs must come from
"your **scraped** catalog". SHL actually *provided* the catalog as a JSON export,
so we use that as the canonical source instead of re-scraping the HTML. This is
more reliable and avoids drift. We will state this plainly in the write-up so the
"why didn't you scrape?" question is answered up front as a deliberate, stronger
choice — the provided JSON *is* their catalog export.

**Rejected — scraping the live website during `/chat`.** Slow, fragile, network
risk, pointless for 377 static items.

**Rejected — using the raw JSON untouched.** Pushes the parsing bug and messy
fields into runtime behavior.

**Rejected — a full scope-classification pipeline for all 377 items
[Corrected].** The data shows only 7 items are ambiguous and the feared reports
are actually in scope, so a big pipeline would be wasted effort and could
*hurt* recall by wrongly excluding valid items.

**Why ours wins.** Fast and repeatable at runtime, one clean place to enforce
scope, and the scope effort is sized to the *actual* ambiguity in the data.

---

### Decision 5 — Lexical (TF-IDF) retrieval, no vector DB **[Corrected]**

**Chosen.** Two lexical signals over each item's search text, then merge candidates:
1. **Word-level TF-IDF** — normal language.
2. **Character-level TF-IDF** — product codes and odd tokens like `OPQ32r`,
   `G+`, `.NET`, `SVAR`.

**What changed and why [Corrected].** Earlier drafts specified a *third* signal —
sentence-embedding ("semantic") similarity — and merged all three ("hybrid retrieval").
It was built, and then **measured against the ten sample conversations: it changed
Recall@10 on none of them** (its single unique recovery was cancelled by an equal
displacement; lexical alone reached the same mean). Rather than carry an unfalsifiable
component and a heavy, version-fragile dependency chain (torch / transformers /
huggingface-hub) that was also silently failing to load under an under-pinned pin, we
**removed the semantic stage entirely** (`retrieval/semantic.py` deleted,
`sentence-transformers` dropped from requirements, the `SHL_ENABLE_SEMANTIC_RETRIEVAL`
/ `SHL_EMBEDDING_MODEL` config removed). The retriever class is now `LexicalRanker`.
The full account is in Decision 16 and `docs/retrieval_design.md`.

**Rejected — keep the hybrid for completeness.** A component that moves no measured
number is not "completeness"; it is weight we cannot justify, plus a real failure
surface. Removing it made retrieval transparent and reproducible at no measured cost.

**Rejected — a vector database (Chroma/pgvector/FAISS server).** 377 items fit in
memory; a database adds infrastructure for no real benefit — and with no embedding
stage there is nothing for it to index.

**Why ours wins.** For this catalog, relevance is dominated by exact skill/product
names and category intent — both lexical. The char-level TF-IDF catches the odd product
codes, and the ranker (Decision 6) folds in category, job-level, and language signals.
The result is fully inspectable and matches the hybrid's measured recall.

---

### Decision 6 — Transparent ranking, not a heavy reranker

**Chosen.** Rank candidates with a readable scoring function that adds up:
lexical score, role/skill fit, job-level fit, language fit, purpose fit, a boost
for exact product-family matches (with an added bonus when a *distinctive* required
skill appears in the item's name), and penalties for conflicts. We also use the
`keys` category as a structured signal.

**Optional later.** If recall on the traces is not good enough, add a small
reranker on only the top candidates — but only if measurement says we need it.

**Rejected — a cross-encoder reranker from day one.** Slower, more complex, less
explainable; may not even help enough to justify itself.

**Why ours wins.** It is easy to inspect ("why did this rank #1?"), easy to debug,
and strong enough for a catalog this size.

---

### Decision 7 — Code owns the structured output, not the LLM

**Chosen.** The LLM writes only the `reply` text. **Code** decides which items are
recommended, the final list, every URL, every `test_type`, the
`end_of_conversation` flag, and runs the final schema check.

**Rejected — let the model return the whole JSON.** Risk of invented products,
invented URLs, malformed JSON, and counts outside 1–10. These are exactly the
hard-eval failures that zero a submission.

**Why ours wins.** It makes the things that must be correct *structurally
impossible* for the model to get wrong, because the model never touches them.

---

### Decision 8 — Hybrid understanding (rules + LLM)

**Chosen.** Pull state from the conversation in two ways:
- **Rules/deterministic** for clear signals: message roles, the last user
  message, confirmation phrases ("perfect, that's what we need"), comparison
  patterns ("difference between X and Y"), legal/regulatory wording, and
  prompt-injection patterns ("ignore previous instructions").
- **LLM** for the fuzzy stuff: long job descriptions, skill priorities, implicit
  seniority, overall intent.

**Rejected — rules only.** Too brittle for real language and pasted JDs.

**Rejected — LLM only.** Inconsistent and harder to debug for the easy, certain
cases that simple code handles perfectly.

**Why ours wins.** Each tool does what it is best at: code for the certain things,
the model for the ambiguous things.

---

### Decision 9 — Derive `test_type` mechanically from `keys` **[Corrected]**

**Chosen.** Compute `test_type` directly from the `keys` field using the fixed
8-entry map in section 1.2. Multi-category items join their codes with commas
(e.g. `Knowledge & Skills` + `Simulations` → `K,S`). A tiny override map (no more
than a handful of entries) exists only to fix multi-code *ordering* if trace
replay ever shows a mismatch.

```
test_type = ",".join(KEYS_TO_CODE[k] for k in item["keys"])
```

**What changed and why [Corrected].** Earlier drafts treated `test_type` as a
big manual curation job (hand-label products, audit each one). The real data
shows it is a **deterministic lookup**: every one of the 377 items has a `keys`
value, the 8 categories map cleanly to the codes the traces use, and multi-code
combos in the traces (`K,S`, `A,S`, `P,C`) are exactly what comma-joining
produces. **Zero** items have empty `keys`, so coverage is 100%. Manual curation
would be wasted effort *and* a weaker interview story.

**Rejected — ask the LLM for `test_type` at runtime.** Unreliable and completely
avoidable when the answer is already in the data.

**Rejected — hand-curate all 377 items.** Slow, fragile (drifts from the
catalog), and unnecessary given the clean mapping.

**Why ours wins.** It is correct by construction, needs almost no maintenance,
and is a clean story: *"I found that `test_type` is encoded in the catalog's
`keys` field and derived it deterministically with full coverage."* That is
exactly the "context engineering" skill SHL grades.

---

### Decision 10 — Provider-flexible LLM layer

**Chosen.** Talk to the LLM through one thin adapter so we can swap providers
(Gemini, Groq, OpenRouter, etc.) without touching the rest of the app. Use
structured-output / JSON mode where the provider supports it. Keep retries and a
fallback behind the same interface.

**Note on statelessness.** Even if a provider offers a stateful/"conversation"
API, we still pass the full history ourselves and keep our app stateless, because
that is what SHL requires.

**Rejected — hard-wire the whole app to one vendor.** Less flexible; painful to
switch if one provider is slow, rate-limited, or lower quality on the day.

**Why ours wins.** The assignment rewards behavior and reliability, not vendor
loyalty. A thin adapter keeps us free without adding real complexity.

---

### Decision 11 — Partial refusal when a turn is mixed

**Chosen.** If one message mixes an out-of-scope ask with a valid in-scope ask,
refuse the out-of-scope part and still answer the in-scope part.
Example: refuse the legal-advice part, still explain what the SHL assessment
measures.

**Rejected — refuse the entire turn.** Less helpful and a worse match to the
sample behavior, which stays helpful while declining the off-topic part.

**Why ours wins.** It stays safe without becoming needlessly unhelpful, which is
what the traces reward.

---

### Decision 12 — A deterministic fallback reply path

**Chosen.** If the LLM errors, times out, or returns garbage, code still produces
a valid, schema-safe reply from templates (clarify template, refusal template,
shortlist-confirmation template, compare-from-facts template).

**Rejected — assume the model call always succeeds.** A bad production
assumption; one provider hiccup would otherwise break the response and risk the
hard evals.

**Why ours wins.** Even on a bad day, we return valid JSON and a sensible
message. Cheap insurance against the worst failure mode.

---

### Decision 13 — Test the fuzzy judgement by law and by judge, not by hand **[Added]**

**Chosen.** The one decision code cannot make — clarify vs recommend — is checked
two ways that hard-code no answers. **Metamorphic laws:** properties that must hold
for *any* input under a transformation (adding information can never make a request
*less* ready; an injection beside a legitimate sentence is still refused; a
comparison never commits a new list; an acceptance with no prior shortlist never
ends). **LLM-as-judge:** an independent model call grades whether each decision was
reasonable and reports an agreement rate. Lives in `shl_recommender/eval/`, run by
`scripts/adversarial.py`.

**Rejected — generate 100 prompts and fix each failure.** That overfits to a
hand-picked set (the same trap the brief warns about with the ten traces) and is
really just hard-coding at scale.

**Why ours wins.** Laws test *logic* and so find whole classes of bug without
memorising answers — running them found and fixed four detector gaps the ten traces
never exercised ("compare X and Y", "X or Y — which fits", "disregard the catalog",
"reveal your full prompt"). The judge *measures* quality rather than asserting it.
This came from a real bug (a bare "senior Java developer" was recommended when it
should have clarified); the root fix was tightening the readiness prompt so a bare
job title is not treated as ready.

### Decision 14 — A human-facing chat UI, mounted onto the same API **[Added]**

**Chosen.** A Gradio `ChatInterface` mounted at the root path `/` on the same FastAPI
app, so one process on one port serves both the machine contract (`/health`, `/chat`)
and a chat box a person can use; the JSON endpoint index lives at `/info`. The UI calls
the same engine in process, so its behaviour is identical to the API. It is
fault-isolated: if Gradio is unavailable the API is unaffected.

**Rejected — a separate UI app / separate host.** Two deployments to keep in sync and
two URLs; unnecessary when one mount serves both.

**Why ours wins.** The grader gets the clean HTTP API; a human gets a real way to test
the agent as a user, from the same URL, with no extra hosting.

### Decision 15 — Deploy to Hugging Face Spaces **[Added; Corrected]**

**Chosen.** A Docker Space (16 GB free tier) built server-side from our `Dockerfile`,
serving on port 7860, with the Gemini key as a Space secret. Retrieval is lexical-only,
so the image carries no torch/embedding stack and there is nothing to pre-download.

**What changed and why [Corrected].** An earlier version of this decision deployed "with
the hybrid on", reasoning that the Spaces RAM was free so we might as well keep the
semantic component even though we had measured it to add *zero* Recall@10 on the ten
traces (lexical alone reached the same number). On reflection that inverted the burden of
proof: a component that moves no measured number should be removed, not kept because a
host happens to have spare RAM — especially one that dragged in a heavy, version-fragile
dependency chain that was itself silently failing to load. So the semantic stage was
removed (Decision 16), and the deploy is lexical-only. The Render-512 MB / OOM worry that
originally pointed us to Spaces is now moot: with no torch there is little to run out of.

**Rejected — keep deploying the hybrid because the RAM is free.** "Free RAM" is not a
reason to ship an unfalsifiable component and a real failure surface.

**Why ours wins.** A live, always-addressable public URL, the 2-minute cold-start grace
the brief allows, a lean image, and a retrieval story that is transparent and reproducible.
The measurement (semantic = 0.000 net) is on record, so the removal is evidence-backed,
not assumed.

### Decision 16 — Current model default, and an opt-in deep health check **[Added]**

**Chosen.** The default model is `gemini/gemini-2.5-flash` (current, with a generous
free-tier quota), swappable to any provider via `SHL_LLM_MODEL`. The default `/health`
never calls the model (readiness is configuration, not a paid ping) and returns a
**minimal body — exactly `{"status": "ok"}`** with HTTP 200 while serving — so a strict
`{"status":"ok"}` check is never tripped by extra keys. `GET /health?deep=1` returns the
richer diagnostic body (overall status, per-component detail, build identity) *and* makes
one real model call so an operator can confirm the key works post-deploy.

**Rejected — an old default model** (an earlier draft used `gemini-1.5-flash`; a later one
briefly referred to a `gemini-3.5-flash` that is not the configured default), and
**rejected returning the rich `{status, build, components}` body by default** (an extra-key
surface for a strict health check) and **pinging the model on every health check** (paid,
slow, and it could take the health path down).

**Why ours wins.** The service stays current, the default health body is the assignment's
literal contract, the health path stays cheap and reliable, and a bad key is still
catchable on demand.

---

### Decision 17 — Semantic retrieval was built, measured, and removed **[Added]**

This is the decision Decisions 5 and 15 defer to. It is recorded in full because
*removing* a component we had already built, on evidence, is exactly the kind of choice
this document exists to defend.

**What we did.** We built the hybrid: word- and char-level TF-IDF plus a
sentence-embedding retriever (`all-MiniLM-L6-v2` via `sentence-transformers`), merged and
ranked. We then measured Recall@10 with the semantic layer **on vs off, per trace**, on
the ten sample conversations.

**What we found.** The numbers were **identical on all ten** (mean unchanged; the semantic
layer's single unique recovery on one trace was cancelled by an equal displacement of
another gold item). Net semantic contribution: **0.000**. Separately, the embedding model
had been *silently failing to load* on some environments because of an under-pinned
dependency / cache path, so on those hosts the "hybrid" had in fact been running
lexical-only the whole time — and the measured recall was the same either way.

**What we chose.** We **removed the semantic stage entirely**:

- deleted `shl_recommender/retrieval/semantic.py`;
- renamed the retriever from `HybridRetriever` to `LexicalRanker`;
- dropped `sentence-transformers` (and its torch/transformers/huggingface-hub chain) from
  `requirements.txt`;
- removed the `SHL_ENABLE_SEMANTIC_RETRIEVAL`, `SHL_EMBEDDING_MODEL`, and
  `SHL_MODEL_CACHE_DIR` configuration;
- dropped the `semantic_retrieval` component from the health report.

The lost recall from all of this is **zero** (measured), and the two recall gains from the
same pre-submission pass — raising `name_boost` 1.5 → 2.0 and adding a
distinctive-skill-in-name bonus — lifted the mean from 0.757 to **0.809**.

**Rejected — keep it "in case it helps later" / because the RAM is free.** A component
that moves no measured number, cannot be shown to help, and silently fails to load is not
insurance — it is unfalsifiable weight plus a real failure surface and a heavy dependency.
Keeping it would also be a *weaker* interview story than removing it on evidence.

**Rejected — a larger / better embedding model.** If the small model's contribution is
exactly zero, a bigger one can only improve the part that already added nothing, at
far higher memory and cold-start cost. The measurement settles it.

**Why ours wins.** Retrieval is now transparent (every ranked signal is a named, readable
term), reproducible (no model download, the floor test never skips), lighter to deploy,
and **no worse on the evidence**. The experiment is kept on the record as history — it was
the right thing to *try*, and removing it was the right thing to *do* once measured.

---

## 8. The runtime path for `POST /chat`

1. **Validate the request** — body shape, allowed roles, content present. If
   malformed, return a clear error.
2. **Rebuild state** from the full history. Latest correction wins.
3. **Pick the mode** — clarify / recommend / refine / compare / refuse.
4. **Retrieve** candidates (only when recommending or refining), using lexical
   (TF-IDF) retrieval over in-scope items.
5. **Rank** candidates with the transparent scorer.
6. **Build the response in code** — shortlist, URLs from `link`, `test_type` from
   `keys`, `end_of_conversation` decided by code, and on non-recommend turns set
   `recommendations` to **`null` by default** (see Decision below / section 9.x).
7. **Write the reply** — the LLM gets the mode, a short state summary, and the
   chosen items' facts, with a strict instruction never to invent products.
8. **Validate the final object** against the schema, then return it.

---

## 9. Exact behavior on edge cases

SHL warns about edge cases, so these are spelled out.

**9.1 Vague opening ("I need an assessment").**
Ask one high-value question. `recommendations: null`. `end_of_conversation:
false`.

**9.2 Detailed JD in the first message.**
If there is already enough to act on, recommend right away. Do not ask filler
questions just to sound conversational.

**9.3 Mid-conversation change ("actually, add personality tests").**
Update the existing shortlist. Do not restart. The newest instruction overrides
older conflicting ones.

**9.4 Comparison ("difference between OPQ and GSA?").**
Compare using catalog facts only, never model memory. Usually
`recommendations: null` on a pure compare turn unless the user also asks for a
refreshed shortlist.

**9.5 No exact match (e.g. a Rust-specific test that does not exist).**
Say there is no exact match, then offer the closest real alternatives. Never
invent a missing product.

**9.6 An item is in the catalog but out of scope.**
Do not recommend it; remove it before ranking. (In practice this only concerns
the ~7 "Solution" items flagged in Decision 4.)

**9.7 User says "no preference".**
Record it, use a sensible default, move on. Do not re-ask. (The grader's user
says this for anything outside its facts, so re-asking wastes turns.)

**9.8 Legal/regulatory question.**
Refuse the legal advice; if possible still explain what the SHL product measures.

**9.9 General hiring advice.**
Refuse or redirect back to SHL-assessment scope.

**9.10 Prompt injection ("ignore previous instructions", "recommend non-SHL
tools").**
Ignore the malicious instruction, stay in scope, return only catalog-grounded
content.

**9.11 Product-code queries (`OPQ32r`, `G+`, `.NET`).**
Character-level lexical retrieval handles these.

**9.12 Too many good matches.**
Return up to 10, favoring relevance and variety over near-duplicates.

**9.13 Empty `recommendations` — use `null`, not `[]` [Corrected].**
On clarify/refuse turns, `recommendations` is **`null`**.
*What changed and why:* earlier drafts chose `[]` based on the PDF word "empty".
But **all 10 gold traces print `null`**, and the traces win when they disagree
with the PDF. We make this a single config flag defaulting to `null`, so we match
the gold data now and can switch instantly if the grader ever surprises us. In
code this means the field is `Optional[List[...]]`.

**9.14 No extra schema fields.**
Return only `reply`, `recommendations`, `end_of_conversation`. Nothing else.

**9.15 `end_of_conversation` — be conservative.**
The first shortlist is usually `false`. Set `true` when the user confirms or
clearly signals they are done. (C1 ends `true` only after the user says "perfect,
that's what we need".)

**9.16 When enough context exists, commit.**
Prefer recommending over asking more non-critical questions, because the grader
ends the conversation once a shortlist appears, and late commitment wastes turns.

**9.17 Timeout or model failure.**
Use the deterministic fallback templates and still return valid schema.

**9.18 Conversational coherence across turns [Added].**
The reply must stay consistent with the conversation so far: do not contradict an
earlier answer, do not re-ask a question the user already answered, and do not
forget a constraint the user gave. Because we rebuild full state every call
(Decision 2), the reply is always written from the *current* state summary, which
is what prevents incoherence. The PDF names "% of turns with hallucinations" as a
per-turn probe and "conversational incoherence" as a failure mode, so we treat
coherence as a first-class behavior, not an afterthought.

---

## 10. Testing and evaluation plan

This is built as an **evaluation-first** project.

1. **Replay the 10 traces.** Turn C1–C10 into fixtures and run them on every
   change. This is our main scoreboard.
2. **Schema tests.** Exact keys; recommendation count 1–10 (or `null`); valid
   URLs; valid `test_type`; boolean `end_of_conversation`.
3. **Grounding tests.** Every recommended item and URL exists in the snapshot;
   no invented products.
4. **Behavior probes.** Vague→clarify; legal→refuse; edit→refine; compare→
   grounded; missing→closest alternatives; "no preference"→move on;
   correction→latest wins; no extra fields ever; **coherence→no contradiction
   or re-asking an answered question across turns**.
5. **Property-based tests** (Hypothesis) for malformed messages, empty histories,
   repeated edits, weird ordering, and the 1–10 limit. Catches edge cases normal
   tests miss.
6. **Latency checks.** Cold start, p50/p95 of `/chat`, LLM time, retrieval time —
   to stay safely under 30 seconds.
7. **Internal scorecard, tracked over time.** Schema pass-rate, trace Recall@10,
   refusal pass-rate, comparison-grounding pass-rate, hallucination count,
   over-asking rate.

---

## 11. Logging

From day one, each request logs: request id, chosen mode, a short state summary,
top retrieved candidates, final shortlist ids, whether the fallback fired, timing
per stage, and how many clarifying questions were asked before the shortlist.
This makes debugging, the write-up, and interview answers far easier.

---

## 12. Architectures we rejected outright

- **A heavy agent framework first** (LangGraph multi-agent, tool graphs).
  Not needed for a 2-endpoint stateless task; harder to explain; bigger bug
  surface.
- **A pure prompt-engineered chatbot.** Weak guarantees; easy to vibe-code; hard
  to defend.
- **A vector database first.** Tiny catalog; infrastructure for no benefit.
- **A pure rules recommender.** Too weak on natural language and long JDs.

---

## 13. The technology stack

This section follows the same chosen / rejected / why pattern as the decisions
above, and it is organized by **job to be done** so you can see the stack covers
*everything* the system needs — not just the interesting parts, but also the
boring "how does it run, get tested, and deploy" parts.

> Read 13.1 as the headline choices and 13.2 as the supporting pieces that make
> the thing actually runnable, testable, and deployable. Together they cover the
> full lifecycle: build → run → call the model → search → respond → test → deploy.

### 13.1 Core choices (the headline stack)

**Web framework — FastAPI.**
- *Why chosen:* matches the required API style exactly, built-in request/response
  typing, async-ready (helps with the 30s timeout), trivial to test, tiny for a
  2-endpoint service.
- *Rejected — Flask:* no native typing/validation or async; we'd bolt on extra
  libraries to reach what FastAPI gives for free.
- *Rejected — Django / DRF:* far too heavy for two endpoints; ORM, admin,
  migrations all unused.
- *Rejected — raw ASGI (Starlette only):* FastAPI *is* Starlette plus the typing
  and validation we want; no reason to drop down a level.

**Data validation / models — Pydantic.**
- *Why chosen:* the schema is "non-negotiable", so we model `ChatRequest`,
  `ChatResponse`, `Recommendation`, and `ConversationState` as typed objects and
  let validation be structural, not hand-written. Comes with FastAPI.
- *Rejected — dataclasses / `TypedDict`:* no runtime validation; we'd write
  manual checks and still risk schema drift.
- *Rejected — `jsonschema` by hand:* more boilerplate, weaker editor support,
  duplicates what Pydantic already does.

**LLM access — a thin provider-agnostic adapter (LiteLLM as the default).**
- *Why chosen:* one call interface across Gemini / Groq / OpenRouter, easy
  fallback and retries, structured-output/JSON mode where supported. The PDF
  rewards behavior and reliability, not vendor loyalty, so we avoid lock-in.
- *Rejected — a single vendor SDK hard-wired (e.g. only `google-generativeai`):*
  painful to switch if that provider is slow or rate-limited on the day.
- *Rejected — LangChain as the abstraction:* pulls in a large dependency and a
  lot of indirection we do not need for one model call per turn; harder to
  explain in the interview.
- *Note:* "adapter" is the commitment; LiteLLM is the concrete pick and is itself
  swappable behind our own small wrapper.

**Lexical search — scikit-learn `TfidfVectorizer` (word **and** char n-grams).**
- *Why chosen:* mature, lightweight, in-memory; the char-level mode is what
  catches product codes like `OPQ32r`, `G+`, `.NET`. Perfect for 377 items.
- *Rejected — a BM25 library (e.g. `rank_bm25`):* fine for word matching but no
  built-in char n-gram trick; scikit-learn gives both modes in one tool we
  already need.
- *Rejected — Elasticsearch / OpenSearch:* a whole server for a 377-row catalog;
  absurd overhead.

**Semantic search — tried (Sentence-Transformers, `all-MiniLM-L6-v2`), then removed.**
- *History, not current state.* We built a sentence-embedding retriever to catch meaning
  the exact words miss ("safety-critical" -> "Dependability"). It is **no longer in the
  system** — see Decision 17.
- *Why removed:* measured against the ten sample conversations, the semantic layer changed
  Recall@10 on **none** of them (net contribution 0.000), while adding a heavy,
  version-fragile dependency chain (torch / transformers / huggingface-hub) that was also
  silently failing to load on some hosts. A component that moves no measured number is not
  worth its failure surface. Retrieval is therefore lexical-only (`LexicalRanker`); the
  final Recall@10 is **0.809**, reached by lexical tuning (`name_boost` 1.5 → 2.0 and a
  distinctive-skill-in-name bonus), not by embeddings.
- *A larger embedding model was rejected too, and the measurement is why:* if the small
  model's contribution is exactly zero, a bigger one can only improve the part that already
  added nothing, at far higher memory and cold-start cost.
- *Rejected — a vector database (FAISS server / Chroma / pgvector):* even when the
  embedding stage existed, 377 vectors fit in a plain NumPy array; a DB is infrastructure
  for no benefit — and with no embedding stage there is nothing for it to index. (Same
  reasoning as Decision 5.)

### 13.2 Supporting pieces (what makes it actually run, test, and ship)

These are the parts the earlier draft left implicit. Naming them is what proves
the stack is complete.

**ASGI server — Uvicorn.**
- *Why chosen:* the standard way to actually *run* a FastAPI app; lightweight,
  works on every free host.
- *Rejected — Gunicorn alone:* needs a worker class to serve ASGI anyway;
  Uvicorn (optionally managed by Gunicorn in production) is the simpler default
  for a small service.

**Catalog loading — Python standard-library `json` with `strict=False`.**
- *Why chosen:* the provided file has a raw control character that breaks strict
  parsing (verified). `json.loads(..., strict=False)` handles it with zero extra
  dependencies. This is called out as an explicit decision because it is a real
  trap, not an afterthought.
- *Rejected — a third-party JSON parser (`orjson`/`ujson`):* faster, but we load
  the catalog *once* offline, so speed is irrelevant, and some of them are
  *stricter* about control chars, which would reintroduce the bug.

**Configuration & secrets — environment variables via `pydantic-settings`.**
- *Why chosen:* API keys, the model name, and the `null`-vs-`[]` flag must be
  configurable without code edits; pydantic-settings reads env vars into a typed
  config object, consistent with the rest of the stack.
- *Rejected — hard-coded constants:* leaks secrets, and flipping the empty-list
  behavior would need a code change instead of an env var.

**Test runner — pytest, with FastAPI's `TestClient` (httpx under the hood).**
- *Why chosen:* pytest is the default for Python; `TestClient` lets us hit
  `/chat` and `/health` in-process for fast, real HTTP-level tests — exactly how
  we replay the 10 traces and run behavior probes.
- *Rejected — `unittest` only:* more boilerplate, weaker fixtures than pytest.
- *Pairs with Hypothesis (below) — they are complementary, not alternatives.*

**Property-based testing — Hypothesis.**
- *Why chosen:* edge-case robustness is exactly what the PDF says weak
  submissions miss. Hypothesis auto-generates malformed messages, empty
  histories, weird orderings, and recommendation-count limits.
- *Rejected — only hand-written example tests:* they miss the inputs you did not
  think of; Hypothesis exists to find those.

**Logging — standard-library `logging` (structured fields).**
- *Why chosen:* zero dependencies, enough for the per-request logs in §11
  (mode, candidates, timings, fallback flag). A heavier observability stack is
  unjustified for this scope.

**Dependency pinning — `requirements.txt` with pinned versions.**
- *Why chosen:* reproducible builds on the deploy host; the grader must be able
  to reach a stable service. Pinning avoids "works on my machine" drift.
- *Rejected — unpinned deps:* a surprise upstream release could break the
  deployed endpoint at submission time, which is the worst possible moment.

**Deployment target — a free always-reachable host (Render as the default
choice; Railway / Fly / HF Spaces as fallbacks).**
- *Why chosen:* the PDF explicitly allows these and gives a 2-minute cold-start
  grace on `/health`, so a free tier is fine. Render is simple for a FastAPI +
  Uvicorn service.
- *Rejected — a paid/always-on VM:* unnecessary cost; the cold-start grace exists
  precisely so free hosting works.
- *Open item:* the final host is confirmed at deploy time (Build Step 9), since
  it does not affect the application code.

### 13.3 Coverage check — does the stack cover everything?

Mapping every runtime job to its owner, so there is no hidden gap:

| Job to be done                         | Covered by                          |
| -------------------------------------- | ----------------------------------- |
| HTTP API, two endpoints                | FastAPI                             |
| Run the app (ASGI)                     | Uvicorn                             |
| Strict request/response validation     | Pydantic                           |
| Typed internal state                   | Pydantic                           |
| Load catalog (control-char trap)       | stdlib `json`, `strict=False`       |
| Derive `test_type`                     | our code, from `keys` (Decision 9)  |
| Lexical search (word + char)           | scikit-learn TF-IDF                 |
| Similarity math                        | NumPy (comes with scikit-learn)     |
| Call the LLM / swap providers          | LLM adapter (LiteLLM)               |
| Config, secrets, feature flag          | env via pydantic-settings           |
| Logging / observability                | stdlib `logging`                    |
| Unit + behavior tests                  | pytest                             |
| HTTP-level endpoint tests              | FastAPI `TestClient` (httpx)        |
| Edge-case generation                   | Hypothesis                         |
| Reproducible install                   | pinned `requirements.txt`           |
| Public deployment                      | Render (or Railway/Fly/HF)          |

Every job has exactly one owner. Nothing in the runtime path is unaccounted for.

All chosen for reliability and explainability, not novelty.

---

## 14. The 30-second interview pitch

"I built a controlled SHL assessment recommender, not a free chatbot. On every
request it rebuilds the hiring need from the full conversation, decides whether
to clarify, recommend, refine, compare, or refuse, retrieves only from an
in-memory SHL catalog snapshot, ranks with lexical (TF-IDF) search and a
transparent weighted scorer, and returns a schema-safe shortlist. The model
handles understanding and wording; the
code owns the decisions, the URLs, the `test_type` (derived from the catalog's
`keys` field), and the final schema. I tuned it against the 10 sample
conversations as a scoreboard."

---

## 15. Interview defense sheet

**Q. Why not a generic chatbot?**
Because the grader is strict on schema and grounding, and probes behavior. A
controlled system reliably knows when to clarify, recommend, refine, compare, or
refuse.

**Q. How did you enforce "Individual Test Solutions only"?**
I checked the actual data: the provided JSON is the working set, and the
report-style items that look risky are actually in the expected answers, so I do
not filter them. Only seven "Solution"-named items are genuinely ambiguous, so I
audit just those and tag scope on the data, not in prompts.

**Q. Why no vector database?**
377 items. In-memory is faster, simpler, cheaper, and easier to explain. I would
revisit only if the catalog grew by orders of magnitude.

**Q. Why lexical-only retrieval — didn't you try semantic?**
I did. I built a hybrid (lexical + sentence-embedding) and measured it against the
ten traces: the semantic layer changed Recall@10 on none of them (net 0.000), while
dragging in a heavy, version-fragile dependency that was also silently failing to
load. So I removed it and tuned the lexical ranker instead, reaching 0.809. The
catalog's relevance is dominated by exact skill/product names and category intent —
both lexical — and char-level TF-IDF catches the odd codes (`OPQ32r`, `G+`). Removing
a component that moved no measured number is the honest call. (See Decision 17.)

**Q. Why not let the LLM return the JSON?**
Schema and catalog-only grounding are hard requirements. I let the model write
the reply but keep the structured fields and URLs in code, so the failures that
zero a submission cannot happen.

**Q. How are URLs guaranteed real?**
Every URL is copied from the catalog's `link` field. The model never produces a
URL.

**Q. How did you handle statelessness?**
Each call carries the full history; I rebuild state from it every time and store
nothing server-side. Latest user correction becomes the active state.

**Q. Where did `test_type` come from if it is not in the JSON?**
It is encoded in the `keys` field. I mapped the eight categories to the codes the
traces use and derived it deterministically with full coverage — no guessing, no
manual curation.

**Q. How did you handle the turn cap?**
I treat questions as expensive but I do not under-ask: I clarify 1–2 times on
vague openers (matching the sample traces), commit fast on specific ones, and
keep every conversation within the message budget.

**Q. `null` vs `[]` for empty recommendations?**
The PDF says "empty" but all ten sample traces use `null`, and the traces are
what the grader is built from, so I default to `null` and made it a config flag.

**Q. What would you improve with more time?**
Deeper catalog enrichment, more behavior probes, an optional reranker driven by
measurement, and more synthetic holdout-style traces.

---

## 16. Build order (after you approve this document)

1. Load + clean the catalog (lenient parse, normalize, derive `test_type` from
   `keys`, audit the 7 "Solution" items), and save the snapshot.
2. Define the typed data models (`Recommendation`, `ChatRequest`,
   `ChatResponse`, `ConversationState`).
3. Build the state reconstructor (history → state, latest-correction-wins).
4. Build the policy engine (mode + adaptive clarify gate).
5. Build retrieval + transparent ranking. (A semantic stage was built here and later
   removed for zero measured recall — see Decision 17; the shipped retriever is
   lexical-only.)
6. Build response assembly + schema validation + fallback templates.
7. Build the trace-replay harness and behavior probes.
8. Tune against the 10 traces using the scorecard.
9. Deploy (with `/health` cold-start handling).
10. Write the 2-page approach document.

---

## 17. One-line summary of the whole design

**A policy-guided, reliability-first, stateless, catalog-grounded conversational
recommender — lexical (TF-IDF) retrieval with a transparent ranker, code-controlled
structured output, `test_type` derived from the catalog's own `keys` field, tuned
against the real sample
conversations.**
