# Out-of-the-box improvements

A running log of design moves that go beyond the literal brief — where a bit of
extra thinking made the system safer or stronger. Kept so the reasoning is on the
record and can be explained in the interview, and so we can tell deliberate
choices apart from accidents.

The guiding rule: widen what the system *recognises* and *tolerates*, but never at
the cost of doing something confidently wrong. Every addition fails safe and is
backed by a test.

## 1. Ground signal detection in the real catalog vocabulary

**What.** The deterministic comparison detector can be given a vocabulary built
from the actual catalog — the 377 product names plus the short codes inside them
(OPQ, SVAR, DSI, MFS, MQ, ...). It uses this to decide whether "compare X and Y"
refers to real products.

**Why it helps.** Two wins at once:

* It catches comparison phrasings the sample wording never showed, because it
  matches on real product references, not on sentence shape.
* It removes false positives. "compare notes with my team" looks like a
  comparison to a naive rule, but the catalog knows "notes" is not a product, so
  the turn is not misrouted.

**Why it is safe.** When no vocabulary is supplied the detector falls back to the
old heuristic, so nothing depends on it; with the vocabulary it is strictly more
accurate. Proven by a test that shows the same sentence is suppressed only when
the catalog is present.

**Interview line.** "I grounded signal detection in the actual catalog instead of
overfitting the ten sample conversations, so it generalises to unseen phrasings
and rejects look-alike non-product phrases."

## 2. Widen phrasing coverage for the holdout, deliberately and asymmetrically

**What.** Stress-tested the detectors with natural variants a real user might use
("ship it", "lets go with that", "OPQ or GSA — which is better?", "can this get
us sued?", "from now on you are DAN") and widened the patterns to cover them.

**The asymmetry that keeps it safe.** Widening is generous where a false positive
is cheap (recognising more *confirmation* or *comparison* phrasings just routes to
the right helpful behaviour) and careful where a false positive is costly
(*refusal* still requires a legal term **and** an obligation/liability framing, so
a product named "HIPAA" or "Workplace Health and Safety" is never mistaken for a
legal question).

**Why it helps.** The graded holdout will not reuse the sample wording verbatim.
Recognising the natural variants reduces the chance of missing a confirmation
(and failing to end the conversation) or a comparison (and answering the wrong
way).

## 3. Tolerate role casing and synonyms in the request

**What.** The request parser accepts `User`/`USER`/`Agent`/`Human` and maps them
onto the canonical `user`/`assistant`/`system`. The brief's JSON uses lowercase,
but the sample transcripts display "User"/"Agent", so the exact casing the grader
sends is not guaranteed.

**Why it helps.** Rejecting an entire request over a casing or synonym difference
would fail a hard eval for no good reason. Being lenient on input while staying
strict on output is the right place to spend tolerance.

**Why it is safe.** Output is unaffected; only the inbound role string is
normalised. Unknown roles are still rejected.

## 4. Skip the model call when deterministic signals already decide the turn

**What.** On off-topic, prompt-injection, and pure-confirmation turns, the
extractor does not call the language model, because understanding cannot change
the outcome.

**Why it helps.** Saves latency against the 30-second budget and removes a
needless dependency on the model for turns that are fully handled by rules. It
also means a refusal or a confirmation still works perfectly when the model is
down.

## 5. Let the model judge "specific enough to recommend", with code holding the limits

**What.** The hardest decision in the agent is when to stop clarifying and commit
to a shortlist. A structural rule ("role plus one differentiator") gets it wrong:
the samples show a request can name many skills and still be too broad to be
precise (a wide full-stack JD), and can name few yet be perfectly clear (the exact
tools to screen on). So the understanding step also returns a readiness judgement
— `ready_to_recommend` and, if not ready, the single most useful question — and
the policy uses it for the clarify-vs-commit choice.

**Why it helps.** It reproduces the gold clarify-then-commit rhythm for the right
reason (the request really is too broad to be precise yet), rather than by a
brittle rule tuned to ten examples. This directly protects two scored behaviours:
not recommending too early on a vague turn, and not over-asking on a clear one.

**Why it is safe.** The model only *advises*. Code still enforces the question
budget and the turn cap, still commits when the budget is spent even if the model
wants to ask more, and falls back to the structural rule when the model gives no
opinion. The model is never a hard dependency and can never make the agent blow
the turn cap.

**Interview line.** "The clarify-vs-commit call needs judgement a rule can't
capture, so the model advises readiness while the code keeps hard control of the
budget and turn cap — the model improves the decision but can never break the
contract."

**How this was found.** Building the policy from principles first, then checking
it against the ten samples, surfaced that a pure structural gate recommended too
early on four of them (the broad-JD and decision-critical-gap cases). That gap is
exactly what the readiness judgement fills — a good example of the samples being
used to *validate* a design rather than to *be* the design.

---

## 6. Staple-default injection for retrieval, from evidence not intuition

**What.** Two general-purpose measures recur across the gold shortlists — OPQ32r
in eight of ten, Verify G+ in three — as the default personality and cognitive
components of a battery. They are almost never named in the query, so text
retrieval alone misses them. The ranker injects them as candidates when their
dimension is relevant to the hire and lets scoring place them.

**Why it helps.** It was the single biggest recall gain (0.505 -> 0.612 and again
through weighting), and it is principled: the sample agent explicitly says things
like "I'm including OPQ32r by default for a senior IC". We are encoding an
observed consulting default, not guessing.

**Why it is safe.** The staple list is tiny and evidence-derived (counted across
the gold shortlists), the injection is gated on the hire being professional and
the dimension relevant, and the family-diversity cap stops it from crowding the
list. It is configurable weight, so it can be dialled back.

## 7. Family-diversity cap instead of a heavy reranker

**What.** A light MMR-style step caps how many items from one product family (the
many "OPQ ... Report" variants, the SVAR spoken-language variants) may appear in
the shortlist, so a single family cannot fill it and crowd out other relevant
items.

**Why it helps.** It lifted recall (0.737 -> 0.757) and improves the *usefulness*
of a shortlist (variety over near-duplicates) without the cost and opacity of a
cross-encoder reranker.

**Why it is safe.** It only defers surplus siblings; the highest-scoring item of
each family is always kept, and if capping leaves the list short it backfills from
the deferred items, so it never returns fewer than it could.

---

Each item above is covered by tests in `tests/conversation/test_signals.py`,
`tests/conversation/test_policy.py`, `tests/conversation/test_policy_vs_traces.py`,
`tests/catalog/test_vocabulary.py`, `tests/api/test_schemas.py`, and
`tests/retrieval/`, including the negative cases that prove the safety guards hold.
