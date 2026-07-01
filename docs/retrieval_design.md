# Retrieval and ranking design

How the agent turns a hiring need into a ranked shortlist of catalog items. The
goal is Recall@10: get the relevant assessments into the top ten. The design
follows from what "relevant" actually looks like in the sample conversations.

## What makes an item relevant (from the samples)

A gold shortlist is never one kind of item. Across the ten traces, relevance comes
from several different signals at once:

* **Exact skill / tool match** — the strongest signal. "Excel and Word" -> MS Excel,
  MS Word. "Java, Spring, SQL" -> Core Java, Spring, SQL. A named skill maps to a
  dedicated catalog test with almost the same name. This is lexical, and often
  exact.
* **Category intent** — the user names a dimension: "cognitive" -> Ability (A),
  "personality" -> Personality (P), "situational judgement" -> Biodata (B),
  "simulation" -> Simulations (S). This maps onto the `keys`/`test_type` of items.
* **Role / context semantics** — "safety-critical plant operators" -> Dependability
  and Safety Instrument; "senior leadership" -> OPQ Leadership Report;
  "contact centre" -> Contact Center Call Simulation. The words in the query do
  not appear in the product name, so this is the hardest signal to catch with words
  alone.
* **Staple defaults** — some items recur as sensible defaults: OPQ32r appears in
  seven of ten gold shortlists as the personality component, Verify G+ in four as
  the cognitive component. A good shortlist often includes a general
  personality/ability measure even when not explicitly requested.
* **Job level** — "graduate" -> Graduate Scenarios and graduate-calibrated
  variants; "entry level" -> entry-level items.

The takeaway: **relevance is dominated by exact skill/product names and category
intent**, both of which are lexical. Char-level TF-IDF catches the odd product codes
(`OPQ32r`, `SQL`, `.NET`) that a purely semantic method would miss, and the ranker
folds in the category, job-level, language, and staple-default signals. The
role/context cases (e.g. "safety-critical" -> "Dependability") are the hardest, and
we did try sentence-embeddings for them — but that stage was measured to add nothing
and was removed (see *The semantic experiment*, below). Retrieval is therefore
**lexical**, then a transparent ranker.

## The pipeline

1. **Scope filter.** Only `in_scope` items are ever candidates.
2. **Lexical retrieval.** TF-IDF over each item's `search_text`, in two flavours:
   word n-grams (normal language) and character n-grams (product codes and odd
   tokens like `OPQ32r`, `G+`, `.NET`). Cosine similarity to the query. This is the
   only retrieval stage — there is no embedding model to load, so the pipeline has no
   optional runtime dependency and behaves identically on every host.
3. **Transparent ranking.** A readable weighted sum over: lexical score,
   category-intent match (requested `test_type` vs item), language match,
   job-level match, a boost for exact name/code hits, and an additive bonus when a
   *distinctive* required skill (one that names only a handful of products, e.g.
   `AWS`, `HIPAA`) appears in the item's name. Every term is inspectable, so "why did
   this rank here?" always has an answer.
4. **Shortlist.** Take the top K (1..10), honouring the max from the contract.

The class that implements retrieval + ranking is `LexicalRanker`
(`shl_recommender/retrieval/ranker.py`).

## Why a transparent ranker rather than a learned one

The catalog is tiny and the scoring must be explainable in an interview and easy
to debug when a trace under-recalls. A weighted sum of named signals gives both.
If measured recall demands more later, a reranker over just the top candidates is
the upgrade path — but only if the numbers justify it.

## How we measure

Recall@10 against the ten sample conversations is the scoreboard: for each trace,
the fraction of its gold items that appear in our top ten, averaged. We track it
as we tune the weights, and we never tune so hard that the design overfits the ten
visible traces — the holdout must benefit too. `scripts/measure_recall.py` runs
it; `tests/retrieval/test_recall_floor.py` guards the result against regressions.

## Tuning history (what moved the number, and what did not)

Starting mean Recall@10 was **0.505** (at that point the pipeline still carried the
semantic stage, later removed — its contribution was measured to be zero, so the arc
below is unaffected by its presence or absence). Each change was made because a
diagnosis pointed to it, and kept because the number improved — always via a general
mechanism, never a trace-specific hack.

1. **Staple defaults (0.505 -> 0.612).** Diagnosis: OPQ32r was missed in several
   traces despite appearing in eight of ten gold shortlists, because the query
   never names it. Fix: inject the recurring defaults (OPQ32r for personality,
   Verify G+ for cognitive) as candidates when their dimension is relevant.
2. **Stronger, additive staple weight (0.612 -> 0.677).** The default must
   actually reach the top ten, and asking for one category (e.g. cognitive) must
   not suppress the personality default — both are additive defaults, as the
   sample agent treats them.
3. **Proportional exact-name score (0.677 -> 0.717).** Diagnosis: a short, exact
   product ("MS Excel", "SQL (New)") was losing to longer variants that merely
   shared a word. Fix: score by the share of the item's own name that the query
   matched, so a tight match beats an incidental one.
4. **Query enrichment with extracted skills (0.717 -> 0.737).** Diagnosis: skills
   buried in a long brief (HIPAA, Linux) were not retrievable because only the raw
   text was searched. Fix: append the understanding step's skills/role/categories
   to the retrieval query.
5. **Family diversity cap (0.737 -> 0.757).** Diagnosis: near-duplicate families
   (the many "OPQ ... Report" products, the SVAR language variants) filled the
   shortlist and crowded out other relevant items. Fix: cap how many items of one
   product family may appear, a light MMR-style diversity step.
6. **Raise `name_boost` 1.5 -> 2.0 (0.757 -> 0.797).** *(pre-submission pass)*
   Diagnosis: two golds — C4 Basic Statistics and C8 MS Word — sat just below the
   top-ten cut, out-scored by longer variants that shared an incidental word. Fix:
   weight an exact-name match more heavily so a tight product name wins its slot.
7. **Distinctive-skill-in-name bonus (0.797 -> 0.809).** *(pre-submission pass)*
   Diagnosis: C9 AWS was missed — the query named "AWS" but the AWS product did not
   surface. Fix: add a small additive bonus when an explicitly-required *distinctive*
   skill (one that names only a handful of products, so `AWS`/`HIPAA` but not the
   ~nine-product `Java`) appears in an item's name. Guarded so a broad skill cannot
   promote a whole near-duplicate family.

**Where we stopped, and why.** The final mean is **0.809**. Eight gold items across
the ten traces are still missed; each was diagnosed (e.g. one is a gold item the user
themselves later dropped; the rest are reachable but semantically indirect, such as
Global Skills Assessment for a "re-skill sales" query or Medical Terminology for a
healthcare-admin role). Closing them would mean tuning to the ten visible traces
specifically, which the brief warns against and which would not help the holdout.
0.809 was reached entirely through general mechanisms, so it is the honest stopping
point. `scripts/measure_recall.py` reproduces it; `tests/retrieval/test_recall_floor.py`
guards a floor beneath it (and, being lexical-only, never skips). A top-candidate
reranker remains the measured upgrade path if a larger evaluation set later justifies it.

## The semantic experiment (built, measured, removed)

Early versions of this pipeline had a third retrieval stage: sentence-embedding
similarity (`all-MiniLM-L6-v2` via `sentence-transformers`), merged with the two
lexical retrievers — a **hybrid**. It was there to catch the role/context cases where
the query words never appear in the product name.

We measured it properly: Recall@10 with the semantic layer **on vs off, per trace**.
The result was **identical on all ten** (net contribution **0.000** — its one unique
recovery on a single trace was cancelled by an equal displacement of another gold
item). At the same time, the embedding model was *silently failing to load* on some
environments (an under-pinned dependency and a cache-path assumption), so on those
hosts the "hybrid" had in fact been running lexical-only all along — at the same
recall.

So we **removed the semantic stage**: `shl_recommender/retrieval/semantic.py` is
deleted, `sentence-transformers` (and its torch/transformers/huggingface-hub chain) is
out of `requirements.txt`, the `SHL_ENABLE_SEMANTIC_RETRIEVAL` / `SHL_EMBEDDING_MODEL`
config is gone, and the retriever is now `LexicalRanker`. The recall cost of removing
it was zero (measured); the two lexical gains above (0.757 -> 0.809) came from the same
pre-submission pass. A *larger* embedding model was rejected by the same logic — if the
small model adds exactly nothing, a bigger one can only improve the part that already
added nothing, at far higher memory and cold-start cost. The full decision is
Decision 17 in `ARCHITECTURE_DECISIONS.md`.
