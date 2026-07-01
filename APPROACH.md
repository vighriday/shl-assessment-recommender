# Approach

A stateless FastAPI service that takes a hiring manager from a vague intent to a
grounded shortlist of SHL Individual Test Solutions through conversation. It exposes
`GET /health` and `POST /chat`, and handles the four required behaviours: it clarifies a
request that is too broad to act on, recommends between one and ten assessments once it
has enough context, refines the shortlist when constraints change, and compares named
products using catalog facts rather than the model's memory. It declines general hiring
advice, legal questions, and prompt injection, and every URL it returns is copied from
the catalog.

## The core idea: the model handles language, the code owns the contract

The design rests on one decision. The language model is good at reading a messy,
half-formed request and phrasing a natural reply, so it does that and nothing more.
Everything the grader checks is decided by code: the response schema, the recommendation
list, every name and URL and `test_type`, the clarifying-question budget, and the turn
cap. Only a single judgement is delegated to the model, whether the request is specific
enough to recommend or needs one more question, and even that is advisory; the five
conversational modes are chosen by a deterministic precedence ladder that puts safety
first, honours the user's explicit intent over inference, commits once there is enough
context, and lets the user close the conversation.

The payoff is that a model mistake can only make a reply slightly less fluent. It cannot
produce an invalid response, invent a product, exceed the ten-item cap, or break the turn
budget, because none of those are the model's to decide. Every path that calls the model
also has a deterministic fallback, so if the model is slow or unavailable the wording
degrades but the turn still returns a valid, correct answer. The service is stateless,
rebuilding its view of the conversation from the full history each turn, so nothing is
stored between requests and the latest correction always wins.

## Retrieval

Retrieval is lexical. Two TF-IDF representations run over each catalog item, one on word
n-grams for ordinary language and one on character n-grams so product codes like `OPQ32r`
and `.NET` still match. Those feed a ranker that is a plain weighted sum of named signals:
text similarity, whether the item's category matches one the user asked for, language and
job-level fit, a boost when a query word is a large fraction of an item's own name, a
small bonus when a distinctive required skill such as `AWS` appears in the name, and an
additive floor for the two staple assessments a competent consultant reaches for by
default. Because every term is a number I can read, any recommendation can be explained.

Mean Recall@10 on the ten sample conversations is 0.809. A script measures it and a test
locks it against regression. I originally built a hybrid that added sentence embeddings on
top of the lexical layer, then measured what the embeddings contributed and found it was
zero: on every trace the hybrid scored the same as lexical alone, because the one gold
item the embeddings uniquely recovered was cancelled by a different one they pushed out.
The embedding model was also failing to load on a clean install because of an
under-pinned dependency, so the hybrid was often not even running. I removed it. Carrying
a heavy, hard-to-reproduce component that moves no measured number is the kind of choice
the brief warns against, and the lexical system is smaller, reproducible, and no worse on
the evidence.

## Prompt design

There are two model calls per turn, both tightly scoped. The understanding call extracts
a small JSON object of role, seniority, skills, requested categories and languages, plus
the readiness judgement, and is told to leave a field empty when the user did not state
it rather than guess. It is also told that a bare job title is not enough to recommend on,
not to re-ask something already answered, and to stop asking once the user pushes back.
The reply call writes only the short framing sentence for the turn; it is told not to list
products or URLs, because the real list is built by code, and on a comparison turn it is
handed the compared products' actual catalog attributes so the comparison is grounded
rather than recalled. Clear signals like injection, off-topic asks, confirmations and
edits are detected in code, so those turns behave correctly even with the model down.

## Evaluation

The brief is explicit that submissions fail on weak evaluation, so testing is where I
invested most, and I measure four things directly.

**Retrieval quality** is Mean Recall@10 against the ten sample conversations, currently
0.809. A script computes it and a test locks a floor so a change cannot quietly regress
it. **Recommendation relevance** is measured the same way but end to end: all ten traces
are replayed through the full engine and the committed shortlist is scored against each
trace's labelled gold set, and a separate test confirms every gold URL actually exists in
the catalog so the ceiling is real. **Groundedness** is enforced structurally and then
tested: the recommendation list is built entirely by code from the catalog, so the model
cannot introduce a product or URL, and a probe drives a deliberately mischievous model
that tries to smuggle a fabricated link into its reply and asserts it never reaches the
recommendations; comparison turns are handed the compared products' real catalog
attributes so the answer is drawn from catalog evidence rather than the model's prior.
**Overall response accuracy and effectiveness** is covered by around four hundred
deterministic tests (unit, HTTP-contract, a Hypothesis property test that `/chat` never
returns a server error on arbitrary input, behaviour probes for the named edge cases, and
a large edge-case battery over every mode plus whitespace, very long job descriptions,
unicode, and role casing), plus two methods for the one genuinely fuzzy decision, clarify
versus recommend: metamorphic laws that assert properties holding for any input (adding
information can never make a request less ready, for instance), and an independent model
call that judges whether each decision was reasonable and reports an agreement rate.

## What did not work

Several things did not work, and I kept only what I could measure. The embedding layer
added nothing and was removed. Raising the category weight measured about ten points
worse, because a flat category flag pulls in broad, unfocused tests, so I kept it low. A
grid search over the weights found no regression-free gain beyond the tuned point. My own
adversarial testing found real bugs the happy path hid, including a prompt-injection check
that missed the most common phrasing of the attack, and I fixed each and locked it with a
test. Eight gold items are still missed at 0.809; I diagnosed each as either an item the
user dropped, a genuine vocabulary gap, or a case buried in a dense cluster of similar
products, none closable without tuning to the visible traces at the cost of the held-out
set.

## Stack and use of AI tools

The service uses FastAPI and Pydantic for a typed, self-validating contract, and
scikit-learn for TF-IDF, which keeps it free of any heavy machine-learning runtime. Model
access goes through LiteLLM so the provider is a configuration choice, with a failover
chain from the primary Gemini model to a second key and then to Groq if a provider is
rate-limited, without letting a real timeout blow the thirty-second budget. The default
model is `gemini-2.5-flash`, and the service is deployed as a Hugging Face Docker Space
with the key held as a secret. Dependencies are pinned for a reproducible build.

I used an AI coding assistant extensively, for end-to-end implementation rather than just
scaffolding: it wrote much of the FastAPI service, the retrieval and ranking code, the
conversation policy, the test suite, and the documentation, working under my direction,
review, and iteration. What was mine throughout was the thinking. I drove the design (the
"code owns the contract, model owns the language" split, and the decision to keep
retrieval lexical after measuring that embeddings added no Recall@10), tuned the retrieval
against the measured number, and diagnosed and directed each bug fix, including the
prompt-injection gap and the chat-history bug that my own testing surfaced. I verified
everything with pytest, Hypothesis, and ruff. Gemini and Groq are the runtime models the
service itself calls. The code is organised and documented so I can explain any part of it
end to end, which is the point: the assistant did the typing, the decisions and their
justification are mine.
