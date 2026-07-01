# Policy design

How the agent decides what to do on each turn. This is written from principles,
not by copying the sample conversations. The samples are used afterwards to
*check* that the principles produce the right behaviour — they are evidence of
what good looks like, not a script to reproduce line by line. A policy built from
sound principles generalises to the unseen holdout; a policy that memorises ten
transcripts does not.

## The question the policy answers

Every turn, the agent answers one question: **what does this turn need from me?**
The hard part is that a single user message can carry more than one signal (a
comparison that also confirms, an edit that also asks a question), so the policy
is a **precedence of intents**, resolved top to bottom. The first that applies
wins.

## The model of a good assessment consultant

The precedence below is the behaviour of a competent, trustworthy consultant,
expressed as rules:

1. **Safety is non-negotiable, but refusal is surgical.**
   A prompt-injection attempt or an out-of-scope ask (legal interpretation,
   general hiring advice) is handled first — you cannot be helpful by being
   manipulated, and you must not answer questions outside your remit. But
   declining the bad part never throws away good work already done: if a shortlist
   exists, it stays. (This is "partial refusal".)

2. **The user's explicit intent outranks the agent's inference.**
   If the user clearly asks to compare two products, clearly accepts the
   shortlist, or clearly edits it, honour *that* before asking the more abstract
   question "do I have enough context to recommend?". Stated intent beats inferred
   need. Concretely, once a shortlist is on the table, an "add/drop/which-is-right"
   message is a refinement, and a "compare X and Y" message is a comparison —
   neither restarts the conversation.

3. **Commitment is a one-way gate, and the bar is adaptive.**
   The moment there is enough to act on, commit to a shortlist rather than
   continuing to interrogate — extra questions cost turns and the user ends the
   conversation once a shortlist appears. But "enough" depends on how specific the
   opener was: a one-line vague request needs a clarifying question or two; a
   detailed brief or pasted job description is already enough to commit on the
   first turn. We never clarify just to seem thorough.

4. **Closure belongs to the user.**
   The agent does not end the conversation unilaterally. It marks the task
   complete only when the user signals acceptance. A first shortlist is an offer,
   not an ending.

## The precedence, concretely

Given the reconstructed state for the turn, choose the first that matches:

1. **REFUSE** — prompt injection, or an off-topic/legal/general-advice ask.
   Returns no new shortlist. Any existing shortlist is preserved and may be
   re-shown alongside the refusal (partial refusal).

2. **CONFIRM → end** — the user accepts and a shortlist already exists. Re-show the
   final shortlist and mark the task complete. (Acceptance with nothing to accept
   is not closure; it falls through.)

3. **COMPARE** — the user asks how products differ. Answer from catalog facts.
   Returns no new shortlist on the comparison turn; an existing shortlist persists
   in context.

4. **REFINE** — a shortlist already exists and the user adds, drops, or questions
   an item. Update (or deliberately decline to change, with a reason) and re-show
   the shortlist.

5. **RECOMMEND** — there is enough context and no shortlist yet. Commit to the
   first shortlist.

6. **CLARIFY** — none of the above and context is insufficient. Ask the single
   most useful question, provided the question budget allows.

7. **Fallback RECOMMEND** — if context is insufficient *but* the question budget is
   exhausted (we are near the turn cap), commit to the best shortlist we can
   rather than asking another question we have no room for. Never burn the last
   turn on a question.

## The adaptive clarify gate (rule 5 vs 6)

Whether to commit or clarify on an early turn turns on two things:

* **Is the request specific enough to recommend well?** This is the subtle part.
  A simple structural rule ("has a role plus one differentiator") is not enough:
  the samples show a request can name *many* skills and still be too broad to be
  precise (the wide full-stack JD in C9, where the agent asks "backend or
  frontend?" before committing), and can name *few* and be perfectly clear (C8,
  "screen admin assistants for Excel and Word" — commit immediately). Telling
  these apart is a judgement, so we ask the language model for it: it returns
  `ready_to_recommend` plus, if not ready, the single most useful question. The
  policy uses that judgement.

  Crucially, **code stays in control**. The model only advises readiness; it never
  bypasses the budget or the turn cap, and if the model gives no opinion (it
  failed, or the turn did not need it) the policy falls back to the structural
  `has_minimum_context()` rule. So the decision is always defined and the model is
  never a hard dependency.

* **Is there budget to ask?** We keep clarification to roughly one or two
  questions before the first shortlist, and we always leave room to actually
  deliver within the turn cap. If the model wants to clarify but the budget is
  spent, we commit anyway (rule 7) rather than waste the last turn.

So: not-specific-enough and budget left -> clarify; specific enough, or budget
spent -> commit. This reproduces the sample behaviour, but the *reason* is the
consultant model above (commit once you know what to measure; ask only when a
decision-critical choice is open), not the samples themselves.

## end_of_conversation

True only on a CONFIRM turn (rule 2). Every other turn — including the first
shortlist — is False. Closure is the user's signal, never the agent's
assumption.

## Why this is defensible

It is a small, ordered set of rules that each express a clear principle, it is
deterministic and testable, and it reproduces the sample behaviour as a
consequence rather than as a target. In an interview it can be defended as "a
model of how a careful consultant prioritises what a conversation needs", which is
stronger than "it matches the ten examples".
