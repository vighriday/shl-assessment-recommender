# Conversation signals

The phrasings the agent must recognise, taken from the ten sample conversations
(C1–C10). These drive the deterministic part of state extraction. The list is
descriptive — it records what real users actually say in the traces — so the
detectors are built for observed language, not invented patterns.

## Comparison (turn returns no new shortlist)

The user asks how two named products differ. Observed forms:

* "What's the difference between OPQ and OPQ MQ Sales Report?" (C5)
* "What's the difference between the DSI and the Safety & Dependability 8.0?" (C6)
* "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?" (C3)

Signal: a difference/compare phrase plus two product references. The two products
are usually joined by "between ... and ..." or "X different from Y".

## Refusal / off-topic (legal, regulatory, general advice)

The user asks for something outside scope — typically legal/compliance
interpretation. The agent declines that part but stays helpful about the
assessments (partial refusal). Canonical example, C7 turn 3:

* "Are we legally required under HIPAA to test all staff who touch patient
  records? And does this SHL test satisfy that requirement?"

Signal words: "legally required", "legal", "regulation/regulatory", "compliance"
framed as an obligation question, "satisfy that requirement", "lawsuit",
"discrimination law". Note: a product *named* "HIPAA (Security)" or "Workplace
Health and Safety" is in scope — the refusal is about interpreting the law, not
about the topic.

## Confirmation / closure (turn ends the conversation; re-emit the shortlist)

The user accepts the shortlist. Observed forms:

* "That works. Thanks." (C2)
* "Perfect — new simulation for volume, old solution for finalists. Confirmed." (C3)
* "Clear. We'll use OPQ for everyone ... keeping the five solutions as our audit stack." (C5)
* "We're industrial. The 8.0 bundle is the right fit. Confirmed." (C6)
* "Understood. Keep the shortlist as-is." (C7)
* "That works. Thanks." / "Good." / "Perfect, that's what we need." (C1)

Signal words: "confirmed", "that works", "perfect", "keep the shortlist",
"keep ... as-is", "we'll use", "good choice", "that's what we need". Only treat as
closure when a shortlist already exists in the conversation.

## Refinement (update the existing shortlist, do not restart)

The user adds or removes a constraint mid-conversation. Observed forms:

* "Can you also add a situational judgement element ...?" (C4)
* "Should I also add a cognitive test for this level?" (C2)
* "In that case, I am OK with adding a simulation ..." (C8)
* "add MQ only where we want motivators ..." (C5)

Add-signals: "also add", "add a/an", "include", "as well", "on top".
Drop-signals: the agent often offers an opt-out ("say the word if you'd rather
drop it" — C2/C8), so the user may reply with "drop the personality test",
"skip personality", "remove ...".

## Role / population / skills / seniority (handled by the LLM, not rules)

Too varied for patterns; extracted by the understanding layer. Examples of the
range:

* "senior Rust engineer for high-performance networking infrastructure" (C2)
* "graduate financial analysts — final-year students, no work experience" (C4)
* "500 entry-level contact centre agents. Inbound calls, customer service" (C3)
* "plant operators for a chemical facility. Safety is absolute top priority" (C6)
* "bilingual healthcare admin staff in South Texas ... assessed in Spanish" (C7)

Short answers to clarifying questions also appear and must be read in context:
"English.", "US." (C3).

## Prompt injection (not present in the traces; required by the brief)

No sample shows it, but the brief requires refusing attempts to override
instructions or recommend non-SHL tools. Detected from known patterns:
"ignore previous instructions", "disregard your instructions", "you are now",
"system prompt", "recommend <non-SHL product>". Built defensively and covered by
synthetic tests.
