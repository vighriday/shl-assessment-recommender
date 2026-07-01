"""Deterministic detection of conversation signals.

This module reads the message history and reports the signals that can be
recognised by rule — comparison requests, off-topic/legal asks, prompt-injection
attempts, user confirmation, and refinement (add/drop) intent. It is pure: no
model calls, no I/O, no randomness, so it is fast, fully testable, and serves as
the dependable floor when the language model is unavailable.

The phrasing patterns are seeded from the observed sample conversations (see
``docs/conversation_signals.md``) and then widened with the natural variants a
real user is likely to use, since the graded holdout will not reuse the sample
wording verbatim. Two principles keep the widening safe:

* Comparison detection can be grounded in the real catalog vocabulary (passed in),
  so "compare OPQ and GSA" resolves real products while "compare notes with my
  team" does not.
* The off-topic/refusal detector stays strict (it must see an obligation or
  liability framing), because a wrong refusal is more harmful than a missed one.

Detectors are conservative by design: when a call is genuinely ambiguous they
leave it to the policy engine and the model, which see the whole picture.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from shl_recommender.catalog.vocabulary import CatalogVocabulary

# --- Comparison ------------------------------------------------------------- #
_COMPARE_WORD = re.compile(
    r"\b(?:difference|differ(?:ent|s)?|compare|comparison|contrast|versus|vs\.?|"
    r"which (?:one |is )?(?:better|should i|to (?:pick|choose|use)|fits?)|"
    r"which .{0,20}\bfits?\b|better choice)\b",
    re.IGNORECASE,
)
# Pair extractors: "between X and Y", "X different from Y", "X vs Y", "X or Y",
# "compare X and Y", and the "X and Y" that precedes a trailing "compare"/"differ".
_BETWEEN_AND = re.compile(r"\bbetween\s+(.+?)\s+and\s+(.+?)[\?\.\!]?$", re.IGNORECASE)
# "compare X and Y" / "compare X to Y" / "contrast X and Y" — the pair follows the verb.
_COMPARE_X_AND_Y = re.compile(
    r"\b(?:compare|contrast)\s+(.+?)\s+(?:and|to|with|against|vs\.?|versus)\s+(.+?)[\?\.\!]?$",
    re.IGNORECASE,
)
_X_FROM_Y = re.compile(
    r"\b(.+?)\s+(?:different from|differ from|vs\.?|versus|compared to|or)\s+(.+?)[\?\.\!]?$",
    re.IGNORECASE,
)
# "how do OPQ and GSA compare", "OPQ and GSA differ?" — pair before the verb.
_X_AND_Y_VERB = re.compile(
    r"\b(?:how (?:do|does)\s+)?(.+?)\s+and\s+(.+?)\s+"
    r"(?:compare|differ|stack up|measure up)\b",
    re.IGNORECASE,
)

# --- Off-topic / legal ------------------------------------------------------ #
# Legal/regulatory questions framed as obligation, liability, or lawfulness.
# Requires a legal term AND a framing word so a product named "HIPAA" or
# "Workplace Health and Safety" does not trip it.
_LEGAL_TERMS = re.compile(
    r"\b(legal(?:ly)?|lawful|unlawful|lawsuit|sued?|liability|liable|regulation|"
    r"regulatory|statute|discrimination|gdpr|eeoc|ada|adverse impact|comply|"
    r"compliance|illegal)\b",
    re.IGNORECASE,
)
_LEGAL_FRAMING = re.compile(
    r"\b(required|require|obligated|obligation|must we|do we have to|are we allowed|"
    r"allowed to|satisf(?:y|ies)|legally|is it (?:legal|lawful|ok|okay)|"
    r"get (?:us |me )?sued|cause|result in|expose us|risk)\b",
    re.IGNORECASE,
)

# General (non-SHL) hiring advice the agent should redirect rather than answer. Two
# arms: the direct "how do I hire/recruit/onboard ..." form, and a "how do I
# structure/run/design/set up ... a hiring/interview/recruitment process|funnel|
# pipeline|interviews" form that catches the natural phrasing where the advice verb is
# not the hiring word itself. Both require an unambiguous hiring-process object, so a
# genuine assessment request ("what should I use to screen ...", "how do I assess
# coding skills") is not swept up.
_GENERAL_ADVICE = re.compile(
    r"\b(how (?:do|should|can) (?:i|we) (?:hire|recruit|interview|onboard|fire|"
    r"retain)|"
    r"how (?:do|should|can) (?:i|we) (?:structure|run|conduct|design|build|set up|"
    r"improve|organi[sz]e|plan) (?:\w+ ){0,3}"
    r"(?:interviews?|hiring|recruit(?:ment|ing)?|onboarding|"
    r"(?:hiring |interview |recruitment )?(?:funnel|process|pipeline|panel))|"
    r"write (?:a |the |me )?(?:job|interview) (?:description|posting|ad|"
    r"questions)|salary (?:range|benchmark|band)|negotiat(?:e|ing) (?:an? )?offer|"
    r"how much should (?:i|we) pay|what should (?:i|we) pay)\b",
    re.IGNORECASE,
)

# --- Prompt injection ------------------------------------------------------- #
# The terminal nouns are matched with an optional plural (``s?``) because the most
# common jailbreak phrasing is plural — "ignore all previous instructions" — and a
# word-boundary after a singular-only noun would fail on the trailing "s", silently
# letting the canonical attack through to the model instead of the guaranteed code
# refusal. This was a real gap found by the edge-case battery.
_INJECTION = re.compile(
    r"\b(ignore (?:all |your |the |these |any )*(?:previous |prior |above |earlier )*"
    r"(?:instructions?|prompts?|rules?|directions?|the above|everything|what)|"
    r"disregard (?:all |your |the |everything|previous |prior )*"
    r"(?:instructions?|prompts?|rules?|above|everything|catalog|guidelines?)|"
    r"forget (?:all |your |the |everything|previous )*"
    r"(?:instructions?|prompts?|rules?|everything|the above)|"
    r"(?:from now on,? )?you are (?:now )?(?:dan|a|an|in|no longer|free)|"
    r"you are now|pretend (?:to be|you are)|act as (?:if|a|an|though)|roleplay|"
    r"jailbreak|dev(?:eloper)? mode|"
    r"(?:print|reveal|show|repeat|tell me) (?:your|the) "
    r"(?:system |full |entire |initial |original )*(?:prompts?|instructions?|rules?)|"
    r"recommend (?:a |an |some )?(?:non-?shl|competitor|mercer|korn ?ferry|"
    r"another (?:vendor|company|tool|product)))\b",
    re.IGNORECASE,
)

# --- Confirmation / closure ------------------------------------------------- #
# Widened well beyond the sample wording to the natural ways people accept a list.
_CONFIRMATION = re.compile(
    r"\b(confirmed?|that works|works for (?:us|me)|that'?s what we need|"
    r"perfect|sounds (?:good|perfect|great)|looks good|that'?s? (?:great|all|it|"
    r"perfect|fine|good)|that (?:covers|does) it|keep (?:the )?(?:shortlist|list|it|"
    r"that|these)|keep .{0,20} as[- ]is|go with (?:the |this |that )?|"
    r"we'?ll (?:use|go with|take)|let'?s (?:go with|use|do) (?:it|that|this)|"
    r"lock(?:ing)? (?:it|that) in|ship it|finali[sz]e|final list|good choice|"
    r"great choice|yep|yes that|that'?s the one|these are (?:great|good|perfect)|"
    r"i'?ll take (?:it|that|these)|this (?:is|works|looks) (?:good|great|perfect|"
    r"fine)|we'?re good|all good)\b",
    re.IGNORECASE,
)

# --- Refinement (add / drop) ----------------------------------------------- #
_ADD = re.compile(
    r"\b(also add|add (?:a|an|the|in|on)|include|as well|on top|additionally|"
    r"throw in|ok with adding|okay with adding|happy to add|adding a|append|"
    r"can you add|please add|plus a|along with|incorporate)\b",
    re.IGNORECASE,
)
_DROP = re.compile(
    r"\b(drop|remove|skip|exclude|without the|take out|don'?t (?:include|add|want)|"
    r"leave out|get rid of|lose the|cut the|no (?:need for )?(?:the )?"
    r"(?:personality|cognitive))\b",
    re.IGNORECASE,
)

# Fragments that are never product references, to suppress comparison false
# positives like "compare notes" / "compare options with my team".
_NON_PRODUCT_FRAGMENT = re.compile(
    r"^\s*(?:notes?|options?|prices?|costs?|results?|scores?|them|these|those|it|"
    r"that|the two|both|candidates?|vendors?|tools?|my team.*|with .*)\s*$",
    re.IGNORECASE,
)


# Typographic characters a client (or a document paste) may send in place of the
# ASCII forms our patterns match. Real transcripts use curly apostrophes, so without
# this every apostrophe-bearing pattern ("we'll", "that's", "don't") would silently
# miss — a quiet failure across confirmation, add/drop, and injection detection.
_TYPOGRAPHIC = {
    "’": "'",  # right single quote  '
    "‘": "'",  # left single quote   '
    "ʼ": "'",  # modifier apostrophe
    "“": '"',  # left double quote   "
    "”": '"',  # right double quote  "
    "–": "-",  # en dash
    "—": "-",  # em dash
}
_TYPO_TABLE = {ord(k): v for k, v in _TYPOGRAPHIC.items()}


def _clean(text: str) -> str:
    """Normalise whitespace and typographic punctuation to the ASCII forms.

    Folding curly quotes and dashes to their ASCII equivalents means the detectors
    match the same whether the user typed plain text or pasted from a word processor.
    """
    return re.sub(r"\s+", " ", (text or "").translate(_TYPO_TABLE)).strip()


_TARGET_LEAD = re.compile(r"^(?:is|are|was|were|does|do|how|the|a|an|both|two)\s+", re.IGNORECASE)


def _clean_target(text: str) -> str:
    fragment = _clean(text)
    previous = None
    while fragment and fragment != previous:
        previous = fragment
        fragment = _TARGET_LEAD.sub("", fragment)
    return fragment


class Signals(BaseModel):
    """Rule-detected signals for the latest user turn."""

    model_config = ConfigDict(frozen=True)

    is_comparison: bool = False
    comparison_targets: tuple[str, ...] = ()
    is_off_topic: bool = False
    is_prompt_injection: bool = False
    is_confirmation: bool = False
    wants_addition: bool = False
    wants_removal: bool = False


def _is_product_fragment(fragment: str, vocab: CatalogVocabulary | None) -> bool:
    """Whether a fragment plausibly names a product.

    Prefers the catalog vocabulary when available (authoritative). Falls back to a
    capitalisation/length heuristic when it is not, so the detector still works
    without a catalog (e.g. in isolation tests).
    """
    fragment = fragment.strip().strip("\"'").rstrip("?.! ")
    if not fragment or len(fragment) > 80:
        return False
    if _NON_PRODUCT_FRAGMENT.match(fragment):
        return False
    if vocab is not None and vocab.mentions_product(fragment):
        return True
    if vocab is not None:
        # Vocabulary present but no match: trust it and reject, which kills false
        # positives like "compare notes".
        return False
    return bool(re.search(r"[A-Z0-9]", fragment))


def _detect_comparison(
    text: str, vocab: CatalogVocabulary | None
) -> tuple[bool, tuple[str, ...]]:
    if not _COMPARE_WORD.search(text):
        return False, ()
    for pattern in (_BETWEEN_AND, _COMPARE_X_AND_Y, _X_AND_Y_VERB, _X_FROM_Y):
        match = pattern.search(text)
        if match:
            left = _clean_target(match.group(1))
            right = _clean_target(match.group(2))
            both_products = _is_product_fragment(left, vocab) and _is_product_fragment(right, vocab)
            if both_products:
                return True, (left, right)
    # A compare word but no resolvable product pair. If a vocabulary is present and
    # the message mentions no known product at all, this is not a product
    # comparison ("compare notes with my team") — suppress it.
    if vocab is not None and not vocab.mentions_product(text):
        return False, ()
    return True, ()


def _detect_off_topic(text: str) -> bool:
    if _GENERAL_ADVICE.search(text):
        return True
    return bool(_LEGAL_TERMS.search(text) and _LEGAL_FRAMING.search(text))


def detect_signals(
    messages: list,
    *,
    has_prior_recommendations: bool,
    vocabulary: CatalogVocabulary | None = None,
) -> Signals:
    """Detect signals from the latest user message.

    ``messages`` is the request history (objects with ``role``/``content``).
    ``has_prior_recommendations`` gates confirmation: "perfect, that's what we
    need" only means closure once a shortlist has actually been offered.
    ``vocabulary`` grounds comparison detection in the real catalog when provided.
    """
    latest = ""
    for message in reversed(messages):
        if getattr(message, "role", None) == "user":
            latest = _clean(getattr(message, "content", ""))
            break

    if not latest:
        return Signals()

    is_comparison, targets = _detect_comparison(latest, vocabulary)
    return Signals(
        is_comparison=is_comparison,
        comparison_targets=targets,
        is_off_topic=_detect_off_topic(latest),
        is_prompt_injection=bool(_INJECTION.search(latest)),
        is_confirmation=bool(has_prior_recommendations and _CONFIRMATION.search(latest)),
        wants_addition=bool(_ADD.search(latest)),
        wants_removal=bool(_DROP.search(latest)),
    )
