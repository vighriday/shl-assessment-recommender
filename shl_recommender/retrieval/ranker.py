"""Lexical retrieval and transparent ranking.

Takes the lexical (TF-IDF) retriever's candidates and ranks them with a readable
weighted sum of named signals:

* lexical similarity (exact skills, product codes, role/context words);
* category intent — the assessment categories the user asked for vs the item's;
* language match — a requested language that the item offers;
* job-level match — a stated seniority that the item targets;
* an exact name/skill boost — a query token that appears in the item name.

Every term is inspectable, so a shortlist can always be explained and debugged.
The retriever scores are min-max normalised to a common [0, 1] scale first so the
weighted sum combines comparable quantities.

Only in-scope items are ever ranked, which is where the "Individual Test
Solutions only" restriction is finally enforced in the request path.

A sentence-embedding ("semantic") stage was built and measured against the ten
sample conversations. It changed Recall@10 on none of them — its single unique
recovery was cancelled by an equal displacement — so it was removed rather than
carried as unfalsifiable weight and an extra (heavy, version-fragile) dependency.
The retrieval story is therefore deliberately lexical: transparent, reproducible,
and no worse on the evidence. See ``docs/retrieval_design.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shl_recommender.catalog.models import CatalogItem
from shl_recommender.conversation.state import ConversationState
from shl_recommender.retrieval.lexical import LexicalRetriever
from shl_recommender.retrieval.types import ScoredItem

# Map free-text category words a user might use onto the catalog codes, so
# "cognitive"/"aptitude" -> A, "personality" -> P, and so on.
_CATEGORY_WORD_TO_CODE = {
    "knowledge": "K", "skill": "K", "skills": "K", "technical": "K",
    "personality": "P", "behaviour": "P", "behavior": "P", "behavioural": "P",
    "cognitive": "A", "aptitude": "A", "ability": "A", "reasoning": "A", "numerical": "A",
    "competency": "C", "competencies": "C",
    "biodata": "B", "situational": "B", "judgement": "B", "judgment": "B", "sjt": "B",
    "simulation": "S", "simulations": "S", "sim": "S",
    "development": "D", "360": "D",
    "exercise": "E", "exercises": "E",
}

_WORD = re.compile(r"[A-Za-z0-9+#.]+")

# Tokens that appear in many product names and so carry no exact-hit signal on
# their own; excluded when deciding whether a name is "made of" a query token.
_NAME_BOILERPLATE = frozenset({"new", "test", "assessment", "level", "the", "and"})

# A skill token is treated as *distinctive* — worth an exact-name bonus when the
# user explicitly required it — only if it names at most this many catalog
# products. The intent is to separate specific technologies ("AWS", "Docker",
# "Spring", "HIPAA": each names one or a few items) from broad skills ("Java",
# which names ~nine Java products); rewarding the latter would just re-promote a
# whole family of near-duplicates. The cut is not knife-edge: any value from a
# few up to the high single digits gives the same result on the sample set,
# because real skill-token frequencies cluster at the extremes (1–3 vs 9+).
_MAX_SKILL_NAME_DF = 8


def _name_tokens(name: str) -> set[str]:
    """Significant (non-boilerplate) tokens of a product name, lower-cased."""
    return {
        t for t in _WORD.findall(name.lower())
        if len(t) >= 2 and t not in _NAME_BOILERPLATE
    }


def _family_key(name: str) -> str:
    """A coarse product-family key from a name's leading significant words.

    "OPQ Candidate Report 2.0" and "OPQ Profile Report" share "opq report";
    "SVAR - Spoken English (US)" and "SVAR - Spoken Spanish" share "svar spoken".
    Uses the first two meaningful tokens plus a following 'report'/'simulation'
    marker so families collapse but distinct products do not.
    """
    tokens = [t for t in _WORD.findall(name.lower()) if t not in {"new", "the", "and", "of"}]
    if not tokens:
        return name.lower()
    key = tokens[:2]
    for marker in ("report", "simulation", "solution"):
        if marker in tokens and marker not in key:
            key.append(marker)
            break
    return " ".join(key)


# Staple assessments: general-purpose measures that a competent consultant adds by
# default, and that recur across the sample shortlists (OPQ32r in eight of ten,
# Verify G+ in three). They are rarely named in the query, so text retrieval alone
# misses them; we inject them as candidates when their dimension is relevant and
# let the ranker place them. Keyed by entity_id with the codes that trigger them.
_STAPLES = (
    # OPQ32r — the default personality/behaviour measure for professional hires.
    ("720", frozenset({"P"})),
    # Verify G+ — the default cognitive/ability measure.
    ("3971", frozenset({"A"})),
)


@dataclass(frozen=True)
class RankingWeights:
    """Weights for the ranking signals. Tuned against the sample conversations."""

    lexical: float = 1.0
    category: float = 0.6
    language: float = 0.3
    job_level: float = 0.3
    # A tight name match (a query token that is a large fraction of an item's own name)
    # is a strong exact-hit signal — "Basic Statistics", "MS Word" — that a short,
    # focused product should not lose to a longer, incidentally-overlapping neighbour.
    # Weighted so such a match wins the last shortlist slots; the effect plateaus above
    # ~2.0 (it is not knife-edge tuning) and, being orthogonal to the category flag, it
    # never promotes broad multi-category noise.
    name_boost: float = 2.0
    # An additive bonus for an item whose name contains a *distinctive* skill the
    # user explicitly required — an "AWS", "Docker", "Spring" that names only a
    # handful of catalog products (see ``_MAX_SKILL_NAME_DF``). Unlike ``name_boost``
    # (a fraction of the name), this is not diluted by a verbose official product
    # name, so "Amazon Web Services (AWS) Development" is rewarded for the required
    # "AWS" as much as a short "Docker" is. It is deliberately smaller than
    # ``name_boost``/``staple`` so it breaks ties among genuinely relevant candidates
    # rather than dominating; the effect is flat across a wide range (~0.5–1.4).
    skill_name: float = 0.8
    # Large, because a relevant staple is a default the shortlist should almost
    # always include; it must outrank incidental text matches, not merely nudge.
    staple: float = 3.0


def _normalise(scored: list[ScoredItem]) -> dict[str, float]:
    """Min-max normalise scores to [0, 1], keyed by entity_id."""
    if not scored:
        return {}
    values = [s.score for s in scored]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= 0:
        return {s.item.entity_id: 1.0 for s in scored}
    return {s.item.entity_id: (s.score - lo) / span for s in scored}


def _requested_codes(state: ConversationState) -> set[str]:
    """Category codes the user explicitly asked for, from their preference words."""
    codes: set[str] = set()
    for phrase in state.test_type_preferences:
        for token in _WORD.findall(phrase.lower()):
            if token in _CATEGORY_WORD_TO_CODE:
                codes.add(_CATEGORY_WORD_TO_CODE[token])
    return codes


def _query_tokens(state: ConversationState) -> set[str]:
    """Meaningful tokens from the query and skills, for the exact-name boost."""
    tokens: set[str] = set()
    sources = [state.query_text, *state.must_have_skills, *state.optional_skills]
    for text in sources:
        for token in _WORD.findall((text or "").lower()):
            if len(token) >= 3:
                tokens.add(token)
    return tokens


class LexicalRanker:
    """Retrieve lexical candidates and rank them transparently.

    Named for what it is: a lexical retriever feeding a transparent weighted-sum
    ranker. (An embedding stage was trialled and dropped for adding no measurable
    recall — see the module docstring.)
    """

    def __init__(
        self,
        items: list[CatalogItem],
        *,
        lexical: LexicalRetriever | None = None,
        weights: RankingWeights | None = None,
    ) -> None:
        self._items = items
        self._by_id = {item.entity_id: item for item in items}
        self._lexical = lexical or LexicalRetriever(items)
        self._weights = weights or RankingWeights()
        # How many product names each name-token appears in, computed once. Used to
        # tell a distinctive skill (names a few products) from a broad one (names
        # many), for the explicit-skill-in-name bonus. See ``_MAX_SKILL_NAME_DF``.
        self._name_token_df: dict[str, int] = {}
        for item in items:
            for token in _name_tokens(item.name):
                self._name_token_df[token] = self._name_token_df.get(token, 0) + 1

    def retrieve(
        self, state: ConversationState, *, candidate_k: int = 60, top_k: int = 10
    ) -> list[ScoredItem]:
        """Return up to ``top_k`` in-scope items ranked for the current state."""
        # Search over the user's words plus the skills the understanding step
        # pulled out. A skill named in a long JD ("HIPAA", "medical") may be diluted
        # in the raw text; repeating the extracted skills makes those items
        # retrievable instead of being missed entirely.
        query = self._search_query(state)

        lexical_hits = self._lexical.search(query, top_k=candidate_k)
        lexical_norm = _normalise(lexical_hits)

        requested_codes = _requested_codes(state)

        # Candidate ids from the lexical retriever, in scope only, plus the staple
        # defaults when their dimension is relevant to this hire.
        candidate_ids = {
            eid for eid in lexical_norm.keys() if self._by_id[eid].in_scope
        }
        candidate_ids |= self._staple_candidates(state, requested_codes)
        if not candidate_ids:
            return []

        query_tokens = _query_tokens(state)
        seniority = (state.seniority or "").lower()
        wanted_languages = {lang.lower() for lang in state.languages}
        skill_tokens = self._distinctive_skill_tokens(state)

        staple_ids = self._staple_candidates(state, requested_codes)

        ranked: list[ScoredItem] = []
        for eid in candidate_ids:
            item = self._by_id[eid]
            score = (
                self._weights.lexical * lexical_norm.get(eid, 0.0)
                + self._weights.category * self._category_score(item, requested_codes)
                + self._weights.language * self._language_score(item, wanted_languages)
                + self._weights.job_level * self._job_level_score(item, seniority)
                + self._weights.name_boost * self._name_score(item, query_tokens)
            )
            # A distinctive required skill appearing in the item's name is a strong
            # exact hit that a verbose official name would otherwise dilute; give it
            # a flat additive bonus so, e.g., "Amazon Web Services (AWS) Development"
            # is not out-ranked by shorter neighbours when the user asked for "AWS".
            if skill_tokens and skill_tokens & _name_tokens(item.name):
                score += self._weights.skill_name
            # A relevant staple gets a floor score so it survives into the shortlist
            # even when the query never named it (it is a default, not a match).
            if eid in staple_ids:
                score += self._weights.staple
            ranked.append(ScoredItem(item=item, score=score))

        ranked.sort(key=lambda s: s.score, reverse=True)
        return self._diversify(ranked, top_k=top_k)

    def _distinctive_skill_tokens(self, state: ConversationState) -> set[str]:
        """The explicitly-required skill tokens that distinctively name a product.

        A token qualifies when the user named it as a required or optional skill
        *and* it appears in at most ``_MAX_SKILL_NAME_DF`` product names — i.e. it
        picks out a specific technology ("aws", "docker", "spring", "hipaa") rather
        than a broad skill shared by a whole family ("java"). Broad skills are left
        to ordinary lexical/name scoring so the bonus never promotes near-duplicates
        wholesale.
        """
        tokens: set[str] = set()
        for skill in (*state.must_have_skills, *state.optional_skills):
            for token in _WORD.findall(skill.lower()):
                if len(token) < 2:
                    continue
                df = self._name_token_df.get(token, 0)
                if 0 < df <= _MAX_SKILL_NAME_DF:
                    tokens.add(token)
        return tokens

    @staticmethod
    def _diversify(ranked: list[ScoredItem], *, top_k: int, per_family: int = 2) -> list[ScoredItem]:
        """Take the top items while limiting near-duplicates from one product family.

        A catalog family like the many "OPQ ... Report" products, or the SVAR
        spoken-language variants, can otherwise fill the whole shortlist and crowd
        out other relevant items. Grouping by a name prefix and capping how many of
        each may appear keeps the shortlist varied without a heavy reranker. Highest
        scores are still preferred; only surplus siblings are deferred.
        """
        selected: list[ScoredItem] = []
        deferred: list[ScoredItem] = []
        family_counts: dict[str, int] = {}
        for scored in ranked:
            family = _family_key(scored.item.name)
            if family_counts.get(family, 0) < per_family:
                selected.append(scored)
                family_counts[family] = family_counts.get(family, 0) + 1
            else:
                deferred.append(scored)
            if len(selected) == top_k:
                return selected
        # If capping left us short of top_k, backfill with the highest deferred.
        for scored in deferred:
            if len(selected) == top_k:
                break
            selected.append(scored)
        return selected[:top_k]

    @staticmethod
    def _search_query(state: ConversationState) -> str:
        """The text handed to the retrievers: user words plus extracted signals.

        Skills, role and requested categories are appended so an item named after a
        skill is retrievable even when that skill is buried in a long brief. The raw
        query is kept first so ordinary-language matching is unaffected.
        """
        parts = [state.query_text]
        parts.extend(state.must_have_skills)
        parts.extend(state.optional_skills)
        if state.role:
            parts.append(state.role)
        parts.extend(state.test_type_preferences)
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()

    def _staple_candidates(
        self, state: ConversationState, requested_codes: set[str]
    ) -> set[str]:
        """Which staple defaults are relevant to this hire.

        A staple applies when the user asked for its category, or — for a general
        professional hire (a role or named skills, without a narrowed set of
        requested categories) — the personality default applies, mirroring how the
        sample agent adds OPQ32r by default for professional roles.
        """
        is_professional_hire = bool(state.role or state.must_have_skills)
        relevant: set[str] = set()
        for entity_id, trigger_codes in _STAPLES:
            if entity_id not in self._by_id or not self._by_id[entity_id].in_scope:
                continue
            asked_for_category = bool(trigger_codes & requested_codes)
            # The personality (OPQ32r) and cognitive (Verify G+) defaults apply to
            # any professional hire (a role or named skills). They are additive
            # defaults — the sample agent adds OPQ32r even when the user asked only
            # for, say, cognitive — so a requested category does not suppress them.
            general_default = is_professional_hire
            if asked_for_category or general_default:
                relevant.add(entity_id)
        return relevant

    @staticmethod
    def _category_score(item: CatalogItem, requested_codes: set[str]) -> float:
        if not requested_codes:
            return 0.0
        item_codes = set(item.test_type.split(","))
        return 1.0 if item_codes & requested_codes else 0.0

    @staticmethod
    def _language_score(item: CatalogItem, wanted_languages: set[str]) -> float:
        if not wanted_languages:
            return 0.0
        item_languages = " ".join(item.languages).lower()
        return 1.0 if any(lang in item_languages for lang in wanted_languages) else 0.0

    @staticmethod
    def _job_level_score(item: CatalogItem, seniority: str) -> float:
        if not seniority or not item.job_levels:
            return 0.0
        levels = " ".join(item.job_levels).lower()
        # Match on the meaningful seniority word (graduate, manager, executive, ...).
        for word in _WORD.findall(seniority):
            if len(word) >= 4 and word in levels:
                return 1.0
        return 0.0

    @staticmethod
    def _name_score(item: CatalogItem, query_tokens: set[str]) -> float:
        """Reward items whose name is a tight match to a query skill token.

        A short, focused product ("SQL (New)", "MS Excel (New)") whose name is
        mostly made of matched query tokens is a strong exact hit and should beat a
        longer, incidentally-overlapping variant. So the score is the share of the
        item's own name tokens that the query matched, not a flat flag.
        """
        if not query_tokens:
            return 0.0
        name_tokens = _name_tokens(item.name)
        if not name_tokens:
            return 0.0
        matched = name_tokens & query_tokens
        if not matched:
            return 0.0
        # Fraction of the name that the query accounts for: a two-word name fully
        # matched scores ~1.0; one match in a long name scores low.
        return len(matched) / len(name_tokens)
