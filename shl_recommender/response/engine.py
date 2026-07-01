"""The response engine: one conversation turn, end to end.

This is where every prior phase converges. Given the message history, it produces
the :class:`ChatResponse` the API returns, by running the turn through the pipeline
each phase built:

    reconstruct state  ->  decide policy  ->  retrieve (if committing)  ->
    build shortlist (code)  ->  write reply (model + fallback)  ->  ChatResponse

The division of labour is the project's core principle made concrete in one place:

* **Code owns the contract.** The policy decides the mode and whether the turn
  commits a shortlist; the shortlist builder produces the 1..10 (or null) list with
  every field copied from the catalog; ``end_of_conversation`` comes straight from
  the policy. None of these can be altered by the language model.
* **The model owns the language.** It supplies the understanding that feeds the
  state, and it phrases the reply — with a deterministic fallback for every mode, so
  a model outage degrades the wording, never the correctness.

The engine holds no per-request state (the service is stateless); it is constructed
once with its dependencies and called per turn. Keeping it free of the web framework
and of the startup wiring makes the whole turn testable without a server.
"""

from __future__ import annotations

import re

from shl_recommender.api.schemas import ChatResponse
from shl_recommender.catalog.models import CatalogItem
from shl_recommender.catalog.vocabulary import CatalogVocabulary
from shl_recommender.conversation.extractor import reconstruct_state
from shl_recommender.conversation.policy import PolicyDecision, decide
from shl_recommender.conversation.state import ConversationState, Mode
from shl_recommender.llm.client import LLMClient
from shl_recommender.observability import get_logger
from shl_recommender.response.reply import ReplyWriter
from shl_recommender.response.shortlist import (
    build_recommendations,
    comparison_facts,
    recover_prior_shortlist,
)
from shl_recommender.response.trace import TurnTrace, build_trace
from shl_recommender.retrieval.ranker import LexicalRanker
from shl_recommender.retrieval.types import ScoredItem

log = get_logger(__name__)


class ResponseEngine:
    """Assembles a full :class:`ChatResponse` for one turn.

    Dependencies are injected rather than constructed here so the engine can be
    unit-tested with a fake model and a small catalog, and so startup owns the
    lifecycle of the expensive objects (the retriever, the embedding model).
    """

    def __init__(
        self,
        retriever: LexicalRanker,
        client: LLMClient,
        *,
        catalog: list[CatalogItem] | None = None,
        vocabulary: CatalogVocabulary | None = None,
        max_recommendations: int = 10,
    ) -> None:
        self._retriever = retriever
        self._client = client
        # Needed to recover a previously-offered shortlist on a confirmation turn.
        # Defaults to the retriever's own items so callers do not pass it twice.
        self._catalog = catalog if catalog is not None else retriever._items
        self._vocabulary = vocabulary
        self._reply = ReplyWriter(client)
        self._max_recommendations = max_recommendations

    def respond(self, messages: list) -> ChatResponse:
        """Run one turn and return the response to send to the client."""
        response, _ = self.respond_with_trace(messages, trace=False)
        return response

    def respond_with_trace(
        self, messages: list, *, trace: bool = True
    ) -> tuple[ChatResponse, TurnTrace | None]:
        """Run one turn, optionally also returning the turn's reasoning trace.

        The trace is strictly additive: it is assembled from the same state, decision,
        and scores the turn already produced, so requesting it cannot change the
        contract fields. When ``trace`` is False the trace is not built and ``None`` is
        returned in its place — this is the path :meth:`respond` uses.
        """
        state = reconstruct_state(messages, self._client, vocabulary=self._vocabulary)
        decision = decide(state)

        scored, recommendations = self._shortlist_for(decision, state, messages)

        # On a comparison turn, resolve the named products to catalog items and hand
        # their real attributes to the reply writer, so the comparison is grounded in
        # catalog facts rather than the model's memory.
        facts = self._comparison_facts(decision, state)

        reply, reply_from_model = self._reply.write_traced(
            decision,
            state,
            messages=messages,
            recommendation_count=len(recommendations) if recommendations else 0,
            comparison_facts=facts,
        )

        log.info(
            "turn handled",
            extra={
                "mode": decision.mode.value,
                "reason": decision.reason,
                "recommendations": len(recommendations) if recommendations else 0,
                "end_of_conversation": decision.end_of_conversation,
            },
        )

        response = ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=decision.end_of_conversation,
        )

        turn_trace = (
            build_trace(
                state,
                decision,
                scored,
                reply_from_model=reply_from_model,
                comparison_facts_resolved=facts is not None,
            )
            if trace
            else None
        )
        return response, turn_trace

    def _shortlist_for(
        self, decision: PolicyDecision, state: ConversationState, messages: list
    ) -> tuple[list[ScoredItem], list | None]:
        """Retrieve and build the shortlist when the turn commits one.

        Returns ``(scored, recommendations)`` — the ranked candidates (for the trace)
        and the built shortlist. Only RECOMMEND and REFINE commit a shortlist. For
        every other mode (CLARIFY, COMPARE, REFUSE) both are empty/``None``: the schema
        represents no shortlist as JSON ``null``, and retrieval is skipped entirely on
        those turns — it would be wasted work and the model is not consulted for the
        list anyway.
        """
        if not decision.commits_shortlist:
            return [], None

        # On a confirmation close the user accepted the shortlist already offered, so
        # re-show exactly those items rather than retrieving on a bare "yes" (which
        # carries no requirements). Recover them from the history; if none can be
        # recovered, fall through to normal retrieval so we still return something.
        if decision.reason == "user_confirmed":
            recovered = recover_prior_shortlist(messages, self._catalog)
            if recovered is not None:
                return [], recovered[: self._max_recommendations]

        scored = self._retriever.retrieve(state, top_k=self._max_recommendations)
        return scored, build_recommendations(scored, limit=self._max_recommendations)

    def _comparison_facts(
        self, decision: PolicyDecision, state: ConversationState
    ) -> str | None:
        """Resolve the compared product names to catalog items and format their facts.

        Only runs on a comparison turn. Each named target is resolved to its best
        catalog match by a lexical lookup over the target text (reusing the retriever,
        so fuzzy names like "Safety & Dependability 8.0" resolve without bespoke
        matching). Duplicates are removed. Returns ``None`` if nothing resolves, so the
        reply keeps to safe framing rather than an empty facts block.
        """
        if decision.mode is not Mode.COMPARE or not state.comparison_targets:
            return None

        resolved = []
        seen: set[str] = set()
        for target in state.comparison_targets:
            match = self._resolve_one(target)
            if match is not None and match.entity_id not in seen:
                seen.add(match.entity_id)
                resolved.append(match)

        return comparison_facts(resolved)

    def _resolve_one(self, target: str):
        """Best catalog item for a single comparison target string, or ``None``.

        Direct name/code overlap first (the reliable signal for a named product), then
        a lexical fallback for looser references. The retriever is *not* reused here
        because its hiring-oriented scoring (staple defaults, category boosts) can pull
        an unrelated general measure to the top of a bare product name — wrong for a
        pure name lookup.
        """
        target = (target or "").strip()
        if not target:
            return None

        target_lower = target.lower()
        target_tokens = {t for t in re.findall(r"[a-z0-9+]+", target_lower) if len(t) > 1}
        if not target_tokens:
            return None

        best = None
        best_score = 0.0
        for item in self._catalog:
            if not item.in_scope:
                continue
            name_lower = item.name.lower()
            # Strong signal: the target text appears in the name (or vice versa).
            if target_lower in name_lower or name_lower in target_lower:
                return item
            # Otherwise score by shared word overlap, normalised by target length so a
            # tight match beats an incidental one.
            name_tokens = set(re.findall(r"[a-z0-9+]+", name_lower))
            overlap = len(target_tokens & name_tokens)
            if overlap:
                score = overlap / len(target_tokens)
                if score > best_score:
                    best_score, best = score, item

        # Require a majority of the target's words to match, so a weak coincidental
        # overlap does not resolve to the wrong product.
        return best if best_score >= 0.5 else None
