"""Run a single conversational input through the real decision pipeline.

Both the metamorphic laws (A) and the judge (B) need the same thing: given a piece of
user text (and optionally a prior-shortlist context), what does the agent decide? This
module provides that one seam — ``probe_decision`` — so the two evaluators exercise the
identical path the live service uses (understanding → signals → policy), not a
re-implementation of it.

It is deliberately independent of the web layer and of retrieval: the judgement under
test is the *decision* (which mode, ready or not), not which items come back. Keeping it
this narrow makes the laws fast and unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

from shl_recommender.api.schemas import Message
from shl_recommender.catalog.vocabulary import CatalogVocabulary
from shl_recommender.conversation.extractor import reconstruct_state
from shl_recommender.conversation.policy import decide
from shl_recommender.conversation.state import Mode
from shl_recommender.llm.client import LLMClient


@dataclass(frozen=True)
class TurnProbe:
    """The decision the agent reached for one probed input, flattened for assertions."""

    text: str
    mode: Mode
    reason: str
    ready_to_recommend: bool | None
    commits_shortlist: bool
    end_of_conversation: bool
    # A couple of extracted signals worth asserting on directly.
    is_refusal: bool
    is_comparison: bool

    @property
    def clarifies(self) -> bool:
        return self.mode is Mode.CLARIFY

    @property
    def recommends(self) -> bool:
        return self.mode is Mode.RECOMMEND


# A synthetic assistant turn that presents a shortlist, used when a probe needs to
# simulate "a shortlist was already offered" (for confirmation/refine laws). It carries
# a real catalog URL so the same detector the live service uses fires.
_PRIOR_SHORTLIST = (
    "Here is a shortlist:\n"
    "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/"
)


def probe_decision(
    text: str,
    client: LLMClient,
    *,
    vocabulary: CatalogVocabulary | None = None,
    with_prior_shortlist: bool = False,
) -> TurnProbe:
    """Run one user turn through understanding + policy and flatten the decision.

    ``with_prior_shortlist`` prepends a synthetic assistant turn that already offered a
    shortlist, so laws about confirmation and refinement can be exercised.
    """
    messages: list[Message] = []
    if with_prior_shortlist:
        messages.append(Message(role="user", content="earlier request"))
        messages.append(Message(role="assistant", content=_PRIOR_SHORTLIST))
    messages.append(Message(role="user", content=text))

    state = reconstruct_state(messages, client, vocabulary=vocabulary)
    decision = decide(state)
    return TurnProbe(
        text=text,
        mode=decision.mode,
        reason=decision.reason,
        ready_to_recommend=state.ready_to_recommend,
        commits_shortlist=decision.commits_shortlist,
        end_of_conversation=decision.end_of_conversation,
        is_refusal=decision.mode is Mode.REFUSE,
        is_comparison=state.is_comparison,
    )
