"""Response assembly: turn the conversation into the API response.

Brings the phases together — state, policy, retrieval, and the language model — into
one :class:`ResponseEngine` that produces a validated :class:`ChatResponse` per
turn. Code owns the recommendation list and the turn's outcome; the model owns the
reply's wording, with a fallback for every mode.
"""

from .engine import ResponseEngine
from .reply import ReplyWriter
from .shortlist import build_recommendations, recover_prior_shortlist

__all__ = [
    "build_recommendations",
    "recover_prior_shortlist",
    "ReplyWriter",
    "ResponseEngine",
]
