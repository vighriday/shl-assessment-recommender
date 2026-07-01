"""Lexical retrieval over the catalog with TF-IDF.

Two vectorisers, fit once over the catalog's ``search_text``:

* a **word** vectoriser for ordinary language ("customer service", "graduate");
* a **character n-gram** vectoriser for product codes and odd tokens that word
  tokenisation fragments or drops — ``OPQ32r``, ``G+``, ``.NET``, ``SVAR``. This
  is what lets an exact product-code query find its item.

A query is scored against both and the two cosine similarities are combined. The
character signal is weighted a little lower by default because it also fires on
incidental substring overlap; the weight is adjustable.

Fitting happens at construction (startup), so per-request retrieval is just two
sparse matrix-vector products — fast and predictable within the latency budget.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from shl_recommender.catalog.models import CatalogItem
from shl_recommender.retrieval.types import ScoredItem


class LexicalRetriever:
    """TF-IDF retriever combining word-level and character-level matching."""

    def __init__(
        self,
        items: list[CatalogItem],
        *,
        char_weight: float = 0.4,
        min_score: float = 1e-6,
    ) -> None:
        if not items:
            raise ValueError("cannot build a retriever over an empty catalog")
        self._items = items
        self._char_weight = char_weight
        self._min_score = min_score

        corpus = [item.search_text for item in items]

        # Word vectoriser: unigrams and bigrams, English stop words removed.
        self._word_vectoriser = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
        )
        self._word_matrix = self._word_vectoriser.fit_transform(corpus)

        # Character vectoriser: 3-5 char n-grams within word boundaries, which
        # captures codes and partial tokens without exploding across whitespace.
        self._char_vectoriser = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=True,
            sublinear_tf=True,
        )
        self._char_matrix = self._char_vectoriser.fit_transform(corpus)

    def search(self, query: str, *, top_k: int = 30) -> list[ScoredItem]:
        """Return the top ``top_k`` items for ``query`` by combined TF-IDF score.

        An empty or whitespace query yields no results rather than an error, so
        callers do not need to special-case it.
        """
        query = (query or "").strip()
        if not query:
            return []

        word_scores = cosine_similarity(
            self._word_vectoriser.transform([query]), self._word_matrix
        )[0]
        char_scores = cosine_similarity(
            self._char_vectoriser.transform([query]), self._char_matrix
        )[0]
        combined = word_scores + self._char_weight * char_scores

        ranked = sorted(
            (ScoredItem(item=self._items[i], score=float(combined[i]))
             for i in range(len(self._items))),
            key=lambda scored: scored.score,
            reverse=True,
        )
        return [scored for scored in ranked if scored.score > self._min_score][:top_k]
