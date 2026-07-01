"""Recall@10 regression guard.

Locks in the measured retrieval quality against the ten sample conversations so a
later change cannot silently degrade it. Retrieval is lexical (TF-IDF) with a
transparent ranker — there is no embedding model to load — so this test has no
optional dependency and **never skips**: the number is enforced on every run.

The floor (0.75) sits a little below the current measured mean (0.809) to tolerate
tokeniser/version noise without inviting regressions. The exact current mean is
reported by ``python -m scripts.measure_recall``.
"""

from __future__ import annotations

import glob
import os

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.config import settings
from shl_recommender.conversation.state import ConversationState
from shl_recommender.retrieval.ranker import LexicalRanker

# Reuse the measurement harness' gold parsing and per-trace hints so the test and the
# scoreboard measure the exact same thing.
from scripts.measure_recall import _HINTS, _gold_urls, _user_text

# Current measured mean is ~0.809; guard a floor a little beneath it.
_MEAN_FLOOR = 0.75


def test_mean_recall_at_10_meets_floor():
    retriever = LexicalRanker(load_catalog(settings.raw_catalog_path))
    total = 0.0
    count = 0
    trace_dir = settings.project_root / "data" / "traces"
    for path in glob.glob(str(trace_dir / "*.md")):
        name = os.path.basename(path).replace(".md", "")
        gold = _gold_urls(path)
        if not gold:
            continue
        state = ConversationState(query_text=_user_text(path), **_HINTS.get(name, {}))
        found = {s.item.url.rstrip("/").lower() for s in retriever.retrieve(state, top_k=10)}
        total += len(gold & found) / len(gold)
        count += 1
    assert count == 10
    mean = total / count
    assert mean >= _MEAN_FLOOR, f"mean recall@10 {mean:.3f} below floor {_MEAN_FLOOR}"
