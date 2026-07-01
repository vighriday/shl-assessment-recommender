"""Measure Recall@10 against the sample conversations.

The scoreboard for retrieval. For each trace it builds the state the agent would
hold at commit time — the user's own words, plus the categories/skills/languages
the user stated — retrieves the top 10 in-scope items, and computes the fraction
of the trace's gold items that appear. Mean Recall@10 is reported.

The per-trace hints below are the requirements the *user* states in each trace
(role words, requested categories, languages). They are not the answers; they are
what a correct understanding step would extract from the conversation, supplied
here so retrieval can be measured without running the model.
"""

from __future__ import annotations

import glob
import os
import re

from shl_recommender.catalog.loader import load_catalog
from shl_recommender.conversation.state import ConversationState
from shl_recommender.retrieval.ranker import LexicalRanker
from shl_recommender.config import settings

_URL = re.compile(r"https?://www\.shl\.com[^\s>|)]+")


def _gold_urls(path: str) -> set[str]:
    urls = set()
    for line in open(path, encoding="utf-8"):
        if line.count("|") >= 6:
            match = _URL.search(line)
            if match:
                urls.add(match.group(0).rstrip("/").lower())
    return urls


def _user_text(path: str) -> str:
    """Concatenate all user turns — the words the agent has heard by the end."""
    text = open(path, encoding="utf-8").read()
    blocks = re.findall(r"\*\*User\*\*\s*\n+((?:>.*\n?)+)", text)
    quoted = " ".join(re.sub(r"^>\s?", "", line) for b in blocks for line in b.splitlines())
    return re.sub(r"\s+", " ", quoted).strip()


# Categories / languages the user explicitly asks for, per trace.
_HINTS = {
    "C1": dict(test_type_preferences=("personality",), seniority="executive"),
    "C2": dict(must_have_skills=("Rust", "networking", "Linux"), seniority="senior"),
    "C3": dict(test_type_preferences=("simulation",), languages=("English (USA)",)),
    "C4": dict(test_type_preferences=("cognitive", "numerical"), seniority="graduate",
               must_have_skills=("finance", "statistics")),
    "C5": dict(test_type_preferences=("personality", "competency"), domain="sales"),
    "C6": dict(test_type_preferences=("personality",), domain="manufacturing"),
    "C7": dict(must_have_skills=("HIPAA", "medical"), languages=("Spanish",)),
    "C8": dict(must_have_skills=("Excel", "Word"), test_type_preferences=("simulation",)),
    "C9": dict(must_have_skills=("Java", "Spring", "SQL", "AWS", "Docker"), seniority="senior"),
    "C10": dict(test_type_preferences=("cognitive", "personality", "situational judgement"),
                seniority="graduate"),
}


def main() -> None:
    items = load_catalog(settings.raw_catalog_path)
    retriever = LexicalRanker(items)

    total = 0.0
    count = 0
    print(f"{'trace':6} {'recall@10':>10}  {'found/gold':>10}")
    for path in sorted(glob.glob(str(settings.project_root / "data" / "traces" / "*.md")),
                       key=lambda p: (len(p), p)):
        name = os.path.basename(path).replace(".md", "")
        gold = _gold_urls(path)
        if not gold:
            continue
        state = ConversationState(query_text=_user_text(path), **_HINTS.get(name, {}))
        top = retriever.retrieve(state, top_k=10)
        found = {s.item.url.rstrip("/").lower() for s in top}
        hit = len(gold & found)
        recall = hit / len(gold)
        total += recall
        count += 1
        print(f"{name:6} {recall:>10.2f}  {hit:>4}/{len(gold):<4}")

    print("-" * 30)
    print(f"{'MEAN':6} {total / count:>10.3f}")


if __name__ == "__main__":
    main()
