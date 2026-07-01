"""Replay the sample conversations through the *live* engine.

This is the pre-submission check: it runs each trace's real user turns through a
fully wired :class:`ResponseEngine` — the real language model, real retrieval — and
prints, turn by turn, what the agent actually does against what the trace expected
(did it end the conversation on the right turn, did it show a shortlist when the
trace did, how well did the final shortlist recall the gold).

It is a script, not a test, on purpose: it needs a model API key, costs a little to
run, and is not deterministic, so it does not belong in the automated suite. The
offline replay test (`tests/eval/test_trace_replay.py`) is the deterministic version
that runs in CI; this is the human-in-the-loop confirmation that the real model
behaves before submitting.

Usage:
    python -m scripts.replay_traces          # all traces
    python -m scripts.replay_traces C6 C9    # selected traces

Requires the provider key in the environment (see .env.example). Without it, the
model calls fall back to deterministic behaviour and the run still completes — but
the point of this script is to exercise the real model, so set the key.
"""

from __future__ import annotations

import sys
import time

from shl_recommender.api.schemas import Message
from shl_recommender.catalog.loader import load_catalog
from shl_recommender.catalog.vocabulary import build_vocabulary
from shl_recommender.config import settings
from shl_recommender.llm.client import LiteLLMClient
from shl_recommender.response.engine import ResponseEngine
from shl_recommender.retrieval.ranker import LexicalRanker

from scripts.trace_utils import all_gold_urls, messages_up_to, parse_turns

_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Seconds to wait between turns, to stay under a rate-limited free tier. Set via the
# ``--delay N`` argument. Default 0 (no pacing) for a paid tier or a quick check.
_DELAY = 0.0


def _mark(ok: bool) -> str:
    return f"{_GREEN}ok{_RESET}" if ok else f"{_RED}MISMATCH{_RESET}"


def _build_engine() -> tuple[ResponseEngine, list]:
    items = load_catalog(settings.raw_catalog_path)
    retriever = LexicalRanker(items)  # the real retrieval path
    engine = ResponseEngine(
        retriever, LiteLLMClient(), catalog=items, vocabulary=build_vocabulary(items),
        max_recommendations=settings.max_recommendations,
    )
    return engine, items


def _replay_one(engine: ResponseEngine, path: str, name: str) -> None:
    turns = parse_turns(path)
    gold = {u.rstrip("/").lower() for u in all_gold_urls(path)}
    print(f"\n{'=' * 70}\n{name}  ({len(turns)} turns)\n{'=' * 70}")

    final_recs: list = []
    for i, turn in enumerate(turns):
        # Pace the turns to respect a rate-limited free tier (e.g. Gemini free tier is
        # ~5 requests/minute and each turn makes up to two model calls). ``_DELAY`` is
        # read from the module so the caller can tune it.
        if i > 0 and _DELAY:
            time.sleep(_DELAY)
        messages = [Message(role=m["role"], content=m["content"]) for m in messages_up_to(turns, i)]
        resp = engine.respond(messages)

        has_recs = resp.recommendations is not None
        eoc_ok = (i < len(turns) - 1) or (resp.end_of_conversation == turn.end_of_conversation)
        recs_ok = has_recs == turn.shows_recommendations or turn.index in _tolerant_turns(name)

        print(f"\n  Turn {turn.index}: {_DIM}{turn.user_text[:72]}{_RESET}")
        print(f"    reply: {resp.reply[:90]}")
        print(
            f"    recs: {len(resp.recommendations) if has_recs else 0} "
            f"(trace showed: {'yes' if turn.shows_recommendations else 'no'}) {_mark(recs_ok)}"
            f"   | end={resp.end_of_conversation} (trace: {turn.end_of_conversation}) {_mark(eoc_ok)}"
        )
        if i == len(turns) - 1 and has_recs:
            final_recs = resp.recommendations

    if final_recs and gold:
        found = {r.url.rstrip("/").lower() for r in final_recs}
        recall = len(gold & found) / len(gold)
        print(f"\n  final shortlist recall@{len(final_recs)}: {recall:.2f} ({len(gold & found)}/{len(gold)} gold)")


def _tolerant_turns(name: str) -> set[int]:
    # Turns where showing-or-not is a judgement call the live model may differ on
    # without being wrong (e.g. re-showing a list on a refine mid-conversation).
    return set()


def main(argv: list[str]) -> None:
    global _DELAY
    args = list(argv)
    # Optional "--delay N" to pace turns under a rate-limited free tier.
    if "--delay" in args:
        idx = args.index("--delay")
        try:
            _DELAY = float(args[idx + 1])
            del args[idx : idx + 2]
        except (IndexError, ValueError):
            print("usage: --delay <seconds>")
            return

    wanted = {a.upper() for a in args}
    engine, _ = _build_engine()
    paths = sorted(
        (settings.project_root / "data" / "traces").glob("*.md"),
        key=lambda p: (len(p.name), p.name),
    )
    for path in paths:
        name = path.stem
        if wanted and name.upper() not in wanted:
            continue
        _replay_one(engine, str(path), name)
    print()


if __name__ == "__main__":
    main(sys.argv[1:])
