"""Parsing helpers for the sample conversation traces.

The traces are Markdown transcripts (``data/traces/C*.md``). Both the offline replay
tests and the live replay script need to read them the same way, so the parsing
lives here in one place: turn boundaries, each turn's user text, whether the agent
showed a shortlist that turn, the stated ``end_of_conversation`` flag, and the gold
catalog URLs.

The parsers are intentionally forgiving about surrounding prose — they key off the
transcript's fixed markers (``### Turn``, ``**User**``, the recommendation table's
pipe columns, the ``end_of_conversation`` line) rather than any exact wording, so a
trace's narrative text can vary without breaking parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A catalog product URL as it appears in the transcripts (angle-bracketed, inside a
# pipe-delimited table row).
_URL = re.compile(r"https?://www\.shl\.com[^\s>|)]+")
# Split the transcript into per-turn chunks on the "### Turn N" headings.
_TURN_SPLIT = re.compile(r"^### Turn\b.*$", re.MULTILINE)
# The user's quoted message inside a turn: a run of "> ..." lines after **User**.
_USER_BLOCK = re.compile(r"\*\*User\*\*\s*\n+((?:>.*\n?)+)")
# The stated end_of_conversation flag for a turn.
_EOC = re.compile(r"end_of_conversation.*?\*\*(true|false)\*\*", re.IGNORECASE)


@dataclass(frozen=True)
class TraceTurn:
    """One turn of a trace: what the user said and what the agent's response was."""

    index: int
    user_text: str
    shows_recommendations: bool
    end_of_conversation: bool
    gold_urls: frozenset[str]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _user_text_of(chunk: str) -> str:
    match = _USER_BLOCK.search(chunk)
    if not match:
        return ""
    lines = match.group(1).splitlines()
    return _clean(" ".join(re.sub(r"^>\s?", "", line) for line in lines))


def _gold_urls_of(chunk: str) -> frozenset[str]:
    urls: set[str] = set()
    for line in chunk.splitlines():
        # Recommendation rows are the only 6+ pipe lines that carry a product URL.
        if line.count("|") >= 6:
            match = _URL.search(line)
            if match:
                urls.add(match.group(0).rstrip("/").lower())
    return frozenset(urls)


def parse_turns(path: str) -> list[TraceTurn]:
    """Parse a trace file into its ordered turns."""
    text = open(path, encoding="utf-8").read()
    # Drop everything before the first turn heading (the "## Conversation" preamble).
    chunks = _TURN_SPLIT.split(text)[1:]
    turns: list[TraceTurn] = []
    for i, chunk in enumerate(chunks, start=1):
        user = _user_text_of(chunk)
        if not user:
            continue
        gold = _gold_urls_of(chunk)
        eoc_match = _EOC.search(chunk)
        end = bool(eoc_match) and eoc_match.group(1).lower() == "true"
        turns.append(
            TraceTurn(
                index=i,
                user_text=user,
                shows_recommendations=bool(gold),
                end_of_conversation=end,
                gold_urls=gold,
            )
        )
    return turns


def all_gold_urls(path: str) -> frozenset[str]:
    """Every gold URL anywhere in the trace (union across turns)."""
    urls: set[str] = set()
    for turn in parse_turns(path):
        urls |= turn.gold_urls
    return frozenset(urls)


def messages_up_to(turns: list[TraceTurn], upto: int) -> list[dict]:
    """Build the chat history the agent would see at ``turns[upto]``.

    Includes every user turn up to and including ``upto``. Prior agent turns are
    reconstructed minimally: for a turn where the agent showed a shortlist, a synthetic
    assistant message carrying that turn's gold URLs is inserted, so the stateless
    reconstruction can detect "a shortlist was already offered" exactly as it would in
    production (it keys off catalog URLs in assistant turns). Agent prose is not
    replayed because the model writes it fresh; only the structural fact of a prior
    shortlist matters to state.
    """
    history: list[dict] = []
    for turn in turns[: upto + 1]:
        history.append({"role": "user", "content": turn.user_text})
        if turn.index == turns[upto].index:
            break
        if turn.shows_recommendations:
            urls = " ".join(sorted(turn.gold_urls))
            history.append(
                {"role": "assistant", "content": f"Here is a shortlist: {urls}"}
            )
    return history
