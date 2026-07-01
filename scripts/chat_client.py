"""A small terminal chat client for the recommender API.

The service is *stateless*: it stores nothing between calls, so a conversation is
maintained entirely by the caller resending the full message history each turn. This
client does exactly that — it is a thin convenience loop that keeps the history on the
client side and posts it to ``/chat`` every turn. It changes nothing about the server's
statelessness; it is a client of the stateless API, the same role a browser or the
built-in chat UI plays.

Usage::

    # talk to the deployed Space (default)
    python -m scripts.chat_client

    # talk to a local server (uvicorn shl_recommender.api.app:app)
    python -m scripts.chat_client --url http://127.0.0.1:8000

    # show the reasoning trace (?debug=1) under each reply
    python -m scripts.chat_client --debug

At the prompt: type a message and press Enter. Commands: ``:debug`` toggles the trace,
``:reset`` starts a fresh conversation, ``:history`` prints the raw message list being
sent, and ``:quit`` (or Ctrl-D) exits.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

DEFAULT_URL = "https://hriday29-shl-assessment-recommender.hf.space"


def _print_reply(data: dict, *, show_trace: bool) -> None:
    """Render one response: the reply, the shortlist if any, and the trace if asked."""
    print(f"\nassistant: {data.get('reply', '')}")

    recs = data.get("recommendations")
    if recs:
        print(f"\n  recommendations ({len(recs)}):")
        for i, rec in enumerate(recs, 1):
            print(f"    {i:>2}. {rec['name']}  [{rec['test_type']}]")
            print(f"        {rec['url']}")

    if data.get("end_of_conversation"):
        print("\n  (the assistant considers this conversation complete)")

    trace = data.get("_trace")
    if show_trace and trace:
        print("\n  --- trace ---------------------------------------------------")
        state = trace["state"]
        print(f"  mode={trace['mode']}  reason={trace['reason']}  "
              f"commits={trace['commits_shortlist']}  ends={trace['end_of_conversation']}")
        print(f"  ready_to_recommend={state['ready_to_recommend']}  "
              f"reply_from_model={trace['reply_from_model']}")
        signals = [
            name for name in (
                "is_comparison", "wants_addition", "wants_removal",
                "is_off_topic", "is_prompt_injection", "user_confirmed",
            )
            if state.get(name)
        ]
        if signals:
            print(f"  signals: {', '.join(signals)}")
        if state.get("must_have_skills"):
            print(f"  skills: {', '.join(state['must_have_skills'])}")
        if trace["retrieval"]:
            print("  top candidates:")
            for cand in trace["retrieval"][:5]:
                print(f"    {cand['score']:>7.4f}  {cand['name']}  [{cand['test_type']}]")
        print("  -------------------------------------------------------------")


def _post(client: httpx.Client, url: str, messages: list, *, debug: bool) -> dict | None:
    """Post the full history to /chat; return the parsed body or None on failure."""
    endpoint = f"{url.rstrip('/')}/chat"
    params = {"debug": "1"} if debug else None
    try:
        response = client.post(endpoint, json={"messages": messages}, params=params, timeout=60)
    except httpx.HTTPError as exc:
        print(f"\n[network error: {exc}]", file=sys.stderr)
        return None
    if response.status_code != 200:
        print(f"\n[server returned {response.status_code}: {response.text[:300]}]", file=sys.stderr)
        return None
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal chat client for the recommender API.")
    parser.add_argument("--url", default=DEFAULT_URL, help="base URL of the service")
    parser.add_argument("--debug", action="store_true", help="show the ?debug=1 trace under each reply")
    args = parser.parse_args()

    show_trace = args.debug
    messages: list[dict] = []

    print(f"Connected to {args.url}")
    print("Type a message. Commands: :debug  :reset  :history  :quit\n")

    with httpx.Client() as client:
        while True:
            try:
                text = input("you: ").strip()
            except EOFError:
                print()
                break

            if not text:
                continue
            if text == ":quit":
                break
            if text == ":debug":
                show_trace = not show_trace
                print(f"[trace {'on' if show_trace else 'off'}]")
                continue
            if text == ":reset":
                messages = []
                print("[conversation reset]")
                continue
            if text == ":history":
                print(json.dumps(messages, indent=2))
                continue

            # Append this user turn and resend the WHOLE history — this is what keeps
            # the conversation coherent against a server that stores nothing.
            messages.append({"role": "user", "content": text})
            data = _post(client, args.url, messages, debug=show_trace)
            if data is None:
                messages.pop()  # drop the turn that never got a reply
                continue

            _print_reply(data, show_trace=show_trace)
            # Record the assistant's reply so the next turn carries full context.
            messages.append({"role": "assistant", "content": data.get("reply", "")})

    print("bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
