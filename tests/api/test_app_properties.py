"""Property-based robustness tests for ``/chat``.

The brief warns that weak submissions fall over on inputs the author did not think
of. Example tests cover the cases we imagined; these cover the ones we did not, by
generating arbitrary message histories and asserting the invariants that must hold
for *every* input:

* the endpoint never returns a 5xx from bad input — it is always either a valid
  200 or a shaped 4xx (a crash that leaked a stack trace would be a contract
  violation);
* a 200 body always has exactly the three contract fields, and ``recommendations``
  is always ``null`` or a 1..10 list — never ``[]`` and never oversized;
* a 4xx body always uses the stable error envelope.

The model is faked (offline, deterministic) so the properties test our code, not the
provider. Semantic retrieval is disabled for speed.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings as hyp_settings
from hypothesis import strategies as st

from tests.api.conftest import FakeLLM, make_client

# One client for the whole module; building it per example would be far too slow.
_CLIENT = make_client(FakeLLM(json={"ready_to_recommend": False}))

# Arbitrary role strings: the valid three, common synonyms, and pure noise — so we
# exercise both the synonym-normalising path and the rejection path.
_roles = st.one_of(
    st.sampled_from(["user", "assistant", "system", "User", "Agent", "Human", "bot"]),
    st.text(max_size=12),
)
_messages = st.lists(
    st.fixed_dictionaries({"role": _roles, "content": st.text(max_size=200)}),
    max_size=8,
)


def _assert_valid_contract_or_error(resp) -> None:
    # Never a server error from input we generated.
    assert resp.status_code < 500, resp.text
    body = resp.json()
    if resp.status_code == 200:
        assert set(body) == {"reply", "recommendations", "end_of_conversation"}
        assert isinstance(body["reply"], str)
        assert isinstance(body["end_of_conversation"], bool)
        recs = body["recommendations"]
        assert recs is None or (isinstance(recs, list) and 1 <= len(recs) <= 10)
        if recs:
            for rec in recs:
                assert set(rec) == {"name", "url", "test_type"}
    else:
        # Any non-200 must still be our stable error envelope.
        assert "error" in body and "type" in body["error"]


@hyp_settings(max_examples=150, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(messages=_messages)
def test_chat_is_robust_to_arbitrary_histories(messages):
    resp = _CLIENT.post("/chat", json={"messages": messages})
    _assert_valid_contract_or_error(resp)


@hyp_settings(max_examples=50, deadline=None)
@given(body=st.dictionaries(st.text(max_size=8), st.text(max_size=20), max_size=4))
def test_chat_is_robust_to_arbitrary_top_level_bodies(body):
    # Bodies that are objects but not our schema must be a shaped 4xx, never a crash.
    resp = _CLIENT.post("/chat", json=body)
    assert resp.status_code < 500
    if resp.status_code != 200:
        assert "error" in resp.json()


@hyp_settings(max_examples=30, deadline=None)
@given(content=st.text(min_size=1, max_size=500))
def test_single_user_turn_always_yields_valid_contract(content):
    # A single well-formed user turn (any content) must always produce a valid 200.
    resp = _CLIENT.post("/chat", json={"messages": [{"role": "user", "content": content}]})
    assert resp.status_code == 200
    _assert_valid_contract_or_error(resp)
