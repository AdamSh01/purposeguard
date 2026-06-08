"""Tests for the Mem0 adapter.

Uses a fake Mem0 client and the LexicalScorer so the suite stays hermetic — no
real mem0 install, no LLM, no network. The adapter must: forward every write to
the client (detection-first — even flagged ones), tag flagged writes' metadata,
forward Mem0's own kwargs (user_id, infer, …) and the original messages object
unchanged, delegate everything else, and raise a clear ImportError-with-hint when
someone asks it to build a client but mem0 isn't installed.
"""

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard
from purposeguard.adapters import GuardedMemory, guard_mem0, guarded_memory


class FakeMem0Client:
    """Minimal stand-in for mem0.Memory / mem0.MemoryClient."""

    def __init__(self):
        self.added = []     # list of (messages, metadata, kwargs)
        self.searched = []  # list of (query, kwargs)

    def add(self, messages, metadata=None, **kwargs):
        self.added.append((messages, metadata, kwargs))
        return {"results": [{"id": len(self.added)}]}

    def search(self, query, **kwargs):
        self.searched.append((query, kwargs))
        return {"results": []}

    def delete_all(self, **kwargs):  # extra method, to exercise delegation
        return "deleted"


ON_PURPOSE = "billing payment invoice refund subscription"
OFF_PURPOSE = "weather forecast sourdough recipe football puppy"


def make_guard():
    return PurposeGuard(
        purpose="billing payment subscription invoice refund account support",
        scorer=LexicalScorer(),
        threshold=0.12,
    )


def make_guarded():
    guard = make_guard()
    client = FakeMem0Client()
    return client, guard, guard_mem0(client, guard)


def test_guard_mem0_returns_wrapper():
    client, guard, guarded = make_guarded()
    assert isinstance(guarded, GuardedMemory)


def test_on_purpose_passes_through_untagged():
    client, guard, guarded = make_guarded()
    guarded.add(ON_PURPOSE, user_id="u1")
    assert len(client.added) == 1
    messages, metadata, kwargs = client.added[0]
    assert messages == ON_PURPOSE
    assert kwargs == {"user_id": "u1"}        # Mem0 kwargs forwarded
    assert not (metadata or {}).get("purposeguard")


def test_off_purpose_tagged_but_not_dropped():
    client, guard, guarded = make_guarded()
    guarded.add(OFF_PURPOSE, user_id="u1")
    # Detection-first: still written.
    assert len(client.added) == 1
    messages, metadata, kwargs = client.added[0]
    assert messages == OFF_PURPOSE
    tag = metadata["purposeguard"]
    assert tag["decision"] == Decision.FLAG.value
    assert tag["alignment"] == pytest.approx(0.0)
    assert "drift" in tag and "reason" in tag


def test_list_of_dicts_is_scored_and_forwarded_verbatim():
    client, guard, guarded = make_guarded()
    msgs = [
        {"role": "user", "content": OFF_PURPOSE},
        {"role": "assistant", "content": "sure"},
    ]
    guarded.add(msgs, user_id="u1", infer=False)
    sent_messages, metadata, kwargs = client.added[0]
    # The original messages object is forwarded unchanged...
    assert sent_messages is msgs
    # ...but it was scored (off-purpose content) and tagged.
    assert metadata["purposeguard"]["decision"] == Decision.FLAG.value
    assert kwargs == {"user_id": "u1", "infer": False}


def test_caller_metadata_is_preserved_and_merged():
    client, guard, guarded = make_guarded()
    guarded.add(OFF_PURPOSE, metadata={"category": "support"}, user_id="u1")
    _, metadata, _ = client.added[0]
    assert metadata["category"] == "support"   # caller's data kept
    assert "purposeguard" in metadata          # our tag added alongside


def test_search_and_other_methods_delegate():
    client, guard, guarded = make_guarded()
    assert guarded.search("how do refunds work", user_id="u1") == {"results": []}
    assert client.searched == [("how do refunds work", {"user_id": "u1"})]
    assert guarded.delete_all() == "deleted"   # delegated to the client


def test_drift_meter_updates_through_adapter():
    client, guard, guarded = make_guarded()
    guarded.add(ON_PURPOSE, user_id="u1")
    guarded.add(OFF_PURPOSE, user_id="u1")
    assert guard.drift().samples == 2


def test_guarded_memory_raises_clear_importerror_without_mem0():
    """The build-a-client path must fail loudly with an install hint."""
    try:
        import mem0  # noqa: F401

        pytest.skip("mem0 is installed; the ImportError path is not exercised here")
    except ImportError:
        pass

    with pytest.raises(ImportError) as excinfo:
        guarded_memory(make_guard())
    assert "mem0ai" in str(excinfo.value)
