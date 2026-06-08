"""Tests for the raw (bring-your-own-store) adapter.

Uses a fake in-memory store and the LexicalScorer so the suite stays hermetic —
no real vector store, no model download, no network. The adapter must: pass every
write through to the store (detection-first — even flagged ones), tag flagged
writes' metadata, leave on-purpose writes untouched, and transparently delegate
everything else to the wrapped store.
"""

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard
from purposeguard.adapters import GuardedStore, guard_store


class FakeStore:
    """Minimal stand-in for a vector store / memory backend."""

    def __init__(self):
        self.added = []     # list of (text, metadata)
        self.searched = []  # list of queries

    def add(self, text, metadata=None, **kwargs):
        self.added.append((text, metadata))
        return f"id-{len(self.added)}"

    def search(self, query, **kwargs):
        self.searched.append(query)
        return [f"result::{query}"]

    def reset(self):  # extra method, to exercise transparent delegation
        self.added.clear()
        return "reset-done"


ON_PURPOSE = "billing payment invoice refund subscription"
OFF_PURPOSE = "weather forecast sourdough recipe football puppy"


def make_guarded():
    guard = PurposeGuard(
        purpose="billing payment subscription invoice refund account support",
        scorer=LexicalScorer(),
        threshold=0.12,
    )
    store = FakeStore()
    return store, guard, guard_store(store, guard)


def test_guard_store_returns_wrapper():
    store, guard, guarded = make_guarded()
    assert isinstance(guarded, GuardedStore)


def test_on_purpose_passes_through_untagged():
    store, guard, guarded = make_guarded()
    guarded.add(ON_PURPOSE)
    assert len(store.added) == 1
    text, metadata = store.added[0]
    assert text == ON_PURPOSE
    assert not (metadata or {}).get("purposeguard")  # no flag breadcrumb


def test_off_purpose_is_tagged_but_not_dropped():
    store, guard, guarded = make_guarded()
    guarded.add(OFF_PURPOSE)
    # Detection-first: the write is STILL stored, never dropped.
    assert len(store.added) == 1
    text, metadata = store.added[0]
    assert text == OFF_PURPOSE
    tag = metadata["purposeguard"]
    assert tag["decision"] == Decision.FLAG.value
    assert tag["alignment"] == pytest.approx(0.0)  # no token overlap with purpose
    assert "drift" in tag and "reason" in tag


def test_caller_metadata_is_preserved_and_merged():
    store, guard, guarded = make_guarded()
    guarded.add(OFF_PURPOSE, metadata={"user_id": "u1"})
    _, metadata = store.added[0]
    assert metadata["user_id"] == "u1"   # caller's data kept
    assert "purposeguard" in metadata    # our tag added alongside


def test_search_passes_through():
    store, guard, guarded = make_guarded()
    out = guarded.search("how do refunds work")
    assert out == ["result::how do refunds work"]
    assert store.searched == ["how do refunds work"]


def test_drift_meter_updates_through_adapter():
    store, guard, guarded = make_guarded()
    guarded.add(ON_PURPOSE)
    guarded.add(OFF_PURPOSE)
    assert guard.drift().samples == 2  # check() ran for each add


def test_unknown_attributes_delegate_to_store():
    store, guard, guarded = make_guarded()
    assert guarded.reset() == "reset-done"  # delegated to FakeStore.reset
