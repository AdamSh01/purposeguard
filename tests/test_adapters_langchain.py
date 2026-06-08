"""Tests for the LangChain adapter.

Uses a fake chat history + fake messages and the LexicalScorer so the suite stays
hermetic — no real langchain dependency in CI. The adapter must: screen every
message as it's added (human AND ai turns, one drift stream), tag flagged
messages' additional_kwargs, never drop them, screen the async path too (no
bypass), delegate reads/clear to the inner history, and raise a clear
ImportError-with-hint when asked to build a message but langchain isn't installed.
"""

import asyncio

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard
from purposeguard.adapters import GuardedChatMessageHistory, guard_chat_history


class FakeMessage:
    """Duck-typed stand-in for a langchain_core BaseMessage."""

    def __init__(self, content, type="human"):
        self.content = content
        self.type = type
        self.additional_kwargs = {}


class FakeHistory:
    """Duck-typed stand-in for a langchain_core BaseChatMessageHistory."""

    def __init__(self):
        self._messages = []

    @property
    def messages(self):
        return list(self._messages)

    def add_message(self, message):
        self._messages.append(message)

    def add_messages(self, messages):
        self._messages.extend(messages)

    def clear(self):
        self._messages.clear()

    async def aadd_messages(self, messages):
        self._messages.extend(messages)


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
    history = FakeHistory()
    return history, guard, guard_chat_history(history, guard)


def _tag(message):
    return message.additional_kwargs.get("purposeguard")


def test_guard_chat_history_returns_wrapper():
    history, guard, guarded = make_guarded()
    assert isinstance(guarded, GuardedChatMessageHistory)


def test_on_purpose_message_passes_through_untagged():
    history, guard, guarded = make_guarded()
    guarded.add_message(FakeMessage(ON_PURPOSE, "human"))
    assert len(history.messages) == 1
    assert _tag(history.messages[0]) is None


def test_off_purpose_message_tagged_but_not_dropped():
    history, guard, guarded = make_guarded()
    guarded.add_message(FakeMessage(OFF_PURPOSE, "human"))
    # Detection-first: still stored.
    assert len(history.messages) == 1
    tag = _tag(history.messages[0])
    assert tag["decision"] == Decision.FLAG.value
    assert tag["alignment"] == pytest.approx(0.0)
    assert "drift" in tag and "reason" in tag


def test_ai_turns_are_also_scored():
    """The deliberate design choice: the agent's OWN replies are screened too."""
    history, guard, guarded = make_guarded()
    guarded.add_message(FakeMessage(OFF_PURPOSE, "ai"))  # an AI turn that wandered
    assert _tag(history.messages[0])["decision"] == Decision.FLAG.value
    assert guard.drift().samples == 1  # the AI turn fed the drift meter


def test_user_and_ai_feed_one_drift_stream():
    history, guard, guarded = make_guarded()
    guarded.add_message(FakeMessage(ON_PURPOSE, "human"))
    guarded.add_message(FakeMessage(OFF_PURPOSE, "ai"))
    assert guard.drift().samples == 2  # both turns counted, one stream


def test_add_messages_bulk_screens_each_and_forwards():
    history, guard, guarded = make_guarded()
    guarded.add_messages(
        [FakeMessage(ON_PURPOSE, "human"), FakeMessage(OFF_PURPOSE, "ai")]
    )
    assert len(history.messages) == 2
    assert _tag(history.messages[0]) is None                       # on-purpose
    assert _tag(history.messages[1])["decision"] == Decision.FLAG.value
    assert guard.drift().samples == 2


def test_async_aadd_messages_screens_and_forwards():
    """Async writes must be screened too — not a bypass."""
    history, guard, guarded = make_guarded()
    asyncio.run(guarded.aadd_messages([FakeMessage(OFF_PURPOSE, "ai")]))
    assert len(history.messages) == 1
    assert _tag(history.messages[0])["decision"] == Decision.FLAG.value
    assert guard.drift().samples == 1


def test_reads_and_clear_delegate_to_inner():
    history, guard, guarded = make_guarded()
    guarded.add_message(FakeMessage(ON_PURPOSE, "human"))
    assert guarded.messages == history.messages   # read delegates
    guarded.clear()                               # clear delegates
    assert history.messages == []


def test_add_user_message_raises_clear_importerror_without_langchain(monkeypatch):
    """Simulate langchain-core being absent and assert a helpful ImportError.

    Monkeypatching __import__ makes this deterministic whether or not langchain
    happens to be installed in the current environment.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langchain_core" or name.startswith("langchain_core."):
            raise ImportError("simulated: langchain_core not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    history, guard, guarded = make_guarded()
    with pytest.raises(ImportError) as excinfo:
        guarded.add_user_message("hello there")
    assert "langchain-core" in str(excinfo.value)
