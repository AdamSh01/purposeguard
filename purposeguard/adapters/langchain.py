"""LangChain adapter — guard messages on their way into a chat history.

A specialisation of the raw adapter (see ``raw.py``) for LangChain's
``BaseChatMessageHistory``. Same boring shape: as each message is added, score
its text against the guard's purpose, tag *flagged* messages, and pass everything
through to the underlying history untouched. Detection-first: a flagged message
is tagged, never dropped.

**Which messages are scored — both user and AI turns, as one drift stream.**
A LangChain chat history accumulates the whole conversation, including the
agent's *own* replies. We deliberately score **every** message added — human and
AI alike — and feed them all into one drift meter. Rationale: a drifting agent's
own responses wander off-mission, so its AI turns are a *first-class* drift
signal, not noise; scoring only user inputs would miss exactly the failure this
library exists to catch. Treating both as one stream is also the simplest
defensible default (no per-role branching).

Trade-off (stated honestly): legitimately terse or broad AI turns ("Sure, done!")
will score low and can nudge the meter. The baseline window and threshold absorb
this; an agent with very short replies may want a larger ``drift_baseline_window``.
If a caller truly wants to exclude a role, they simply don't route it through the
guarded history.

We mirror **OWASP Agent Memory Guard**'s ``GuardedChatMessageHistory`` shape so
the two compose: AMG screens markers (injection, secrets, protected keys), we
screen semantic drift. Because our wrapper accepts *any* ``BaseChatMessageHistory``
as its inner store, you can nest them — ``guard_chat_history(amg_history, guard)``
runs both checks on every message. The breadcrumb goes in the message's
``additional_kwargs`` (LangChain's per-message metadata bag).

LangChain is imported lazily, only when we must *construct* a message
(``add_user_message`` / ``add_ai_message``); the duck-typed ``add_message`` path
needs nothing installed, which is what lets the tests run with a fake history.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..guard import PurposeGuard
from ..types import Decision, Write
from .raw import _verdict_tag

_INSTALL_HINT = (
    "The LangChain adapter requires 'langchain-core', which is not installed. "
    "Install it with:  pip install langchain-core"
)


def _require_langchain_messages():
    """Lazy-import LangChain's message classes, with a clear install hint."""
    try:
        from langchain_core.messages import AIMessage, HumanMessage
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return HumanMessage, AIMessage


def _message_text(message: Any) -> str:
    """Pull the scorable text out of a LangChain message (duck-typed).

    ``content`` is usually a string; for multimodal messages it can be a list of
    parts, so we concatenate the text parts. We only read — never mutate content.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


class GuardedChatMessageHistory:
    """Wraps a ``BaseChatMessageHistory`` so every added message is scored.

    Construct via :func:`guard_chat_history`. Presents the full chat-history
    interface: writes (``add_message``/``add_messages``/``add_user_message``/
    ``add_ai_message``/``aadd_messages``) are screened, while reads and ``clear``
    delegate to the inner history. Every write entry point is overridden so the
    guard can never be bypassed; only reads fall through to the wrapped history.
    """

    def __init__(self, history: Any, guard: PurposeGuard) -> None:
        self._inner = history
        self._guard = guard

    # -- screening (the only logic this adapter adds) ----------------------

    def _screen(self, message: Any) -> Any:
        """Score one message and tag it in place if flagged. Never drops it."""
        verdict = self._guard.check(Write(content=_message_text(message)))
        if verdict.decision is Decision.FLAG:
            kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(kwargs, dict):
                kwargs["purposeguard"] = _verdict_tag(verdict)
            else:  # exotic message without the usual bag: attach one
                try:
                    message.additional_kwargs = {"purposeguard": _verdict_tag(verdict)}
                except Exception:
                    pass  # detection still happened via the meter; tag is best-effort
        return message

    def _screen_all(self, messages: Iterable[Any]) -> list:
        msgs = list(messages)
        for message in msgs:
            self._screen(message)
        return msgs

    # -- write surface (all screened) --------------------------------------

    def add_message(self, message: Any) -> None:
        self._screen(message)
        self._inner.add_message(message)

    def add_messages(self, messages: Iterable[Any]) -> None:
        self._inner.add_messages(self._screen_all(messages))

    def add_user_message(self, message: str) -> None:
        HumanMessage, _ = _require_langchain_messages()
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        _, AIMessage = _require_langchain_messages()
        self.add_message(AIMessage(content=message))

    async def aadd_messages(self, messages: Iterable[Any]) -> None:
        """Async bulk add — screened too, so async callers aren't a bypass."""
        screened = self._screen_all(messages)
        ainner = getattr(self._inner, "aadd_messages", None)
        if ainner is not None:
            await ainner(screened)
        else:
            self._inner.add_messages(screened)

    # -- everything else (reads, clear, async reads) delegates -------------

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:  # accessed during/before __init__
            raise AttributeError(name)
        return getattr(inner, name)


def guard_chat_history(history: Any, guard: PurposeGuard) -> GuardedChatMessageHistory:
    """Wrap a LangChain chat history so each added message is scored.

    Usage::

        from langchain_core.chat_history import InMemoryChatMessageHistory
        from purposeguard import PurposeGuard
        from purposeguard.adapters import guard_chat_history

        guard = PurposeGuard(purpose="A billing support assistant")
        history = guard_chat_history(InMemoryChatMessageHistory(), guard)
        history.add_user_message("How do I get a refund?")  # scored, then stored

    Compose with OWASP Agent Memory Guard by wrapping its history as the inner
    store, so marker screening and semantic-drift screening both run.
    """
    return GuardedChatMessageHistory(history, guard)
