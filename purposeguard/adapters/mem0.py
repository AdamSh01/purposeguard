"""Mem0 adapter — guard writes on their way into a Mem0 memory.

A specialisation of the raw adapter (see ``raw.py``) for Mem0's client API. Same
boring shape: translate the incoming write into a `Write`, run ``guard.check()``,
tag *flagged* writes' metadata with the verdict breadcrumb, and pass everything
through to Mem0 untouched. Detection-first: a flagged write is tagged, never
dropped.

Two design choices worth stating, both deliberately simple:

1. **We check the raw input text, pre-add — not Mem0's LLM-extracted facts.**
   Mem0's ``add(..., infer=True)`` (the default) runs an LLM to extract facts,
   and those facts only exist *after* a network/model round-trip and are
   non-deterministic. Checking the input before it is sent keeps the adapter
   deterministic, offline-testable, and on the write hot path. The input text is
   a faithful proxy for "what is this write about?", which is exactly what
   purpose-drift cares about. (A post-add re-check of the extracted facts is
   possible later, but that is core/feature work, not something a thin adapter
   should do.)

2. **We attach the breadcrumb via Mem0's ``metadata`` parameter**, which
   ``Memory.add()`` accepts as an optional dict. If a client ever dropped
   metadata support, the breadcrumb simply could not be persisted there —
   detection still happens (the drift meter updates and the verdict is returned),
   but the tag would have nowhere to live. Every Mem0 version we target supports
   ``metadata``, so we rely on it.

Mem0 is imported lazily, only inside :func:`guarded_memory`; importing this
module never requires Mem0. Wrapping an existing client (:func:`guard_mem0`) does
not import Mem0 at all — holding a client already proves it is installed.
"""

from __future__ import annotations

from typing import Any

from ..guard import PurposeGuard
from ..types import Decision, Write
from .raw import _verdict_tag

_INSTALL_HINT = (
    "The Mem0 adapter requires the 'mem0ai' package, which is not installed. "
    "Install it with:  pip install mem0ai"
)


def _require_mem0():
    """Lazy-import Mem0, raising a clear, actionable error if it is missing."""
    try:
        import mem0
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return mem0


def _extract_text(messages: Any) -> str:
    """Flatten Mem0's ``messages`` argument into a single string to score.

    Mem0 accepts either a plain string or a list of ``{"role", "content"}`` dicts.
    For drift scoring we only care about the textual content being written, so we
    concatenate the content of every message. We never mutate ``messages`` — the
    original is forwarded to Mem0 verbatim.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        return str(messages.get("content", ""))
    if isinstance(messages, (list, tuple)):
        parts = []
        for m in messages:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(p for p in parts if p)
    return str(messages)


class GuardedMemory:
    """Wraps a Mem0 client so every ``add`` is scored against the guard's purpose.

    Construct via :func:`guard_mem0` (wrap an existing client) or
    :func:`guarded_memory` (build one for you). Behaves like the wrapped client
    for everything else — ``search``, ``get_all``, ``delete``, … all delegate.
    """

    def __init__(self, client: Any, guard: PurposeGuard) -> None:
        self._client = client
        self._guard = guard

    def add(
        self,
        messages: Any,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Score the write, tag it if flagged, then forward to Mem0's ``add``.

        ``messages`` and every other keyword (``user_id``, ``agent_id``,
        ``run_id``, ``infer``, …) are forwarded to Mem0 unchanged. Returns
        whatever Mem0's ``add`` returns, so the wrapper is a drop-in replacement.
        """
        write = Write(
            content=_extract_text(messages),
            metadata=dict(metadata) if metadata else {},
        )
        verdict = self._guard.check(write)

        meta = dict(write.metadata)
        if verdict.decision is Decision.FLAG:
            # Tag, never drop: the write still goes to Mem0 below.
            meta["purposeguard"] = _verdict_tag(verdict)

        # Forward a metadata kwarg only when there's something to forward, so a
        # plain add(messages, user_id=...) still works on the on-purpose path.
        if meta:
            return self._client.add(messages, metadata=meta, **kwargs)
        return self._client.add(messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes GuardedMemory doesn't define itself; makes
        # the wrapper a true drop-in (search/get_all/delete/... keep working).
        client = self.__dict__.get("_client")
        if client is None:  # accessed during/before __init__
            raise AttributeError(name)
        return getattr(client, name)


def guard_mem0(client: Any, guard: PurposeGuard) -> GuardedMemory:
    """Wrap an existing Mem0 client so each ``add`` is scored against ``guard``.

    Does not import Mem0 — you already hold a client. Usage::

        from mem0 import Memory
        from purposeguard import PurposeGuard
        from purposeguard.adapters import guard_mem0

        guard = PurposeGuard(purpose="A billing support assistant")
        memory = guard_mem0(Memory(), guard)
        memory.add("User asked about a refund", user_id="u1")
    """
    return GuardedMemory(client, guard)


def guarded_memory(
    guard: PurposeGuard, *, config: dict[str, Any] | None = None
) -> GuardedMemory:
    """Build a Mem0 ``Memory`` (lazy-importing Mem0) and wrap it with ``guard``.

    Use this when you don't already have a client. Raises ImportError with an
    install hint if Mem0 is not installed. If you already have a client, prefer
    :func:`guard_mem0`.
    """
    mem0 = _require_mem0()
    client = mem0.Memory.from_config(config) if config is not None else mem0.Memory()
    return guard_mem0(client, guard)
