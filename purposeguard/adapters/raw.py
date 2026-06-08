"""Raw store adapter — the framework-agnostic reference integration.

This is the "bring your own store" path: wrap ANY object that exposes
``add(text, *, metadata=...)`` and ``search(query)`` so that every write is
scored against the guard's purpose before it lands. It is deliberately the
thinnest possible adapter — translate the incoming write into a `Write`, run
`guard.check()`, tag the metadata of *flagged* writes with the verdict, and
otherwise pass everything through to the underlying store untouched.

Detection-first (guardrail #1): a flagged write is **never** dropped or blocked.
It is stored exactly as it would have been, with an extra
``metadata["purposeguard"]`` breadcrumb recording why it looked off-purpose.
Adopters can alert on that field or watch the live trend via ``guard.drift()`` —
but their data is never silently lost just because they added the guard.

Every other adapter (mem0, langchain) is a specialisation of this same
translate-check-tag shape. Understand this file and you understand the pattern.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..guard import PurposeGuard
from ..types import Decision, Verdict, Write


class Store(Protocol):
    """The minimal contract a store must satisfy to be guarded.

    ``add`` takes the content plus an optional ``metadata`` mapping — that is
    where flag breadcrumbs go; ``search`` is passed through verbatim. Vector
    stores, Mem0, and most memory backends already match this shape.
    """

    def add(self, text: str, **kwargs: Any) -> Any: ...
    def search(self, query: str, **kwargs: Any) -> Any: ...


def _verdict_tag(verdict: Verdict) -> dict[str, Any]:
    """Translate a Verdict into a compact, JSON-friendly metadata breadcrumb.

    Kept small and primitive-typed so it survives serialization into whatever
    backend the store uses (JSON columns, vector-store metadata, etc.).
    """
    drift = verdict.details.get("drift", {})
    return {
        "decision": verdict.decision.value,
        "alignment": verdict.score,
        "drift": drift.get("drift"),
        "drifting": drift.get("drifting"),
        "reason": verdict.reason,
    }


class GuardedStore:
    """A transparent wrapper that scores writes before they reach ``store``.

    Constructed via :func:`guard_store`. It behaves like the wrapped store for
    every method (unknown attributes are delegated), but ``add`` first runs the
    write through the guard and tags flagged writes. The wrapper holds no state
    of its own beyond the store and guard — the drift trend lives in the guard.
    """

    def __init__(self, store: Store, guard: PurposeGuard) -> None:
        self._store = store
        self._guard = guard

    def add(
        self,
        text: "str | Write",
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Score one write, tag it if flagged, then forward it to the store.

        Returns whatever the underlying store's ``add`` returns, so the wrapper
        is a drop-in replacement. ``metadata`` (if given) is merged with the
        write's own and, on FLAG, with the guard breadcrumb.
        """
        write = Write.coerce(text)
        if metadata:
            write.metadata = {**write.metadata, **metadata}

        verdict = self._guard.check(write)

        meta = dict(write.metadata)
        if verdict.decision is Decision.FLAG:
            # Tag, never drop: the write still goes to the store below.
            meta["purposeguard"] = _verdict_tag(verdict)

        # Only forward a metadata kwarg when there is something to forward, so a
        # store whose add() takes just text still works on the common path.
        if meta:
            return self._store.add(write.content, metadata=meta, **kwargs)
        return self._store.add(write.content, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Any:
        """Pure pass-through: reads are never gated by the guard."""
        return self._store.search(query, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes GuardedStore doesn't define itself; makes
        # the wrapper a true drop-in (delete/reset/get/... all keep working).
        store = self.__dict__.get("_store")
        if store is None:  # accessed during/before __init__
            raise AttributeError(name)
        return getattr(store, name)


def guard_store(store: Store, guard: PurposeGuard) -> GuardedStore:
    """Wrap ``store`` so every ``add`` is scored against ``guard``'s purpose.

    The reference adapter. Usage::

        from purposeguard import PurposeGuard
        from purposeguard.adapters import guard_store

        guard = PurposeGuard(purpose="A billing support assistant")
        memory = guard_store(my_vector_store, guard)
        memory.add("User asked about a refund")   # scored, then passed through

    Returns a :class:`GuardedStore` you use exactly like the original store.
    """
    return GuardedStore(store, guard)
