"""OWASP Agent Memory Guard composition adapter.

Composes two complementary layers on a single memory write:

  * PurposeGuard -> semantic DRIFT detection (is this write on the agent's mission?)
  * OWASP Agent Memory Guard (AMG) -> marker/policy POISONING enforcement
    (prompt-injection markers, secrets, protected/immutable keys, size anomalies).

They cover different failure modes; together you get drift + poisoning coverage
from one wrapper.

Verified against agent-memory-guard 0.2.2 (Incubator, v0.x):
  - guard class: ``MemoryGuard(store=None, *, policy=None, detectors=None, ...)``
  - screening: ``guard.write(key, value, *, source="agent") -> Action``
  - ``Action`` values: allow / redact / block / quarantine
  - behavior: write() RETURNS the Action for allow/redact/quarantine, but RAISES
    ``PolicyViolation`` (a ``MemoryGuardError``) when the action is BLOCK.
We handle both the returned-Action and raised-exception paths.

KNOWN FRAGILITY (flagged so it isn't an unexamined assumption)
-------------------------------------------------------------
To stay framework-free we do NOT hard-import AMG, so a BLOCK is recognized by
exception *name* ("PolicyViolation" / "MemoryGuardError" -- see
``_AMG_BLOCK_EXCEPTIONS``). That is pinned to AMG's CURRENT (0.2.x) exception
taxonomy: if a future AMG version renames or restructures those exceptions, the
block path would be missed -- a raised block would surface as an unexpected error
instead of ``effective_action="block"``. Re-verify ``_AMG_BLOCK_EXCEPTIONS``
against the installed package when bumping AMG. The real-AMG integration test in
the CI embeddings job is what should catch such a taxonomy change.

VERDICT-COMBINATION POLICY (the design decision)
------------------------------------------------
Each layer keeps its own authority; composition does NOT blur them:

  * PurposeGuard is ADVISORY and stays detection-first (guardrail #1): it only
    ever ALLOW/FLAGs and NEVER blocks. Its drift flag is surfaced, never enforced.
  * OWASP AMG owns ENFORCEMENT on its domain: the combined ``effective_action`` is
    ALWAYS AMG's action (allow/redact/block/quarantine). PurposeGuard's drift flag
    never escalates a write to a block.

Disagreement cases, made explicit:
  - AMG=block,  PG=allow  -> effective=block  (AMG blocks poisoning; no drift)
  - AMG=allow,  PG=flag   -> effective=allow  (STORED) + drift flag surfaced for
                             the operator to watch/alert. PurposeGuard does NOT
                             turn this into a block.
  - AMG=allow,  PG=allow  -> effective=allow, clean
  - AMG=redact/quarantine -> AMG's action stands; any drift flag rides alongside

So a security-literate reader can always tell *who* decided *what*: AMG decided
the enforcement action; PurposeGuard contributed the (non-blocking) drift signal.

AMG is a LAZY, OPTIONAL dependency: never imported at module load, never in the
base install. Wrapping an existing ``MemoryGuard`` imports nothing (you already
hold one); the build-it-for-you constructor lazy-imports AMG and raises a clear
ImportError-with-hint if it's missing. The core stays framework-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..guard import PurposeGuard
from ..types import Decision, Verdict

_INSTALL_HINT = (
    "The OWASP Agent Memory Guard composition adapter requires the "
    "'agent-memory-guard' package, which is not installed. Install it with:  "
    "pip install agent-memory-guard"
)

# AMG raises these on a BLOCK action. Matched by NAME so we can recognize a block
# without importing agent_memory_guard (keeps the adapter framework-free and lets
# the tests use a fake in CI). Anything else from write() is a real error -> re-raised.
_AMG_BLOCK_EXCEPTIONS = {"PolicyViolation", "MemoryGuardError"}


@dataclass
class CombinedVerdict:
    """The result of screening one write through both layers.

    ``amg_action`` is AMG's enforcement decision ("allow"/"redact"/"block"/
    "quarantine"); ``drift`` is PurposeGuard's (advisory, never-blocking) verdict.
    """

    amg_action: str
    drift: Verdict
    drift_flagged: bool

    @property
    def effective_action(self) -> str:
        """Enforcement is AMG's alone — PurposeGuard never escalates to a block."""
        return self.amg_action

    @property
    def stored(self) -> bool:
        """Was the write persisted to the live store? (block/quarantine are not.)"""
        return self.amg_action in ("allow", "redact")

    def __str__(self) -> str:  # ASCII-only (guardrail #4)
        d = "drift:FLAG" if self.drift_flagged else "drift:ok"
        return (
            f"CombinedVerdict(amg={self.amg_action}, {d}, "
            f"score={self.drift.score:.2f})"
        )


class ComposedGuard:
    """Runs PurposeGuard drift detection AND an AMG MemoryGuard on each write.

    Construct via :func:`guard_with_amg` (wrap existing instances) or
    :func:`composed_guard` (build the MemoryGuard for you). The drift layer is
    advisory; the AMG layer enforces. See the module docstring for the policy.
    """

    def __init__(self, purpose_guard: PurposeGuard, memory_guard: Any) -> None:
        self._pg = purpose_guard
        self._amg = memory_guard

    def write(self, key: str, value: Any, *, source: str = "agent") -> CombinedVerdict:
        """Screen one write through both layers and return a combined verdict.

        PurposeGuard scores the content for drift (detection-first, never blocks);
        AMG screens + enforces (it may allow/redact/block/quarantine, and it owns
        storage). Drift is recorded for the *attempted* write regardless of AMG's
        action, so the trend reflects what the agent tried to write.
        """
        text = value if isinstance(value, str) else str(value)

        # Drift first: pure scoring, no side effects; never blocks.
        drift = self._pg.check(text)

        # Enforcement: AMG's authority. Returns an Action for allow/redact/
        # quarantine, or raises a PolicyViolation/MemoryGuardError on BLOCK.
        try:
            action = self._amg.write(key, value, source=source)
            amg_action = getattr(action, "value", str(action))
        except Exception as exc:
            if type(exc).__name__ in _AMG_BLOCK_EXCEPTIONS:
                amg_action = "block"
            else:
                raise  # an unexpected error, not an AMG policy block

        return CombinedVerdict(
            amg_action=amg_action,
            drift=drift,
            drift_flagged=(drift.decision is Decision.FLAG),
        )

    def __getattr__(self, name: str) -> Any:
        # Delegate reads/snapshots/rollback/etc. to the wrapped MemoryGuard, so the
        # composed object is a drop-in for it. Only write() is intercepted.
        amg = self.__dict__.get("_amg")
        if amg is None:  # during/before __init__
            raise AttributeError(name)
        return getattr(amg, name)


def guard_with_amg(purpose_guard: PurposeGuard, memory_guard: Any) -> ComposedGuard:
    """Compose an existing PurposeGuard with an existing AMG ``MemoryGuard``.

    Imports nothing — you already hold both. Usage::

        from agent_memory_guard import MemoryGuard, Policy
        from purposeguard import PurposeGuard
        from purposeguard.adapters import guard_with_amg

        guard = guard_with_amg(
            PurposeGuard(purpose="A billing support assistant"),
            MemoryGuard(policy=Policy.strict()),
        )
        verdict = guard.write("note", "User asked about a refund")
        print(verdict.effective_action, verdict.drift_flagged)
    """
    return ComposedGuard(purpose_guard, memory_guard)


def composed_guard(
    purpose: "str | PurposeGuard",
    *,
    policy: Any = None,
    store: Any = None,
    memory_guard: Any = None,
    **purpose_guard_kwargs: Any,
) -> ComposedGuard:
    """First-class drift + scope + AMG-poisoning guard from a SINGLE config.

    One call gives the whole composed surface instead of wiring the layers by hand:

        from agent_memory_guard import Policy
        from purposeguard.adapters import composed_guard

        guard = composed_guard(
            "A customer-support assistant for billing and payments",
            allowed_topics=["invoices", "refunds"],
            blocked_topics=["legal advice", "medical advice"],
            mode="redirect",
            policy=Policy.strict(),
        )
        v = guard.write("note", "for your lawsuit, here's the legal advice you need")
        # v.drift.recommended_action -> redirect (advisory drift/scope signal)
        # v.effective_action         -> AMG's enforcement (allow/redact/block/quarantine)

    - ``purpose`` is a purpose string (a PurposeGuard is built from it plus any
      drift/scope kwargs: ``allowed_topics``, ``blocked_topics``, ``mode``,
      ``threshold``, ``scorer``, ``require_embeddings``, ...), OR an already-built
      PurposeGuard (then the extra kwargs are ignored).
    - The AMG ``MemoryGuard`` is built from ``policy``/``store`` (lazy-importing
      AMG; clear ImportError if absent), unless you pass an existing
      ``memory_guard`` (e.g. one with custom detectors/snapshots).

    The verdict-combination policy is unchanged (guardrail #1): PurposeGuard is
    advisory and never blocks; AMG owns enforcement (``CombinedVerdict``'s
    ``effective_action`` is always AMG's). This **composes defense-in-depth layers
    — it is not a security guarantee.** AMG catches known markers/policies and is
    defeated by novel obfuscation/encoding/paraphrase; PurposeGuard catches
    topical drift/scope. See THREAT_MODEL.md.
    """
    pg = (
        purpose
        if isinstance(purpose, PurposeGuard)
        else PurposeGuard(purpose, **purpose_guard_kwargs)
    )
    if memory_guard is None:
        amg = _require_amg()
        memory_guard = amg.MemoryGuard(store, policy=policy)
    return guard_with_amg(pg, memory_guard)


def _require_amg():
    """Lazy-import OWASP Agent Memory Guard, with a clear install hint."""
    try:
        import agent_memory_guard
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return agent_memory_guard
