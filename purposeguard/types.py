"""Core data types for PurposeGuard.

The whole library speaks two tiny shapes: a `Write` going into memory, and a
`Verdict` coming back. Everything else — adapters, scorers, policies — is built
around these. Keeping them framework-agnostic is what makes the core general:
Mem0, LangChain, a raw vector store, or a plain dict can all produce a `Write`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Decision(str, Enum):
    """The guard's DETECTION outcome for a write: is it in scope?

    The guard only ever returns ALLOW or FLAG — `decision` is detection, not
    enforcement. What the caller should DO about a FLAG (given the guard's mode)
    is carried separately in `Verdict.recommended_action`, so even in block mode
    `decision` stays ALLOW/FLAG and never BLOCK (guardrail #1: the guard
    recommends, the caller enforces). BLOCK is kept in the enum for adapters that
    map to a full action set; the guard itself does not emit it.
    """

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"  # not emitted by PurposeGuard; see Verdict.recommended_action


class RecommendedAction(str, Enum):
    """What the guard RECOMMENDS the caller do, given its mode. Advisory only —
    PurposeGuard never acts on the caller's data; the caller reads this and
    enforces (guardrail #1).

    - ALLOW    : in scope; proceed (any mode).
    - MONITOR  : off scope, mode=monitor; just log/observe (the v0.1 default).
    - REDIRECT : off scope, mode=redirect; replace the output with a safe fallback.
    - BLOCK    : off scope, mode=block; stop / suppress the off-scope output.
    """

    ALLOW = "allow"
    MONITOR = "monitor"
    REDIRECT = "redirect"
    BLOCK = "block"


@dataclass
class Write:
    """A single piece of content on its way into agent memory.

    `content` is the only required field. `key` and `metadata` are optional so
    that the simplest possible caller — `guard.check("some text")` — works, while
    richer callers (an adapter that knows the memory key, the source, the session)
    can pass more for better policy decisions and forensics.
    """

    content: str
    key: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, write: "Write | str") -> "Write":
        """Accept either a bare string or a full Write. Keeps the API friendly."""
        if isinstance(write, Write):
            return write
        if isinstance(write, str):
            return cls(content=write)
        raise TypeError(
            f"check() expects a str or Write, got {type(write).__name__}"
        )


@dataclass
class Verdict:
    """The guard's conclusion about one write.

    `score` is the alignment with the declared purpose in [0.0, 1.0], where 1.0
    is perfectly on-mission. `decision` is the DETECTION outcome (ALLOW/FLAG).
    `recommended_action` is what the guard recommends the caller DO about it given
    the mode (advisory — the caller enforces; the guard never acts). `reason` is a
    short human-readable explanation; `details` carries the raw signals (similarity,
    blocked/allowed anchors, judge result, threshold, mode) for debugging/forensics.
    `fallback` is a SUGGESTED safe replacement message, populated only when the
    recommendation is REDIRECT or BLOCK — text the caller MAY use; the guard never
    sends it (guardrail #1). It is None for ALLOW/MONITOR.
    """

    score: float
    decision: Decision
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    recommended_action: RecommendedAction = RecommendedAction.ALLOW
    fallback: Optional[str] = None

    @property
    def aligned(self) -> bool:
        """Convenience: did this write pass as on-purpose?"""
        return self.decision == Decision.ALLOW

    def __str__(self) -> str:
        return (
            f"Verdict({self.decision.value}, score={self.score:.2f}, "
            f"reason={self.reason!r})"
        )
