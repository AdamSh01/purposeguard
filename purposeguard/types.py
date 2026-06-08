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
    """What the guard concluded about a write.

    v0.1 is detection-first, so the guard only ever returns ALLOW or FLAG.
    BLOCK exists in the enum so adapters can be written against the full set,
    but no v0.1 policy returns it — nothing can silently drop a user's data.
    """

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"  # reserved for opt-in enforcement (post-v0.1)


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
    is perfectly on-mission. `decision` is the policy's action. `reason` is a
    short human-readable explanation suitable for logging. `details` carries the
    raw signals (similarity, judge result, threshold) for debugging and for the
    forensic trail.
    """

    score: float
    decision: Decision
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def aligned(self) -> bool:
        """Convenience: did this write pass as on-purpose?"""
        return self.decision == Decision.ALLOW

    def __str__(self) -> str:
        return (
            f"Verdict({self.decision.value}, score={self.score:.2f}, "
            f"reason={self.reason!r})"
        )
