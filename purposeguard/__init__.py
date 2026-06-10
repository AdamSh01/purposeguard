"""PurposeGuard — a framework-agnostic drift guard for AI agent memory.

This module is the public surface of the library. Everything an adopter needs is
re-exported here so the common case is a single import:

    from purposeguard import PurposeGuard

    guard = PurposeGuard(purpose="A customer-support agent for billing questions")
    verdict = guard.check("User asked how to bake sourdough bread")

The internal modules (``types``, ``scoring``, ``drift``, ``guard``) are kept
small and framework-agnostic on purpose; pulling their names up to the package
root means callers never have to know which file a symbol lives in, and we keep
the freedom to move things internally without breaking imports.
"""

from __future__ import annotations

from .drift import DriftMeter, DriftReading
from .guard import PurposeGuard
from .presets import BALANCED, BROAD, NARROW, Preset, get_preset
from .scoring import EmbeddingScorer, LexicalScorer, Scorer, default_scorer
from .types import Decision, RecommendedAction, Verdict, Write

__version__ = "0.3.0"

__all__ = [
    "PurposeGuard",
    "Write",
    "Verdict",
    "Decision",
    "RecommendedAction",
    "DriftMeter",
    "DriftReading",
    "Scorer",
    "EmbeddingScorer",
    "LexicalScorer",
    "default_scorer",
    "Preset",
    "NARROW",
    "BALANCED",
    "BROAD",
    "get_preset",
    "__version__",
]
