"""Hermetic comparison baselines for the benchmark.

A benchmark with no baseline can't show PurposeGuard *beats* anything — its only
comparator used to be its own degraded lexical floor. These are the trivial
alternatives an adopter would otherwise reach for, so PurposeGuard's lift over them
is meaningful:

  * always-allow      — the do-nothing null model (FPR 0, recall 0). The floor any
                        detector must clear to justify existing.
  * keyword-blocklist — flag a write that shares too few of the purpose's salient
                        keywords. The obvious non-ML alternative.

Both are pure-stdlib and deterministic, so they run in CI with no model.
"""

from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
# Generic words that carry no topical signal — dropped when picking purpose keywords.
_STOP = {
    "a", "an", "the", "and", "or", "for", "of", "to", "in", "on", "with", "that",
    "helps", "help", "assistant", "agent", "questions", "about", "customer", "support",
}


def purpose_keywords(purpose: str, top: int = 12) -> set:
    """The salient keywords of a purpose string (most-common non-stopword tokens)."""
    toks = [w for w in _WORD.findall(purpose.lower()) if w not in _STOP and len(w) > 2]
    return {w for w, _ in Counter(toks).most_common(top)}


def keyword_flags(writes: list, purpose: str, min_overlap: int = 1) -> list:
    """Flag a write that shares fewer than ``min_overlap`` of the purpose keywords."""
    kw = purpose_keywords(purpose)
    out = []
    for w in writes:
        toks = set(_WORD.findall(w.lower()))
        out.append(len(toks & kw) < min_overlap)
    return out


def always_allow_flags(writes: list) -> list:
    """The null model: never flags anything."""
    return [False] * len(writes)


def metrics_from_flags(flags: list, labels: list) -> dict:
    """FPR / precision / recall of a per-write flag vector vs ground-truth labels
    (1 = off-mission). Returns None for an undefined ratio (no positives, etc.)."""
    n = len(labels)
    on_mission = labels.count(0)
    has_pos = labels.count(1) > 0
    tp = sum(1 for i in range(n) if flags[i] and labels[i] == 1)
    fp = sum(1 for i in range(n) if flags[i] and labels[i] == 0)
    fn = sum(1 for i in range(n) if not flags[i] and labels[i] == 1)
    return {
        "fpr": (fp / on_mission) if on_mission else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "recall": (tp / (tp + fn)) if has_pos else None,
    }
