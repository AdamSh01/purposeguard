"""Alignment scoring: how on-purpose is a piece of content?

Design goals, in priority order:
  1. Easy to adopt: works with zero model downloads if the user wants, via a
     dependency-light lexical fallback. Better accuracy is opt-in, not required.
  2. Honest: returns a score in [0,1] plus the raw signal, never a black box.
  3. Cheap by default: local embeddings, no API calls. An optional LLM judge can
     be layered on top by the policy for borderline cases, but the scorer itself
     never makes a network call.

The scorer answers one question: how semantically close is `content` to the
declared `purpose`? It does NOT decide allow/flag — that's the policy's job.
Separating "measure" from "act" is what lets v0.2 reuse this unchanged with a
different reference (memory consensus instead of purpose).
"""

from __future__ import annotations

import math
import re
from typing import Optional, Protocol


class Scorer(Protocol):
    """Anything that can rate content against a reference string in [0,1]."""

    def score(self, content: str, reference: str) -> float: ...


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LexicalScorer:
    """Dependency-light fallback scorer using token-overlap cosine.

    Not as good as embeddings at catching paraphrase or semantic drift, but it
    needs nothing beyond the standard library, so `pip install purposeguard`
    runs out of the box. We treat each text as a bag-of-words vector and take the
    cosine of the two. It's a floor, not a ceiling — good enough to demo and to
    keep the library usable in locked-down environments.
    """

    def score(self, content: str, reference: str) -> float:
        a, b = _tokens(content), _tokens(reference)
        if not a or not b:
            return 0.0
        from collections import Counter

        ca, cb = Counter(a), Counter(b)
        shared = set(ca) & set(cb)
        dot = sum(ca[t] * cb[t] for t in shared)
        na = math.sqrt(sum(v * v for v in ca.values()))
        nb = math.sqrt(sum(v * v for v in cb.values()))
        if na == 0 or nb == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (na * nb)))


def _rescale_cosine(cos: float, floor: float = 0.0, ceil: float = 0.5) -> float:
    """Clamp-and-stretch a raw cosine into [0,1] using a realistic band.

    Sentence-embedding cosines for natural text occupy a narrow band: with
    all-MiniLM, unrelated content sits near 0.0 and even strongly on-topic content
    only reaches ~0.5. Mapping that band [floor, ceil] onto [0,1] keeps scores in
    the unit interval while using their real dynamic range, so a small cosine gap
    becomes a usable score gap. (The earlier (cos+1)/2 remap instead squashed
    everything into ~[0.45, 0.75], shrinking both the per-write margin and the
    drift delta — see benchmark/RESULTS.md.) Values outside the band clamp.
    """
    if ceil <= floor:
        return 0.0
    return max(0.0, min(1.0, (cos - floor) / (ceil - floor)))


class EmbeddingScorer:
    """Local sentence-embedding scorer (preferred when available).

    Uses all-MiniLM-L6-v2 by default: small, fast, runs on CPU, no API key.

    Scoring is cosine similarity followed by a clamp-and-stretch into [0,1] (see
    :func:`_rescale_cosine`). The default band [``cos_floor``, ``cos_ceil``] =
    [0.0, 0.5] was chosen from the benchmark's measured cosine distributions:
    off-mission writes cluster at ~0.0 and strongly on-mission writes top out
    near ~0.5 for this model. The model is loaded lazily so importing the library
    is cheap and a caller who never scores never pays the load cost.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        cos_floor: float = 0.0,
        cos_ceil: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.cos_floor = cos_floor
        self.cos_ceil = cos_ceil
        self._model = None  # lazy
        self._ref_cache: dict[str, "object"] = {}
        # If the model can't be loaded (offline, firewalled, missing weights),
        # we degrade to a lexical scorer instead of crashing. An adopter behind
        # a corporate proxy should get a working library, not a stack trace.
        self._fallback: Optional["LexicalScorer"] = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score(self, content: str, reference: str) -> float:
        if self._fallback is not None:
            return self._fallback.score(content, reference)
        # numpy is part of the [embeddings] extra, so importing it can fail for
        # the same reasons the model can — keep it INSIDE the try so a missing
        # optional dep degrades to the lexical floor exactly like a missing model,
        # never a crash (guardrail #4; EmbeddingScorer is the cited example).
        try:
            import numpy as np

            model = self._ensure_model()
        except Exception:
            # First failure (missing model OR a missing optional dep like numpy):
            # warn once, then quietly use the lexical floor.
            import warnings

            warnings.warn(
                f"EmbeddingScorer could not initialize '{self.model_name}' "
                "(offline, model unavailable, or missing optional dependency); "
                "falling back to LexicalScorer. Install the 'embeddings' extra "
                "for full accuracy.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fallback = LexicalScorer()
            return self._fallback.score(content, reference)
        # Cache the reference (purpose) embedding — it rarely changes, and this
        # is the hot path: every write scores against the same purpose.
        ref_vec = self._ref_cache.get(reference)
        if ref_vec is None:
            ref_vec = model.encode(reference, normalize_embeddings=True)
            self._ref_cache[reference] = ref_vec
        con_vec = model.encode(content, normalize_embeddings=True)
        cos = float(np.dot(con_vec, ref_vec))  # already normalized -> cosine
        return _rescale_cosine(cos, self.cos_floor, self.cos_ceil)


def default_scorer(prefer_embeddings: bool = True) -> Scorer:
    """Pick the best available scorer.

    Tries the embedding scorer; if sentence-transformers isn't installed, falls
    back to lexical so the library always works. This is the single place that
    decides "fancy vs. floor", so the rest of the code never branches on it.
    """
    if prefer_embeddings:
        try:
            import sentence_transformers  # noqa: F401

            return EmbeddingScorer()
        except Exception:
            pass
    return LexicalScorer()
