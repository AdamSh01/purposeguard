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


def _rescale_cosine(cos: float, floor: float = 0.0, ceil: float = 0.61) -> float:
    """Clamp-and-stretch a raw cosine into [0,1] using a realistic band.

    Sentence-embedding cosines for natural text occupy a narrow band: with
    all-MiniLM, unrelated content sits near 0.0 and strongly on-topic content
    reaches only ~0.6. Mapping that band [floor, ceil] onto [0,1] keeps scores in
    the unit interval while using their real dynamic range, so a small cosine gap
    becomes a usable score gap. (The earlier (cos+1)/2 remap instead squashed
    everything into ~[0.45, 0.75], shrinking both the per-write margin and the
    drift delta — see benchmark/RESULTS.md.) Values outside the band clamp.

    The default band [0.0, 0.61] is calibrated on the benchmark's TRAIN traces
    only (ceil = 95th percentile of narrow on-mission cosines), NOT on the
    reported test set — see benchmark/calibrate.py. It is domain-sensitive; tune
    via EmbeddingScorer(cos_floor=..., cos_ceil=...) for your corpus.
    """
    if ceil <= floor:
        return 0.0
    return max(0.0, min(1.0, (cos - floor) / (ceil - floor)))


class EmbeddingScorer:
    """Local sentence-embedding scorer (preferred when available).

    Uses all-MiniLM-L6-v2 by default: small, fast, runs on CPU, no API key.

    Scoring is cosine similarity followed by a clamp-and-stretch into [0,1] (see
    :func:`_rescale_cosine`). The default band [``cos_floor``, ``cos_ceil``] =
    [0.0, 0.61] is calibrated on the benchmark's TRAIN traces only (off-mission
    writes cluster at ~0.0; ceil is the 95th percentile of narrow on-mission
    cosines). The model is loaded lazily so importing the library is cheap and a
    caller who never scores never pays the load cost.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        cos_floor: float = 0.0,
        cos_ceil: float = 0.61,
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


# Module-level so the "you're on the floor" notice fires at most once per process
# (guardrail #4's warn-once pattern, applied to the function instead of an object).
_warned_lexical_fallback = False


def default_scorer(prefer_embeddings: bool = True) -> Scorer:
    """Pick the best available scorer.

    Tries the embedding scorer; if sentence-transformers isn't installed, falls
    back to lexical so the library always works. This is the single place that
    decides "fancy vs. floor", so the rest of the code never branches on it.

    When it falls back to lexical *because embeddings were unavailable* (not
    because the caller opted out), it warns ONCE — the silent floor is a
    misleading first impression. A caller who deliberately wants the floor passes
    ``LexicalScorer()`` explicitly (or ``prefer_embeddings=False``), which never
    reaches this warning.
    """
    global _warned_lexical_fallback
    if prefer_embeddings:
        try:
            import sentence_transformers  # noqa: F401

            return EmbeddingScorer()
        except Exception:
            if not _warned_lexical_fallback:
                _warned_lexical_fallback = True
                import warnings

                warnings.warn(
                    "PurposeGuard is using the lexical fallback scorer because the "
                    "'embeddings' extra is not installed; verdicts will be rough. "
                    "Install purposeguard[embeddings] for realistic results "
                    "(or pass scorer=LexicalScorer() to choose the floor and "
                    "silence this).",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return LexicalScorer()


def require_embedding_scorer(model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingScorer":
    """Return a verified-working EmbeddingScorer, or raise — never the floor.

    For callers who must NOT silently run on the lexical floor (see PurposeGuard's
    ``require_embeddings=True``). Fails loudly if the 'embeddings' extra is missing
    or the model can't actually be loaded, instead of degrading.
    """
    import warnings

    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "require_embeddings=True but the 'embeddings' extra is not installed. "
            'Install it with:  pip install "purposeguard[embeddings]"'
        ) from exc

    scorer = EmbeddingScorer(model_name)
    with warnings.catch_warnings():  # the probe would warn-on-fallback; we raise instead
        warnings.simplefilter("ignore")
        scorer.score("probe", "probe")  # force the model to load
    if scorer._fallback is not None:
        raise RuntimeError(
            f"require_embeddings=True but the embedding model '{model_name}' could "
            "not be loaded (offline or unavailable). Make the model available, or "
            "set require_embeddings=False to allow the lexical fallback."
        )
    return scorer


def resolve_scorer(scorer: Optional[Scorer] = None, *, require_embeddings: bool = False) -> Scorer:
    """Decide which scorer a guard should use.

    Default: the caller's scorer, else `default_scorer()` (which may warn-once if
    it falls back to lexical). With ``require_embeddings=True`` the lexical floor
    is never used silently — a working embedding scorer is guaranteed or a clear
    error is raised. Opt-in only; the default path is unchanged.
    """
    if not require_embeddings:
        return scorer or default_scorer()

    if scorer is None:
        return require_embedding_scorer()
    if isinstance(scorer, LexicalScorer):
        raise ValueError(
            "require_embeddings=True but a LexicalScorer was passed explicitly. "
            "Omit scorer to auto-load embeddings, pass a working EmbeddingScorer, "
            "or set require_embeddings=False."
        )
    if isinstance(scorer, EmbeddingScorer):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scorer.score("probe", "probe")
        if scorer._fallback is not None:
            raise RuntimeError(
                "require_embeddings=True but the provided EmbeddingScorer fell back "
                "to lexical (its model could not be loaded)."
            )
        return scorer
    return scorer  # a custom scorer: not the lexical floor, so trust the caller
