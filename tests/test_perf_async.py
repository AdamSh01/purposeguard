"""Performance (embed-once) + async API tests. Hermetic.

The core saving: one check scores the content against purpose + every allowed +
every blocked anchor, but the CONTENT should be embedded ONCE — not once per
anchor. Verified here with a counting scorer. Also covers the async wrappers and
the bounded reference cache.
"""

import asyncio
import math
import warnings
from collections import Counter

from purposeguard import EmbeddingScorer, LexicalScorer, PurposeGuard


class CountingScorer:
    """Counts how many times CONTENT is embedded. Implements the score_many fast
    path (content embedded once) and a score() fallback (content embedded per call)."""

    def __init__(self):
        self.content_embeds = 0

    @staticmethod
    def _vec(text):
        return Counter(text.lower().split())

    @staticmethod
    def _sim(a, b):
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b[t] for t in set(a) & set(b))
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def score(self, content, reference):
        self.content_embeds += 1
        return self._sim(self._vec(content), self._vec(reference))

    def score_many(self, content, references):
        self.content_embeds += 1  # content embedded ONCE per check
        c = self._vec(content)
        return [self._sim(c, self._vec(r)) for r in references]


class NoBatchScorer(CountingScorer):
    score_many = None  # force the guard onto the per-reference fallback path


def _guard(scorer, **kw):
    kw.setdefault("threshold", 0.05)
    kw.setdefault("blocked_threshold", 0.9)
    return PurposeGuard(
        "billing payment refund invoice subscription account",
        scorer=scorer,
        allowed_topics=["a one", "b two", "c three"],
        blocked_topics=["x nine", "y eight", "z seven"],
        **kw,
    )


def test_check_embeds_content_once_regardless_of_anchor_count():
    scorer = CountingScorer()
    guard = _guard(scorer)
    scorer.content_embeds = 0
    guard.check("billing payment refund")
    # 1 purpose + 3 allowed + 3 blocked references, but content embedded ONCE.
    assert scorer.content_embeds == 1


def test_without_score_many_falls_back_to_per_reference():
    scorer = NoBatchScorer()
    guard = _guard(scorer)
    scorer.content_embeds = 0
    guard.check("billing payment refund")
    # No fast path -> one score() (one content embed) per reference: 1+3+3.
    assert scorer.content_embeds == 7


def test_lexical_score_many_matches_score():
    s = LexicalScorer()
    content = "billing payment refund invoice"
    refs = ["billing invoice", "weather forecast", "payment subscription charge"]
    assert s.score_many(content, refs) == [s.score(content, r) for r in refs]


def test_embed_once_does_not_change_verdicts():
    """The embed-once refactor must be behavior-preserving vs per-anchor scoring."""
    fast = _guard(CountingScorer())
    slow = _guard(NoBatchScorer())
    for text in ["billing payment refund", "x nine y eight", "totally unrelated text"]:
        vf, vs = fast.check(text, record=False), slow.check(text, record=False)
        assert vf.decision == vs.decision
        assert vf.score == vs.score


def test_acheck_equals_check():
    g1 = _guard(CountingScorer())
    g2 = _guard(CountingScorer())
    sync = g1.check("billing payment refund")
    a = asyncio.run(g2.acheck("billing payment refund"))
    assert (a.decision, a.score) == (sync.decision, sync.score)


def test_acheck_response_equals_check_response():
    g1 = _guard(CountingScorer())
    g2 = _guard(CountingScorer())
    sync = g1.check_response("hi", "billing payment refund")
    a = asyncio.run(g2.acheck_response("hi", "billing payment refund"))
    assert (a.decision, a.score) == (sync.decision, sync.score)


def test_check_many_matches_sequential_checks():
    g1 = _guard(CountingScorer())
    g2 = _guard(CountingScorer())
    writes = ["billing payment", "weather forecast", "refund invoice"]
    batch = g1.check_many(writes)
    seq = [g2.check(w) for w in writes]
    assert [v.score for v in batch] == [v.score for v in seq]


def test_reference_cache_is_bounded():
    """The EmbeddingScorer ref cache is a bounded LRU. Force the lexical fallback
    (no model in this env) so the cache path runs hermetically, with a tiny cap."""
    scorer = EmbeddingScorer()
    scorer._ref_cache_max = 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress the one-time fallback warning
        scorer.score_many("some content here", ["alpha", "beta", "gamma", "delta"])
    assert len(scorer._ref_cache) <= 2
