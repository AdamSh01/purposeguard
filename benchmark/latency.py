"""Per-check latency microbenchmark (Phase-2).

Answers the question every adopter asks before putting a guard on the write path:
"what does a check cost?" Reports p50/p95 ms/check for the lexical floor (always)
and the embedding scorer (if it loads), and how cost scales with the number of
allowed/blocked anchors — which, thanks to the embed-once fast path, should add
cheap reference look-ups, not a re-embed of the content.

    python benchmark/latency.py
"""

from __future__ import annotations

import time
import warnings
from statistics import median

try:
    from traces import BILLING_ON_MISSION, PURPOSE_BILLING
except ImportError:  # allow `python -m benchmark.latency`
    from benchmark.traces import BILLING_ON_MISSION, PURPOSE_BILLING

from purposeguard import EmbeddingScorer, LexicalScorer, PurposeGuard


def _percentile(xs: list, p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def time_checks(guard: PurposeGuard, writes: list, repeats: int = 5) -> list:
    samples = []
    for _ in range(repeats):
        for w in writes:
            t0 = time.perf_counter()
            guard.check(w, record=False)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def run(label: str, scorer) -> None:
    writes = BILLING_ON_MISSION[:10]
    for n_anchors in (0, 3, 10):
        guard = PurposeGuard(
            PURPOSE_BILLING,
            scorer=scorer,
            allowed_topics=[f"in-scope topic number {i}" for i in range(n_anchors)],
        )
        # one warm-up check so model/cache costs aren't charged to the first sample
        guard.check(writes[0], record=False)
        s = time_checks(guard, writes)
        print(f"  {label:<10} anchors={n_anchors:<3} "
              f"p50={median(s):.3f}ms  p95={_percentile(s, 95):.3f}ms  (n={len(s)})")


def main() -> None:
    print("PurposeGuard per-check latency (ms/check, record=False). Lower is better.\n")
    run("lexical", LexicalScorer())

    emb = EmbeddingScorer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb.score("probe", "probe")  # force a real load (or fallback)
    if getattr(emb, "_fallback", None) is None:
        run("embedding", emb)
    else:
        print("\n[note] embeddings unavailable (model not loaded); reporting lexical only.")


if __name__ == "__main__":
    main()
