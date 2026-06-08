"""Shared helper for the examples: pick the best available scorer, honestly.

Examples should reflect REAL behavior. With the `[embeddings]` extra installed you
get realistic semantic scoring; without it you fall back to the lexical FLOOR,
which is crash-proof but rough (it keys on word overlap and misclassifies a lot
of natural prose). This prints a visible notice in the fallback case so an example
never silently pretends the floor is the real thing.

(ASCII-only output on purpose: a non-ASCII glyph here would crash a Windows
cp1252 console — guardrail #4.)
"""

from __future__ import annotations

import warnings


def pick_scorer():
    """Return an embedding scorer if the model actually loads, else the lexical
    floor — printing a visible notice when it falls back."""
    from purposeguard import EmbeddingScorer, LexicalScorer

    emb = EmbeddingScorer()
    with warnings.catch_warnings():     # the scorer warns on fallback; we print our own notice
        warnings.simplefilter("ignore")
        emb.score("probe", "probe")     # forces a real model load (or graceful fallback)
    if getattr(emb, "_fallback", None) is None:
        return emb                       # embeddings genuinely loaded

    print(
        "[note] embeddings unavailable -- using lexical floor; verdicts will be "
        "rough.\n        Install  purposeguard[embeddings]  for realistic results.\n"
    )
    return LexicalScorer()
