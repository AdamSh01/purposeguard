"""Tests for the embedding-score rescale and the threshold presets (Task 3).

Hermetic: the rescale math is tested via the pure `_rescale_cosine` helper (no
model load, no network), and presets are plain data + a constructor. The live
embedding accuracy is validated separately by the benchmark.
"""

import pytest

from purposeguard import (
    BALANCED,
    BROAD,
    NARROW,
    EmbeddingScorer,
    LexicalScorer,
    PurposeGuard,
    get_preset,
)
from purposeguard.scoring import _rescale_cosine


# -- the score-compression fix -------------------------------------------------

def test_rescale_maps_band_to_unit_interval():
    # Default band [0.0, 0.61]: floor -> 0, ceil -> 1, midpoint -> 0.5.
    assert _rescale_cosine(0.0) == pytest.approx(0.0)
    assert _rescale_cosine(0.61) == pytest.approx(1.0)
    assert _rescale_cosine(0.305) == pytest.approx(0.5)


def test_rescale_clamps_outside_band():
    assert _rescale_cosine(-0.3) == 0.0
    assert _rescale_cosine(0.9) == 1.0


def test_rescale_is_monotonic():
    assert _rescale_cosine(0.05) < _rescale_cosine(0.20) < _rescale_cosine(0.55)


def test_rescale_restores_dynamic_range_vs_old_remap():
    """The whole point of the rescale: stop squashing the real cosine band.

    Over the realistic band [0, 0.61], the old (cos+1)/2 remap spanned only
    [0.5, ~0.805] (range ~0.305); the rescale spans the full [0,1] (range 1.0).
    """
    old = lambda c: (c + 1.0) / 2.0
    new_spread = _rescale_cosine(0.61) - _rescale_cosine(0.0)
    old_spread = old(0.61) - old(0.0)
    assert new_spread > old_spread
    assert new_spread == pytest.approx(1.0)


def test_rescale_degenerate_band_is_safe():
    assert _rescale_cosine(0.3, floor=0.5, ceil=0.5) == 0.0  # ceil <= floor


# -- graceful degradation (guardrail #4) ---------------------------------------

@pytest.mark.parametrize("missing", ["numpy", "sentence_transformers"])
def test_embedding_scorer_falls_back_when_optional_dep_missing(monkeypatch, missing):
    """EmbeddingScorer must degrade to the lexical floor — never crash — when an
    optional dependency can't be imported (the model lib OR numpy).

    Closes the class of guardrail-#4 defects rather than patching one dep:
    whichever optional import is simulated-missing, scoring still returns a valid
    value and warns once. Hermetic: we block the import instead of uninstalling,
    and the lexical fallback path never touches the blocked module.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == missing or name.startswith(missing + "."):
            raise ImportError(f"simulated: {missing} not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    scorer = EmbeddingScorer()
    with pytest.warns(RuntimeWarning):
        score = scorer.score("how do refunds work", "billing support assistant")
    assert isinstance(score, float) and 0.0 <= score <= 1.0
    # warn-once: the fallback is now wired in, later calls stay quiet and work.
    again = scorer.score("another billing question", "billing support assistant")
    assert isinstance(again, float) and 0.0 <= again <= 1.0


# -- presets -------------------------------------------------------------------

def test_preset_thresholds_are_ordered():
    # Narrower agent -> stricter (higher) threshold; broader -> more lenient.
    assert NARROW.threshold > BALANCED.threshold > BROAD.threshold


def test_get_preset_is_case_insensitive():
    assert get_preset("BROAD") is BROAD
    assert get_preset("balanced") is BALANCED
    assert get_preset(NARROW) is NARROW  # pass-through


def test_get_preset_unknown_raises():
    with pytest.raises(ValueError):
        get_preset("aggressive")


def test_default_threshold_is_balanced():
    g = PurposeGuard("billing support", scorer=LexicalScorer())
    assert g.threshold == pytest.approx(BALANCED.threshold)


def test_from_preset_applies_preset_threshold():
    g = PurposeGuard.from_preset("broad", "billing support", scorer=LexicalScorer())
    assert g.threshold == pytest.approx(BROAD.threshold)


def test_from_preset_explicit_threshold_overrides():
    g = PurposeGuard.from_preset(
        "broad", "billing support", scorer=LexicalScorer(), threshold=0.33
    )
    assert g.threshold == pytest.approx(0.33)
