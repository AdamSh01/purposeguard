"""Tests for honest scorer selection: warn-once on the accidental lexical floor,
and opt-in require_embeddings strictness.

Hermetic: embedding availability is simulated by blocking the import, so these
run in CI with nothing installed and never download a model.
"""

import warnings

import pytest

import purposeguard.scoring as scoring
from purposeguard import LexicalScorer, PurposeGuard
from purposeguard.scoring import default_scorer


def _block_import(monkeypatch, name):
    """Make `import <name>` raise, to simulate the extra not being installed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(n, *args, **kwargs):
        if n == name or n.startswith(name + "."):
            raise ImportError(f"simulated: {name} not installed")
        return real_import(n, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _runtime_warnings(records):
    return [r for r in records if issubclass(r.category, RuntimeWarning)]


# -- Step 1: warn-once on the ACCIDENTAL lexical floor --------------------------

def test_default_scorer_warns_on_accidental_lexical_fallback(monkeypatch):
    scoring._warned_lexical_fallback = False
    _block_import(monkeypatch, "sentence_transformers")
    with pytest.warns(RuntimeWarning, match="embeddings"):
        s = default_scorer()
    assert isinstance(s, LexicalScorer)


def test_default_scorer_warns_only_once(monkeypatch):
    scoring._warned_lexical_fallback = False
    _block_import(monkeypatch, "sentence_transformers")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        default_scorer()
        default_scorer()
    assert len(_runtime_warnings(rec)) == 1  # warn-once, not per-call


def test_default_scorer_opted_out_does_not_warn(monkeypatch):
    """prefer_embeddings=False is a deliberate choice, not an accident."""
    scoring._warned_lexical_fallback = False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        s = default_scorer(prefer_embeddings=False)
    assert isinstance(s, LexicalScorer)
    assert _runtime_warnings(rec) == []


def test_explicit_lexical_scorer_does_not_warn():
    """The key requirement: a user who deliberately passes LexicalScorer() (e.g.
    firewalled, knows what they're doing) is NOT nagged — default_scorer isn't
    even reached."""
    scoring._warned_lexical_fallback = False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        PurposeGuard("billing payment refund", scorer=LexicalScorer())
    assert _runtime_warnings(rec) == []


# -- Step 3: opt-in require_embeddings -----------------------------------------

def test_require_embeddings_raises_when_extra_missing(monkeypatch):
    _block_import(monkeypatch, "sentence_transformers")
    with pytest.raises(RuntimeError) as excinfo:
        PurposeGuard("billing payment refund", require_embeddings=True)
    assert "embeddings" in str(excinfo.value)


def test_require_embeddings_rejects_explicit_lexical():
    with pytest.raises(ValueError):
        PurposeGuard(
            "billing payment refund",
            scorer=LexicalScorer(),
            require_embeddings=True,
        )


def test_require_embeddings_accepts_custom_nonlexical_scorer():
    """A custom (non-lexical) scorer is not the floor, so it's accepted."""

    class FakeScorer:
        def score(self, content, reference):
            return 0.5

    g = PurposeGuard("billing payment refund", scorer=FakeScorer(), require_embeddings=True)
    assert isinstance(g.scorer, FakeScorer)


def test_default_does_not_require_embeddings():
    """Zero-dep behavior unchanged: no require_embeddings -> lexical is fine."""
    g = PurposeGuard("billing payment refund", scorer=LexicalScorer())
    assert isinstance(g.scorer, LexicalScorer)
