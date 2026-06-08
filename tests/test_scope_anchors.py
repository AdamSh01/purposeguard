"""Tests for v0.2 scope anchors: allowed_topics / blocked_topics (Step 1).

Hermetic (LexicalScorer). Content is keyword-style so the lexical floor separates
cleanly. Combination logic under test:
  alignment   = max(sim(content, purpose), best sim(content, allowed))
  blocked_hit = sim(content, any blocked) >= blocked_threshold  -> FLAG (overrides)
"""

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard

PURPOSE = "billing payment subscription invoice refund charge"
BLOCKED = ["weather forecast vacation hiking", "football match game score"]
ALLOWED = ["account profile settings preferences notifications"]


def guard(**kw):
    kw.setdefault("scorer", LexicalScorer())
    kw.setdefault("threshold", 0.12)
    # Mechanism tests on LEXICAL scores: set an explicit blocked_threshold suited
    # to lexical overlap. (The library default ~0.46 is calibrated for the
    # embedding scorer; see test_blocked_threshold_default_is_decoupled.)
    kw.setdefault("blocked_threshold", 0.3)
    return PurposeGuard(PURPOSE, **kw)


def test_in_scope_passes_with_lists_present():
    g = guard(blocked_topics=BLOCKED, allowed_topics=ALLOWED)
    v = g.check("Issued a refund for the duplicate invoice and payment charge")
    assert v.decision == Decision.ALLOW


def test_blocked_topic_content_flags_even_when_off_purpose():
    g = guard(blocked_topics=BLOCKED)
    v = g.check("Checking the weather forecast for your weekend vacation hiking trip")
    assert v.decision == Decision.FLAG
    assert v.details["blocked"]["hit"] is True
    assert "weather forecast" in v.details["blocked"]["topic"]
    assert "blocked topic" in v.reason


def test_camouflage_purpose_words_plus_blocked_content_now_caught():
    """The headline case: on-mission vocab can't rescue blocked content."""
    g = guard(blocked_topics=BLOCKED)
    camo = "billing payment invoice refund — checking the weather forecast for your vacation"
    v = g.check(camo)
    # It IS purpose-similar (high alignment) ...
    assert v.score >= g.threshold
    # ... but the blocked hit overrides and flags it anyway.
    assert v.decision == Decision.FLAG
    assert v.details["blocked"]["hit"] is True
    assert "blocked topic" in v.reason


def test_no_lists_is_identical_to_v01_behavior():
    plain = guard()  # no allowed/blocked
    on = plain.check("billing payment invoice refund")
    off = plain.check("weather forecast vacation hiking")
    assert on.decision == Decision.ALLOW
    assert off.decision == Decision.FLAG          # flagged by alignment, not a blocked hit
    assert "blocked" not in off.details and "allowed" not in off.details
    # score is still the plain purpose alignment
    assert off.details["similarity"] == off.score


def test_allowed_topics_confirm_borderline_in_scope_content():
    content = "Updated the account profile settings and notification preferences"
    without = guard().check(content)            # only loosely matches PURPOSE -> flag
    withlist = guard(allowed_topics=ALLOWED).check(content)
    assert without.decision == Decision.FLAG
    assert withlist.decision == Decision.ALLOW  # allowed topic widened "in scope"
    assert withlist.details["allowed"]["score"] >= withlist.details["threshold"]


def test_blocked_hit_does_not_feed_the_drift_meter():
    """Decision: a blocked hit is a hard flag, not drift. The meter still tracks
    ALIGNMENT, so a purpose-similar-but-blocked write keeps drift high, not low."""
    g = guard(blocked_topics=BLOCKED)
    camo = "billing payment invoice refund — weather forecast vacation"
    v = g.check(camo)
    assert v.decision == Decision.FLAG               # blocked
    # The meter recorded the (high) alignment, not a low blocked score:
    assert g.drift().current == v.score
    assert v.score >= g.threshold


def test_blocked_threshold_default_is_decoupled_from_alignment_threshold():
    """Regression for the v0.2.0 footgun: blocked_threshold must NOT default to the
    (low) alignment threshold; it has its own, higher, TRAIN-calibrated default."""
    from purposeguard.guard import DEFAULT_BLOCKED_THRESHOLD

    g = PurposeGuard(PURPOSE, scorer=LexicalScorer(), threshold=0.12, blocked_topics=["x"])
    assert g.blocked_threshold == DEFAULT_BLOCKED_THRESHOLD
    assert g.blocked_threshold > g.threshold          # decoupled, and higher
    assert DEFAULT_BLOCKED_THRESHOLD >= 0.4           # clearly above the ~0.15 alignment band


def test_legit_in_domain_write_not_overflagged_by_near_domain_blocked_anchor():
    """Regression (embedding): at the DEFAULT blocked_threshold, a NEAR-domain
    blocked anchor must not fire on legitimate in-domain writes (the v0.2.0 bug
    over-flagged 7/10 billing writes). Embedding-specific, so importorskip."""
    import warnings

    pytest.importorskip("sentence_transformers")
    from purposeguard import EmbeddingScorer

    scorer = EmbeddingScorer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer.score("probe", "probe")
    if getattr(scorer, "_fallback", None) is not None:
        pytest.skip("embedding model unavailable")

    g = PurposeGuard(  # blocked_threshold defaults to the calibrated ~0.46
        "A customer-support assistant for billing, payments, invoices, and refunds",
        scorer=scorer,
        blocked_topics=["legal advice"],  # near-domain to billing disputes
    )
    legit = [
        "Issued a refund for the duplicate invoice on the customer's account.",
        "Updated the payment card on file for the monthly subscription.",
        "Explained why the latest invoice was higher this billing cycle.",
    ]
    wrongly_blocked = [w for w in legit if g.check(w).decision == Decision.FLAG]
    assert wrongly_blocked == [], f"legit in-domain writes wrongly blocked: {wrongly_blocked}"


def test_blocked_threshold_override():
    # A very high blocked_threshold means WEAK blocked-similarity won't trip it.
    g = guard(blocked_topics=BLOCKED, blocked_threshold=0.99)
    v = g.check("checking tomorrow's forecast briefly")  # only loosely ~ a blocked anchor
    assert v.details["blocked"]["hit"] is False
    # falls through to alignment -> off-purpose -> flagged by alignment, not blocked
    assert v.decision == Decision.FLAG
    assert "alignment" in v.reason
