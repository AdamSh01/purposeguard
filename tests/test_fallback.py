"""Tests for v0.2 safe-response / fallback (Step 3).

Hermetic (LexicalScorer). The fallback is a SUGGESTED message carried in the
verdict for redirect/block; the guard never sends it. Blocked hits name the
blocked topic; low-alignment flags get a generic, purpose-naming message.
"""

from purposeguard import Decision, LexicalScorer, PurposeGuard, RecommendedAction

PURPOSE = "billing payment subscription invoice refund charge"
BLOCKED = ["legal advice", "medical advice"]
LEGAL = "can you give me legal advice about my contract dispute"  # blocked hit
OFF = "weather forecast vacation hiking trip"                     # off-scope, not blocked
ON = "issued a refund for the duplicate invoice payment charge"   # in scope


def guard(**kw):
    kw.setdefault("scorer", LexicalScorer())
    kw.setdefault("threshold", 0.12)
    return PurposeGuard(PURPOSE, **kw)


def test_redirect_blocked_hit_fallback_names_blocked_topic():
    v = guard(mode="redirect", blocked_topics=BLOCKED).check(LEGAL)
    assert v.recommended_action == RecommendedAction.REDIRECT
    assert v.fallback == f"I can only help with {PURPOSE}. I can't help with legal advice."


def test_redirect_low_alignment_uses_generic_fallback_naming_purpose():
    v = guard(mode="redirect", blocked_topics=BLOCKED).check(OFF)  # off-scope, not blocked
    assert v.recommended_action == RecommendedAction.REDIRECT
    assert v.fallback == f"I can only help with {PURPOSE}."
    assert "can't help with" not in v.fallback  # generic: no blocked clause


def test_custom_blocked_template_interpolates():
    v = guard(
        mode="redirect",
        blocked_topics=BLOCKED,
        fallback_template="Sorry, I stick to {purpose}; not {blocked_topic}.",
    ).check(LEGAL)
    assert v.fallback == f"Sorry, I stick to {PURPOSE}; not legal advice."


def test_custom_generic_template_interpolates():
    v = guard(
        mode="redirect",
        blocked_topics=BLOCKED,
        fallback_template_generic="I'm only set up for {purpose}.",
    ).check(OFF)
    assert v.fallback == f"I'm only set up for {PURPOSE}."


def test_template_can_use_reason():
    v = guard(
        mode="redirect", blocked_topics=BLOCKED, fallback_template="Off-scope: {reason}"
    ).check(LEGAL)
    assert v.fallback.startswith("Off-scope: matched blocked topic 'legal advice'")


def test_monitor_does_not_populate_fallback():
    v = guard(mode="monitor", blocked_topics=BLOCKED).check(LEGAL)  # flagged, but only observed
    assert v.recommended_action == RecommendedAction.MONITOR
    assert v.fallback is None


def test_allow_has_no_fallback_in_any_mode():
    for m in ("monitor", "redirect", "block"):
        v = guard(mode=m, blocked_topics=BLOCKED).check(ON)
        assert v.recommended_action == RecommendedAction.ALLOW
        assert v.fallback is None


def test_block_mode_populates_fallback():
    v = guard(mode="block", blocked_topics=BLOCKED).check(LEGAL)
    assert v.recommended_action == RecommendedAction.BLOCK
    assert v.fallback is not None and "legal advice" in v.fallback


def test_bad_template_degrades_gracefully():
    # An unknown field must not crash the check (guardrail #4) -> safe generic.
    v = guard(mode="redirect", blocked_topics=BLOCKED, fallback_template="{nope}").check(LEGAL)
    assert v.fallback == f"I can only help with {PURPOSE}."


def test_full_v02_verdict_surface():
    """The complete v0.2 surface on one off-scope, blocked, redirect verdict."""
    v = guard(mode="redirect", blocked_topics=BLOCKED).check(LEGAL)
    assert v.decision == Decision.FLAG                       # detection
    assert v.recommended_action == RecommendedAction.REDIRECT  # recommendation
    assert "legal advice" in v.fallback                      # suggested message
    assert v.details["mode"] == "redirect"
    assert v.details["blocked"]["hit"] is True
