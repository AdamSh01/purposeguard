"""Tests for v0.2 modes: monitor / redirect / block (Step 2).

Hermetic (LexicalScorer). Modes change ONLY the verdict's recommended_action;
the guard never acts on the caller's data. monitor is the default and must match
no-mode behavior exactly.
"""

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard, RecommendedAction, Write

PURPOSE = "billing payment subscription invoice refund charge"
ON = "Issued a refund for the duplicate invoice payment charge"
OFF = "weather forecast vacation hiking trip"


def guard(**kw):
    kw.setdefault("scorer", LexicalScorer())
    kw.setdefault("threshold", 0.12)
    return PurposeGuard(PURPOSE, **kw)


def test_monitor_is_default():
    g = guard()
    assert g.mode == "monitor"


def test_monitor_flags_only_and_recommends_monitor():
    g = guard(mode="monitor")
    on = g.check(ON)
    off = g.check(OFF)
    assert on.decision == Decision.ALLOW and on.recommended_action == RecommendedAction.ALLOW
    assert off.decision == Decision.FLAG and off.recommended_action == RecommendedAction.MONITOR
    assert off.details["mode"] == "monitor"


def test_redirect_mode_recommends_redirect():
    g = guard(mode="redirect")
    off = g.check(OFF)
    assert off.decision == Decision.FLAG               # detection unchanged
    assert off.recommended_action == RecommendedAction.REDIRECT
    # on-scope still just allows, regardless of mode
    assert g.check(ON).recommended_action == RecommendedAction.ALLOW


def test_block_mode_recommends_block_but_decision_is_still_flag():
    g = guard(mode="block")
    off = g.check(OFF)
    assert off.recommended_action == RecommendedAction.BLOCK
    # CRITICAL: the guard's DECISION stays detection (FLAG), never Decision.BLOCK.
    assert off.decision == Decision.FLAG
    assert off.decision != Decision.BLOCK


def test_on_scope_recommends_allow_in_every_mode():
    for m in ("monitor", "redirect", "block"):
        v = guard(mode=m).check(ON)
        assert v.decision == Decision.ALLOW
        assert v.recommended_action == RecommendedAction.ALLOW


def test_decision_is_never_block_in_any_mode():
    for m in ("monitor", "redirect", "block"):
        for text in (ON, OFF, "", "x" * 300):
            v = guard(mode=m).check(text or "placeholder")
            assert v.decision != Decision.BLOCK


def test_modes_never_mutate_or_drop_caller_data():
    """The guard returns a recommendation; it never acts. Even in block mode it
    raises nothing and leaves the caller's input untouched."""
    g = guard(mode="block")
    w = Write(content=OFF, key="mem:1", metadata={"src": "agent"})
    v = g.check(w)  # must not raise, must not stop anything
    assert v.recommended_action == RecommendedAction.BLOCK
    # input object is unchanged — the guard added nothing, removed nothing.
    assert w.content == OFF
    assert w.key == "mem:1"
    assert w.metadata == {"src": "agent"}


def test_monitor_matches_no_mode_default_exactly():
    default = guard()             # no mode kwarg
    explicit = guard(mode="monitor")
    for text in (ON, OFF):
        a, b = default.check(text), explicit.check(text)
        assert (a.decision, a.score, a.recommended_action) == (
            b.decision, b.score, b.recommended_action
        )


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        guard(mode="nuke")


def test_check_response_honors_mode():
    g = guard(mode="redirect")
    v = g.check_response("what's the weather?", OFF)
    assert v.recommended_action == RecommendedAction.REDIRECT
    assert v.details["kind"] == "response"
