"""Tests for the PurposeGuard core. Run: python -m pytest tests/ -v

These use the LexicalScorer so they run offline with no model downloads, which
also keeps CI fast and hermetic. The embedding path is exercised separately in
environments that can load the model.
"""

import pytest

from purposeguard import Decision, DriftReading, LexicalScorer, PurposeGuard, Write


def make_guard(threshold=0.12):
    return PurposeGuard(
        purpose="billing payment subscription invoice refund account support",
        scorer=LexicalScorer(),
        threshold=threshold,
    )


def test_on_purpose_allows():
    g = make_guard()
    v = g.check("How do I update my subscription payment and billing details?")
    assert v.decision == Decision.ALLOW
    assert v.aligned
    assert 0.0 <= v.score <= 1.0


def test_off_purpose_flags():
    g = make_guard()
    v = g.check("What is the capital of France?")
    assert v.decision == Decision.FLAG
    assert not v.aligned


def test_never_blocks_in_v01():
    """Detection-first guarantee: v0.1 must never emit BLOCK."""
    g = make_guard()
    for text in ["totally unrelated", "billing payment", "", "x" * 1000]:
        v = g.check(text or "placeholder")
        assert v.decision != Decision.BLOCK


def test_accepts_str_and_write():
    g = make_guard()
    v1 = g.check("billing invoice refund")
    v2 = g.check(Write(content="billing invoice refund", key="mem:1"))
    assert v1.score == pytest.approx(v2.score)


def test_purpose_is_immutable():
    g = make_guard()
    with pytest.raises(AttributeError):
        g.purpose = "something else"  # type: ignore[misc]


def test_empty_purpose_rejected():
    with pytest.raises(ValueError):
        PurposeGuard(purpose="   ", scorer=LexicalScorer())


def test_drift_rises_when_agent_wanders():
    g = PurposeGuard(
        purpose="billing payment subscription invoice refund",
        scorer=LexicalScorer(),
        threshold=0.10,
        drift_baseline_window=3,
        drift_alpha=0.5,
    )
    for on in ["billing payment invoice", "subscription refund billing", "payment invoice refund"]:
        g.check(on)
    baseline = g.drift().baseline
    for off in ["weather forecast vacation", "sourdough bread recipe", "favorite football team game"]:
        g.check(off)
    reading = g.drift()
    assert reading.drift > 0.0
    assert reading.current < baseline


def test_judge_only_consulted_near_threshold():
    calls = []

    def judge(content, purpose):
        calls.append(content)
        return True  # always rescue

    g = PurposeGuard(
        purpose="billing payment subscription",
        scorer=LexicalScorer(),
        threshold=0.12,
        judge=judge,
        judge_band=0.05,
    )
    # Clearly off-purpose: far below threshold, judge should NOT be consulted.
    g.check("the capital of France is Paris and the weather is nice")
    assert calls == []


def test_dry_run_does_not_affect_drift():
    g = make_guard()
    g.check("billing payment invoice")  # records
    n_before = g.drift().samples
    g.check("weather forecast", record=False)  # dry run
    assert g.drift().samples == n_before


def test_driftreading_str_is_ascii_safe():
    """Guardrail #4 (never crash): emitted strings must survive a cp1252 console.

    DriftReading.__str__ is printed and logged by adopters; a non-ASCII glyph in
    it raises UnicodeEncodeError on a default Windows console. Assert the rendered
    string is ASCII-encodable in BOTH the drifting and non-drifting branches so no
    future contributor can reintroduce a glyph without this test failing.
    """
    not_drifting = DriftReading(current=0.90, baseline=0.90, drift=0.00, samples=5)
    drifting = DriftReading(current=0.10, baseline=0.50, drift=0.40, samples=10)
    # Sanity: we are actually exercising both branches of the flag.
    assert not not_drifting.drifting
    assert drifting.drifting
    # The real assertion: neither rendering raises UnicodeEncodeError.
    str(not_drifting).encode("ascii")
    str(drifting).encode("ascii")


def test_reanchor_wraps_purpose_front_and_back():
    g = make_guard()
    out = g.reanchor("conversation body here")
    assert out.startswith("[PURPOSE]")
    assert out.rstrip().endswith(g.purpose)
    assert "conversation body here" in out
