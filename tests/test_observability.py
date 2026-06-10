"""Observability + persistence tests (2.5). Hermetic (LexicalScorer)."""

import json
import warnings

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard, Write

PURPOSE = "billing payment subscription invoice refund"


def g(**kw):
    kw.setdefault("scorer", LexicalScorer())
    kw.setdefault("threshold", 0.12)
    return PurposeGuard(PURPOSE, **kw)


def _drifting_guard():
    return PurposeGuard(
        PURPOSE, scorer=LexicalScorer(), threshold=0.10,
        drift_baseline_window=3, drift_alpha=0.5,
    )


def test_verdict_to_dict_is_json_serializable():
    v = g().check("weather forecast vacation")
    d = v.to_dict()
    json.dumps(d)  # must not raise
    assert d["decision"] in ("allow", "flag")
    assert d["recommended_action"] in ("allow", "monitor", "redirect", "block")
    assert "details" in d and "context" in d and "aligned" in d


def test_driftreading_to_dict_is_json_serializable():
    guard = g()
    guard.check("billing payment invoice")
    d = guard.drift().to_dict()
    json.dumps(d)
    assert {"current", "baseline", "drift", "samples", "drifting", "anchored_drifting"} <= set(d)


def test_context_carried_onto_verdict():
    v = g().check("billing payment", context={"session_id": "s1", "user_id": "u1"})
    assert v.context == {"session_id": "s1", "user_id": "u1"}
    # also via Write.context, merged with an explicit context arg
    v2 = g().check(Write(content="billing payment", context={"run_id": "r1"}),
                   context={"session_id": "s2"})
    assert v2.context == {"run_id": "r1", "session_id": "s2"}


def test_on_verdict_hook_fires_for_writes_and_responses():
    seen = []
    guard = g(on_verdict=seen.append)
    guard.check("billing payment invoice")
    guard.check_response("hi", "weather forecast vacation")
    assert len(seen) == 2
    assert all(hasattr(x, "decision") for x in seen)


def test_on_verdict_hook_exception_does_not_break_guard():
    def boom(_v):
        raise RuntimeError("hook boom")

    guard = g(on_verdict=boom)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = guard.check("billing payment invoice")  # must NOT raise
    assert v.decision in (Decision.ALLOW, Decision.FLAG)
    assert any("on_verdict" in str(w.message) for w in caught)


def test_state_roundtrip_restores_drift_across_restart():
    guard = _drifting_guard()
    for on in ["billing payment invoice", "subscription refund billing", "payment invoice refund"]:
        guard.check(on)
    for off in ["weather forecast vacation", "sourdough bread recipe"]:
        guard.check(off)
    before = guard.drift()
    blob = json.dumps(guard.state())  # state is JSON-serializable

    # Simulate a restart: a brand-new guard that has seen nothing, then restore.
    fresh = _drifting_guard()
    assert fresh.drift().samples == 0
    fresh.restore(json.loads(blob))
    after = fresh.drift()
    assert after.current == pytest.approx(before.current)
    assert after.baseline == pytest.approx(before.baseline)
    assert after.samples == before.samples


def test_restore_rejects_mismatched_purpose():
    saved = g()
    saved.check("billing payment")
    other = PurposeGuard("a personal gardening and houseplant assistant", scorer=LexicalScorer())
    with pytest.raises(ValueError):
        other.restore(saved.state())


def test_health_flips_to_drifting():
    guard = _drifting_guard()
    for on in ["billing payment invoice", "subscription refund billing", "payment invoice refund"]:
        guard.check(on)
    assert guard.health()["status"] == "ok"
    for off in ["weather forecast vacation", "sourdough bread recipe",
                "football team game", "puppy training tips"]:
        guard.check(off)
    h = guard.health()
    assert h["status"] == "drifting"
    json.dumps(h)  # the composite is JSON-friendly end-to-end
