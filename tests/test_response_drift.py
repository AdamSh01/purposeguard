"""Tests for response-level drift detection: PurposeGuard.check_response().

Hermetic — LexicalScorer, no model/network. Covers on-mission vs off-mission
answers, detection-first (never blocks), that responses feed a SEPARATE meter
from writes (attribution), and that user_input is recorded context, not a
scoring input.
"""

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard

PURPOSE = "billing payment subscription invoice refund account support"
ON_MISSION_RESPONSE = (
    "Your billing invoice was corrected and the subscription refund posted to "
    "your payment account."
)
OFF_MISSION_RESPONSE = (
    "Here is a great sourdough bread recipe with an overnight rise and a crispy crust."
)


def make_guard(threshold=0.12):
    return PurposeGuard(purpose=PURPOSE, scorer=LexicalScorer(), threshold=threshold)


def test_check_response_on_mission_allows():
    g = make_guard()
    v = g.check_response("How do I get a refund on my invoice?", ON_MISSION_RESPONSE)
    assert v.decision == Decision.ALLOW
    assert v.aligned
    assert v.details["kind"] == "response"


def test_check_response_off_mission_flags():
    g = make_guard()
    v = g.check_response("How do I get a refund?", OFF_MISSION_RESPONSE)
    assert v.decision == Decision.FLAG
    assert not v.aligned


def test_check_response_never_blocks():
    """Detection-first applies to responses too."""
    g = make_guard()
    for resp in [OFF_MISSION_RESPONSE, "", "x" * 500, ON_MISSION_RESPONSE]:
        v = g.check_response("anything", resp or "placeholder")
        assert v.decision != Decision.BLOCK


def test_response_meter_is_separate_from_write_meter():
    """The point of the feature: answer-drift is tracked independently of writes,
    so a healthy memory stream can't mask drifting answers (and vice versa)."""
    g = make_guard()
    for _ in range(3):
        g.check_response("any question", OFF_MISSION_RESPONSE)
    assert g.response_drift().samples == 3
    assert g.drift().samples == 0          # writes untouched -> attribution

    g.check("billing invoice refund payment")  # a write moves only the write meter
    assert g.drift().samples == 1
    assert g.response_drift().samples == 3


def test_response_drift_rises_when_answers_wander():
    g = PurposeGuard(
        purpose="billing payment subscription invoice refund",
        scorer=LexicalScorer(),
        threshold=0.10,
        drift_baseline_window=3,
        drift_alpha=0.5,
    )
    for on in ["billing payment invoice", "subscription refund billing", "payment invoice refund"]:
        g.check_response("q", on)
    baseline = g.response_drift().baseline
    for off in ["weather forecast vacation", "sourdough bread recipe", "favorite football team game"]:
        g.check_response("q", off)
    reading = g.response_drift()
    assert reading.drift > 0.0
    assert reading.current < baseline


def test_user_input_is_recorded_context_not_a_scoring_input():
    g = make_guard()

    # Off-mission question, but the agent stayed on mission -> ALLOW (the user's
    # question does not drag the score down), and the input is recorded.
    v1 = g.check_response("What's the weather tomorrow?", ON_MISSION_RESPONSE)
    assert v1.decision == Decision.ALLOW
    assert v1.details["user_input"] == "What's the weather tomorrow?"

    # On-mission question, but the agent wandered -> FLAG. The response drives the
    # verdict, not the on-topic question.
    v2 = g.check_response("How do I get a refund on my invoice?", OFF_MISSION_RESPONSE)
    assert v2.decision == Decision.FLAG


def test_check_response_dry_run_does_not_affect_meter():
    g = make_guard()
    g.check_response("q", ON_MISSION_RESPONSE)            # records
    n_before = g.response_drift().samples
    g.check_response("q", OFF_MISSION_RESPONSE, record=False)  # dry run
    assert g.response_drift().samples == n_before
