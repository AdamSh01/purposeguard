"""Tests for the purpose-anchored drift signal (v0.3 Step 2).

Hermetic (LexicalScorer). The signal is OPT-IN (off by default): at a floor that
catches off-purpose poisoning it over-flags broad agents (measured 27/30 in the
adversarial benchmark), so it must not be on by default. When enabled it catches
OFF-purpose streams the relative baseline trend misses; it is blind to ON-purpose
content by construction.
"""

import inspect

from purposeguard import LexicalScorer, PurposeGuard
from purposeguard.guard import DEFAULT_PURPOSE_FLOOR

PURPOSE = "billing payment subscription invoice refund charge"
OFF = [
    "weather forecast vacation hiking", "sourdough bread recipe baking",
    "football match game night", "puppy training tips today",
    "movie night recommendations", "hiking trail gear list",
    "gardening in the spring season", "jazz music study playlist",
    "chess opening rules beginner", "beach travel budget trip",
    "coffee brewing pour over", "fantasy novel series plot",
]
ON = [
    "billing payment invoice", "subscription refund charge",
    "invoice payment billing", "refund charge subscription",
    "payment billing invoice refund",
]


def test_anchored_signal_is_off_by_default():
    """Footgun-safe: with no purpose_floor, the anchored signal never fires."""
    g = PurposeGuard(PURPOSE, scorer=LexicalScorer())
    assert g.meter.purpose_floor is None
    v = g.check("weather forecast vacation hiking")  # off-purpose, but signal disabled
    assert v.details["drift"]["anchored_drifting"] is False


def test_purpose_floor_param_defaults_to_none():
    """Lock the calibration decision: opt-in (None), not the over-flagging 0.28."""
    assert DEFAULT_PURPOSE_FLOOR >= 0.2  # the calibrated narrow-agent value still exists
    assert inspect.signature(PurposeGuard.__init__).parameters["purpose_floor"].default is None


def test_anchored_fires_on_off_purpose_when_enabled():
    g = PurposeGuard(PURPOSE, scorer=LexicalScorer(), purpose_floor=0.1)
    v = None
    for w in OFF:
        v = g.check(w)
    assert v.details["drift"]["anchored_drifting"] is True


def test_anchored_silent_on_on_purpose_when_enabled():
    g = PurposeGuard(PURPOSE, scorer=LexicalScorer(), purpose_floor=0.1)
    v = None
    for w in ON:
        v = g.check(w)
    assert v.details["drift"]["anchored_drifting"] is False


def test_anchored_catches_baseline_poisoning_where_relative_evades():
    """The core win, hermetic: an OFF-purpose stream from the start poisons the
    baseline -> the RELATIVE drift trend evades; the absolute purpose floor FIRES."""
    g = PurposeGuard(PURPOSE, scorer=LexicalScorer(), purpose_floor=0.1)
    v = None
    for w in OFF:
        v = g.check(w)
    d = v.details["drift"]
    assert d["drifting"] is False          # relative: baseline poisoned -> no drop seen
    assert d["anchored_drifting"] is True  # anchored to the fixed purpose -> fires
