"""Deterministic synthetic-trace generator (Phase-2 benchmark machinery).

Turns the fixed sentence pools in traces.py into a FAMILY of traces that vary by
drift onset, drift profile, and length — so detection metrics can be reported as a
distribution (mean ± spread over many onsets/profiles) instead of the old N=1
point estimate. Fully deterministic (no RNG): the same parameters always yield
byte-identical traces, so CI stays hermetic.

Unlike traces.Trace (which encodes a strict on→off STEP), GenTrace allows gradual
profiles, so each write carries its own ground-truth label (1 = predominantly
off-mission) and `onset` is simply the first off-mission index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    from traces import BILLING_ON_MISSION, OFF_MISSION, PURPOSE_BILLING
except ImportError:  # allow `python -m benchmark.generate`
    from benchmark.traces import BILLING_ON_MISSION, OFF_MISSION, PURPOSE_BILLING

PROFILES = ("step", "ramp", "sigmoid", "oscillating")
ONSETS = (5, 10, 15, 20, 25)
LENGTHS = (20, 30, 40)


@dataclass
class GenTrace:
    name: str
    purpose: str
    writes: list
    labels: list           # 1 = predominantly off-mission
    onset: Optional[int]   # first index labelled off-mission (None if it never drifts)
    profile: str


def _off_fraction(profile: str, onset: int, t: int, length: int) -> float:
    """Fraction of a write that is off-mission at step t, per drift profile."""
    if t < onset:
        return 0.0
    span = max(1, length - onset)
    p = (t - onset + 1) / span  # 0..1 across the drift tail
    if profile == "step":
        return 1.0
    if profile == "ramp":
        return min(1.0, p)
    if profile == "sigmoid":
        x = (t - (onset + span / 2)) / max(1.0, span / 6)
        return 1.0 / (1.0 + math.exp(-x))
    if profile == "oscillating":
        return 1.0 if (t - onset) % 2 == 0 else 0.25
    raise ValueError(f"unknown profile {profile!r}")


def _blend(on_sentence: str, off_sentence: str, off_fraction: float) -> str:
    """Word-level blend of an on-mission and off-mission sentence — the realistic
    'mixed content within one write' case at intermediate fractions."""
    on_w, off_w = on_sentence.split(), off_sentence.split()
    n_off = round(len(off_w) * off_fraction)
    n_on = round(len(on_w) * (1.0 - off_fraction))
    words = on_w[:n_on] + off_w[:n_off]
    return " ".join(words) if words else on_sentence


def make_trace(profile: str, onset: int, length: int = 30,
               purpose: str = PURPOSE_BILLING) -> GenTrace:
    writes, labels = [], []
    for t in range(length):
        on = BILLING_ON_MISSION[t % len(BILLING_ON_MISSION)]
        off = OFF_MISSION[t % len(OFF_MISSION)]
        frac = _off_fraction(profile, onset, t, length)
        writes.append(_blend(on, off, frac))
        labels.append(1 if frac >= 0.5 else 0)
    onset_idx = next((i for i, lab in enumerate(labels) if lab == 1), None)
    return GenTrace(f"{profile}-onset{onset}-len{length}", purpose, writes, labels,
                    onset_idx, profile)


def generate_suite(profiles=PROFILES, onsets=ONSETS, lengths=LENGTHS) -> list:
    """A deterministic grid of traces (default 4 profiles x 5 onsets x 3 lengths = 60)."""
    return [
        make_trace(p, o, length)
        for p in profiles
        for length in lengths
        for o in onsets
        if o < length  # an onset must leave room for a drift tail
    ]


if __name__ == "__main__":
    suite = generate_suite()
    print(f"generated {len(suite)} traces across "
          f"{len(PROFILES)} profiles x {len(ONSETS)} onsets x {len(LENGTHS)} lengths")
    for t in suite[:5]:
        print(f"  {t.name:<26} onset={t.onset} off={sum(t.labels)}/{len(t.labels)}")
