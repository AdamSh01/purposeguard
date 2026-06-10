"""Tests for the Phase-2 benchmark machinery (generator, baselines, --assert).

Hermetic: the generator + baselines + collect_metrics all run on fixed strings /
the lexical scorer, so this needs no model and no network.
"""

import os
import sys

# The benchmark/ scripts are run as a directory, not an installed package, so make
# them importable here the same way `python benchmark/run.py` resolves them.
_BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark"))
if _BENCH not in sys.path:
    sys.path.insert(0, _BENCH)

import baselines  # noqa: E402
import generate  # noqa: E402
import run  # noqa: E402


def test_generator_produces_a_family_of_valid_traces():
    suite = generate.generate_suite()
    assert len(suite) >= 30  # a distribution, not the old N=1
    for t in suite:
        assert len(t.writes) == len(t.labels)
        assert all(lab in (0, 1) for lab in t.labels)
        # onset (if any) is exactly the first off-mission index
        first = next((i for i, lab in enumerate(t.labels) if lab == 1), None)
        assert t.onset == first


def test_step_profile_is_a_clean_step():
    t = generate.make_trace("step", onset=10, length=30)
    assert t.labels[:10] == [0] * 10
    assert all(lab == 1 for lab in t.labels[10:])
    assert t.onset == 10


def test_generator_is_deterministic():
    a = generate.make_trace("ramp", onset=8, length=24)
    b = generate.make_trace("ramp", onset=8, length=24)
    assert a.writes == b.writes and a.labels == b.labels


def test_always_allow_baseline_flags_nothing():
    flags = baselines.always_allow_flags(["x", "y", "z"])
    assert flags == [False, False, False]
    m = baselines.metrics_from_flags(flags, [0, 1, 1])
    assert m["fpr"] == 0.0 and m["recall"] == 0.0


def test_keyword_baseline_flags_off_topic_writes():
    purpose = "billing payment subscription invoice refund"
    writes = ["refund the duplicate invoice payment", "the weather looks sunny today"]
    flags = baselines.keyword_flags(writes, purpose)
    assert flags == [False, True]  # on-topic kept, off-topic flagged


def test_collect_metrics_shape():
    m = run.collect_metrics()
    assert m["scorer"] == "lexical"
    assert set(m["traces"]) == {"narrow-drifting", "broad-on-mission", "narrow-stable"}
    for tm in m["traces"].values():
        assert {"purposeguard", "keyword_baseline", "always_allow_baseline"} <= set(tm)


def test_assert_matches_committed_baseline():
    """The committed lexical baseline must reproduce exactly (regression lock)."""
    assert os.path.exists(run.DEFAULT_BASELINE_PATH), "baseline_lexical.json missing"
    assert run.assert_against_baseline(run.DEFAULT_BASELINE_PATH) == 0
