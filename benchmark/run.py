"""Run PurposeGuard over the synthetic traces and report the v0.1 benchmark.

Every metric serves one question: does PurposeGuard catch drift early enough to
be useful without over-flagging a legitimately broad agent?

  * detection lead-time - writes from true onset to the DRIFTING meter firing.
  * false-positive rate - fraction of on-mission writes FLAGged (per-write).
  * precision / recall  - per-write FLAG vs ground-truth off-mission labels.

Two signals, reported separately on purpose:
  - the DRIFTING TREND (lead-time, "drift?") reads the raw alignment score, so it
    is INDEPENDENT of the per-write threshold/preset. Reported once per trace.
  - per-write FLAG (FPR/precision/recall) depends on the threshold, so it is
    reported across all three presets to make the tradeoff explicit.

Scorers: LexicalScorer always; EmbeddingScorer too IF the model loads (never
blocks on it). Presets are calibrated for the EmbeddingScorer.

Reproduce:  python benchmark/run.py
"""

from __future__ import annotations

import argparse
import json
import os

try:
    import baselines
    from traces import TRACES
except ImportError:  # allow `python -m benchmark.run`
    from benchmark import baselines
    from benchmark.traces import TRACES

from purposeguard import (
    BALANCED,
    BROAD,
    NARROW,
    Decision,
    EmbeddingScorer,
    LexicalScorer,
    PurposeGuard,
)

DEFAULT_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline_lexical.json")

DRIFT_ALPHA = 0.2
DRIFT_BASELINE_WINDOW = 10
PRESETS = [NARROW, BALANCED, BROAD]


def evaluate(trace, scorer, threshold) -> dict:
    """Run the guard over one trace and compute trend + per-write metrics."""
    guard = PurposeGuard(
        trace.purpose,
        scorer=scorer,
        threshold=threshold,
        drift_alpha=DRIFT_ALPHA,
        drift_baseline_window=DRIFT_BASELINE_WINDOW,
    )
    flagged, drifting = [], []
    for text in trace.writes:
        verdict = guard.check(text)
        flagged.append(verdict.decision == Decision.FLAG)
        drifting.append(bool(verdict.details["drift"]["drifting"]))

    labels = trace.labels
    n = len(labels)
    tp = sum(1 for i in range(n) if flagged[i] and labels[i] == 1)
    fp = sum(1 for i in range(n) if flagged[i] and labels[i] == 0)
    fn = sum(1 for i in range(n) if not flagged[i] and labels[i] == 1)
    on_mission = labels.count(0)
    has_pos = labels.count(1) > 0

    first_drift = next((i for i, d in enumerate(drifting) if d), None)
    return {
        "fpr": fp / on_mission if on_mission else None,
        "precision": (tp / (tp + fp)) if has_pos and (tp + fp) else None,
        "recall": (tp / (tp + fn)) if has_pos and (tp + fn) else None,
        "first_drift": first_drift,
        "lead_time": (
            first_drift - trace.onset
            if trace.onset is not None and first_drift is not None
            else None
        ),
    }


def get_scorers():
    """[(label, scorer)] - always lexical; embeddings only if the model loads."""
    out = [("lexical", LexicalScorer())]
    emb = EmbeddingScorer()
    try:
        emb.score("probe", "probe")  # force a real load (or fallback)
    except Exception as exc:
        print(f"[note] EmbeddingScorer raised {exc!r}; lexical only.\n")
        return out
    if getattr(emb, "_fallback", None) is not None:
        print(
            "[note] all-MiniLM-L6-v2 could not load (offline/uncached); "
            "EmbeddingScorer fell back. Reporting LEXICAL ONLY.\n"
        )
        return out
    out.append(("embedding", emb))
    return out


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _lead(x):
    return "n/a" if x is None else (f"+{x}" if x >= 0 else str(x))


def _drift(x):
    return "no" if x is None else f"@{x}"


def _recommended(trace):
    return BROAD if trace.name.startswith("broad") else NARROW


def collect_metrics() -> dict:
    """Deterministic LEXICAL metrics for the held-out TEST traces, plus the trivial
    baselines, as a machine-readable dict. Lexical so it reproduces offline and in
    CI with no model; the embedding numbers are regenerated in the embeddings env."""
    scorer = LexicalScorer()
    out = {"scorer": "lexical", "threshold": BALANCED.threshold, "traces": {}}
    for t in TRACES:
        pg = evaluate(t, scorer, BALANCED.threshold)
        out["traces"][t.name] = {
            "purposeguard": {
                k: pg[k] for k in ("fpr", "precision", "recall", "first_drift", "lead_time")
            },
            "keyword_baseline": baselines.metrics_from_flags(
                baselines.keyword_flags(t.writes, t.purpose), t.labels
            ),
            "always_allow_baseline": baselines.metrics_from_flags(
                baselines.always_allow_flags(t.writes), t.labels
            ),
        }
    return out


def _diff(a, b, path: str) -> list:
    """Recursively diff two metric trees with a float tolerance (lexical metrics are
    deterministic, so any real difference is a regression)."""
    if isinstance(a, dict):
        keys = sorted(set(a) | (set(b) if isinstance(b, dict) else set()))
        out = []
        for k in keys:
            bv = b.get(k, "<<missing>>") if isinstance(b, dict) else "<<missing>>"
            out += _diff(a.get(k, "<<missing>>"), bv, f"{path}.{k}" if path else k)
        return out
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return [] if abs(a - b) <= 1e-9 else [f"{path}: {a} != {b}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def assert_against_baseline(path: str = DEFAULT_BASELINE_PATH) -> int:
    """Compare freshly-computed lexical metrics to the committed baseline. Returns a
    process exit code (0 = match) so RESULTS/numbers can't silently rot."""
    if not os.path.exists(path):
        print(f"[assert] no baseline at {path}; create it with:\n"
              f"         python benchmark/run.py --json > {path}")
        return 1
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)
    mismatches = _diff(collect_metrics(), ref, "")
    if mismatches:
        print("[assert] FAIL -- lexical metrics drifted from the committed baseline:")
        for m in mismatches:
            print("   " + m)
        return 1
    print(f"[assert] OK -- lexical metrics match {os.path.basename(path)}")
    return 0


def main() -> None:
    scorers = dict(get_scorers())
    print("PurposeGuard v0.1 benchmark: synthetic drift traces")
    print("reporting on the HELD-OUT TEST set; band/presets calibrated on TRAIN "
          "only (see calibrate.py)")
    print(f"scorers run: {', '.join(scorers)}")
    print(
        f"config: drift_alpha={DRIFT_ALPHA}, drift_baseline_window="
        f"{DRIFT_BASELINE_WINDOW}; presets NARROW={NARROW.threshold} "
        f"BALANCED={BALANCED.threshold} BROAD={BROAD.threshold}"
    )

    emb = scorers.get("embedding")
    if emb is not None:
        # 1) Trend signal — threshold-independent (uses the raw alignment score).
        print("\n[A] DRIFTING trend (embedding) - independent of preset/threshold")
        h = f"{'profile':<18}{'onset':>6}{'lead':>7}{'drift?':>8}"
        print(h + "\n" + "-" * len(h))
        for t in TRACES:
            m = evaluate(t, emb, BALANCED.threshold)
            onset = "n/a" if t.onset is None else str(t.onset)
            print(f"{t.name:<18}{onset:>6}{_lead(m['lead_time']):>7}{_drift(m['first_drift']):>8}")

        # 2) Per-write FLAG — depends on the preset; show the tradeoff in full.
        print("\n[B] per-write FLAG (embedding) across presets  (* = recommended for that agent)")
        h = f"{'profile':<18}{'preset':<10}{'thr':>5}{'FLAG-FPR':>10}{'prec':>8}{'recall':>8}"
        print(h + "\n" + "-" * len(h))
        for t in TRACES:
            rec = _recommended(t)
            for p in PRESETS:
                m = evaluate(t, emb, p.threshold)
                star = " *" if p is rec else ""
                print(
                    f"{t.name:<18}{p.name:<10}{p.threshold:>5.2f}"
                    f"{_f(m['fpr']):>10}{_f(m['precision']):>8}{_f(m['recall']):>8}{star}"
                )
            print()
    else:
        print("\n[note] embeddings unavailable; presets are embedding-calibrated, sweep skipped.")

    # 3) Lexical floor — confirm it stays a floor (not a detector).
    lex = scorers.get("lexical")
    if lex is not None:
        print("[C] lexical floor at BALANCED threshold (shown only as a floor)")
        h = f"{'profile':<18}{'FLAG-FPR':>10}{'prec':>8}{'recall':>8}{'drift?':>8}"
        print(h + "\n" + "-" * len(h))
        for t in TRACES:
            m = evaluate(t, lex, BALANCED.threshold)
            print(
                f"{t.name:<18}{_f(m['fpr']):>10}{_f(m['precision']):>8}"
                f"{_f(m['recall']):>8}{_drift(m['first_drift']):>8}"
            )

    # 4) Baselines — PurposeGuard must beat the trivial alternatives, not just its
    # own degraded lexical floor. Lexical PurposeGuard vs always-allow vs a keyword
    # filter over the purpose's salient terms (all hermetic/deterministic).
    print("\n[D] hermetic baselines vs PurposeGuard (lexical, BALANCED) - FPR / recall")
    h = f"{'profile':<18}{'PG fpr':>8}{'PG rec':>8}{'kw fpr':>8}{'kw rec':>9}{'allow rec':>11}"
    print(h + "\n" + "-" * len(h))
    metrics = collect_metrics()["traces"]
    for t in TRACES:
        pg = metrics[t.name]["purposeguard"]
        kw = metrics[t.name]["keyword_baseline"]
        aa = metrics[t.name]["always_allow_baseline"]
        print(
            f"{t.name:<18}{_f(pg['fpr']):>8}{_f(pg['recall']):>8}"
            f"{_f(kw['fpr']):>8}{_f(kw['recall']):>9}{_f(aa['recall']):>11}"
        )

    print(
        "\nlegend: lead = writes from onset to DRIFTING firing; FLAG-FPR = "
        "on-mission writes flagged; drift? = index DRIFTING first fired."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PurposeGuard benchmark")
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable lexical metrics + baselines (JSON)",
    )
    parser.add_argument(
        "--assert", dest="assert_", action="store_true",
        help="fail (exit 1) if lexical metrics drift from the committed baseline",
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE_PATH)
    args = parser.parse_args()
    if args.json:
        print(json.dumps(collect_metrics(), indent=2, sort_keys=True))
    elif args.assert_:
        raise SystemExit(assert_against_baseline(args.baseline))
    else:
        main()
