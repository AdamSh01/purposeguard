"""Derive the cosine rescale band and threshold presets from the TRAIN traces ONLY.

This is the calibration step of the train/test split. It reads ONLY
``TRAIN_TRACES`` (router + wellness domains) and never touches ``TEST_TRACES`` —
that is what keeps the reported numbers honest. Its output is what justifies the
values baked into the library (EmbeddingScorer's `cos_floor`/`cos_ceil` and the
presets in purposeguard/presets.py).

Needs the `[embeddings]` extra. Run once when re-calibrating:

    python benchmark/calibrate.py
"""

from __future__ import annotations

import statistics as st

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from traces import TRAIN_TRACES
except ImportError:
    from benchmark.traces import TRAIN_TRACES


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    per = {}
    narrow_on = []
    for t in TRAIN_TRACES:
        ref = model.encode(t.purpose, normalize_embeddings=True)
        cos = [float(np.dot(model.encode(w, normalize_embeddings=True), ref)) for w in t.writes]
        on = [c for c, l in zip(cos, t.labels) if l == 0]
        off = [c for c, l in zip(cos, t.labels) if l == 1]
        per[t.name] = (cos, t.labels, on, off)
        if "narrow" in t.name:
            narrow_on += on

    print("=== raw cosine distributions (TRAIN only) ===")
    for name, (_, _, on, off) in per.items():
        print(f"{name}")
        print(f"  on : n={len(on):2d} min={min(on):.3f} med={st.median(on):.3f} max={max(on):.3f}")
        if off:
            print(f"  off: n={len(off):2d} min={min(off):.3f} med={st.median(off):.3f} max={max(off):.3f}")

    # Band from TRAIN: floor 0 (unrelated text sits ~0), ceil = 95th percentile of
    # narrow on-mission cosines (so strongly on-topic content maps near 1.0).
    floor = 0.0
    ceil = round(float(np.percentile(narrow_on, 95)), 2)
    print(f"\nderived band (TRAIN): cos_floor={floor}  cos_ceil={ceil}  (ceil = p95 narrow on-mission)")

    def rescale(c):
        return max(0.0, min(1.0, (c - floor) / (ceil - floor)))

    print("\n=== threshold sweep on RESCALED train scores (FPR / P / R) ===")
    print(f"{'thr':>5} | {'narrow-drift FPR/P/R':>22} | {'broad FPR':>9} | {'stable FPR':>10}")
    print("-" * 56)
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        cells = {"d": "", "b": "", "s": ""}
        for name, (cos, labels, on, off) in per.items():
            sc = [rescale(c) for c in cos]
            flags = [s < thr for s in sc]
            onc = labels.count(0)
            fp = sum(1 for f, l in zip(flags, labels) if f and l == 0)
            fpr = fp / onc if onc else float("nan")
            if labels.count(1):
                tp = sum(1 for f, l in zip(flags, labels) if f and l == 1)
                fn = sum(1 for f, l in zip(flags, labels) if not f and l == 1)
                prec = tp / (tp + fp) if (tp + fp) else float("nan")
                rec = tp / (tp + fn) if (tp + fn) else float("nan")
                cells["d"] = f"{fpr:.2f}/{prec:.2f}/{rec:.2f}"
            elif "broad" in name:
                cells["b"] = f"{fpr:.2f}"
            else:
                cells["s"] = f"{fpr:.2f}"
        print(f"{thr:>5.2f} | {cells['d']:>22} | {cells['b']:>9} | {cells['s']:>10}")


if __name__ == "__main__":
    main()
