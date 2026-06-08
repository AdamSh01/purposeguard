# PurposeGuard v0.1 — Benchmark Results

**The one question this benchmark answers:** does PurposeGuard catch drift early
enough to be useful *without* over-flagging a legitimately broad agent?

Three synthetic traces with per-write ground-truth labels (see `traces.py`):

| profile | purpose | writes | drifts? |
| --- | --- | --- | --- |
| `narrow-drifting` | billing support | 15 on-mission + 15 off-mission | yes, onset @15 |
| `broad-on-mission` | general assistant | 30 diverse, all on-mission | no |
| `narrow-stable` | billing support | 30 on-mission | no |

**Out of scope (on purpose):** adversarial / poisoning traces — that's the v0.2
consensus benchmark.

## Two signals, reported separately

PurposeGuard emits two things, and they behave differently:

- **The DRIFTING trend** (lead-time): a rolling drop below the agent's own
  baseline. It reads the *raw alignment score*, so it is **independent of the
  threshold/preset**.
- **Per-write FLAG** (FPR / precision / recall): an instantaneous threshold
  decision, so it **depends on the preset**.

This split is the headline: presets tune per-write FLAG; the trend is untouched
by them. That is why, for a broad agent, **the trend is more trustworthy than
per-write FLAG** — the trend stays quiet no matter the preset, while per-write
FLAG is necessarily noisy on an agent that legitimately roams.

## Two fixes in this pass (in order)

1. **Fixed score compression.** `EmbeddingScorer` previously mapped cosine via
   `(cos+1)/2`, squashing this model's real cosine band (~0.0 unrelated to ~0.5
   on-topic) into ~[0.45, 0.75]. It now clamp-and-stretches `[0.0, 0.5] -> [0,1]`
   (band chosen from the measured distributions below). **Caveat: the `[0.0, 0.5]`
   band is fitted to these synthetic traces and may need re-tuning on real-world
   corpora — it is not a universal constant; override it via
   `EmbeddingScorer(cos_floor=..., cos_ceil=...)`.** This restored dynamic
   range and — with **no change to the drift meter's `alpha`** — cut detection
   lead-time from **+12 to +1** while the broad/stable agents kept *not* firing.
2. **Added presets** `NARROW=0.20 / BALANCED=0.15 / BROAD=0.10` (default is now
   BALANCED). These move the per-write FLAG operating point per agent type.

Measured raw cosine distributions that set the rescale band and presets:

```
narrow on-mission : min 0.124  median 0.33  max 0.51
off-mission       : min -0.05  median 0.01  max 0.07
broad  on-mission : min 0.042  median 0.14  max 0.285   <- low & overlaps off-mission
```

## Before / after (embedding scorer)

| metric | before (compression bug, single 0.55 threshold) | after |
| --- | --- | --- |
| narrow-drifting lead-time | **+12** | **+1** |
| narrow-drifting precision / recall | 1.00 / 1.00 | 1.00 / 1.00 (NARROW/BALANCED) |
| narrow-drifting FLAG-FPR | 0.00 | 0.00 |
| broad-on-mission FLAG-FPR | **0.30** | **0.07** (BROAD) |
| broad-on-mission DRIFTING fired? | no | no (preserved) |
| narrow-stable FLAG-FPR / DRIFTING | 0.00 / no | 0.00 / no |

## After: full results

Config: `drift_alpha=0.2`, `drift_baseline_window=10`; presets NARROW=0.20,
BALANCED=0.15, BROAD=0.10; embedding model `all-MiniLM-L6-v2`.

**[A] DRIFTING trend (embedding) — independent of preset/threshold**

| profile | onset | lead | drift? |
| --- | --- | --- | --- |
| narrow-drifting | 15 | **+1** | @16 |
| broad-on-mission | — | n/a | **no** |
| narrow-stable | — | n/a | **no** |

**[B] per-write FLAG (embedding) across presets** (`*` = recommended for that agent)

| profile | preset | thr | FLAG-FPR | prec | recall |
| --- | --- | --- | --- | --- | --- |
| narrow-drifting | NARROW `*` | 0.20 | 0.00 | 1.00 | 1.00 |
| narrow-drifting | BALANCED | 0.15 | 0.00 | 1.00 | 1.00 |
| narrow-drifting | BROAD | 0.10 | 0.00 | 1.00 | 0.80 |
| broad-on-mission | NARROW | 0.20 | 0.30 | — | — |
| broad-on-mission | BALANCED | 0.15 | 0.23 | — | — |
| broad-on-mission | BROAD `*` | 0.10 | **0.07** | — | — |
| narrow-stable | NARROW `*` | 0.20 | 0.00 | — | — |
| narrow-stable | BALANCED | 0.15 | 0.00 | — | — |
| narrow-stable | BROAD | 0.10 | 0.00 | — | — |

**[C] lexical floor at BALANCED — shown only to confirm it stays a floor**

| profile | FLAG-FPR | prec | recall | drift? |
| --- | --- | --- | --- | --- |
| narrow-drifting | 0.80 | 0.43 | 0.60 | no |
| broad-on-mission | 0.90 | — | — | no |
| narrow-stable | 0.70 | — | — | no |

## Interpretation (the tradeoff, shown not hidden)

- **All three profiles are satisfied at once — but only with the right preset
  per agent.** Narrow agents @ NARROW: FPR 0.00, P/R 1.00. Broad agent @ BROAD:
  FPR 0.07 (down from 0.30). Stable agent: FPR 0.00. And the DRIFTING trend
  catches the real drift in +1 write while never firing on the two non-drifting
  agents.
- **The coupling held.** Cutting lead-time (+12 -> +1) came from the rescale, not
  from making the meter twitchier — `alpha` is unchanged — so broad-tolerance was
  preserved (broad never fires DRIFTING). We did *not* fix detection by breaking
  broad-tolerance.
- **There is a genuine tradeoff, and it is real, not papered over.** No single
  preset is best for everyone: BROAD on a *narrow drifting* agent drops recall to
  0.80 (too lenient — it misses the weakest off-mission writes); NARROW on a
  *broad* agent pushes FPR to 0.30 (too strict — it flags legitimate range).
  Picking per agent type is the supported answer.
- **Lexical is a floor, not a detector** (FPR 0.70–0.90, never fires DRIFTING).
  Install the `embeddings` extra for real use; presets are calibrated for it.

## Reproduce

```bash
pip install -e ".[dev]"
pip install "purposeguard[embeddings]"   # enables the embedding rows
python benchmark/run.py
```

Content is hand-authored and fixed (no RNG), so runs are byte-for-byte
reproducible. If `all-MiniLM-L6-v2` can't load, the script prints a note and
reports lexical only — it never blocks on embeddings. Expect harmless Hugging
Face loading warnings on stderr the first time.
