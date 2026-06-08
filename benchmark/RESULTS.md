# PurposeGuard v0.1 — Benchmark Results

**The one question:** does PurposeGuard catch drift early enough to be useful
*without* over-flagging a legitimately broad agent?

## Train / test split (added in v0.1.2 — fixes eval overfitting)

A reviewer-flagged credibility problem: the cosine rescale band and the presets
were originally calibrated on the **same three traces** the headline numbers were
then reported on. That overfits the eval — the numbers partly measured the fit,
not the method. Fixed by splitting into disjoint sets in **different domains**:

| set | domains | traces | used for |
| --- | --- | --- | --- |
| **TRAIN** (calibration only) | Wi-Fi router support, wellness coach | `train-narrow-drifting`, `train-broad-on-mission`, `train-narrow-stable` | picking the band + presets (`benchmark/calibrate.py`) |
| **TEST** (held out — reported) | billing support, general assistant | `narrow-drifting`, `broad-on-mission`, `narrow-stable` | every number below |

All traces are hand-authored and fixed (no RNG), so both sets are reproducible
(`benchmark/traces.py`). **The TEST traces are never used for calibration.**

**Caveat (still synthetic-to-synthetic):** the TRAIN domains (router + wellness)
are themselves hand-authored synthetic traces. The split removes the *overfitting*
(calibrating and reporting on the same data), but it is not validation on real
agent logs — both sides are author-generated. Real-corpus validation remains
future work.

### Calibration on TRAIN only (`python benchmark/calibrate.py`)

Measured on TRAIN: off-mission cosines cluster at ~0.0; narrow on-mission runs
0.35–0.62; broad on-mission is lower/wider (0.09–0.43). From this:

- **Rescale band `[0.0, 0.61]`** — floor 0.0 (unrelated ≈ 0), ceil 0.61 (95th
  percentile of narrow on-mission). *(The previously-shipped 0.50 was fit to the
  billing test traces; 0.61 comes from the held-out-separate train domains.)*
- **Presets** NARROW=0.25, BALANCED=0.15, BROAD=0.10 (from the train threshold
  sweep). NARROW rose from 0.20 → 0.25: the train domain needs 0.25 for full
  recall, because its off-mission writes score a bit higher than billing's did.

These train-derived values are now the library defaults — so the shipped defaults
are no longer fit to the reported test set.

## Held-out TEST results (`python benchmark/run.py`)

Config: `drift_alpha=0.2`, `drift_baseline_window=10`; band `[0.0, 0.61]`;
presets NARROW=0.25, BALANCED=0.15, BROAD=0.10; model `all-MiniLM-L6-v2`.

**[A] DRIFTING trend (embedding) — independent of preset/threshold**

| profile | onset | lead | drift? |
| --- | --- | --- | --- |
| narrow-drifting | 15 | **+1** | @16 |
| broad-on-mission | — | n/a | **no** |
| narrow-stable | — | n/a | **no** |

**[B] per-write FLAG (embedding) across presets** (`*` = recommended for that agent)

| profile | preset | thr | FLAG-FPR | prec | recall |
| --- | --- | --- | --- | --- | --- |
| narrow-drifting | NARROW `*` | 0.25 | 0.07 | 0.94 | 1.00 |
| narrow-drifting | BALANCED | 0.15 | 0.00 | 1.00 | 1.00 |
| narrow-drifting | BROAD | 0.10 | 0.00 | 1.00 | 0.93 |
| broad-on-mission | NARROW | 0.25 | 0.50 | — | — |
| broad-on-mission | BALANCED | 0.15 | 0.30 | — | — |
| broad-on-mission | BROAD `*` | 0.10 | **0.13** | — | — |
| narrow-stable | NARROW `*` | 0.25 | 0.03 | — | — |
| narrow-stable | BALANCED | 0.15 | 0.00 | — | — |
| narrow-stable | BROAD | 0.10 | 0.00 | — | — |

**[C] lexical floor at BALANCED — shown only to confirm it stays a floor**

| profile | FLAG-FPR | prec | recall | drift? |
| --- | --- | --- | --- | --- |
| narrow-drifting | 0.80 | 0.43 | 0.60 | no |
| broad-on-mission | 0.90 | — | — | no |
| narrow-stable | 0.70 | — | — | no |

## Before / after — the held-out numbers are WORSE, and that's the point

The earlier numbers (band/presets fit to these same traces) were optimistic.
On held-out TEST with train-calibrated params:

| metric (embedding) | old (overfit) | held-out (honest) |
| --- | --- | --- |
| narrow-drifting @ NARROW — FPR / precision | 0.00 / 1.00 | **0.07 / 0.94** |
| narrow-stable @ NARROW — FPR | 0.00 | **0.03** |
| broad-on-mission @ BROAD — FPR | 0.07 | **0.13** |
| narrow-drifting @ BALANCED — FPR / P / R | 0.00 / 1.00 / 1.00 | 0.00 / 1.00 / 1.00 (held) |
| DRIFTING lead-time | +1 | +1 (held) |
| broad / stable DRIFTING false-fire | no | no (held) |

Stated plainly: the recommended **NARROW** preset, evaluated on a held-out narrow
agent, now shows **~7% false positives and 0.94 precision** (not the previously
claimed 0.00 / 1.00), and the broad agent's **BROAD** false-positive rate is
**0.13** (not 0.07). We did **not** re-tune to recover the old numbers — these are
the honest held-out results.

## What held up, and what to learn

- **The DRIFTING trend is robust.** Still +1-write detection on the held-out
  drifting agent, and it never false-fires on the held-out broad/stable agents —
  unchanged by the recalibration. For broad agents especially, the trend remains
  more trustworthy than per-write FLAG.
- **Per-write FLAG is domain-sensitive, more than the overfit eval implied.**
  Two honest artifacts: (1) the higher train ceil (0.61) lowers all rescaled
  scores on the lower-scoring billing domain, nudging FPRs up; (2) NARROW=0.25,
  calibrated on the higher-scoring router domain, is slightly too strict for
  billing — on the held-out narrow agent, **BALANCED (0.00 FPR, 1.00 P/R)
  actually beat NARROW**. Lesson: pick the preset for *your* domain's score
  distribution; a single global calibration won't be optimal everywhere. (This is
  also the case for a lighter, more domain-robust embedding backend — see
  BUILD_PLAN.)
- **Lexical is still a floor** (FPR 0.70–0.90), unchanged by the band (it affects
  embeddings only). Install `[embeddings]`.

## Known adversarial limitations (where the detector is fooled)

PurposeGuard is a reliability guardrail, **not a security boundary** (guardrail
#5). `benchmark/adversarial.py` measures attack success rate (ASR = % of
adversarial events NOT flagged) for four families, on a held-out attack set, with
**nothing tuned to catch them**. The bypasses below are the honest result —
documenting them is the point.

| attack family | N | per-write FLAG ASR | DRIFTING trend |
| --- | --- | --- | --- |
| (a) keyword-stuffing / camouflage | 10 | **100%** | evaded |
| (b) gradual drift (boiling frog) | 6 | 17% | caught |
| (c) baseline poisoning | 15 | 0% | **evaded (100% trend ASR)** |
| (d) on-topic-but-wrong-policy | 10 | **100%** | evaded |

(Embedding scorer, default BALANCED threshold; "evaded" = the trend never fired
during the attack = 100% trend ASR.)

- **(a) Keyword-stuffing — full bypass (100%).** Prefixing an off-mission or
  malicious payload with billing vocabulary keeps cosine-to-purpose high, so
  neither per-write FLAG nor the trend reacts. A semantic-similarity detector is
  blind to this by construction — it measures topical overlap, which the attacker
  simply supplies.
- **(b) Gradual drift — largely caught *here*, but read the mechanism.** This ramp
  ended clearly off-mission, so the FIXED early baseline produced a large eventual
  delta and the trend fired (and 5/6 tail writes were per-write-flagged). The
  boiling-frog evades only when the drift stays within ~0.15 of the baseline (i.e.
  only *mildly* off), or when it corrupts the baseline itself — which is (c). Do
  not read "caught" as "gradual drift is solved."
- **(c) Baseline poisoning — defeats the TREND by construction (100% trend ASR).**
  Fill the first N writes (the baseline window) with adversarial content and the
  trend has no "normal" to drift from, so it never fires. Note the split result:
  per-write FLAG ASR is **0%** — because per-write FLAG scores against the
  *immutable* purpose, not the baseline, so it still flags every poisoned write.
  The immutable purpose is exactly what confines this attack to the trend.
- **(d) On-topic-but-wrong-policy — full bypass (100%).** "Approve all refunds
  without verification" is topically billing (high cosine) but behaviorally wrong.
  PurposeGuard measures topical alignment, not policy/behavioral correctness, so
  it cannot see this class at all. Catching it needs rule/policy checks, not drift.

**Takeaway:** PurposeGuard catches *topical* drift toward clearly different
subject matter. It does NOT catch semantic camouflage, on-topic policy/behavioral
violations, or trend evasion via baseline poisoning. A security-literate adopter
must handle those with other controls — and they are part of what the planned
v0.2 consensus reference + OWASP Agent Memory Guard marker composition target.

## Reproduce

```bash
pip install -e ".[dev]"
pip install "purposeguard[embeddings]"   # enables the embedding rows + calibration
python benchmark/calibrate.py            # derive band/presets from TRAIN only
python benchmark/run.py                  # report on held-out TEST only
python benchmark/adversarial.py          # attack success rates (needs embeddings)
```

If `all-MiniLM-L6-v2` can't load, `run.py` prints a note and reports lexical only;
`calibrate.py` requires the model.
