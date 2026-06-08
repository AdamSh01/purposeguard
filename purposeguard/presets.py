"""Named threshold presets, calibrated for the EmbeddingScorer.

One threshold can't fit every agent. A narrow single-purpose bot's on-mission
writes score high and consistently, so it can afford a strict threshold; a
legitimately broad assistant's on-mission writes score *lower and more variably*
(they only loosely match a broad purpose statement), so it needs a lenient
threshold or it drowns in false positives. The benchmark quantifies this: at a
strict threshold the broad agent's per-write false-positive rate climbs steeply,
while at a lenient one a narrow agent starts missing real drift. There is no
single value that is perfect for both — these presets are the supported way to
pick the right point on that curve for *your* agent.

Values were chosen from `benchmark/_calibrate.py` sweeps over the three profiles
(see `benchmark/RESULTS.md`):

  * NARROW  (0.20) — narrow agents: FPR ~0.00, precision/recall ~1.00.
  * BALANCED(0.15) — default: narrow still ~perfect; broad FPR moderate.
  * BROAD   (0.10) — broad agents: broad FPR drops sharply (~0.30 -> ~0.07),
                     at the cost of some recall on a *drifting* agent.

These are tuned for the EmbeddingScorer's rescaled scores. The LexicalScorer is a
floor and too noisy to preset meaningfully — install the `embeddings` extra for
real use.

Standing guidance: for broad agents, the **DRIFTING trend is more trustworthy
than per-write FLAG**. Per-write FLAG is necessarily noisy on a broad agent
(its writes legitimately roam), but a sustained drop below the agent's own
baseline still signals real drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """A named alignment threshold and what it's for."""

    name: str
    threshold: float
    description: str


NARROW = Preset(
    "narrow",
    0.20,
    "Single-purpose agent: strict. Flags anything not clearly on its one topic.",
)
BALANCED = Preset(
    "balanced",
    0.15,
    "Default: a focused agent with some natural range.",
)
BROAD = Preset(
    "broad",
    0.10,
    "Wide-ranging agent: lenient, to avoid over-flagging. For broad agents, "
    "trust the DRIFTING trend over per-write FLAG.",
)

PRESETS = {p.name: p for p in (NARROW, BALANCED, BROAD)}


def get_preset(preset: "str | Preset") -> Preset:
    """Resolve a preset by name (case-insensitive) or pass a Preset through."""
    if isinstance(preset, Preset):
        return preset
    try:
        return PRESETS[preset.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown preset {preset!r}; choose from {sorted(PRESETS)}"
        )
