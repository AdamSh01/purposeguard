"""The drift meter: turn a stream of per-write scores into one watchable number.

A single write being off-purpose is noise. *Sustained* decline in alignment is
drift — the failure the user described: an agent that slowly forgets what it's
for. The meter exists to make that trend observable and alertable.

It keeps an exponential moving average (EMA) of alignment scores. EMA is chosen
over a simple window because it's O(1) memory, needs no history buffer, and
naturally weights recent behavior more — which is exactly what "is the agent
drifting *now*" wants. We also track a baseline (early alignment) so "drift" can
be reported as a drop relative to where the agent started, not just an absolute
score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DriftReading:
    """A snapshot of the agent's current alignment health."""

    current: float       # EMA of recent alignment, [0,1]
    baseline: float      # early-life alignment, [0,1]
    drift: float         # baseline - current, clamped at 0; higher = worse
    samples: int         # how many writes have been scored

    @property
    def drifting(self) -> bool:
        """Heuristic: meaningful drop from baseline. Tunable by the caller."""
        return self.drift >= 0.15

    def __str__(self) -> str:
        # ASCII-only marker on purpose: __str__ is an emitted output path, and a
        # non-ASCII glyph here crashes on a Windows cp1252 console (guardrail #4:
        # never crash). test_driftreading_str_is_ascii_safe locks this in.
        flag = " [DRIFTING]" if self.drifting else ""
        return (
            f"DriftReading(current={self.current:.2f}, baseline={self.baseline:.2f}, "
            f"drift={self.drift:.2f}, n={self.samples}){flag}"
        )


class DriftMeter:
    """Rolling alignment tracker.

    `alpha` is the EMA smoothing factor: higher reacts faster to recent writes,
    lower is steadier. 0.2 is a reasonable default — roughly a 10-write memory.
    `baseline_window` is how many initial writes establish the "this is what
    on-mission looks like for this agent" reference, so a legitimately broad
    agent sets its own bar instead of being judged against an absolute.
    """

    def __init__(self, alpha: float = 0.2, baseline_window: int = 10) -> None:
        self.alpha = alpha
        self.baseline_window = baseline_window
        self._ema: float | None = None
        self._baseline_sum = 0.0
        self._baseline_n = 0
        self._samples = 0

    def update(self, score: float) -> DriftReading:
        self._samples += 1

        # EMA of current alignment
        if self._ema is None:
            self._ema = score
        else:
            self._ema = self.alpha * score + (1 - self.alpha) * self._ema

        # Baseline accumulates only during the opening window
        if self._baseline_n < self.baseline_window:
            self._baseline_sum += score
            self._baseline_n += 1

        baseline = (
            self._baseline_sum / self._baseline_n if self._baseline_n else self._ema
        )
        drift = max(0.0, baseline - self._ema)
        return DriftReading(
            current=self._ema,
            baseline=baseline,
            drift=drift,
            samples=self._samples,
        )

    def reading(self) -> DriftReading:
        """Current state without recording a new sample."""
        if self._ema is None:
            return DriftReading(current=1.0, baseline=1.0, drift=0.0, samples=0)
        baseline = (
            self._baseline_sum / self._baseline_n if self._baseline_n else self._ema
        )
        return DriftReading(
            current=self._ema,
            baseline=baseline,
            drift=max(0.0, baseline - self._ema),
            samples=self._samples,
        )
