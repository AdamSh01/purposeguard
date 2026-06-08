"""PurposeGuard — the single public interface.

Usage is meant to be one line to construct, one line to check:

    guard = PurposeGuard(purpose="A customer-support agent for billing questions")
    verdict = guard.check("User asked how to bake sourdough bread")
    # -> Verdict(flag, score=0.10, reason='alignment 0.10 below threshold 0.55')

Everything is framework-agnostic: `check()` takes a str or a Write, so Mem0,
LangChain, a raw store, or a plain script can all use it. The guard composes
three already-built pieces — a Scorer (how on-purpose?), a threshold policy
(allow/flag — never block in v0.1), and a DriftMeter (trend over time).

Detection-first by design: the only decisions v0.1 emits are ALLOW and FLAG.
The guard never mutates or drops the caller's write. It observes and reports.
"""

from __future__ import annotations

from typing import Callable, Optional

from .drift import DriftMeter, DriftReading
from .presets import BALANCED, Preset, get_preset
from .scoring import Scorer, default_scorer
from .types import Decision, Verdict, Write

# Optional LLM judge: a function (content, purpose) -> bool ("is this on-mission?").
# The library never ships one — the user supplies it if they want a second
# opinion on borderline scores. Keeping it injectable means no API dependency
# and no opinion baked in about which model to call.
LLMJudge = Callable[[str, str], bool]


class PurposeGuard:
    def __init__(
        self,
        purpose: str,
        *,
        threshold: Optional[float] = None,
        scorer: Optional[Scorer] = None,
        judge: Optional[LLMJudge] = None,
        judge_band: float = 0.10,
        drift_alpha: float = 0.2,
        drift_baseline_window: int = 10,
    ) -> None:
        """
        purpose: the immutable mission statement. This is the reference every
            write is scored against. Stored once; the guard never lets it change
            after construction (mirrors OWASP AMG's immutable-key idea — purpose
            drift can't come from purpose *edits*).
        threshold: alignment below this is FLAGged. Defaults to the BALANCED
            preset (calibrated for the EmbeddingScorer's rescaled scores). Use
            BROAD for wide-ranging agents and NARROW for single-purpose ones, or
            the `from_preset` constructor. See purposeguard/presets.py.
        scorer: defaults to local embeddings if available, else lexical.
        judge: optional callable for a second opinion on borderline writes only.
        judge_band: how close to the threshold counts as "borderline" and worth
            spending a judge call on. Keeps LLM usage rare and cheap.
        """
        if not purpose or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        self._purpose = purpose.strip()
        self.threshold = BALANCED.threshold if threshold is None else threshold
        self.scorer = scorer or default_scorer()
        self.judge = judge
        self.judge_band = judge_band
        self.meter = DriftMeter(alpha=drift_alpha, baseline_window=drift_baseline_window)

    @classmethod
    def from_preset(
        cls, preset: "str | Preset", purpose: str, **kwargs
    ) -> "PurposeGuard":
        """Construct a guard using a named threshold preset.

        ``preset`` is a name ("narrow"/"balanced"/"broad") or a `Preset`. Any
        keyword (including an explicit ``threshold``) overrides the preset's
        value. Presets are calibrated for the EmbeddingScorer — see
        purposeguard/presets.py.

            guard = PurposeGuard.from_preset("broad", purpose="A travel concierge")
        """
        resolved = get_preset(preset)
        kwargs.setdefault("threshold", resolved.threshold)
        return cls(purpose, **kwargs)

    @property
    def purpose(self) -> str:
        """Read-only. The purpose is fixed for the guard's lifetime."""
        return self._purpose

    def check(self, write: "Write | str", *, record: bool = True) -> Verdict:
        """Score one write against the purpose and return a Verdict.

        record=True updates the drift meter. Set it False for a dry-run check
        that shouldn't influence the rolling trend (e.g. re-scoring old data).
        """
        w = Write.coerce(write)
        score = self.scorer.score(w.content, self._purpose)

        details: dict = {"similarity": score, "threshold": self.threshold}

        # Only consult the (optional) judge for borderline cases near the line.
        # This is the cost-control move: embeddings handle the obvious 95%, the
        # judge only arbitrates the ambiguous few.
        decision = Decision.ALLOW if score >= self.threshold else Decision.FLAG
        if self.judge is not None and abs(score - self.threshold) <= self.judge_band:
            on_mission = bool(self.judge(w.content, self._purpose))
            details["judge"] = on_mission
            decision = Decision.ALLOW if on_mission else Decision.FLAG

        if decision == Decision.ALLOW:
            reason = f"alignment {score:.2f} meets threshold {self.threshold:.2f}"
        else:
            reason = f"alignment {score:.2f} below threshold {self.threshold:.2f}"

        if record:
            reading = self.meter.update(score)
            details["drift"] = {
                "current": reading.current,
                "baseline": reading.baseline,
                "drift": reading.drift,
                "drifting": reading.drifting,
            }

        return Verdict(score=score, decision=decision, reason=reason, details=details)

    def drift(self) -> DriftReading:
        """Current drift health without recording a new sample."""
        return self.meter.reading()

    def reanchor(self, context: str, *, max_chars: int = 4000) -> str:
        """Re-inject the purpose at the front AND back of a context string.

        Beats the U-shaped attention curve: models attend best to the start and
        end of their context, so pinning the purpose at both ends keeps the
        agent's mission salient even in long contexts. This is the cheap, always-
        on half of drift defense (detection is the other half).
        """
        anchor = f"[PURPOSE] {self._purpose}"
        body = context if len(context) <= max_chars else context[:max_chars]
        return f"{anchor}\n\n{body}\n\n{anchor}"
