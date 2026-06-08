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
from .scoring import Scorer, resolve_scorer
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
        require_embeddings: bool = False,
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
        scorer: defaults to local embeddings if available, else lexical (which
            warns once — the floor is a fallback, not the intended experience).
        require_embeddings: opt-in. When True, the guard refuses to run silently
            on the lexical floor: it guarantees a working embedding scorer or
            raises a clear error. For production users who'd rather fail loud than
            ship on the floor. Default False keeps the zero-dependency behavior.
        judge: optional callable for a second opinion on borderline writes only.
        judge_band: how close to the threshold counts as "borderline" and worth
            spending a judge call on. Keeps LLM usage rare and cheap.
        """
        if not purpose or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        self._purpose = purpose.strip()
        self.threshold = BALANCED.threshold if threshold is None else threshold
        self.scorer = resolve_scorer(scorer, require_embeddings=require_embeddings)
        self.judge = judge
        self.judge_band = judge_band
        # Two independent meters on the same machinery: one for what the agent
        # STORES (check), one for what it SAYS (check_response). They are kept
        # separate on purpose — see check_response's docstring for why.
        self.meter = DriftMeter(alpha=drift_alpha, baseline_window=drift_baseline_window)
        self._response_meter = DriftMeter(
            alpha=drift_alpha, baseline_window=drift_baseline_window
        )

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

    def _evaluate(
        self,
        content: str,
        *,
        meter: DriftMeter,
        record: bool,
        extra_details: Optional[dict] = None,
    ) -> Verdict:
        """Shared score -> policy -> meter path for every kind of check.

        Both check() and check_response() route through here so there is exactly
        one scorer call, one threshold policy, and one (optional) judge step —
        only the text being scored and which meter it feeds differ. This is what
        keeps response checking from becoming a parallel scoring path.
        """
        score = self.scorer.score(content, self._purpose)

        details: dict = {"similarity": score, "threshold": self.threshold}
        if extra_details:
            details.update(extra_details)

        # Only consult the (optional) judge for borderline cases near the line.
        # This is the cost-control move: embeddings handle the obvious 95%, the
        # judge only arbitrates the ambiguous few.
        decision = Decision.ALLOW if score >= self.threshold else Decision.FLAG
        if self.judge is not None and abs(score - self.threshold) <= self.judge_band:
            on_mission = bool(self.judge(content, self._purpose))
            details["judge"] = on_mission
            decision = Decision.ALLOW if on_mission else Decision.FLAG

        if decision == Decision.ALLOW:
            reason = f"alignment {score:.2f} meets threshold {self.threshold:.2f}"
        else:
            reason = f"alignment {score:.2f} below threshold {self.threshold:.2f}"

        if record:
            reading = meter.update(score)
            details["drift"] = {
                "current": reading.current,
                "baseline": reading.baseline,
                "drift": reading.drift,
                "drifting": reading.drifting,
            }

        return Verdict(score=score, decision=decision, reason=reason, details=details)

    def check(self, write: "Write | str", *, record: bool = True) -> Verdict:
        """Score one memory WRITE against the purpose and return a Verdict.

        record=True updates the write drift meter (read via drift()). Set it
        False for a dry-run check that shouldn't influence the trend (e.g.
        re-scoring old data).
        """
        w = Write.coerce(write)
        return self._evaluate(
            w.content, meter=self.meter, record=record, extra_details={"kind": "write"}
        )

    def check_response(
        self, user_input: str, response: str, *, record: bool = True
    ) -> Verdict:
        """Score the agent's own RESPONSE against the purpose and return a Verdict.

        An agent can drift in what it *says* even while its memory writes still
        look on-mission, so scoring only writes under-delivers the "keep the agent
        on-mission" promise. This scores the agent's answer with the exact same
        scorer and threshold policy as check() — only the text and the meter
        differ.

        Meter (design decision): responses feed a SEPARATE drift meter, read via
        response_drift(), NOT the write meter behind drift(). They are kept apart
        deliberately. The whole reason this method exists is that answer-drift and
        write-drift are independent failure modes; merging them into one meter
        would let a healthy memory stream mask drifting answers (and vice versa) —
        defeating the feature. The cost is that there is no single unified
        on-mission number out of the box; a caller who wants one can combine
        drift() and response_drift() themselves. The gain is attribution: you can
        tell whether the agent is drifting in its memory or in its live answers.

        user_input (design decision): it is recorded in the verdict details for
        context/forensics but does NOT change the score or decision. Drift is
        about whether the agent's *output* stays on its mission, independent of
        what was asked — a billing agent that answers a weather question has
        drifted even though the user asked about weather. Using user_input to
        excuse off-mission answers would mask exactly the user-induced drift this
        is meant to catch. (The on-topic case needs no help: an on-topic answer
        already scores high and passes.)

        Detection-first: this only ever ALLOWs or FLAGs; it never blocks.

        Calibration note: responses currently use the SAME threshold/preset and
        drift config as writes (calibrated jointly for now). Response alignment
        may sit in a different range than write alignment on real data, so a
        future pass may want to tune the response threshold separately.
        """
        return self._evaluate(
            response,
            meter=self._response_meter,
            record=record,
            extra_details={"kind": "response", "user_input": user_input},
        )

    def drift(self) -> DriftReading:
        """Current WRITE drift health (from check()) without recording a sample."""
        return self.meter.reading()

    def response_drift(self) -> DriftReading:
        """Current RESPONSE drift health (from check_response()), kept separate
        from write drift() so the two failure modes don't mask each other."""
        return self._response_meter.reading()

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
