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
from .types import Decision, RecommendedAction, Verdict, Write

_VALID_MODES = ("monitor", "redirect", "block")
# decision==FLAG -> what the guard RECOMMENDS, by mode (the caller enforces).
_FLAG_ACTION_BY_MODE = {
    "monitor": RecommendedAction.MONITOR,
    "redirect": RecommendedAction.REDIRECT,
    "block": RecommendedAction.BLOCK,
}
# The recommendations that suppress/replace the off-scope output, so a safe
# fallback message is rendered for them. (monitor only observes; allow is fine.)
_FALLBACK_ACTIONS = (RecommendedAction.REDIRECT, RecommendedAction.BLOCK)

# Sane default fallback templates. Fields available to a custom template:
# {purpose}, {blocked_topic} (the matched forbidden topic), {reason}.
DEFAULT_FALLBACK_TEMPLATE = "I can only help with {purpose}. I can't help with {blocked_topic}."
DEFAULT_FALLBACK_GENERIC = "I can only help with {purpose}."

# Default blocked-anchor trip threshold, calibrated on the TRAIN domains (router +
# wellness) by benchmark/calibrate.py: ~95th percentile of legit in-domain vs
# OFF-scope similarity (~0.41) plus margin, staying below real blocked-topic
# content (~0.49+). DECOUPLED from the alignment threshold on purpose: reusing the
# low alignment threshold (~0.15) made near-domain blocked anchors over-flag
# legitimate in-domain traffic (the v0.2.0 footgun; see benchmark/RESULTS.md).
DEFAULT_BLOCKED_THRESHOLD = 0.46

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
        allowed_topics: Optional[list] = None,
        blocked_topics: Optional[list] = None,
        blocked_threshold: Optional[float] = None,
        mode: str = "monitor",
        fallback_template: str = DEFAULT_FALLBACK_TEMPLATE,
        fallback_template_generic: str = DEFAULT_FALLBACK_GENERIC,
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
        allowed_topics: optional in-scope topics that WIDEN what counts as
            on-mission. A write's alignment is max(similarity-to-purpose,
            best-similarity-to-an-allowed-topic), so content clearly on an allowed
            topic passes even if it only loosely matches the purpose string.
        blocked_topics: optional forbidden topics. A write whose similarity to ANY
            blocked topic reaches `blocked_threshold` is FLAGged REGARDLESS of
            purpose alignment (blocked OVERRIDES alignment). This partially closes
            keyword-camouflage: on-mission vocabulary cannot rescue content that is
            clearly about a blocked topic. A blocked hit is an immediate hard flag
            and does NOT feed the drift meter (drift tracks purpose alignment, not
            forbidden-topic hits). Absent allowed/blocked lists == exact v0.1
            behavior.

            KNOWN TRADEOFF (intentional, conservative): blocked-overrides-alignment
            means a *legitimately on-purpose* write that sits semantically near a
            blocked anchor will still flag — e.g. a billing agent that mentions
            "legal advice" while otherwise staying on billing. We accept this
            false-positive risk in exchange for camouflage resistance; raise
            `blocked_threshold` to make blocked hits stricter (fewer false flags,
            weaker camouflage coverage). The logic is deliberately not softened.
        blocked_threshold: similarity-to-a-blocked-topic at/above which a write is
            flagged. Defaults to DEFAULT_BLOCKED_THRESHOLD (~0.46), calibrated on
            TRAIN so near-domain blocked anchors don't fire on legitimate in-domain
            writes. DECOUPLED from `threshold` on purpose: the alignment threshold
            (~0.15) is far too low for blocked anchors and over-flags. Tuning lever
            for the conservatism tradeoff above.
        mode: what the guard RECOMMENDS on an off-scope (FLAG) write. "monitor"
            (DEFAULT) = flag only, exactly v0.1/Step-1 behavior; "redirect" =
            recommend replacing the output with a safe fallback; "block" =
            recommend stopping the off-scope output. Modes change ONLY the verdict's
            `recommended_action`; the guard NEVER acts on, mutates, drops, or stops
            the caller's data (guardrail #1) — the CALLER reads the recommendation
            and enforces.
        fallback_template: message rendered into `verdict.fallback` when a blocked
            topic is matched and the recommendation is REDIRECT/BLOCK. Fields:
            {purpose}, {blocked_topic}, {reason}. Default:
            "I can only help with {purpose}. I can't help with {blocked_topic}."
        fallback_template_generic: message used instead when the flag is low
            alignment (no specific blocked topic). Fields: {purpose}, {reason}.
            Default: "I can only help with {purpose}." The fallback is only ever a
            SUGGESTION carried in the verdict; the guard never sends it.
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
        # Scope anchors (fixed for the guard's lifetime, like the purpose).
        self._allowed_topics = [t.strip() for t in (allowed_topics or []) if t and t.strip()]
        self._blocked_topics = [t.strip() for t in (blocked_topics or []) if t and t.strip()]
        self.blocked_threshold = (
            DEFAULT_BLOCKED_THRESHOLD if blocked_threshold is None else blocked_threshold
        )
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")
        self.mode = mode
        self.fallback_template = fallback_template
        self.fallback_template_generic = fallback_template_generic
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

    @property
    def allowed_topics(self) -> tuple:
        """Read-only in-scope topic anchors (fixed for the guard's lifetime)."""
        return tuple(self._allowed_topics)

    @property
    def blocked_topics(self) -> tuple:
        """Read-only forbidden topic anchors (fixed for the guard's lifetime)."""
        return tuple(self._blocked_topics)

    def _best_anchor(self, content: str, anchors: list):
        """Best ``(score, anchor)`` of ``content`` over a list of topic anchors.

        Reuses the SAME scorer as the purpose — anchors are just extra reference
        strings, not a parallel scoring route. Returns ``(0.0, None)`` if empty.
        """
        best_score, best_anchor = 0.0, None
        for anchor in anchors:
            s = self.scorer.score(content, anchor)
            if s > best_score:
                best_score, best_anchor = s, anchor
        return best_score, best_anchor

    def _evaluate(
        self,
        content: str,
        *,
        meter: DriftMeter,
        record: bool,
        extra_details: Optional[dict] = None,
    ) -> Verdict:
        """Shared score -> policy -> meter path for every kind of check.

        check() and check_response() both route through here, so there is exactly
        one scoring route; only the text and the meter differ. Combination logic:

          alignment   = max(sim(content, purpose), best sim(content, allowed_topics))
          blocked_hit = sim(content, any blocked_topic) >= blocked_threshold

        A blocked hit FLAGs the write regardless of alignment (blocked overrides,
        and the judge does not rescue it). Otherwise the write ALLOWs iff
        alignment >= threshold (allowed topics having widened alignment). Only the
        ALIGNMENT score feeds the drift meter — a blocked hit is a discrete hard
        flag, not gradual drift.
        """
        purpose_score = self.scorer.score(content, self._purpose)
        allowed_score, allowed_topic = self._best_anchor(content, self._allowed_topics)
        alignment = max(purpose_score, allowed_score)

        blocked_score, blocked_topic = self._best_anchor(content, self._blocked_topics)
        blocked_hit = bool(self._blocked_topics) and blocked_score >= self.blocked_threshold

        details: dict = {"similarity": alignment, "threshold": self.threshold}
        if extra_details:
            details.update(extra_details)
        if self._allowed_topics:
            details["allowed"] = {"score": allowed_score, "topic": allowed_topic}
        if self._blocked_topics:
            details["blocked"] = {
                "score": blocked_score,
                "topic": blocked_topic,
                "threshold": self.blocked_threshold,
                "hit": blocked_hit,
            }

        if blocked_hit:
            # Blocked overrides alignment; the judge does not rescue a blocked hit.
            decision = Decision.FLAG
            reason = (
                f"matched blocked topic {blocked_topic!r} "
                f"(similarity {blocked_score:.2f} >= {self.blocked_threshold:.2f})"
            )
        else:
            decision = Decision.ALLOW if alignment >= self.threshold else Decision.FLAG
            # Judge only arbitrates borderline PURPOSE alignment (cost-control).
            if self.judge is not None and abs(alignment - self.threshold) <= self.judge_band:
                on_mission = bool(self.judge(content, self._purpose))
                details["judge"] = on_mission
                decision = Decision.ALLOW if on_mission else Decision.FLAG
            reason = (
                f"alignment {alignment:.2f} meets threshold {self.threshold:.2f}"
                if decision == Decision.ALLOW
                else f"alignment {alignment:.2f} below threshold {self.threshold:.2f}"
            )

        # The mode-aware RECOMMENDATION (advisory; the caller enforces). decision
        # stays detection (ALLOW/FLAG) in every mode — even block — so the guard
        # never itself escalates to acting on the caller's data (guardrail #1).
        recommended = (
            RecommendedAction.ALLOW
            if decision == Decision.ALLOW
            else _FLAG_ACTION_BY_MODE[self.mode]
        )
        details["mode"] = self.mode

        # A SUGGESTED safe message for interventions only (redirect/block). It is
        # text the caller MAY use; the guard never sends it. None for allow/monitor.
        fallback = None
        if recommended in _FALLBACK_ACTIONS:
            fallback = self._render_fallback(blocked_hit, blocked_topic, reason)

        if record:
            # Drift tracks PURPOSE ALIGNMENT only; a blocked hit never feeds it.
            reading = meter.update(alignment)
            details["drift"] = {
                "current": reading.current,
                "baseline": reading.baseline,
                "drift": reading.drift,
                "drifting": reading.drifting,
            }

        return Verdict(
            score=alignment,
            decision=decision,
            reason=reason,
            details=details,
            recommended_action=recommended,
            fallback=fallback,
        )

    def _render_fallback(self, blocked_hit, blocked_topic, reason):
        """Render the suggested safe message. Blocked hits name the topic; a
        low-alignment flag uses the generic (purpose-only) template. Never raises:
        a bad custom template degrades to a minimal safe message (guardrail #4)."""
        ctx = {
            "purpose": self._purpose,
            "blocked_topic": blocked_topic or "",
            "reason": reason,
        }
        template = self.fallback_template if blocked_hit else self.fallback_template_generic
        try:
            return template.format(**ctx)
        except Exception:
            return DEFAULT_FALLBACK_GENERIC.format(purpose=self._purpose)

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
