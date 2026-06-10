"""PurposeGuard — the single public interface.

Usage is meant to be one line to construct, one line to check:

    guard = PurposeGuard(purpose="A customer-support agent for billing questions")
    verdict = guard.check("User asked how to bake sourdough bread")
    # -> Verdict(flag, score=0.10, reason='alignment 0.10 below threshold 0.15')

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

# Suggested purpose_floor for the purpose-anchored drift signal, calibrated on the
# TRAIN domains (benchmark/calibrate.py) for NARROW/focused agents (EMA ~0.7+).
# It catches OFF-purpose baseline poisoning the relative trend misses. It is OPT-IN
# (purpose_floor defaults to None) and NOT safe for broad agents: measured, this
# floor false-fired on 27/30 legitimate broad on-mission writes -- a broad agent's
# alignment overlaps the poison level, so NO floor separates them. It is also blind
# by construction to ON-purpose poisoning. See THREAT_MODEL.md.
DEFAULT_PURPOSE_FLOOR = 0.28

# Optional LLM judge: a function (content, purpose) -> bool ("is this on-mission?").
# The library never ships one — the user supplies it if they want a second
# opinion on borderline scores. Keeping it injectable means no API dependency
# and no opinion baked in about which model to call.
LLMJudge = Callable[[str, str], bool]


def _validate_unit(name: str, value: float, *, allow_zero: bool = True) -> None:
    """Validate a similarity-style parameter lies in [0,1] (or (0,1] if not allow_zero).

    Thresholds and the EMA alpha are meaningless outside this band: a threshold >1
    flags everything, <0 allows everything, and an alpha outside (0,1] drives the
    drift EMA outside the [0,1] range every docstring promises. Mirrors the existing
    purpose/mode validation so misconfiguration fails loud at construction instead of
    silently producing nonsense verdicts.
    """
    lo_ok = value >= 0.0 if allow_zero else value > 0.0
    if not (lo_ok and value <= 1.0):
        rng = "[0.0, 1.0]" if allow_zero else "(0.0, 1.0]"
        raise ValueError(f"{name} must be in {rng}, got {value!r}")


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
        purpose_floor: Optional[float] = None,
        scorer: Optional[Scorer] = None,
        require_embeddings: bool = False,
        judge: Optional[LLMJudge] = None,
        judge_band: float = 0.10,
        drift_alpha: float = 0.2,
        drift_baseline_window: int = 10,
        on_verdict: Optional[Callable[[Verdict], None]] = None,
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
            clearly about a blocked topic. A blocked hit is an immediate hard flag;
            it adds no forbidden-topic-specific signal to the drift meter (drift
            tracks purpose alignment only). The write's own alignment score is still
            recorded in the meter even on a blocked hit. Absent allowed/blocked
            lists == exact v0.1 behavior.

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
        purpose_floor: OPT-IN absolute floor for the purpose-ANCHORED drift signal
            (`drift().anchored_drifting`). Fires when the EMA of alignment to the
            immutable purpose drops below it — with NO baseline term, so a poisoned
            baseline can't hide an off-purpose stream. Default None (OFF). For a
            NARROW/focused agent, pass ~DEFAULT_PURPOSE_FLOOR (0.28) to catch
            off-purpose baseline poisoning. It is BLIND to ON-purpose poisoning and
            is NOT safe for broad agents (it false-fired on 27/30 legit broad writes
            in testing) — that's why it's off by default. See THREAT_MODEL.md.
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
        # Range-validate every numeric knob (fail loud, like purpose/mode below).
        _validate_unit("threshold", self.threshold)
        _validate_unit("blocked_threshold", self.blocked_threshold, allow_zero=False)
        _validate_unit("drift_alpha", drift_alpha, allow_zero=False)
        if purpose_floor is not None:
            _validate_unit("purpose_floor", purpose_floor)
        if drift_baseline_window < 1:
            raise ValueError(
                f"drift_baseline_window must be >= 1, got {drift_baseline_window!r}"
            )
        if judge_band < 0:
            raise ValueError(f"judge_band must be >= 0, got {judge_band!r}")
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")
        self.mode = mode
        self.fallback_template = fallback_template
        self.fallback_template_generic = fallback_template_generic
        self.scorer = resolve_scorer(scorer, require_embeddings=require_embeddings)
        self.judge = judge
        self.judge_band = judge_band
        # Optional observability hook fired with every Verdict (a structured event
        # stream so adapters don't each reimplement logging). It must never break
        # the guard: an exception in the hook is swallowed with a warning.
        self.on_verdict = on_verdict
        # Two independent meters on the same machinery: one for what the agent
        # STORES (check), one for what it SAYS (check_response). They are kept
        # separate on purpose — see check_response's docstring for why.
        self.meter = DriftMeter(
            alpha=drift_alpha,
            baseline_window=drift_baseline_window,
            purpose_floor=purpose_floor,
        )
        self._response_meter = DriftMeter(
            alpha=drift_alpha,
            baseline_window=drift_baseline_window,
            purpose_floor=purpose_floor,
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

    def _score_many(self, content: str, references: list) -> list:
        """Score ``content`` against many reference strings, embedding the content
        ONCE when the scorer supports it (``score_many``), else one ``score()`` per
        reference. This is the hot path: every check scores one content against the
        purpose + every allowed + every blocked anchor, so embedding content once
        instead of once-per-anchor is the real per-check saving. Anchors are just
        extra reference strings — not a parallel scoring route."""
        if not references:
            return []
        score_many = getattr(self.scorer, "score_many", None)
        if score_many is not None:
            return list(score_many(content, references))
        return [self.scorer.score(content, r) for r in references]

    @staticmethod
    def _best(scores: list, anchors: list):
        """Best ``(score, anchor)`` over parallel score/anchor lists. Seeds the
        anchor on the first element, so a non-empty list always names a topic — even
        if every score is 0.0 — and blocked_hit can never be True with topic=None.
        Returns ``(0.0, None)`` for an empty list."""
        best_score, best_anchor = 0.0, None
        for s, a in zip(scores, anchors):
            if best_anchor is None or s > best_score:
                best_score, best_anchor = s, a
        return best_score, best_anchor

    def _evaluate(
        self,
        content: str,
        *,
        meter: DriftMeter,
        record: bool,
        extra_details: Optional[dict] = None,
        context: Optional[dict] = None,
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
        # Embed `content` ONCE and score it against the purpose + all anchors in a
        # single pass (see _score_many): index 0 is the purpose, then the allowed
        # topics, then the blocked topics.
        na = len(self._allowed_topics)
        refs = [self._purpose, *self._allowed_topics, *self._blocked_topics]
        scores = self._score_many(content, refs)
        purpose_score = scores[0]
        allowed_score, allowed_topic = self._best(scores[1 : 1 + na], self._allowed_topics)
        alignment = max(purpose_score, allowed_score)

        blocked_score, blocked_topic = self._best(scores[1 + na :], self._blocked_topics)
        # Require an actually-selected anchor: never report a hit with topic=None
        # (defends against a degenerate blocked_threshold; see _best).
        blocked_hit = blocked_topic is not None and blocked_score >= self.blocked_threshold

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
            # Drift tracks PURPOSE ALIGNMENT only. The forbidden-topic signal is
            # never mixed in — but the write's own alignment score IS still recorded
            # even on a blocked hit (a camouflage write is genuinely purpose-aligned;
            # the blocked anchor already fired the hard flag separately).
            reading = meter.update(alignment)
            details["drift"] = {
                "current": reading.current,
                "baseline": reading.baseline,
                "drift": reading.drift,
                "drifting": reading.drifting,
                "anchored_drifting": reading.anchored_drifting,
            }

        verdict = Verdict(
            score=alignment,
            decision=decision,
            reason=reason,
            details=details,
            recommended_action=recommended,
            fallback=fallback,
            context=dict(context) if context else {},
        )
        if self.on_verdict is not None:
            try:
                self.on_verdict(verdict)
            except Exception:
                import warnings

                warnings.warn(
                    "on_verdict hook raised an exception; ignoring it (the "
                    "observability hook must not break the guard).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return verdict

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

    def check(
        self, write: "Write | str", *, record: bool = True, context: Optional[dict] = None
    ) -> Verdict:
        """Score one memory WRITE against the purpose and return a Verdict.

        record=True updates the write drift meter (read via drift()). Set it
        False for a dry-run check that shouldn't influence the trend (e.g.
        re-scoring old data). ``context`` (e.g. {"session_id": ...}) is merged with
        the Write's own context and carried onto the Verdict for observability.
        """
        w = Write.coerce(write)
        ctx = {**w.context, **(context or {})}
        return self._evaluate(
            w.content, meter=self.meter, record=record,
            extra_details={"kind": "write"}, context=ctx,
        )

    def check_many(
        self, writes: "list[Write | str]", *, record: bool = True
    ) -> "list[Verdict]":
        """Score a batch of writes in order. Convenience over a loop; verdicts are
        returned in input order and (when ``record``) feed the write drift meter
        sequentially, exactly as repeated ``check()`` calls would."""
        return [self.check(w, record=record) for w in writes]

    async def acheck(
        self, write: "Write | str", *, record: bool = True, context: Optional[dict] = None
    ) -> Verdict:
        """Async-compatible ``check`` for async agent loops. Scoring is CPU-bound and
        synchronous under the hood; this is an API-compatibility surface (so async
        callers don't block on a sync signature), not a concurrency speedup."""
        return self.check(write, record=record, context=context)

    def check_response(
        self, user_input: str, response: str, *, record: bool = True,
        context: Optional[dict] = None,
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
            context=context or {},
        )

    async def acheck_response(
        self, user_input: str, response: str, *, record: bool = True,
        context: Optional[dict] = None,
    ) -> Verdict:
        """Async-compatible ``check_response`` (see :meth:`acheck` on why it's a thin
        wrapper). Lets async response handlers ``await guard.acheck_response(...)``."""
        return self.check_response(user_input, response, record=record, context=context)

    def drift(self) -> DriftReading:
        """Current WRITE drift health (from check()) without recording a sample."""
        return self.meter.reading()

    def response_drift(self) -> DriftReading:
        """Current RESPONSE drift health (from check_response()), kept separate
        from write drift() so the two failure modes don't mask each other."""
        return self._response_meter.reading()

    def health(self) -> dict:
        """One composite, JSON-friendly snapshot of BOTH drift channels.

        The two meters stay separate by design (write-drift vs response-drift can
        fail independently), but callers who just want a single "is this agent
        on-mission?" signal can read this. ``status`` is ``"drifting"`` if either
        channel is drifting (relative trend) or off-purpose (anchored floor)."""
        w, r = self.drift(), self.response_drift()
        unhealthy = (
            w.drifting or r.drifting or w.anchored_drifting or r.anchored_drifting
        )
        return {
            "status": "drifting" if unhealthy else "ok",
            "write_drift": w.to_dict(),
            "response_drift": r.to_dict(),
        }

    def state(self) -> dict:
        """Serialize the live drift state (both meters) so it survives a process
        restart. Drift is a long-horizon signal: a guard rebuilt fresh each request
        loses its baseline/EMA and starts the trend over. Persist this and rehydrate
        with :meth:`restore`. The purpose is recorded so restore can verify it."""
        return {
            "version": 1,
            "purpose": self._purpose,
            "write_meter": self.meter.state(),
            "response_meter": self._response_meter.state(),
        }

    def restore(self, state: dict) -> None:
        """Restore drift state produced by :meth:`state`. Refuses a state saved under
        a different purpose — the trend is only meaningful against the same immutable
        reference it was measured against."""
        if state.get("purpose") != self._purpose:
            raise ValueError(
                "cannot restore drift state saved under a different purpose"
            )
        self.meter = DriftMeter.from_state(state["write_meter"])
        self._response_meter = DriftMeter.from_state(state["response_meter"])

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
