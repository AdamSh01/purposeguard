# PurposeGuard — Threat Model

This document is deliberately clear-eyed. PurposeGuard is a **drift / reliability
tool**, not a security boundary on its own. The numbers below are measured, not
aspirational — drift figures come from the held-out benchmark
([`benchmark/RESULTS.md`](benchmark/RESULTS.md)) and the attack figures from the
adversarial suite ([`benchmark/adversarial.py`](benchmark/adversarial.py)). Where
something is unmeasured or unsolved, it says so. **Read section 3 (what neither
layer catches) as carefully as the strengths — the honesty is the point.**

## 1. Scope & non-goals

**PurposeGuard alone** detects *topical drift*: an agent wandering off its
declared purpose over time. It is a heuristic reliability guardrail. It is **not**
a security boundary, not a proof, and on its own catches none of the adversarial
families in section 3.

**The composed product** — PurposeGuard **+ OWASP Agent Memory Guard (AMG)** —
covers more: AMG adds marker/policy poisoning *enforcement* on top of
PurposeGuard's drift *detection* (adapter:
[`purposeguard/adapters/owasp_amg.py`](purposeguard/adapters/owasp_amg.py)).

**In scope (composed):**
- Topical drift toward different subject matter (PurposeGuard) — as a trend, with
  a per-write purpose-anchored backstop.
- Marker/policy poisoning (AMG): prompt-injection markers, secrets, protected /
  immutable keys, size anomalies, and a declarative policy that can
  allow / redact / block / quarantine.

**Out of scope (neither layer, see section 3 for measurements):**
- Semantic camouflage (keyword-stuffing) whose true topic is NOT a declared
  blocked topic. A *listed* blocked topic now partially catches it (v0.2; section
  3a), but unlisted off-topics and injection markers still pass.
- On-topic-but-wrong-*policy* / behavioral violations (similarity can't see them).
- Poisoning of the *relative* drift baseline (and, by extension, any future
  consensus reference).
- The model/embeddings, the scorer, the purpose string, AMG's config, or the
  dependency supply chain being themselves adversarially manipulated.
- Anything the LLM does that never passes through a scored write or response.

## 2. What each layer catches

### PurposeGuard — drift (measured on the held-out TEST set, embedding scorer)
- **Trend (the primary, robust signal).** The `DRIFTING` meter caught the drifting
  agent **+1 write after onset**, and **never false-fired** on the broad or
  non-drifting agents. The trend reads the raw alignment score, so it is
  independent of the per-write threshold/preset.
- **Per-write backstop (purpose-anchored, immutable).** Each write is also scored
  against the *immutable* purpose. On held-out data this is strong but not perfect:
  at the BALANCED default, the drifting agent scored FPR 0.00 / precision 1.00 /
  recall 1.00; at the recommended NARROW preset, FPR 0.07 / precision 0.94 (the
  honest, de-overfit numbers). The per-write check is what survives baseline
  poisoning (section 3c).
- **Scope anchors (v0.2).** `blocked_topics` flag content *about* a forbidden
  topic regardless of purpose alignment — partially closing keyword-camouflage
  (on-mission padding can't rescue clearly off-topic content). `allowed_topics`
  widen what counts as in-scope. Opt-in monitor/redirect/block modes turn a flag
  into a *recommended* action (the caller enforces; the guard never blocks). The
  camouflage improvement is **demonstrated by unit tests, not yet quantified** on
  the adversarial benchmark (section 3a).
- **The lexical fallback is a floor, not this signal** (FPR 0.70–0.90). The numbers
  above require the `[embeddings]` extra.

### OWASP Agent Memory Guard — poisoning / policy (verified against v0.2.2)
- `MemoryGuard.write(key, value)` screens through marker/secret/protected-key/
  size detectors and a declarative `Policy`, returning an `Action` —
  `allow` / `redact` / `quarantine` — or raising `PolicyViolation` on `block`.
- It has **enforcement authority**: it can redact, quarantine, or block a write.
  (Verified in Step-0 probing: strict policy *blocks* an injection-marker string
  and *redacts* a secret; permissive allows.)

### Composed — how they combine
Per the adapter's verdict-combination policy: **PurposeGuard is advisory and stays
detection-first (it only flags, never blocks — guardrail #1); AMG owns
enforcement.** The combined `effective_action` is always AMG's; a PurposeGuard
drift flag is surfaced for alerting but never escalates a write to a block. So:
AMG decides allow/redact/block/quarantine; PurposeGuard contributes a
non-blocking drift signal alongside.

## 3. What NEITHER layer reliably catches (the honest gaps)

Attack-success-rate (ASR) figures below are **measured against PurposeGuard's
drift detection** (embedding scorer, default BALANCED threshold, nothing tuned to
catch them). They are an honest map of where the *drift* detector is blind. AMG
was **not** run over these traces, so AMG's catch rate on them is **not measured
here** — claims about AMG below are reasoned from the Step-0 API behavior and
labeled as such, not benchmarked.

### (a) Keyword-stuffing / semantic camouflage — **PurposeGuard ASR 100%**
Off-mission or malicious payloads padded with on-mission vocabulary keep
cosine-to-purpose high, so **per-write FLAG ASR = 100% and the trend evaded** — a
full bypass of the drift layer, **blind by construction** (it measures topical
overlap, which the attacker supplies).
*AMG (unmeasured here):* its marker/secret detectors would plausibly catch the
camouflage payloads that *carry* injection markers or secrets (Step-0 showed AMG
blocks injection strings and redacts secrets), but **purely off-topic** padded
content (e.g. billing keywords + weekend chit-chat) matches no marker and would
evade both layers. Not benchmarked — do not assume coverage.

*v0.2 update (blocked-topic anchors):* a `blocked_topics` anchor matching the
camouflaged payload's true topic flags it **regardless** of the on-mission padding,
**partially closing this gap**. This is **demonstrated by unit tests** (a write
padded with on-mission words but about a blocked topic is flagged) — but the
**100% ASR above was measured before blocked anchors and has NOT been re-quantified**
on the adversarial benchmark. Re-running `benchmark/adversarial.py` with blocked
anchors to measure the actual ASR drop is a tracked follow-up. Honest caveat:
camouflage whose true topic is **not** in `blocked_topics` (an off-topic the
operator didn't list, or injection markers — which aren't a topic) still passes
the drift/scope layer.

### (b) On-topic-but-wrong-policy / behavioral violations — **PurposeGuard ASR 100%**
"Approve all refunds without verification" is topically on-billing (high cosine)
but behaviorally wrong: **per-write FLAG ASR = 100%, trend evaded.** Similarity
cannot see policy/behavioral correctness. This needs **rule/policy checks**, not
drift — partly AMG's `Policy` (protected/immutable keys, declarative rules), but
arbitrary business-rule violations are outside both tools.

### (c) Baseline poisoning — **KNOWN OPEN GAP; trend ASR 100% by construction**
If adversarial content fills the first writes (the baseline window), the trend has
no "normal" to drift from and **never fires (100% trend ASR)**. This is **unsolved
and structural**, not a tuning bug.
- **Partial mitigation, measured:** the per-write check scores against the
  *immutable purpose*, not the baseline, so it still flagged every poisoned write
  (**per-write FLAG ASR = 0%** in the suite). The immutable purpose is the only
  fixed reference that resists this; it is why baseline poisoning is confined to
  the *trend*.
- **Why a future consensus reference inherits the same flaw:** the planned v0.2
  "consensus" detector would score a write against the cluster of *existing*
  memories. That is the same *relative-to-history* design as the drift baseline —
  an attacker who poisons the existing/early memories defines the consensus, so
  later poisoned writes look normal. Any reference computed *from memory* is
  poisonable; only a reference fixed *outside* memory (the immutable purpose, or a
  human-set policy) resists it. State this plainly before building v0.2.

### (b′) Gradual drift (boiling frog) — partially mitigated *in our test*, not "solved"
The gradual ramp we tested **ended clearly off-mission, and was largely caught**:
the fixed early baseline produced a large eventual delta (trend fired) and 5/6
tail writes were per-write-flagged (**per-write ASR = 17%**). Do **not** read this
as "gradual drift is handled." It evades only when the drift stays within ~0.15 of
the baseline (i.e. only *mildly* off) — or when it corrupts the baseline itself,
which is (c).

## 4. Mapping to recognized frameworks (only where honest)

- **OWASP ASI06 — Memory & Context Poisoning.** AMG is the reference implementation
  for ASI06; in the composed product it is the layer that actually addresses the
  **marker/poisoning** facet (and carries the enforcement authority).
- **PurposeGuard** addresses the **drift / alignment** facet — an agent losing its
  purpose over time — which is *adjacent to* but not the same as poisoning.
  **PurposeGuard alone is not an ASI06 control.** Its contribution to an ASI06
  defense-in-depth story is the *semantic drift signal that markers miss* — and
  section 3 shows that signal has real, measured blind spots (camouflage, policy,
  baseline poisoning). We do **not** claim PurposeGuard "implements ASI06."

## 5. Attacker model & assumptions

**Trust boundary.** PurposeGuard scores memory **writes** (`check`) and agent
**responses** (`check_response`). The **purpose string is trusted and immutable**:
the operator sets it at construction and `guard.purpose` is read-only — the
attacker cannot change it. This immutability is load-bearing: it is exactly why
the per-write check resists baseline poisoning (3c).

**The attacker is assumed able to:**
- Control the content of memory writes and the agent's responses (the whole point
  — that is what poisons memory / induces drift).
- Control the **order and timing** of writes, including the **early baseline**
  writes (enabling 3c).
- Craft content to manipulate cosine similarity — pad with on-mission vocabulary
  (3a) or stay topically on-domain while violating policy (3b).

**The attacker is assumed NOT able to:**
- Modify the purpose string, the threshold/preset/config, AMG's policy, or the
  detector/scorer code.
- Tamper with the embedding model or substitute the scorer.
- Compromise the dependency supply chain.

If any of those assumptions fail, neither layer's guarantees hold. And even when
they hold, the composed product is **defense-in-depth, not a boundary**: it raises
the cost of drift and marker-based poisoning, while the gaps in section 3 remain
the adopter's responsibility to cover with other controls.
