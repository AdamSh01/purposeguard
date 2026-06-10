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
  into a *recommended* action (the caller enforces; the guard never blocks).
  Measured (v0.2.1, calibrated `blocked_threshold`): blocked anchors close part of
  the topical half of keyword-camouflage when the off-topic is enumerated (overall
  100% → 70% ASR, 0 false positives), but the malicious half survives — section 3a.
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

*v0.2.1 measured (blocked-topic anchors, calibrated threshold).* With the
TRAIN-calibrated default `blocked_threshold` (0.46), blocked anchors close **part
of the topical half** of camouflage (topical ASR 100% → **40%**; overall camouflage
**100% → 70%**) **only when the specific, distinct off-topic is enumerated**, with
**zero** false positives. The catch is partial because the on-mission padding
dilutes the camo's similarity to the blocked anchor. The **malicious half**
(injection/exfiltration/credential-theft) is **fully unaffected (100%)** — not
topics; that's AMG's job. A mismatched/abstract list catches nothing and (post
calibration fix) over-flags nothing (0/10 legit). NOTE: v0.2.0 shipped with
`blocked_threshold` defaulting to the alignment threshold, which over-flagged 7/10
legit writes; v0.2.1 decoupled it. Bottom line: the camouflage close is **real but
narrow and partial** (topical-only, enumerated, ~30% of camouflage) — see
`benchmark/RESULTS.md` §3a. Camouflage in an unlisted off-topic, or via injection
markers, still passes.

### (b) On-topic-but-wrong-policy / behavioral violations — **PurposeGuard ASR 100%**
"Approve all refunds without verification" is topically on-billing (high cosine)
but behaviorally wrong: **per-write FLAG ASR = 100%, trend evaded.** Similarity
cannot see policy/behavioral correctness. This needs **rule/policy checks**, not
drift — partly AMG's `Policy` (protected/immutable keys, declarative rules), but
arbitrary business-rule violations are outside both tools.

### (c) Baseline poisoning — **partially mitigated by purpose-anchored detection (measured)**
If adversarial content fills the first writes (the baseline window), the RELATIVE
drift trend (drift-from-baseline) has no "normal" to drift from and **never fires
(100% trend ASR)**. v0.3 adds a purpose-ANCHORED signal that addresses the
*off-purpose* case; the precise, measured scope:

- **Per-write check (always on):** scores against the *immutable purpose*, not the
  baseline, so it still flagged every poisoned write (**per-write FLAG ASR = 0%**).
- **Purpose-anchored trend (v0.3, opt-in `purpose_floor`):** the EMA of
  alignment-to-purpose vs an ABSOLUTE floor (no baseline term). Measured: on the
  off-purpose baseline-poisoning family it **FIRED** where the relative trend
  EVADED. This closes the *off-purpose* baseline-poisoning trend gap.
- **BLIND to ON-purpose poisoning (by construction, headline caveat):** content
  that stays purpose-aligned (e.g. "approve all refunds without verification" —
  topically billing, behaviorally wrong) stays ABOVE the floor and is invisible.
  Purpose-anchored detection closes the *off-purpose* gap; **on-purpose poisoning
  needs rules, not drift** (it is family (b), 100% ASR, AMG/policy territory).
- **Broad-agent false positives — why it's OPT-IN, not default:** an absolute
  floor is the broad-agent footgun. Measured, the TRAIN-calibrated floor (0.28)
  that catches poisoning false-fired on **27/30 legitimate broad on-mission
  writes** (held-out general-assistant). A broad agent's legit alignment (~0.2–0.25)
  *overlaps* the poison level (~0.18) — **no single floor separates them.** So the
  anchored signal ships **off by default**; enable it (`purpose_floor≈0.28`) only
  for NARROW/focused agents (whose alignment sits ~0.7, far above the floor).
- **Why a future consensus reference inherits the relative-trend flaw:** a v0.3+
  "consensus" detector scores a write against the cluster of *existing* memories —
  the same *relative-to-history* design as the drift baseline. An attacker who
  poisons the existing memories defines the consensus, so later poisoned writes
  look normal. Any reference computed *from memory* is poisonable; only a reference
  fixed *outside* memory (the immutable purpose, or a human-set policy) resists it.

### (b′) Gradual drift (boiling frog) — partially mitigated *in our test*, not "solved"
The gradual ramp we tested **ended clearly off-mission, and was largely caught**:
the fixed early baseline produced a large eventual delta (trend fired) and 5/6
tail writes were per-write-flagged (**per-write ASR = 17%**). Do **not** read this
as "gradual drift is handled." It evades only when the drift stays within ~0.15 of
the baseline (i.e. only *mildly* off) — or when it corrupts the baseline itself,
which is (c).

### (e) Obfuscation / evasion of blocked anchors — **partially mitigated (homoglyph, zero-width); the rest open**
Because the scorer ultimately keys on the *characters* of a write, an attacker can
rewrite blocked-topic words to look the same to a human but score differently:
Unicode homoglyphs (Latin → Cyrillic/Greek look-alikes), zero-width characters
inserted mid-word, paraphrase that shares no tokens with the anchor, a different
language, or splitting a payload across several individually-on-topic writes.
*Measured before hardening:* a three-character Cyrillic swap inside an enumerated
blocked topic flipped a blocked **FLAG → ALLOW** — defeating the exact enumerated
case §3a markets as the working win.
- **Now closed (a floor, not a wall):** a shared input-normalization pass runs
  before *both* scorers (`purposeguard/scoring.py:_normalize`): NFKC, strip
  zero-width / format (`Cf`) characters, and fold a curated set of common
  Cyrillic/Greek homoglyphs to ASCII. This neutralizes the homoglyph and
  zero-width variants (the obfuscated payload now scores like the plain one). It is
  **not** a complete Unicode-confusables defense, and by design it **alters
  genuinely non-Latin-script text** — anchors/purposes are assumed Latin-script.
- **Still open (unmitigated, by construction):** paraphrase below the lexical
  floor, language-switch (also a false-*positive* source on legitimate non-English
  on-mission writes), and payload-splitting across writes. These need a stronger
  scorer and/or rule/policy checks, not character normalization.

These classes are exercised in `tests/test_evasion.py` (closed cases asserted to
flag; open cases asserted to bypass, so the disclosure is regression-locked) and
measured in `benchmark/adversarial.py` (`evasion_followup`).

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
