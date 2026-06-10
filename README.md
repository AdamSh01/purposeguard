# PurposeGuard

**A drift + scope guardrail — purpose/mission observability that keeps your AI agent in its lane.**

Customer-facing agents wander. Session history piles up and the agent slowly
answers off-mission — a billing assistant that drifts into recipes, weather, or
legal advice. PurposeGuard scores every write and response against the agent's
declared **purpose** (and optional **allowed** / **blocked** topics), surfaces
drift as a watchable number, and — when you opt in — tells your caller to redirect
or block off-scope output. The guard *recommends*; your code decides.

- **Drift + scope.** Catch gradual drift (a trend) AND hard scope violations
  (a write clearly about a blocked topic) — even when on-mission vocabulary is
  used to camouflage it.
- **Safe by default, explicit to enforce.** `monitor` mode (the default) only
  *flags* — it never blocks, mutates, or drops your data. `redirect`/`block` are
  opt-in, and even then the guard returns a *recommended action*; your code
  enforces it. You can't break your agent by adding it.
- **Framework-agnostic & easy to adopt.** Mem0, LangChain, a raw store, or a
  plain script. `pip install`, one line to construct, one line to check.
- **Honest about scope.** A drift + scope guardrail, **not a security tool** —
  see [Honest limitations](#honest-limitations).

## Why agents drift — and why your memory layer won't catch it

A role prompt set at session start gets buried as history grows: models attend best
to the **start and end** of their context (the U-shaped curve), so the mission fades
from the middle. Vector memory (mem0, Chroma, LangChain history) optimizes for
**recall** — "what did we say before?" — not **relevance to the mission**, so it
happily stores and resurfaces off-topic chatter. And generic guardrails check inputs
and outputs against *safety* policy, not against *this* agent's declared purpose.
PurposeGuard is the missing relevance-to-mission meter — a watchable drift number
over both what the agent stores and what it says.

First, `pip install "purposeguard[embeddings]"` (the lexical-only base install is a
rough fallback — see [Install](#install)).

```python
from purposeguard import PurposeGuard, RecommendedAction

guard = PurposeGuard(
    purpose="A customer-support agent for billing and payments",
    allowed_topics=["invoices", "refunds", "subscriptions"],   # widen what counts as in-scope
    blocked_topics=["legal advice", "medical advice"],         # hard out-of-scope
    mode="redirect",             # monitor (default) | redirect | block
    require_embeddings=True,     # fail loud rather than ship on the lexical floor
)

verdict = guard.check_response(
    user_input="my invoice looks wrong — also, can you give me legal advice?",
    response="Sure, for your lawsuit here's the legal advice you need...",
)
print(verdict.decision.value)            # 'flag'      -> detection: off scope
print(verdict.recommended_action.value)  # 'redirect'  -> what to do, given the mode
print(verdict.fallback)                  # (with the [embeddings] extra) "I can only help with ... I can't help with legal advice."
print(guard.drift())                     # DriftReading(current=..., baseline=..., drift=..., n=...)

# The guard never sends the fallback or stops anything — YOUR code enforces:
if verdict.recommended_action is RecommendedAction.REDIRECT:
    response = verdict.fallback
```

`monitor` mode (the default) is exactly the v0.1 behavior — flag only, never block
— so upgrading changes nothing until you opt into a mode.

## What it catches — and what it doesn't

PurposeGuard is **observability for mission-alignment**, not a security boundary
(see [Honest limitations](#honest-limitations)). Be clear-eyed about the line:

| Catches | Doesn't catch |
| --- | --- |
| Topical drift away from the mission, as a **trend** (the robust signal) | Semantic camouflage in an *unlisted* topic (on-mission padding keeps cosine high) |
| Per-write off-purpose content vs the immutable purpose | On-topic-but-wrong-**policy** (e.g. "approve all refunds without verification") |
| Enumerated, distinct **blocked topics** — incl. homoglyph / zero-width obfuscation | Prompt injection / exfiltration markers (compose with [OWASP AMG](#compose-with-owasp-agent-memory-guard-defense-in-depth)) |
| Off-purpose baseline poisoning (opt-in `purpose_floor`, narrow agents) | On-purpose poisoning, paraphrase beyond the scorer's reach, payload split across writes |

The numbers behind every cell are measured in [`benchmark/RESULTS.md`](benchmark/RESULTS.md)
and [`THREAT_MODEL.md`](THREAT_MODEL.md) — including where it gets bypassed.

## Install

```bash
pip install "purposeguard[embeddings]"   # recommended for real use: local-embedding scoring (all-MiniLM-L6-v2)
pip install purposeguard                  # minimal: zero deps, but only the rough lexical floor
```

> **Pre-release note:** `purposeguard` is not on PyPI yet. Until the first release,
> install from source:
> ```bash
> pip install "purposeguard[embeddings] @ git+https://github.com/AdamSh01/purposeguard"
> ```

**Install the `[embeddings]` extra for any real use.** It adds local
sentence-embeddings (no API key, runs on CPU) and is the intended experience.

The base install has **zero required dependencies** and works behind a firewall,
but its only scorer is the **lexical floor** — a crash-proof *fallback*, not the
real thing. It keys on word overlap, so it misjudges a lot of natural prose
(benchmark false-positive rate 0.70–0.90; see [`benchmark/RESULTS.md`](benchmark/RESULTS.md)).
Treat lexical as "works anywhere so the library never hard-fails," not as the
detector you ship on. If the embedding model can't be loaded, PurposeGuard warns
once and falls back to lexical instead of crashing.

## What it does

1. **Declare a purpose** once. It's stored immutably — drift can't come from the
   purpose itself being edited away.
2. **Score each write** for alignment with that purpose, in `[0,1]`.
3. **Track drift** as a rolling metric: a stable baseline established early, then
   a live exponential-moving-average of recent alignment. When recent behavior
   falls meaningfully below the agent's own baseline, it flags `DRIFTING`.
4. **Re-anchor** long contexts by re-injecting the purpose at the front *and*
   back (beating the U-shaped attention curve), so the agent stays on-mission.
5. **Scope it** with `allowed_topics` (widen what counts as in-scope) and
   `blocked_topics` (hard out-of-scope). A write clearly about a blocked topic is
   flagged **even when padded with on-mission vocabulary** — partially closing
   keyword-camouflage. A blocked hit is an immediate flag, not gradual drift: it
   adds no forbidden-topic signal to the drift meter (the meter tracks purpose
   alignment only — though the write's own alignment score is still recorded).
6. **Choose a mode** — `monitor` (default, flag only), `redirect` (the verdict
   carries a `fallback` message), or `block` (the verdict signals stop). The guard
   only ever returns a `recommended_action`; **your code enforces it** — the guard
   never blocks, mutates, or drops anything itself.

## See it catch drift

`python examples/drift_demo.py` simulates an agent that starts on-mission and
wanders. The meter stays flat, then climbs:

```
 #  verdict score  drift  memory write
------------------------------------------------------------------------------
 1  ok      0.17   0.00  User asked how to update their payment card
 5  ok      0.35   0.00  Resolved a failed payment and updated the account
 6  FLAG    0.00   0.08  User asked about the weather forecast
 9  FLAG    0.00   0.17  Discussed the user's favorite football team  <-- DRIFTING
10  FLAG    0.08   0.16  Gave tips on training a new puppy            <-- DRIFTING
```

## Two things to watch: what the agent stores *and* what it says

An agent can drift in its **answers** even while its **memory writes** still look
on-mission. So PurposeGuard scores both, with the same scorer and threshold:

```python
guard = PurposeGuard(purpose="A customer-support agent for billing questions")

# What the agent STORES -> feeds guard.drift()
guard.check("User asked about a refund on their invoice")

# What the agent SAYS -> feeds guard.response_drift()
guard.check_response(
    user_input="How do I get a refund?",
    response="Sure! Here's my favorite sourdough recipe...",   # off-mission answer
)

print(guard.drift())            # drift in stored memories
print(guard.response_drift())   # drift in live answers
```

The two feed **separate** drift meters on purpose, so a healthy memory stream
can't mask drifting answers (or vice versa) — and you can attribute drift to
*memory* vs *live answers*. `user_input` is recorded as context but doesn't change
the score: a billing agent that answers a weather question has drifted regardless
of what it was asked. By default (`monitor` mode) `check_response` only flags;
opt into `redirect`/`block` for an enforcement *recommendation* (your code still
acts — the guard never blocks or mutates anything).

## Adapters & examples

Drop PurposeGuard into the frameworks you already use. The core stays
framework-free; each adapter imports its framework lazily, tags off-mission
writes' metadata, and never drops them (detection-first).

| Framework | Wrap with | Import from `purposeguard.adapters` |
| --- | --- | --- |
| Any store with `add`/`search` | `guard_store(store, guard)` | `guard_store` |
| Mem0 | `guard_mem0(client, guard)` or `guarded_memory(guard)` | `guard_mem0`, `guarded_memory` |
| LangChain | `guard_chat_history(history, guard)` | `guard_chat_history`, `GuardedChatMessageHistory` |

Runnable examples (no real framework or API key needed — they use local
embeddings when available and print a visible notice if they fall back to the
lexical floor):

- [`examples/raw_store_example.py`](examples/raw_store_example.py)
- [`examples/mem0_example.py`](examples/mem0_example.py)
- [`examples/langchain_example.py`](examples/langchain_example.py)

## Compose with OWASP Agent Memory Guard (defense-in-depth)

PurposeGuard catches drift + scope. For marker/policy **poisoning enforcement**,
compose it with [OWASP Agent Memory Guard](https://github.com/OWASP/www-project-agent-memory-guard)
in one config — `composed_guard` builds the drift+scope guard AND wires AMG:

```python
from agent_memory_guard import Policy
from purposeguard.adapters import composed_guard

guard = composed_guard(
    "A customer-support assistant for billing and payments",
    blocked_topics=["legal advice", "medical advice"],
    mode="redirect",
    policy=Policy.strict(),        # AMG (lazy-imported, optional dependency)
)
v = guard.write("note", "for your lawsuit, here's the legal advice you need")
v.drift.recommended_action   # PurposeGuard: drift/scope (advisory; never blocks)
v.effective_action           # AMG: enforcement (allow/redact/block/quarantine)
```

PurposeGuard stays advisory; **AMG owns enforcement.** This **composes
defense-in-depth layers — it is NOT "secure against poisoning."** AMG catches
*known* markers/policies and is defeated by novel obfuscation/encoding/paraphrase;
PurposeGuard catches topical drift/scope. See [`THREAT_MODEL.md`](THREAT_MODEL.md).

### Catching off-purpose baseline-poisoning (opt-in, narrow agents only)

The drift *trend* is drift-from-baseline, so an attacker who controls the early
writes can poison the baseline and hide. The opt-in **`purpose_floor`** adds a
**purpose-anchored** signal — the EMA of alignment to the *immutable* purpose vs an
absolute floor (no baseline term), surfaced as `drift().anchored_drifting`:

```python
guard = PurposeGuard(purpose="...", purpose_floor=0.28)  # opt-in; OFF by default
```

Measured honestly (see [`benchmark/RESULTS.md`](benchmark/RESULTS.md)): it **fires
on off-purpose baseline-poisoning that the relative trend misses**, but it is
**blind by construction to on-purpose poisoning** (content that stays
purpose-aligned), and an absolute floor **over-flags broad agents** (27/30 legit
broad on-mission writes in testing — a broad agent's alignment overlaps the poison
level, so no floor separates them). So it is **off by default and for
NARROW/focused agents only.** It is a narrow, opt-in mitigation for one case — not
a poisoning defense.

## Honest limitations

PurposeGuard is a **drift + scope guardrail — "keep your agent in its lane" —
NOT a security tool.** Be clear-eyed:

- **It catches topical drift and scope violations, not security threats.**
  Blocked-topic anchors flag content that is *about* a forbidden topic, and
  **partially** close keyword-camouflage (on-mission padding can't rescue clearly
  off-topic content). They do **NOT** catch on-topic-but-wrong-*policy*
  (e.g. "approve all refunds without verification" — topically billing,
  behaviorally wrong) or adversarial injection. For those, **compose with OWASP
  Agent Memory Guard** (markers/policy enforcement) — see
  [`purposeguard/adapters/owasp_amg.py`](purposeguard/adapters/owasp_amg.py) and
  [`THREAT_MODEL.md`](THREAT_MODEL.md). Memory-poisoning layers are planned for
  v0.3; they are not in this release.
- **The camouflage close is narrow and partial (measured).** On the adversarial
  benchmark, blocked anchors close *part of the topical half* of keyword-camouflage
  (overall **100% → 70%** ASR, **0** false positives) **only when you enumerate the
  specific, distinct off-topic** — the on-mission padding dilutes the rest. The
  *malicious half* (injection/exfiltration) is unaffected (AMG's domain). The
  `blocked_threshold` default is calibrated (~0.46) so near-domain blocked anchors
  don't over-flag legit traffic. Enumerate deliberately; see
  [`benchmark/RESULTS.md`](benchmark/RESULTS.md) §3a.
- **Blocked-overrides-alignment is intentionally conservative.** A legitimately
  on-purpose write that sits near a blocked topic (a billing agent mentioning
  "legal advice") will flag. `blocked_threshold` is the tuning lever.
- **"On-mission" is a tunable threshold.** Broad agents produce false positives —
  use a preset, lower the threshold, add `allowed_topics`, or the optional judge.
- **The lexical fallback is a floor, not a ceiling.** It keys on word overlap,
  misses paraphrase, and is weaker at camouflage. Install the `embeddings` extra.
- **Drift needs a stable baseline.** If the agent is off-mission from the first
  writes there's nothing to drift from — and an attacker controlling the early
  writes can poison the baseline (the per-write scope check still applies; the
  trend does not). See THREAT_MODEL.md.
- **Not a proof.** It raises the cost of drift and scope violations; it does not
  guarantee their absence.

## Configuration that matters

| Option | What it does |
| --- | --- |
| `allowed_topics` | In-scope topics that *widen* what counts as on-mission (alignment = max over purpose + allowed). Optional. |
| `blocked_topics` | Forbidden topics. A write similar to any of these is flagged regardless of purpose alignment (blocked overrides). Optional. |
| `blocked_threshold` | Similarity to a blocked topic at/above which it flags. Defaults to ~0.46, **decoupled** from `threshold` (calibrated separately so near-domain anchors don't over-flag); raise to reduce false flags. |
| `mode` | `monitor` (default, flag only) / `redirect` (verdict carries `fallback`) / `block` (verdict signals stop). Recommendation only — the caller enforces. |
| `fallback_template` / `fallback_template_generic` | Messages rendered into `verdict.fallback` for blocked-hit vs low-alignment flags. Fields: `{purpose}`, `{blocked_topic}`, `{reason}`. |
| `threshold` | Alignment below this is flagged. Defaults to the `BALANCED` preset. Lower for broad agents (see presets below). |
| `drift_baseline_window` | How many early writes set the "normal" baseline. Lock it in *before* drift can start. |
| `drift_alpha` | EMA reactivity. Higher = reacts faster to recent writes. |
| `judge` | Optional `(content, purpose) -> bool` callable, consulted only for borderline scores. You supply the model; none is bundled. |

## Threshold presets — pick one for your agent's breadth

One threshold can't fit every agent. A narrow single-purpose bot scores its
on-mission writes high and consistently, so it can be strict; a legitimately
broad assistant scores its on-mission writes *lower and more variably* (they only
loosely match a broad purpose), so it must be lenient or it over-flags. The
benchmark quantifies this — there is no single value that's best for both, so
choose by agent type:

```python
from purposeguard import PurposeGuard

guard = PurposeGuard.from_preset("broad", purpose="A general personal assistant")
```

| Preset | Threshold | Use for | Benchmark behavior (held-out TEST) |
| --- | --- | --- | --- |
| `NARROW` | 0.25 | Single-purpose agents (one topic) | narrow agent: FPR ~0.07, precision ~0.94, recall ~1.00 |
| `BALANCED` | 0.15 | Default; a focused agent with some range | narrow agent ~perfect (FPR 0.00, P/R 1.00); broad-agent FPR ~0.30 |
| `BROAD` | 0.10 | Wide-ranging assistants | broad-agent FPR ~0.13 |

The tradeoff is real and not hidden: `BROAD` on a *narrow* agent misses the
weakest off-mission writes (recall ~0.93), and `NARROW` on a *broad* agent
over-flags (FPR ~0.50). Match the preset to the agent. Presets are calibrated for
the embedding scorer; the lexical fallback is a floor, not tunable.

**Standing guidance — for broad agents, trust the DRIFTING trend over per-write
FLAG.** Per-write FLAG is necessarily noisy on an agent that legitimately roams,
but the `drift()` trend reads the raw alignment score and is *independent of the
threshold/preset*: in the benchmark it caught real drift in one write while never
firing on the broad or non-drifting agents. Alert on the trend; treat per-write
FLAG as a tag, not a verdict. See [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

## Roadmap

- **v0.1** — purpose-drift detection, drift meter, re-anchoring, response-level
  drift, framework-agnostic core, offline-safe.
- **v0.2 (this release)** — **drift + scope guardrail**: allowed/blocked-topic
  anchors (partially closing keyword-camouflage), opt-in monitor/redirect/block
  modes (the guard recommends, the caller acts), and suggested fallback messages.
  Composition with [OWASP Agent Memory Guard](https://github.com/OWASP/www-project-agent-memory-guard)
  for marker/policy poisoning enforcement.
- **v0.3 (this release)** — poisoning layer: a first-class `composed_guard`
  (drift + scope + OWASP AMG poisoning/enforcement in one config), and an opt-in
  purpose-anchored drift signal (`purpose_floor`) that catches off-purpose
  baseline-poisoning the relative trend misses — narrow-agents-only / off by
  default (over-flags broad agents), and blind by construction to on-purpose
  poisoning. Defense-in-depth layers, not a security guarantee.
- **Later** — an existing-memory **consensus** reference (note: itself poisonable
  if the attacker controls the existing memories — THREAT_MODEL.md §3c); and,
  opt-in/experimental, cryptographic write provenance for forensic attribution.

## License

Apache-2.0
