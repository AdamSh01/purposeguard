# PurposeGuard — Project Memory (read this first)

This file is auto-loaded by Claude Code. It carries every decision already made
so you don't re-litigate them. Read it fully before writing code.

## What this is

A **framework-agnostic, pip-installable drift + scope guardrail for
customer-facing AI agents.** It keeps an agent in its lane: you declare a
**purpose** (and optionally **allowed** / **blocked** topics), and PurposeGuard
scores every write and response against them — surfacing drift as a watchable
number and, when you opt in, recommending the caller redirect or block off-scope
output.

Lineage: v0.1 was a detection-first drift guard (purpose only, flag-only). v0.2
is the **drift + scope** stance — allowed/blocked-topic anchors and opt-in
monitor/redirect/block modes. Extend along the roadmap; don't redesign what's
settled.

## The one idea (do not violate)

Score every write/response against fixed **reference anchors**, then act on the score.
- v0.1 (drift, DONE): reference = the agent's declared **purpose**.
- v0.2 (scope, THIS): references = purpose + **allowed/blocked-topic anchors**;
  plus opt-in monitor/redirect/block modes.
- v0.3 (poisoning, LATER — NOT now): reference = **existing-memory consensus**,
  composed with OWASP AMG.

Same core, scorer, policy, and meter — v0.2 adds reference *anchors*, not a
parallel scoring route. If you find yourself forking the core, stop — you're
doing it wrong.

## Non-negotiable guardrails

1. **Safe-by-default, explicit-to-enforce.** v0.1 was detection-first (never
   emitted `Decision.BLOCK`). v0.2 **deliberately adds enforcement** — keeping an
   agent in its lane inherently requires intervening — but under hard constraints
   that preserve the original spirit. This is a reasoned reversal, documented
   here, NOT a silent drift:
   - **`monitor` mode is the DEFAULT** and behaves exactly like v0.1: flag only,
     never BLOCK. Upgrading changes no existing agent's behavior. The
     `test_never_blocks_in_v01` test still holds for the default mode — keep it;
     it now guards "the default never blocks."
   - **`redirect` and `block` are OPT-IN** (`mode=...`).
   - **Never silently mutate or drop data, in ANY mode.** The guard returns a
     STRUCTURED verdict carrying a *recommended* action; the CALLER acts.
     PurposeGuard never reaches into the agent to delete or edit. `block` mode
     signals "stop the off-scope response" — it does not perform the stop.
2. **Base install stays dependency-light.** The package must `pip install` and
   run with ZERO required dependencies (lexical fallback). Heavy ML deps
   (sentence-transformers, numpy) live ONLY in the `[embeddings]` optional extra.
3. **The core imports no framework.** `purposeguard/` must never import mem0,
   langchain, etc. Framework code goes in `adapters/` and is optional.
4. **Graceful degradation.** Anything that can fail to load (models, optional
   deps) must warn once and fall back, never crash. See `EmbeddingScorer`.
5. **Honesty in docs.** This is a **drift + scope guardrail — "keep your agent in
   its lane"** — NOT a security tool. Frame it that way, never "secure your agent."
   Blocked-topic scoring catches topical drift into forbidden areas and PARTIALLY
   catches keyword-camouflage; it does NOT catch on-topic-but-wrong-policy
   (business rules are deliberately out of scope) or adversarial injection.
   `block` mode existing does NOT make this a security boundary — enforcing scope
   is not enforcing security. Memory-poisoning defense is v0.3 (markers/rules +
   OWASP AMG); until then public framing is "composes with AMG for security,"
   never "secure against poisoning."
6. **Ship one thing at a time.** Each roadmap step is independently useful and
   shippable. No half-built multi-feature commits.

## Architecture (built, stable)

```
purposeguard/
  types.py     # Write, Verdict, Decision — tiny framework-agnostic shapes
  scoring.py   # EmbeddingScorer (local all-MiniLM) + LexicalScorer fallback
  drift.py     # DriftMeter: EMA + baseline -> DriftReading
  guard.py     # PurposeGuard: scorer + threshold policy + meter + reanchor()
  __init__.py  # public API
```

Public API: `PurposeGuard(purpose, *, threshold, scorer, judge, ...)`, then
`.check(str | Write) -> Verdict`, `.drift() -> DriftReading`, `.reanchor(ctx)`.

## Conventions

- Python >=3.9, `from __future__ import annotations` at the top of every module.
- Dataclasses for data shapes. `Protocol` for pluggable interfaces (see `Scorer`).
- Docstrings explain WHY, not just what — match the existing style.
- Tests use `LexicalScorer` so they run offline/hermetic. Keep CI model-free.
- Run tests: `python -m pytest tests/ -v`. Run demo: `python examples/drift_demo.py`.

## Roadmap — your task queue (in order)

See `TASKS.md` for the detailed, checkable breakdown. Summary:

1. **Adapters** (`adapters/raw.py`, `adapters/mem0.py`, `adapters/langchain.py`) —
   thin translators framework-write -> `Write`, apply verdict. Core untouched.
2. **Benchmark** (`benchmark/`) — synthetic drift traces; report detection
   lead-time + false-positive rate. Doubles as paper eval.
3. **Embedding accuracy pass** — validate thresholds with real embeddings; ship
   per-use-case threshold presets.
4. **v0.2 poisoning detector** — `references/consensus.py`: score a write against
   the cluster of existing trusted memories. Reuse scorer+policy+meter unchanged.
   Compose with OWASP Agent Memory Guard (it does marker detection; we do the
   semantic gap it misses).
5. **Experimental provenance** (`experimental/provenance.py`, opt-in) — tamper-
   evident write log + trusted timestamps for forensic attribution. The research-
   paper artifact. Kept OFF the adoptable path on purpose.

## Context you may want

- This composes with **OWASP Agent Memory Guard**
  (github.com/OWASP/www-project-agent-memory-guard) — a Python pip package that
  does marker-based write screening (prompt-injection markers, secrets, protected
  keys, size anomalies). It does NOT do semantic drift or semantic poisoning
  detection. That gap is our reason to exist. For v0.2, compose with it: run its
  markers AND our semantic check.
- The poisoning research (attacks like MINJA, PoisonedRAG, MemPoison) targets
  semantically-clean-looking writes that markers miss — that's what v0.2's
  consensus reference is designed to catch.
- Set before publishing: package name (if not "purposeguard"), the GitHub org URL
  in `pyproject.toml`, and threshold presets per use case.
