# PurposeGuard — Build Plan

This is the working plan. v0.1 (the drift guard) is **built and runnable** in
this repo. The sections below record the architecture, what's done, and the
sequenced path forward so each step ships standalone — no half-built pile.

## The one idea

Score every memory write against a fixed **reference**, then act on the score.

- **v0.1 (drift):** reference = the agent's declared **purpose**.
- **v0.2 (poisoning):** reference = **existing-memory consensus**.

Same core, same scorer, same policy engine, same drift/anomaly machinery — only
the reference changes. That's what makes "both" reachable without a rewrite.

## Architecture (built)

```
            +------------------------------------------+
  write --> |  PurposeGuard.check(write) -> Verdict     |
  (str or   |                                           |
   Write)   |   Scorer ---> Policy(allow/flag) ---> Verdict
            |     |                                     |
            |     +--> DriftMeter (rolling trend)        |
            +------------------------------------------+
```

- `types.py` — `Write`, `Verdict`, `Decision`. The tiny framework-agnostic shapes.
- `scoring.py` — `EmbeddingScorer` (local all-MiniLM, preferred) with graceful
  fallback to `LexicalScorer` (zero-dependency floor). `default_scorer()` picks.
- `drift.py` — `DriftMeter`: EMA of alignment + early baseline → `DriftReading`.
- `guard.py` — `PurposeGuard`: ties scorer + threshold policy + meter + reanchor.
- `__init__.py` — clean public API.

## Done in v0.1

- [x] Framework-agnostic core (`check(str | Write) -> Verdict`)
- [x] Local-embedding scorer + offline-safe lexical fallback (warns, never crashes)
- [x] Drift meter (baseline + EMA + `DRIFTING` flag)
- [x] Purpose re-anchoring (front+back, beats U-shaped attention curve)
- [x] Immutable purpose (can't drift via purpose edits)
- [x] Optional injectable LLM judge for borderline cases only (cost-controlled)
- [x] Detection-first guarantee (never emits BLOCK in v0.1)
- [x] 10 passing tests, runnable demo, pip-installable, Apache-2.0

## Next steps (sequenced, each shippable)

1. **Adapters (thin, optional).** `adapters/mem0.py`, `adapters/langchain.py`,
   `adapters/raw.py`. Each only translates that framework's write into `Write`
   and applies the verdict. The core never imports a framework. → still v0.1.x.
2. **Benchmark.** `benchmark/` — synthetic drift traces (on-mission → wandering),
   measure detection lead-time and false-positive rate on deliberately-broad
   agents. Doubles as the eval section for any writeup. → v0.1.x.
3. **Embedding accuracy pass.** Validate thresholds with real embeddings (not the
   lexical floor) and ship sensible per-use-case threshold presets. → v0.1.
4. **v0.2 — poisoning detector.** New `references/consensus.py`: score a write
   against the cluster of existing trusted memories instead of the purpose.
   Reuse scorer + policy + meter unchanged. Compose with OWASP Agent Memory
   Guard (markers) so PurposeGuard covers the semantic gap markers miss.
5. **Later, opt-in, experimental.** `experimental/provenance.py`: tamper-evident
   write log + trusted timestamps for forensic attribution. Clearly labeled,
   never required to use the guard. (This is the research-paper artifact, kept
   off the adoptable path on purpose.)

## Near-term task (post-v0.1.1)

- **Evaluate a torch-free embedding backend** (model2vec / static embeddings, or
  an ONNX MiniLM) to make good semantic scoring a *light default* without the
  torch dependency. This is the root-cause fix for the lexical-floor first
  impression (benchmark FPR 0.70-0.90; the examples need a visible fallback
  notice). If it lands, embeddings could become the default without the zero-dep
  tax. Revisit guardrail #2 (zero required deps) ONLY if a light backend proves
  infeasible -- do not change guardrail #2 until then.

## Guardrails (do not violate)

- **Detection-first stays the default.** Enforcement (BLOCK) is always opt-in.
- **Base install stays dependency-light.** Heavy ML deps are an extra, never required.
- **Honesty in the README.** It's a heuristic reliability guardrail, not a proof
  or a security boundary. Say so.
- **Ship one thing at a time.** Each numbered step is independently useful.
