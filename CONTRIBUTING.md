# Contributing to PurposeGuard

Thanks for helping! PurposeGuard is a small, deliberately-scoped library. Before
opening a PR, please read the three invariants below — they are the load-bearing
design decisions, and a change that violates one will be sent back.

## The three invariants (do not break)

1. **Zero required dependencies in the base install.** `pip install purposeguard`
   must work with the standard library alone (the lexical fallback scorer). Heavy
   ML deps live ONLY in optional extras (`[embeddings]`, `[fast]`). Never add a
   top-level `dependencies` entry.
2. **The core imports no framework.** `purposeguard/` must never import mem0,
   langchain, the MCP SDK, agent-memory-guard, etc. Framework code goes in
   `purposeguard/adapters/` and is lazy-imported, raising a clear
   `ImportError`-with-install-hint when the framework is absent.
3. **Detection-first: never silently mutate or drop caller data.** The guard
   returns a *recommended* action; the caller enforces. `monitor` is the default
   and only ever flags. Adapters tag flagged writes — they never drop them.

Two more that keep the project honest:

- **Graceful degradation, never a crash.** Anything that can fail to load (a model,
  an optional dep) must warn once and fall back. Every emitted string
  (`__str__`/logs/CLI) must be ASCII-only so it survives a Windows cp1252 console.
- **Honesty in docs.** This is a *reliability/observability* guardrail, not a
  security boundary. Don't add claims the benchmark doesn't support; if you change
  behavior that the README or `RESULTS.md` describes, update them in the same PR.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -e ".[dev]"            # tests run on the lexical fallback, no model needed
python -m pytest tests/ -q         # should be all green
```

Optional extras:

```bash
pip install -e ".[dev,embeddings]"   # exercise the real embedding scorer + benchmark
pip install -e ".[lint]"             # ruff + mypy
```

## Before you open a PR

- `python -m pytest tests/ -q` is green (tests must stay **hermetic** — use
  `LexicalScorer`, no network/model downloads; gate any embedding test behind
  `pytest.importorskip("sentence_transformers")`).
- `ruff check purposeguard tests benchmark examples` is clean.
- `mypy purposeguard` has no new errors.
- New behavior has a test that closes the **class** of issue, not just one instance
  (and, for a detector gap, an honest test documenting what it still misses).
- Docs updated if you changed user-facing behavior or any reported number.

## What's in / out of scope

In scope: scorer backends, adapters/integrations, observability, benchmark rigor,
docs. Currently **deferred** (see `THREAT_MODEL.md` and the build notes): a
memory-*consensus* reference (poisonable by construction) and cryptographic
provenance. Please open an issue to discuss before building either.
