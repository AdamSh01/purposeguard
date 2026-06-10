# Changelog

All notable changes to PurposeGuard are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (currently 0.x,
so minor versions may still change behavior).

## [Unreleased]

_Nothing yet._

## [0.4.0] - 2026-06-10

The "Trust & Correctness" + "Credibility & Integration" pass.

### Added
- **Input normalization** before scoring (NFKC, zero-width/`Cf` stripping, and a
  curated Cyrillic/Greek homoglyph fold), shared by both scorers — closes the
  homoglyph and zero-width evasions of `blocked_topics`. A floor, not a full
  confusables defense; see `THREAT_MODEL.md` §3(e).
- **Numeric range validation** at construction for `threshold`, `blocked_threshold`,
  `drift_alpha`, `drift_baseline_window`, `judge_band`, and `purpose_floor` — absurd
  values now raise `ValueError` instead of silently producing nonsense.
- **Embed-once fast path**: optional `embed()` / `similarity()` / `score_many()` on
  the `Scorer` protocol; a check now embeds the content once instead of once per
  anchor. Bounded LRU reference cache on `EmbeddingScorer`.
- **Async + batch**: `PurposeGuard.acheck`, `acheck_response`, and `check_many`.
- **Observability**: `Verdict.to_dict()` / `DriftReading.to_dict()` (JSON-friendly);
  an `on_verdict` hook; an optional `context` field on `Write`/`Verdict`
  (session/user/run ids); and a composite `PurposeGuard.health()`.
- **Persistence**: `DriftMeter.state()/from_state()` and `PurposeGuard.state()/restore()`
  so the long-horizon drift trend survives a process restart.
- **MCP integration**: `purposeguard.adapters.mcp` exposes `check`/`check_response`
  as MCP tools (lazy-imported), plus `examples/mcp_server_example.py`.
- **Benchmark machinery**: a deterministic trace generator (`benchmark/generate.py`),
  hermetic baselines (always-allow, keyword-blocklist), a latency harness
  (`benchmark/latency.py`), and machine-readable `run.py --json` / `--assert`.
- New tests: `test_evasion.py`, `test_readme_presets.py`, `test_perf_async.py`,
  `test_observability.py`, `test_benchmark.py`, `test_adapters_mcp.py`.
- Tooling: `ruff` + `mypy` config and a CI lint/type job; coverage in the test job.
  `CONTRIBUTING.md`, `SECURITY.md`, and this changelog.

### Changed
- Repositioned as **purpose/mission observability & reliability**, not a security
  tool. README now leads with a "what it catches / what it misses" table and a
  "why agents drift" section.
- Fixed doc/code contradictions: README `NARROW` preset 0.20 → **0.25** with held-out
  numbers; `blocked_threshold` documented as ~0.46 (decoupled); stale `pyproject` /
  `__init__` descriptions updated to the drift+scope framing.

### Fixed
- Degenerate `blocked_threshold` no longer yields `"matched blocked topic None"`
  with an empty fallback (anchor selection always names a topic).

### Notes
- Publishing to PyPI and moving to a neutral org are still pending; until then,
  install from source (`git+https://...`).
- The light, torch-free default scorer (fastembed/ONNX) is deferred to a focused
  follow-up because it requires re-calibrating the rescale band + presets (currently
  tuned for all-MiniLM).

## [0.3.0]
- Composed **drift + scope + OWASP AMG** guard in one config (`composed_guard`).
- Opt-in **purpose-anchored** drift signal (`purpose_floor`) catching off-purpose
  baseline poisoning the relative trend misses (narrow agents only; off by default).

## [0.2.1]
- Fixed `blocked_threshold` over-flagging (decoupled from the alignment threshold,
  TRAIN-calibrated default ~0.46); measured keyword-camouflage numbers.

## [0.2.0]
- **Drift + scope guardrail**: `allowed_topics` / `blocked_topics`, opt-in
  `monitor`/`redirect`/`block` modes, and suggested fallback messages.

## [0.1.3]
- OWASP Agent Memory Guard composition adapter; `THREAT_MODEL.md`.

## [0.1.2]
- De-overfit benchmark (TRAIN/TEST split in different domains); adversarial
  limitations documented; version + CI fixes.

## [0.1.0] – [0.1.1]
- Initial drift guard: purpose-anchored scoring, EMA drift meter, re-anchoring,
  response-level drift, framework-agnostic core, offline-safe lexical fallback.
