# PurposeGuard — Task Queue for Claude Code

Work top to bottom. Each task is independently shippable, has acceptance
criteria, and a ready-to-paste prompt for Claude Code. Check items off as you go.

---

## Task 1 — Adapters (v0.1.x)

Make PurposeGuard trivial to drop into the frameworks people actually use, while
keeping the core framework-free.

**Build:**
- [ ] `adapters/__init__.py`
- [ ] `adapters/raw.py` — a `guard_store(store, guard)` wrapper for any object
      with `add(text)`/`search(query)`. The reference "bring-your-own-store" path.
- [ ] `adapters/mem0.py` — wrap Mem0's `add()` so each write is `check()`ed; on
      FLAG, tag the memory's metadata with the drift score (do NOT drop it — v0.1
      is detection-first). Import mem0 lazily inside the adapter only.
- [ ] `adapters/langchain.py` — a `GuardedChatMessageHistory`-style wrapper that
      scores messages as they're added. Mirror OWASP AMG's integration shape so
      the two compose naturally.

**Acceptance:**
- Core (`purposeguard/`) still imports zero frameworks (grep to confirm).
- Each adapter works if its framework is installed, and raises a clear
  ImportError-with-install-hint if not.
- A test per adapter using a fake/mock store (no real framework dep in CI).

**Claude Code prompt:**
> Read CLAUDE.md. Build the adapters in Task 1 of TASKS.md. Start with
> `adapters/raw.py` (the framework-agnostic reference), get it tested with a mock
> store, then do mem0 and langchain with lazy imports. Keep the core
> framework-free. Detection-first: adapters tag flagged writes, never drop them.

---

## Task 2 — Benchmark (v0.1.x)

Prove it works with numbers. This doubles as the eval section for any writeup.

**Build:**
- [ ] `benchmark/traces.py` — generate synthetic agent traces: N on-mission
      writes, then a controlled drift into off-topic content, with ground-truth
      labels for where drift begins.
- [ ] `benchmark/run.py` — run PurposeGuard over traces; report **detection
      lead-time** (how many writes after true drift onset until DRIFTING fires),
      **false-positive rate** on a deliberately-broad-but-on-mission agent, and a
      precision/recall on per-write FLAG vs ground truth.
- [ ] `benchmark/RESULTS.md` — committed results table + how to reproduce.

**Acceptance:**
- `python benchmark/run.py` produces a table.
- Includes at least: a narrow agent (should detect fast), a broad agent (should
  not over-flag), and a non-drifting agent (FPR should be low).
- Run with both LexicalScorer and (if available) EmbeddingScorer; report both.

**Claude Code prompt:**
> Read CLAUDE.md and Task 2 of TASKS.md. Build the benchmark. I want detection
> lead-time, false-positive rate, and per-write precision/recall against
> ground-truth drift onset. Test with narrow, broad, and non-drifting agents.
> Commit a RESULTS.md with a reproducible table.

---

## Task 3 — Embedding accuracy pass + threshold presets (v0.1)

The lexical fallback is a floor; embeddings are the real default. Tune for it.

**Build:**
- [ ] Validate thresholds using real embeddings (needs network to fetch
      all-MiniLM-L6-v2 once, or a local model path).
- [ ] `purposeguard/presets.py` — named threshold presets: e.g. `NARROW` (0.6),
      `BALANCED` (0.5), `BROAD` (0.4). Document what each means.
- [ ] Wire presets into `PurposeGuard` (e.g. `PurposeGuard.from_preset("broad", purpose=...)`).

**Acceptance:**
- Demo and benchmark run on embeddings, not just lexical.
- Presets documented in README with guidance on which to pick.

**Claude Code prompt:**
> Read CLAUDE.md and Task 3. With sentence-transformers installed, validate the
> alignment thresholds on the benchmark traces and add named presets
> (NARROW/BALANCED/BROAD). Add a from_preset constructor. Update the README.

---

## Task 4 — v0.2 Poisoning detector (the second capability)

Same core, new reference: score a write against existing-memory **consensus**
instead of the purpose. Catches adversarial, semantically-clean writes that
marker-based tools (OWASP AMG) miss.

**Build:**
- [ ] `purposeguard/references/__init__.py`
- [ ] `purposeguard/references/consensus.py` — given the current memory set,
      score a candidate write by how well it agrees with the existing trusted
      cluster (e.g. mean similarity to k-nearest existing memories; flag
      outliers / contradictions).
- [ ] Extend `PurposeGuard` (or add a sibling `MemoryGuard`) that accepts a
      consensus reference instead of a purpose. Reuse scorer + policy + meter.
- [ ] `adapters/owasp_amg.py` — compose: run OWASP Agent Memory Guard's marker
      detectors AND our semantic consensus check; combine verdicts.
- [ ] Benchmark extension: add semantic-poisoning payloads (MINJA/PoisonedRAG
      style — clean-looking but contradictory) and show markers miss them while
      consensus catches them.

**Acceptance:**
- The core scorer/policy/meter are reused unchanged (diff should show no rewrite).
- Benchmark demonstrates the gap: marker-only detection rate vs. marker+consensus.
- Composes cleanly with OWASP AMG; document the combined pipeline.

**Claude Code prompt:**
> Read CLAUDE.md and Task 4. Add the v0.2 poisoning detector as a new reference
> (consensus.py) reusing the existing scorer/policy/meter — do NOT rewrite the
> core. Add an OWASP Agent Memory Guard composition adapter. Extend the benchmark
> with semantic-poisoning payloads and show marker detection misses them while
> consensus catches them.

---

## Task 5 — Experimental provenance (opt-in, research artifact)

Tamper-evident write log + trusted timestamps for forensic attribution. This is
the paper's distinctive piece. Keep it OFF the adoptable path.

**Build:**
- [ ] `experimental/__init__.py` (clearly labeled experimental)
- [ ] `experimental/provenance.py` — append-only hash-chained log of accepted
      writes (Merkle/CT-style), per-writer signatures, optional RFC-3161
      timestamp anchoring. Async/batched, off the write hot path.
- [ ] Verification tool: given a stored memory, prove inclusion + timestamp +
      writer signature.

**Acceptance:**
- Importing the base package does NOT require any of this.
- README/BUILD_PLAN clearly frame it as forensic attribution + tamper-evidence,
  NOT prevention. It attests "the guard saw and signed this entry," not "this
  entry is safe."

**Claude Code prompt:**
> Read CLAUDE.md and Task 5. Build the experimental provenance module as an
> opt-in, append-only hash-chained log with per-writer signatures and optional
> RFC-3161 timestamping, async/off-hot-path. Add an inclusion-proof verifier.
> Keep it entirely optional and frame it as forensics, not prevention.

---

## Setup checklist before first publish

- [ ] Decide final package name (claim it on PyPI if not "purposeguard").
- [ ] Set the real GitHub org/repo URL in `pyproject.toml`.
- [ ] Add a GitHub Actions CI running `pytest` on the dependency-light install.
- [ ] Pin a version and tag v0.1.0 once Tasks 1-2 land.
