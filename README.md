# PurposeGuard

**Detect when an AI agent forgets its purpose.**

Agents that run for a while drift. Session history piles up, memories get
consolidated, and the agent slowly starts answering off-mission — a billing
assistant that wanders into recipes and weather. PurposeGuard scores every
memory write against the agent's *declared purpose* and surfaces drift as a
single watchable number, before your users notice the agent has lost the plot.

- **Framework-agnostic.** Works with Mem0, LangChain, a raw vector store, or a
  plain script. A "write" is just text — feed it from anywhere.
- **Detection-first, safe by default.** v0.1 only ever *flags*; it never blocks,
  mutates, or drops your data. You can't break your agent by adding it.
- **Easy to adopt.** `pip install purposeguard`, one line to construct, one line
  to check. Runs offline with zero model downloads (a local-embedding upgrade is
  one extra install away).

First, `pip install "purposeguard[embeddings]"` (the lexical-only base install is a
rough fallback — see [Install](#install)).

```python
from purposeguard import PurposeGuard

# Uses local embeddings (from the [embeddings] extra). In production, pass
# require_embeddings=True to fail loudly instead of silently dropping to the
# rough lexical floor if the model isn't available:
guard = PurposeGuard(
    purpose="A customer-support agent for billing questions",
    require_embeddings=True,
)

verdict = guard.check("How do I bake sourdough bread?")
print(verdict)            # Verdict(flag, score=0.08, reason='alignment 0.08 below threshold 0.15')
print(guard.drift())      # DriftReading(current=..., baseline=..., drift=..., n=...)
```

## Install

```bash
pip install "purposeguard[embeddings]"   # recommended for real use: local-embedding scoring (all-MiniLM-L6-v2)
pip install purposeguard                  # minimal: zero deps, but only the rough lexical floor
```

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
of what it was asked. Like everything in v0.1, `check_response` only ever flags —
it never blocks.

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

## Honest limitations

PurposeGuard is a **heuristic reliability guardrail, not a proof and not a
security boundary.**

- "On-mission" is a tunable threshold. Legitimately broad agents will produce
  false positives — lower the threshold, or use the optional LLM judge for
  borderline cases.
- The lexical fallback scorer is a *floor*, not a ceiling: it keys on word
  overlap and misses paraphrase. Install the `embeddings` extra for real use.
- Drift detection depends on a stable baseline. If your agent is off-mission
  from the very first writes, there's no good baseline to drift *from* — set the
  purpose and baseline window deliberately.

## Configuration that matters

| Option | What it does |
| --- | --- |
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

| Preset | Threshold | Use for | Benchmark behavior |
| --- | --- | --- | --- |
| `NARROW` | 0.20 | Single-purpose agents (one topic) | narrow agent: FPR ~0.00, precision/recall ~1.00 |
| `BALANCED` | 0.15 | Default; a focused agent with some range | narrow ~perfect; broad FPR moderate (~0.23) |
| `BROAD` | 0.10 | Wide-ranging assistants | broad-agent FPR drops sharply (~0.30 → ~0.07) |

The tradeoff is real and not hidden: `BROAD` on a *narrow* agent misses the
weakest off-mission writes (recall ~0.80), and `NARROW` on a *broad* agent
over-flags (FPR ~0.30). Match the preset to the agent. Presets are calibrated for
the embedding scorer; the lexical fallback is a floor, not tunable.

**Standing guidance — for broad agents, trust the DRIFTING trend over per-write
FLAG.** Per-write FLAG is necessarily noisy on an agent that legitimately roams,
but the `drift()` trend reads the raw alignment score and is *independent of the
threshold/preset*: in the benchmark it caught real drift in one write while never
firing on the broad or non-drifting agents. Alert on the trend; treat per-write
FLAG as a tag, not a verdict. See [`benchmark/RESULTS.md`](benchmark/RESULTS.md).

## Roadmap

- **v0.1 (this release)** — purpose-drift detection, drift meter, re-anchoring,
  framework-agnostic core, offline-safe.
- **v0.2** — a poisoning detector on the same core: swap the reference from
  "purpose" to "existing-memory consensus" to catch adversarial,
  semantically-clean-looking writes that marker-based tools miss. Composes with
  [OWASP Agent Memory Guard](https://github.com/OWASP/www-project-agent-memory-guard).
- **Later (opt-in, experimental)** — cryptographic write provenance
  (tamper-evident log + trusted timestamps) for forensic attribution. Clearly
  labeled `experimental/`; never required to use the guard.

## License

Apache-2.0
