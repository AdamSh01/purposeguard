# Security Policy

## What PurposeGuard is — and is not

**PurposeGuard is a reliability / observability guardrail, not a security boundary.**
It measures how well an agent's memory writes and responses stay aligned to a
declared purpose, and surfaces drift as a watchable signal. It does **not** stop a
determined adversary.

Please read [`THREAT_MODEL.md`](THREAT_MODEL.md) before relying on it. In particular,
the following are **known, documented limitations**, not vulnerabilities to report:

- Semantic camouflage / keyword-stuffing in an unlisted topic (high cosine to purpose).
- On-topic-but-wrong-**policy** content (e.g. "approve all refunds without verification").
- Baseline poisoning of the *relative* drift trend.
- Paraphrase beyond the scorer's reach, language-switch, and payload split across
  multiple writes.
- The lexical fallback scorer being weak (it is a floor; install an embedding extra).

These are measured in [`benchmark/RESULTS.md`](benchmark/RESULTS.md) and
`benchmark/adversarial.py`. For poisoning/marker enforcement, compose with
[OWASP Agent Memory Guard](https://github.com/OWASP/www-project-agent-memory-guard).

## What IS a security issue worth reporting

A genuine defect in the library's own safety promises, for example:

- A code path that **crashes** instead of degrading (violating graceful-degradation),
  or that **mutates/drops caller data** despite detection-first.
- A way to make the guard **mutate or delete** memory it was only supposed to observe.
- A supply-chain / dependency issue in the package itself.
- A normalization bug (e.g. a new obfuscation) that defeats an **enumerated**
  blocked topic the docs claim is covered.

## Reporting

Until a dedicated security contact is published, please report privately via a
GitHub **Security Advisory** ("Report a vulnerability") on the repository, or open a
minimal issue **without** exploit details and ask a maintainer for a private channel.
Please include a reproduction and the installed versions. We aim to acknowledge
within a few days.

## Supported versions

This is 0.x / alpha software; only the latest release receives fixes.
