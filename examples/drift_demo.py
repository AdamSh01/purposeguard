"""Demo: watch an agent drift off-purpose, and watch the meter catch it.

Run:  python examples/drift_demo.py

Simulates a billing-support agent whose stored memories start on-mission and
gradually wander into unrelated chit-chat — the exact failure described by
users: "after a while the agent forgets its purpose and answers irrelevant
questions." The drift meter should stay low at first, then climb as the off-
purpose writes accumulate.

Uses the lexical scorer so it runs with zero downloads anywhere. In a real
deployment you'd let the default embedding scorer load for much better accuracy.
"""

from purposeguard import LexicalScorer, PurposeGuard

PURPOSE = (
    "A customer-support agent that answers billing payment subscription "
    "invoice refund account questions"
)

# A session-by-session stream: on-mission early, drifting later.
STREAM = [
    # --- on mission ---
    "User asked how to update their payment card on the subscription",
    "Helped user dispute a duplicate invoice charge and issue a refund",
    "Explained the billing cycle and when the next payment is due",
    "Walked user through downgrading their subscription plan",
    "Resolved a failed payment and updated the account billing address",
    # --- starting to wander ---
    "User asked about the weather forecast for their vacation",
    "Chatted about a good recipe for sourdough bread",
    "Recommended some movies to watch this weekend",
    "Discussed the user's favorite football team and last night's game",
    "Gave tips on training a new puppy not to chew furniture",
]


def main() -> None:
    guard = PurposeGuard(
        purpose=PURPOSE,
        scorer=LexicalScorer(),
        threshold=0.10,
        # Lock the baseline in over the first 5 (on-mission) writes so the meter
        # has a stable reference *before* the agent starts to wander. If the
        # baseline window spans the drift itself, current and baseline fall
        # together and the meter under-reports — a real tuning lesson.
        drift_baseline_window=5,
        drift_alpha=0.4,
    )
    print(f"PURPOSE: {guard.purpose}\n")
    print(f"{'#':>2}  {'verdict':<6} {'score':>5}  {'drift':>5}  memory write")
    print("-" * 78)
    for i, content in enumerate(STREAM, 1):
        v = guard.check(content)
        d = v.details["drift"]
        flag = "FLAG" if not v.aligned else "ok"
        warn = "  <-- DRIFTING" if d["drifting"] else ""
        print(
            f"{i:>2}  {flag:<6} {v.score:>5.2f}  {d['drift']:>5.2f}  "
            f"{content[:42]}{warn}"
        )
    print("-" * 78)
    final = guard.drift()
    print(f"\nFinal: {final}")
    if final.drifting:
        print(
            "\nThe meter caught it: alignment fell well below the agent's own "
            "baseline.\nIn production this is where you'd alert, re-anchor, or "
            "quarantine off-purpose memories."
        )


if __name__ == "__main__":
    main()
