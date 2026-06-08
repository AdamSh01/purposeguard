"""Example: guard a Mem0 memory with the mem0 adapter.

`guard_mem0(client, guard)` wraps an existing Mem0 client so every `add()` is
scored against the purpose; off-mission writes get a `purposeguard` breadcrumb in
their metadata but are never dropped (detection-first). The adapter checks the
INPUT text pre-add (deterministic, offline) rather than Mem0's LLM-extracted
facts — see purposeguard/adapters/mem0.py.

Runs as-is against a FAKE client: no real mem0 install, no LLM, no API key. It
uses local embeddings if `[embeddings]` is installed; otherwise it falls back to
the lexical floor and SAYS SO (rough on natural prose — see the printed note).

    python examples/mem0_example.py
"""

from _scorer import pick_scorer

from purposeguard import PurposeGuard
from purposeguard.adapters import guard_mem0


class FakeMem0Client:
    """Stands in for mem0.Memory / mem0.MemoryClient (same add/search shape)."""

    def __init__(self):
        self.added = []  # list of (messages, metadata, kwargs)

    def add(self, messages, metadata=None, **kwargs):
        self.added.append((messages, metadata, kwargs))
        return {"results": [{"id": len(self.added)}]}

    def search(self, query, **kwargs):
        return {"results": []}


def main():
    guard = PurposeGuard(
        purpose="A customer-support assistant for billing, payments, and refunds",
        scorer=pick_scorer(),  # embeddings if available, else lexical floor (with a notice)
    )

    client = FakeMem0Client()
    memory = guard_mem0(client, guard)  # <-- wrap an existing Mem0 client

    # Mem0 accepts a plain string OR a list of {role, content} messages; both work.
    writes = [
        ("Updated the billing subscription and the saved payment method", {"user_id": "u1"}),
        ([{"role": "user", "content": "Can you refund the duplicate charge on my last invoice?"}],
         {"user_id": "u1"}),
        ("Talked about the weekend football scores", {"user_id": "u1"}),          # off-mission
        ([{"role": "assistant", "content": "Here's a great sourdough bread recipe"}],
         {"user_id": "u1"}),                                                      # off-mission
    ]

    print("Adding memories through the guarded Mem0 client:\n")
    for messages, kw in writes:
        memory.add(messages, **kw)  # scored -> tagged-if-off-mission -> forwarded to Mem0
        _, meta, _ = client.added[-1]
        tag = (meta or {}).get("purposeguard")
        label = f"FLAG (score {tag['alignment']:.2f})" if tag else "ok           "
        print(f"  {label} -- sent to Mem0: {messages}")

    print(f"\nAll {len(client.added)} writes reached Mem0 (flagged ones tagged, none dropped).")
    print(f"Drift reading: {guard.drift()}")

    # --- With a REAL Mem0 client (needs `pip install mem0ai`, and an LLM key for
    # --- Mem0 itself), it's the same two lines:
    #
    #   from mem0 import Memory
    #   memory = guard_mem0(Memory(), guard)
    #   memory.add("Refund my invoice", user_id="u1")
    #
    # Or let the adapter build the client for you (raises a clear ImportError with
    # an install hint if mem0 isn't installed):
    #
    #   from purposeguard.adapters import guarded_memory
    #   memory = guarded_memory(guard)            # or guarded_memory(guard, config=...)


if __name__ == "__main__":
    main()
