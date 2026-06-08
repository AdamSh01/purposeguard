"""Example: guard ANY store with the raw adapter.

`guard_store(store, guard)` wraps any object with `add()` / `search()` so every
write is scored against the purpose before it lands. Off-mission writes are
TAGGED (their metadata gets a `purposeguard` breadcrumb) but never dropped —
detection-first. Drift accumulates in the guard, readable via `guard.drift()`.

Runs as-is with no real vector store and no API key. It uses local embeddings if
the `[embeddings]` extra is installed; otherwise it falls back to the lexical
floor and SAYS SO (the floor is rough on natural prose — see the printed note).

    python examples/raw_store_example.py
"""

from _scorer import pick_scorer

from purposeguard import PurposeGuard
from purposeguard.adapters import guard_store


class InMemoryStore:
    """A stand-in for a vector store / memory backend. The adapter treats any
    object with this shape (`add(text, *, metadata=...)`, `search(query)`) the
    same way — swap this for Chroma, FAISS, a dict, whatever you already use."""

    def __init__(self):
        self.items = []  # list of (text, metadata)

    def add(self, text, metadata=None, **kwargs):
        self.items.append((text, metadata))
        return len(self.items)  # a fake id

    def search(self, query, **kwargs):
        return [t for t, _ in self.items if query.lower() in t.lower()]


def main():
    # Default threshold is the BALANCED preset (calibrated for the embedding
    # scorer). pick_scorer() uses embeddings when available, else the lexical floor.
    guard = PurposeGuard(
        purpose="A customer-support assistant for billing, payments, and refunds",
        scorer=pick_scorer(),
    )

    store = InMemoryStore()
    memory = guard_store(store, guard)  # <-- the one line that adds the guard

    memories = [
        "Updated the customer's payment card on their billing subscription",  # on-mission
        "Issued a refund for a duplicate invoice charge",                     # on-mission
        "Chatted about last night's football game",                          # off-mission
        "Recommended a sourdough bread recipe for the weekend",               # off-mission
    ]

    print("Writing memories through the guarded store:\n")
    for text in memories:
        memory.add(text)  # scored -> tagged-if-off-mission -> passed through, never dropped
        stored_text, meta = store.items[-1]
        tag = (meta or {}).get("purposeguard")
        if tag:
            print(f"  FLAG (score {tag['alignment']:.2f}) -- stored + tagged: {stored_text}")
        else:
            print(f"  ok               -- stored clean    : {stored_text}")

    # Detection-first: every write is still in the store, even the flagged ones.
    print(f"\nAll {len(store.items)} writes are still stored (nothing dropped).")
    print(f"Drift reading: {guard.drift()}")


if __name__ == "__main__":
    main()
