"""Example: guard a LangChain chat history + score live answers.

Two layers of drift detection in one realistic flow:

  1. `GuardedChatMessageHistory` wraps a chat history so EVERY message added
     (user AND AI turns) is scored as it lands; off-mission turns get a
     `purposeguard` breadcrumb in their `additional_kwargs`. This feeds the WRITE
     drift meter -> `guard.drift()`.
  2. `guard.check_response(user_input, ai_text)` scores the agent's LIVE answer
     against the purpose. This feeds the SEPARATE response drift meter ->
     `guard.response_drift()`, so answer-drift is tracked independently of what
     gets stored.

Runs as-is against a FAKE history + fake messages: no real langchain, no keys. It
uses local embeddings if `[embeddings]` is installed; otherwise it falls back to
the lexical floor and SAYS SO (rough on natural prose — see the printed note).

    python examples/langchain_example.py
"""

from _scorer import pick_scorer

from purposeguard import PurposeGuard
from purposeguard.adapters import guard_chat_history


class FakeMessage:
    """Duck-typed stand-in for a langchain_core BaseMessage (content/type/kwargs)."""

    def __init__(self, content, type):
        self.content = content
        self.type = type
        self.additional_kwargs = {}


class FakeHistory:
    """Stand-in for a langchain_core BaseChatMessageHistory."""

    def __init__(self):
        self._messages = []

    @property
    def messages(self):
        return list(self._messages)

    def add_message(self, message):
        self._messages.append(message)

    def add_messages(self, messages):
        self._messages.extend(messages)

    def clear(self):
        self._messages.clear()


def main():
    guard = PurposeGuard(
        purpose="A customer-support assistant for billing, payments, and refunds",
        scorer=pick_scorer(),  # embeddings if available, else lexical floor (with a notice)
    )

    history = guard_chat_history(FakeHistory(), guard)  # <-- wrap the chat history

    # A short session: on-mission, on-mission, then the agent WANDERS on turn 3.
    conversation = [
        ("How do I update my payment card?",
         "Open your billing settings to change the payment card on your subscription."),
        ("Can I get a refund for a duplicate charge?",
         "Yes, I've issued a refund for the duplicate invoice to your account."),
        ("What's the weather like this weekend?",
         "It'll be sunny and warm, perfect for a hike, so pack some sunscreen!"),  # off-mission = drift
    ]

    print("Conversation (each turn is scored as it lands in history):\n")
    for user_text, ai_text in conversation:
        # User turn -> stored + scored by the adapter (feeds guard.drift()).
        history.add_message(FakeMessage(user_text, "human"))

        # Agent produces an answer. Score the LIVE answer (the new feature) ->
        # feeds guard.response_drift(), independent of what gets stored.
        rv = guard.check_response(user_text, ai_text)
        verdict = "FLAG" if not rv.aligned else "ok  "
        print(f"  user: {user_text}")
        print(f"  ai  : [{verdict} score {rv.score:.2f}] {ai_text}\n")

        # AI turn -> also stored + scored by the adapter (feeds guard.drift()).
        history.add_message(FakeMessage(ai_text, "ai"))

    # With real langchain you'd wrap an InMemoryChatMessageHistory and use
    # history.add_user_message(...) / history.add_ai_message(...) (which build real
    # HumanMessage / AIMessage objects under the hood). The guarding is identical.

    tagged = sum(1 for m in history.messages if m.additional_kwargs.get("purposeguard"))
    print(f"Stored {len(history.messages)} messages; {tagged} flagged off-mission (tagged, not dropped).")
    print(f"Write   drift (everything stored): {guard.drift()}")
    print(f"Response drift (live AI answers) : {guard.response_drift()}")


if __name__ == "__main__":
    main()
