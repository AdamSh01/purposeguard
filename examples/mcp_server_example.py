"""Run PurposeGuard as an MCP server (purpose/mission observability as a tool).

Any MCP-capable agent (Claude, Cursor, an OpenAI-based agent, ...) can then call
PurposeGuard over the protocol — no Python import on their side.

    pip install "purposeguard[embeddings]" mcp
    python examples/mcp_server_example.py        # serves over stdio

This script also runs WITHOUT the `mcp` package installed: it falls back to calling
the framework-free tool layer directly, so you can see the verdicts an MCP client
would receive. With embeddings unavailable it uses the lexical floor (a visible
notice is printed) — install the extra for real verdicts.
"""

from __future__ import annotations

from purposeguard import PurposeGuard
from purposeguard.adapters import purposeguard_tools

# A reusable example scorer that prints a notice if it falls back to the floor.
try:
    from _scorer import pick_scorer
except ImportError:  # when run as a module
    from examples._scorer import pick_scorer

PURPOSE = "A customer-support assistant for billing, payments, invoices, and refunds"


def main() -> None:
    guard = PurposeGuard(purpose=PURPOSE, blocked_topics=["legal advice", "medical advice"],
                         mode="redirect", scorer=pick_scorer())

    try:
        from purposeguard.adapters import build_mcp_server

        server = build_mcp_server(guard, name="purposeguard")
        print("Serving PurposeGuard over MCP (stdio). Tools: "
              "check_memory_write, check_agent_response, drift_health.")
        server.run()  # blocks, serving an MCP client over stdio
    except ImportError:
        # No `mcp` SDK installed — demonstrate the tool layer directly instead.
        print("[note] the 'mcp' package is not installed; showing the tool layer "
              "output directly (pip install mcp to serve over the protocol).\n")
        tools = purposeguard_tools(guard)
        demo = tools["check_agent_response"](
            user_input="my invoice looks wrong, can you give me legal advice?",
            response="Sure, for your lawsuit here's the legal advice you need...",
        )
        print("check_agent_response ->")
        for k in ("decision", "recommended_action", "score", "fallback"):
            print(f"   {k}: {demo[k]}")


if __name__ == "__main__":
    main()
