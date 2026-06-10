"""Model Context Protocol (MCP) integration.

Exposes PurposeGuard as MCP *tools* so any MCP-capable agent (Claude, Cursor,
OpenAI-based agents, ...) can call it over the protocol — the framework-agnostic
dream realized at the protocol layer, with the core still importing no framework.

Two layers, on purpose:
  * ``purposeguard_tools(guard)`` — the FRAMEWORK-FREE tool layer: plain callables
    returning JSON-friendly verdict dicts. Unit-testable with no MCP SDK installed.
  * ``build_mcp_server(guard)`` — a thin shell that lazy-imports the ``mcp`` SDK
    (``FastMCP``) and registers those callables as tools.

The ``mcp`` package is a LAZY, OPTIONAL dependency: never imported at module load,
never in the base install. ``purposeguard_tools`` imports nothing; only
``build_mcp_server`` touches the SDK, raising a clear ImportError-with-hint if it's
absent (mirrors the mem0/langchain adapters).
"""

from __future__ import annotations

from typing import Any, Callable

from ..guard import PurposeGuard

_INSTALL_HINT = (
    "The MCP integration requires the 'mcp' package, which is not installed. "
    "Install it with:  pip install mcp"
)


def purposeguard_tools(guard: PurposeGuard) -> "dict[str, Callable[..., dict]]":
    """The framework-free tool layer: a name -> callable mapping an MCP server should
    expose. Each callable returns a JSON-friendly verdict/health dict (never raises
    on normal input), so it can be unit-tested without the MCP SDK."""

    def check_memory_write(content: str) -> dict:
        """Score a prospective agent memory write against the agent's declared
        purpose. Returns a verdict (decision, recommended_action, score, reason,
        fallback). Advisory only — the caller decides whether to store it."""
        return guard.check(content).to_dict()

    def check_agent_response(user_input: str, response: str) -> dict:
        """Score an agent's response to a user against the agent's declared purpose.
        Returns a verdict; in redirect/block mode it carries a safe fallback."""
        return guard.check_response(user_input, response).to_dict()

    def drift_health() -> dict:
        """Current drift-health snapshot for the agent: status ('ok'/'drifting') plus
        the write-drift and response-drift readings."""
        return guard.health()

    return {
        "check_memory_write": check_memory_write,
        "check_agent_response": check_agent_response,
        "drift_health": drift_health,
    }


def _require_mcp():
    """Lazy-import the MCP SDK's FastMCP, with a clear install hint."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return FastMCP


def build_mcp_server(guard: PurposeGuard, *, name: str = "purposeguard") -> Any:
    """Build a FastMCP server exposing PurposeGuard as MCP tools.

    Lazy-imports the ``mcp`` SDK (clear ImportError if absent). Usage::

        from purposeguard import PurposeGuard
        from purposeguard.adapters import build_mcp_server

        server = build_mcp_server(PurposeGuard(purpose="A billing-support agent"))
        server.run()   # serves over stdio for an MCP client to connect

    Returns the FastMCP instance so the caller controls transport/run.
    """
    fast_mcp = _require_mcp()
    server = fast_mcp(name)
    for tool_name, fn in purposeguard_tools(guard).items():
        server.tool(name=tool_name)(fn)
    return server
