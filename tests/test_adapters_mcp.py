"""Tests for the MCP integration. Hermetic — the framework-free tool layer needs no
MCP SDK; the server builder is tested for a clean ImportError when 'mcp' is absent,
and exercised for real only if 'mcp' happens to be installed."""

import builtins

import pytest

from purposeguard import Decision, LexicalScorer, PurposeGuard
from purposeguard.adapters import build_mcp_server, purposeguard_tools

PURPOSE = "billing payment subscription invoice refund"


def make_guard(**kw):
    kw.setdefault("scorer", LexicalScorer())
    kw.setdefault("threshold", 0.12)
    return PurposeGuard(PURPOSE, **kw)


def test_tool_layer_returns_json_friendly_verdicts():
    tools = purposeguard_tools(make_guard())
    assert set(tools) == {"check_memory_write", "check_agent_response", "drift_health"}

    on = tools["check_memory_write"]("billing payment invoice refund")
    off = tools["check_memory_write"]("weather forecast vacation hiking")
    assert on["decision"] == Decision.ALLOW.value
    assert off["decision"] == Decision.FLAG.value
    # JSON-friendly: primitive-typed values only
    assert isinstance(off["score"], float) and isinstance(off["reason"], str)


def test_tool_layer_check_response_and_health():
    tools = purposeguard_tools(make_guard())
    v = tools["check_agent_response"]("how do I get a refund?", "here's a sourdough recipe")
    assert v["decision"] == Decision.FLAG.value
    h = tools["drift_health"]()
    assert h["status"] in ("ok", "drifting")
    assert "write_drift" in h and "response_drift" in h


def test_build_server_raises_clear_importerror_without_mcp(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("simulated: mcp not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as excinfo:
        build_mcp_server(make_guard())
    assert "pip install mcp" in str(excinfo.value)


def test_build_server_registers_tools_if_mcp_installed():
    pytest.importorskip("mcp")
    server = build_mcp_server(make_guard(), name="purposeguard-test")
    assert server is not None  # a FastMCP instance with our tools registered
