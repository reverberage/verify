"""Tests for the MCP server module."""

from __future__ import annotations

import pytest

from rvrb_verify import mcp


class TestMCPVerificationFunctions:
    def test_verify_claim_returns_error_on_bad_strategy(self) -> None:
        result = mcp.verify_claim("test", strategy="nonexistent")
        assert "error" in result
        assert "Unknown strategy" in result["error"]

    def test_list_strategies(self) -> None:
        strategies = mcp.list_available_strategies()
        assert isinstance(strategies, list)
        assert "fact-check" in strategies
        assert "legal" in strategies
        assert "research" in strategies

    def test_mcp_server_creation(self) -> None:
        server = mcp._build_server()
        if mcp.HAS_MCP:
            assert server is not None
            assert hasattr(server, "_tool_manager")
        else:
            assert server is None

    def test_mcp_server_has_tools(self) -> None:
        if not mcp.HAS_MCP:
            pytest.skip("mcp package not installed")
        server = mcp._build_server()
        assert server is not None
        tool_names = list(server._tool_manager._tools.keys())
        assert "_verify_claim_tool" in tool_names
        assert "_list_strategies_tool" in tool_names
