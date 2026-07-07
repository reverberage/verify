"""Tests for tool gateway components."""

from __future__ import annotations

from rvrb_verify.tools import MockToolGateway, ToolGateway


class TestToolGatewayProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        assert isinstance(MockToolGateway(), ToolGateway)


class TestMockToolGateway:
    def test_execute_returns_string(self) -> None:
        gw = MockToolGateway()
        result = gw.execute("search_web", {"q": "test"})
        assert isinstance(result, str)

    def test_execute_contains_tool_name(self) -> None:
        gw = MockToolGateway()
        result = gw.execute("search_web", {"q": "hello"})
        assert "search_web" in result

    def test_execute_contains_arguments(self) -> None:
        gw = MockToolGateway()
        result = gw.execute("search_news", {"q": "world", "lang": "en"})
        assert "q='world'" in result
        assert "lang='en'" in result

    def test_execute_empty_arguments(self) -> None:
        gw = MockToolGateway()
        result = gw.execute("search_web", {})
        assert "search_web()" in result

    def test_execute_includes_mock_prefix(self) -> None:
        gw = MockToolGateway()
        result = gw.execute("test", {})
        assert result.startswith("[mock]")

    def test_execute_different_tools(self) -> None:
        gw = MockToolGateway()
        r1 = gw.execute("search_web", {"q": "a"})
        r2 = gw.execute("search_news", {"q": "b"})
        assert r1 != r2
