"""Tests for VerificationEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rvrb_verify.engine import VerificationEngine, VerificationError
from rvrb_verify.models import Verdict, VerdictEnum
from rvrb_verify.tools import MockToolGateway


class TestEngineConstruction:
    def test_default_tool_gateway(self, mock_search_provider, mock_judge_provider) -> None:
        engine = VerificationEngine(
            search_provider=mock_search_provider,
            judge_provider=mock_judge_provider,
        )
        assert engine._tool_gateway is not None

    def test_custom_tool_gateway(self, mock_search_provider, mock_judge_provider) -> None:
        from rvrb_verify.tools import MockToolGateway

        gw = MockToolGateway()
        engine = VerificationEngine(
            search_provider=mock_search_provider,
            judge_provider=mock_judge_provider,
            tool_gateway=gw,
        )
        assert engine._tool_gateway is gw


class TestEngineVerify:
    def test_both_phases_execute(self, engine, strategy) -> None:
        verdict = engine.verify("The sky is blue", strategy)
        assert isinstance(verdict, Verdict)
        assert verdict.verdict == VerdictEnum.TRUE

    def test_search_receives_correct_tools(self, engine, strategy) -> None:
        engine._search_provider = MagicMock()
        engine._search_provider.complete_with_tools.return_value = MagicMock(
            tool_calls=[]
        )
        engine._search_provider.model = "test"

        engine.verify("test claim", strategy)
        call_args = engine._search_provider.complete_with_tools.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert "tools" in kwargs
        assert kwargs["tools"] == strategy.tool_definitions

    def test_judge_receives_correct_output_type(self, engine, strategy) -> None:
        verdict = engine.verify("test claim", strategy)
        assert isinstance(verdict, Verdict)

    def test_judge_never_called_before_search(self, engine, strategy) -> None:
        engine._judge_provider.complete_structured.reset_mock()
        engine.verify("test", strategy)
        engine._judge_provider.complete_structured.assert_called_once()

    def test_tool_gateway_called(self, engine, strategy) -> None:
        gw = MagicMock(spec=MockToolGateway)
        gw.execute.return_value = "result from mock"
        engine._tool_gateway = gw
        engine.verify("test", strategy)
        gw.execute.assert_called()

    def test_returns_verdict(self, engine, strategy) -> None:
        verdict = engine.verify("test", strategy)
        assert isinstance(verdict, Verdict)

    def test_verdict_has_model_used(self, engine, strategy) -> None:
        verdict = engine.verify("test", strategy)
        assert verdict.model_used == "qwen3.7-plus"


class TestEngineEmptyTools:
    def test_no_tool_calls_proceeds_to_judge(self, engine, strategy) -> None:
        engine._search_provider.complete_with_tools.return_value = MagicMock(
            tool_calls=[]
        )
        verdict = engine.verify("test", strategy)
        assert verdict is not None
        assert verdict.verdict == VerdictEnum.TRUE

    def test_no_tool_definitions_still_works(
        self, mock_search_provider, mock_judge_provider
    ) -> None:
        from types import SimpleNamespace

        from rvrb_verify.models import Verdict

        minimal_strategy = SimpleNamespace(
            name="minimal",
            system_prompt_search="Search",
            system_prompt_judge="Judge",
            tool_definitions=[],
            verdict_schema=Verdict,
            thinking_config={},
            model_search=None,
            model_judge=None,
        )
        engine = VerificationEngine(
            search_provider=mock_search_provider,
            judge_provider=mock_judge_provider,
            tool_gateway=MockToolGateway(),
        )
        verdict = engine.verify("test", minimal_strategy)
        assert verdict is not None


class TestEngineErrors:
    def test_search_exhaustion_raises(
        self, exhausted_search_provider, mock_judge_provider, strategy
    ) -> None:
        engine = VerificationEngine(
            search_provider=exhausted_search_provider,
            judge_provider=mock_judge_provider,
            tool_gateway=MockToolGateway(),
        )
        with pytest.raises(VerificationError) as exc:
            engine.verify("test", strategy)
        assert exc.value.phase == "search"

    def test_judge_exhaustion_raises(
        self, mock_search_provider, exhausted_judge_provider, strategy
    ) -> None:
        engine = VerificationEngine(
            search_provider=mock_search_provider,
            judge_provider=exhausted_judge_provider,
            tool_gateway=MockToolGateway(),
        )
        with pytest.raises(VerificationError) as exc:
            engine.verify("test", strategy)
        assert exc.value.phase == "judge"

    def test_search_provider_error_raises(
        self, error_search_provider, mock_judge_provider, strategy
    ) -> None:
        engine = VerificationEngine(
            search_provider=error_search_provider,
            judge_provider=mock_judge_provider,
            tool_gateway=MockToolGateway(),
        )
        with pytest.raises(VerificationError) as exc:
            engine.verify("test", strategy)
        assert exc.value.phase == "search"

    def test_error_carries_original_exception(
        self, mock_search_provider, mock_judge_provider, strategy
    ) -> None:
        from rvrb_verify.models import ProviderError

        err_provider = MagicMock()
        err_provider.complete_with_tools.side_effect = ProviderError("m", 500, "boom")
        err_provider.model = "m"
        engine = VerificationEngine(
            search_provider=err_provider,
            judge_provider=mock_judge_provider,
            tool_gateway=MockToolGateway(),
        )
        with pytest.raises(VerificationError) as exc:
            engine.verify("test", strategy)
        assert "boom" in str(exc.value)
