"""Tests for verification strategies."""

from __future__ import annotations

from rvrb_verify.models import Verdict
from rvrb_verify.strategies import REGISTRY, list_strategies
from rvrb_verify.strategies.base import VerificationStrategy


class TestStrategyProtocol:
    def test_all_strategies_satisfy_protocol(self) -> None:
        for name, strategy in REGISTRY.items():
            assert isinstance(strategy, VerificationStrategy), (
                f"Strategy {name!r} does not satisfy VerificationStrategy protocol"
            )

    def test_each_strategy_has_required_attributes(self) -> None:
        for name, strategy in REGISTRY.items():
            assert hasattr(strategy, "name")
            assert hasattr(strategy, "system_prompt_search")
            assert hasattr(strategy, "system_prompt_judge")
            assert hasattr(strategy, "tool_definitions")
            assert hasattr(strategy, "verdict_schema")
            assert hasattr(strategy, "thinking_config")
            assert hasattr(strategy, "model_search")
            assert hasattr(strategy, "model_judge")


class TestStrategyRegistry:
    def test_registry_has_three_strategies(self) -> None:
        assert len(REGISTRY) == 3

    def test_registry_keys(self) -> None:
        assert set(REGISTRY.keys()) == {"fact-check", "legal", "research"}

    def test_list_strategies_returns_sorted(self) -> None:
        names = list_strategies()
        assert names == sorted(names)
        assert names == ["fact-check", "legal", "research"]


class TestFactCheckStrategy:
    def test_name(self) -> None:
        assert REGISTRY["fact-check"].name == "fact-check"

    def test_has_two_tools(self) -> None:
        tools = REGISTRY["fact-check"].tool_definitions
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "search_web"
        assert tools[1]["function"]["name"] == "search_news"

    def test_verdict_schema_is_verdict(self) -> None:
        assert REGISTRY["fact-check"].verdict_schema is Verdict

    def test_search_prompt_non_empty(self) -> None:
        prompt = REGISTRY["fact-check"].system_prompt_search
        assert len(prompt) > 50
        assert "fact-checker" in prompt.lower()

    def test_judge_prompt_non_empty(self) -> None:
        prompt = REGISTRY["fact-check"].system_prompt_judge
        assert len(prompt) > 50
        assert "judge" in prompt.lower()

    def test_models_configured(self) -> None:
        assert REGISTRY["fact-check"].model_search == "qwen3-coder-plus"
        assert REGISTRY["fact-check"].model_judge == "qwen3.7-plus"


class TestLegalStrategy:
    def test_name(self) -> None:
        assert REGISTRY["legal"].name == "legal"

    def test_has_two_tools(self) -> None:
        tools = REGISTRY["legal"].tool_definitions
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "search_statutes"
        assert tools[1]["function"]["name"] == "search_case_law"

    def test_verdict_schema_is_verdict(self) -> None:
        assert REGISTRY["legal"].verdict_schema is Verdict

    def test_search_prompt_non_empty(self) -> None:
        prompt = REGISTRY["legal"].system_prompt_search
        assert len(prompt) > 50
        assert "legal" in prompt.lower()

    def test_judge_prompt_non_empty(self) -> None:
        prompt = REGISTRY["legal"].system_prompt_judge
        assert len(prompt) > 50
        assert "judge" in prompt.lower()

    def test_models_configured(self) -> None:
        assert REGISTRY["legal"].model_search == "qwen3-coder-plus"
        assert REGISTRY["legal"].model_judge == "qwen3.7-plus"


class TestResearchStrategy:
    def test_name(self) -> None:
        assert REGISTRY["research"].name == "research"

    def test_has_two_tools(self) -> None:
        tools = REGISTRY["research"].tool_definitions
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "search_papers"
        assert tools[1]["function"]["name"] == "search_arxiv"

    def test_verdict_schema_is_verdict(self) -> None:
        assert REGISTRY["research"].verdict_schema is Verdict

    def test_search_prompt_non_empty(self) -> None:
        prompt = REGISTRY["research"].system_prompt_search
        assert len(prompt) > 50
        assert "research" in prompt.lower()

    def test_judge_prompt_non_empty(self) -> None:
        prompt = REGISTRY["research"].system_prompt_judge
        assert len(prompt) > 50
        assert "judge" in prompt.lower()

    def test_models_configured(self) -> None:
        assert REGISTRY["research"].model_search == "qwen3-coder-plus"
        assert REGISTRY["research"].model_judge == "qwen3.7-plus"


class TestStrategyCrossDomain:
    def test_system_prompts_differ(self) -> None:
        """Each strategy should have distinct prompts."""
        prompts = {
            name: (s.system_prompt_search, s.system_prompt_judge)
            for name, s in REGISTRY.items()
        }
        assert len(set(prompts.values())) == len(prompts)

    def test_tool_definitions_differ(self) -> None:
        """Each strategy should have distinct tool names."""
        tool_names = {}
        for name, s in REGISTRY.items():
            names = tuple(t["function"]["name"] for t in s.tool_definitions)
            tool_names[name] = names
        assert len(set(tool_names.values())) == len(tool_names)
