"""Shared fixtures for rvrb_verify tests."""

from __future__ import annotations

import pytest

from rvrb_verify.models import ToolCall, ToolResult


@pytest.fixture
def mock_search_provider(mocker):
    """Returns a MagicMock for the search-phase ModelProvider."""
    provider = mocker.MagicMock()
    result = ToolResult(
        content="",
        tool_calls=[
            ToolCall(id="call_1", name="search_web", arguments={"q": "sky color"})
        ],
    )
    provider.complete_with_tools.return_value = result
    provider.model = "qwen3-coder-plus"
    return provider


@pytest.fixture
def mock_judge_provider(mocker):
    """Returns a MagicMock for the judge-phase ModelProvider."""
    from rvrb_verify.models import Verdict, VerdictEnum

    provider = mocker.MagicMock()
    provider.complete_structured.return_value = Verdict(
        claim="The sky is blue",
        verdict=VerdictEnum.TRUE,
        confidence=0.95,
        summary="The sky appears blue due to Rayleigh scattering.",
    )
    provider.model = "qwen3.7-plus"
    return provider


@pytest.fixture
def engine(mock_search_provider, mock_judge_provider):
    """Pre-wired VerificationEngine with mocked providers."""
    from rvrb_verify.engine import VerificationEngine
    from rvrb_verify.tools import MockToolGateway

    return VerificationEngine(
        search_provider=mock_search_provider,
        judge_provider=mock_judge_provider,
        tool_gateway=MockToolGateway(),
    )


@pytest.fixture
def exhausted_search_provider(mocker):
    """Returns a mock provider that raises QuotaExhaustedError on search."""
    from rvrb_verify.models import QuotaExhaustedError

    provider = mocker.MagicMock()
    provider.complete_with_tools.side_effect = QuotaExhaustedError(
        model_id="qwen3-coder-plus",
        status_code=429,
        body='{"code":"AllocationQuota.FreeTierOnly"}',
    )
    return provider


@pytest.fixture
def exhausted_judge_provider(mocker):
    """Returns a mock provider that raises QuotaExhaustedError on judge."""
    from rvrb_verify.models import QuotaExhaustedError

    provider = mocker.MagicMock()
    provider.complete_structured.side_effect = QuotaExhaustedError(
        model_id="qwen3.7-plus",
        status_code=429,
        body='{"code":"AllocationQuota.FreeTierOnly"}',
    )
    return provider


@pytest.fixture
def error_search_provider(mocker):
    """Returns a mock provider that raises a non-quota ProviderError on search."""
    from rvrb_verify.models import ProviderError

    provider = mocker.MagicMock()
    provider.complete_with_tools.side_effect = ProviderError(
        model_id="qwen3-coder-plus",
        status_code=401,
        body="Unauthorized",
    )
    return provider


@pytest.fixture
def strategy():
    """Returns the default fact-check strategy."""
    from rvrb_verify.strategies.fact_check import fact_check_strategy

    return fact_check_strategy
