"""Tests for provider.py — env-var defaults, fallback resolution."""

from __future__ import annotations

import os

from rvrb_verify.provider import DEFAULT_BASE_URL, DEFAULT_MODEL, get_provider


class TestGetProvider:
    """get_provider() resolution chain."""

    def test_get_provider_no_args(self) -> None:
        """get_provider() with no args returns a ModelProvider."""
        provider = get_provider()
        assert hasattr(provider, "model")
        assert hasattr(provider, "base_url")
        assert hasattr(provider, "complete")
        assert hasattr(provider, "complete_structured")
        assert hasattr(provider, "complete_with_tools")

    def test_get_provider_with_model(self) -> None:
        """get_provider(model=\"gpt-4\") returns a provider with assigned model."""
        provider = get_provider(model="gpt-4")
        assert provider.model == "gpt-4"

    def test_get_provider_protocol_match(self) -> None:
        """Returned provider satisfies ModelProvider protocol structural check."""
        provider = get_provider(model="gpt-4")
        assert callable(getattr(provider, "complete", None))
        assert callable(getattr(provider, "complete_structured", None))
        assert callable(getattr(provider, "complete_with_tools", None))


class TestDefaults:
    """DEFAULT_MODEL and DEFAULT_BASE_URL are strings (read from env or hardcoded)."""

    def test_default_model_is_string(self) -> None:
        assert isinstance(DEFAULT_MODEL, str)

    def test_default_base_url_is_string(self) -> None:
        assert isinstance(DEFAULT_BASE_URL, str)

    def test_default_model_has_fallback(self) -> None:
        """Without env var, DEFAULT_MODEL contains a valid model reference."""
        assert len(DEFAULT_MODEL) > 0

    def test_default_base_url_has_fallback(self) -> None:
        """Without env var, DEFAULT_BASE_URL contains a valid URL."""
        assert DEFAULT_BASE_URL.startswith("http")


class TestFallbackProvider:
    """Validates that _build_fallback_provider handles all supported types."""

    def test_qwen_provider_type(self) -> None:
        """Provider type \"""
qwen\""" resolves without error."""
        provider = get_provider(provider="qwen")
        assert hasattr(provider, "model")
        assert hasattr(provider, "base_url")

    def test_openai_provider_type(self) -> None:
        """Provider type \"""
openai\""" resolves without error."""
        # Skip if no API key — the fallback will raise
        if not os.environ.get("OPENAI_API_KEY"):
            import pytest
            pytest.skip("OPENAI_API_KEY not set")
        provider = get_provider(provider="openai")
        assert hasattr(provider, "model")
        assert hasattr(provider, "base_url")
