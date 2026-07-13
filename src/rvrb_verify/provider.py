"""Provider contract for rvrb-verify.

Follows satellite-protocol-v2.md:
- ModelProvider Protocol (structural, not ABC)
- get_provider() factory with n3rverberage fallback
- DEFAULT_MODEL and DEFAULT_BASE_URL read from env vars
- Generic fallback provider supports any OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from pydantic import BaseModel

from rvrb_verify.models import ProviderError, ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Protocol definition (structural — no inheritance required)
# ---------------------------------------------------------------------------


class ModelProvider(Protocol):
    """Protocol for LLM providers.

    Any object with these attributes/methods is a valid provider.
    No ABC inheritance — uses duck typing.
    """

    model: str
    base_url: str

    def complete(self, messages: list[dict], **kwargs: Any) -> str: ...
    def complete_structured(
        self,
        messages: list[dict],
        output_type: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel: ...
    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs: Any,
    ) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Defaults — env-var-driven with Qwen fallback for backward compatibility
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = os.environ.get(
    "N3RVERBERAGE_DEFAULT_MODEL",
    "qwen3-coder-plus",
)
DEFAULT_BASE_URL: str = os.environ.get(
    "N3RVERBERAGE_DEFAULT_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Map provider type → default model, default URL, API key env var
_PROVIDER_FALLBACKS: dict[str, tuple[str, str, str]] = {
    "qwen": (
        "qwen3-coder-plus",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    "openai": (
        "gpt-4",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
    ),
    "local": (
        "qwen2.5",
        "http://127.0.0.1:11434/v1",
        "",  # no API key required
    ),
}


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def get_provider(
    model: str | None = None,
    provider: str | None = None,
) -> ModelProvider:
    """Resolve a model provider.

    Resolution order:
    1. Try n3rverberage.providers.get_provider() with ``provider:model`` format
    2. Fallback to ``_GenericProvider`` with env-var-driven defaults

    Parameters
    ----------
    model : str | None
        Override the model ID. If None, uses DEFAULT_MODEL.
    provider : str | None
        Provider name (qwen, openai, local).  Overrides
        ``N3RVERBERAGE_PROVIDER`` env var.  If both are unset, defaults to
        Qwen for backward compatibility.

    Returns
    -------
    ModelProvider
        A provider instance matching the ModelProvider Protocol.
    """
    resolved_model = model or DEFAULT_MODEL
    resolved_provider = provider or os.environ.get("N3RVERBERAGE_PROVIDER") or "qwen"

    # Try n3rverberage first (preferred — has fallback chain + quota detection)
    try:
        from n3rverberage.providers import get_provider as n3rv_get_provider

        return n3rv_get_provider(name=f"{resolved_provider}:{resolved_model}")
    except ImportError:
        pass

    # Fallback: generic OpenAI-compatible provider
    return _build_fallback_provider(resolved_provider, resolved_model)


def _build_fallback_provider(provider_type: str, model: str) -> _GenericProvider:
    """Construct a ``_GenericProvider`` from provider type and model.

    Reads the API key and base URL from the appropriate env vars for the
    given provider type.
    """
    provider_type = provider_type.strip().lower()
    if provider_type not in _PROVIDER_FALLBACKS:
        raise ValueError(
            f"Unknown provider type: '{provider_type}'. "
            f"Supported: {', '.join(_PROVIDER_FALLBACKS)}"
        )

    default_model, default_url, api_key_var = _PROVIDER_FALLBACKS[provider_type]

    # Resolve base URL: env var > per-type default
    base_url = os.environ.get("N3RVERBERAGE_DEFAULT_BASE_URL") or default_url

    # Resolve API key
    api_key: str | None = None
    if api_key_var:
        api_key = os.environ.get(api_key_var)
        if not api_key:
            raise ValueError(
                f"{api_key_var} is not set. Set it or install n3rverberage."
            )

    return _GenericProvider(
        model=model or default_model,
        base_url=base_url,
        api_key=api_key or "not-needed",
    )


# ---------------------------------------------------------------------------
# Generic OpenAI-compatible provider (fallback when n3rverberage absent)
# ---------------------------------------------------------------------------


class _GenericProvider:
    """Minimal OpenAI-compatible provider for any endpoint.

    Used as fallback when n3rverberage is not installed.
    No provider-specific error codes — all API errors are wrapped as
    generic ``ProviderError``.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._api_key = api_key

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self.base_url, timeout=60.0)

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        max_tokens = kwargs.pop("max_tokens", 4096)
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            raise _wrap_error(self.model, exc) from exc
        return response.choices[0].message.content or ""

    def complete_structured(
        self,
        messages: list[dict],
        output_type: type[BaseModel],
        **kwargs: Any,
    ) -> BaseModel:
        schema = output_type.model_json_schema()
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", output_type.__name__),
                "schema": schema,
                "strict": True,
            },
        }
        max_tokens = kwargs.pop("max_tokens", 4096)
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise _wrap_error(self.model, exc) from exc

        raw = response.choices[0].message.content
        if not raw:
            raise ProviderError(self.model, 200, "Empty structured response")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(self.model, 200, f"Invalid JSON: {exc}") from exc

        return output_type.model_validate(parsed)

    def complete_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        **kwargs: Any,
    ) -> ToolResult:
        max_tokens = kwargs.pop("max_tokens", 4096)
        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            raise _wrap_error(self.model, exc) from exc

        message = response.choices[0].message
        content = message.content or ""

        if not message.tool_calls:
            return ToolResult(content=content)

        tcs: list[ToolCall] = []
        for tc in message.tool_calls:
            args = _safe_parse_args(tc.function.arguments)
            tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ToolResult(content=content, tool_calls=tcs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_parse_args(raw: str | None) -> dict[str, Any]:
    """Safely parse tool call arguments."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _wrap_error(model_id: str, exc: Exception) -> ProviderError:
    """Wrap any exception as a generic ProviderError.

    This fallback does NOT inspect ``AllocationQuota.FreeTierOnly`` or any
    other provider-specific error code.  Those checks belong in the upstream
    n3rverberage providers, not in satellite fallback code.
    """
    status_code = getattr(exc, "status_code", 500) or 500
    body = str(exc)
    return ProviderError(model_id, int(status_code), body)
