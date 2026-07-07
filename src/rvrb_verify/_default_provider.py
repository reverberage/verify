"""Minimal default provider for rvrb-verify when n3rverberage is not installed."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from rvrb_verify.models import (
    ProviderError,
    QuotaExhaustedError,
    ToolCall,
    ToolResult,
)

_DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen3-coder-plus"
_MAX_TOKENS = 4096


class _DefaultProvider:
    """Minimal provider for DashScope (Qwen) API.

    Used as a fallback when n3rverberage is not installed.
    Detects free-tier quota exhaustion (429 + AllocationQuota.FreeTierOnly).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or _DEFAULT_MODEL
        resolved_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError("DASHSCOPE_API_KEY is required")
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=_DASHSCOPE_BASE_URL,
            timeout=60.0,
        )

    def complete(self, messages: list[dict], **kwargs: Any) -> str:
        max_tokens = kwargs.pop("max_tokens", _MAX_TOKENS)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            raise ProviderError(self.model, 500, str(exc)) from exc
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
        max_tokens = kwargs.pop("max_tokens", _MAX_TOKENS)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise ProviderError(self.model, 500, str(exc)) from exc
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
        max_tokens = kwargs.pop("max_tokens", _MAX_TOKENS)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                **kwargs,
            )
        except Exception as exc:
            if _is_429_quota(exc):
                raise QuotaExhaustedError(self.model, 429, str(exc)) from exc
            raise ProviderError(self.model, 500, str(exc)) from exc
        message = response.choices[0].message
        content = message.content or ""

        if not message.tool_calls:
            return ToolResult(content=content)

        tcs = []
        for tc in message.tool_calls:
            args = _safe_parse_args(tc.function.arguments)
            tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return ToolResult(content=content, tool_calls=tcs)


def _safe_parse_args(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _is_429_quota(exc: Exception) -> bool:
    """Check if an exception is a 429 quota error."""
    body = getattr(exc, "body", None) or ""
    body_str = json.dumps(body) if isinstance(body, dict) else str(body)
    return "AllocationQuota.FreeTierOnly" in body_str
