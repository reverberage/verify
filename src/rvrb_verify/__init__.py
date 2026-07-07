"""Claim verification engine for reverberage."""

from __future__ import annotations

from rvrb_verify.models import Evidence, Source, Verdict
from rvrb_verify.tools import MockToolGateway, ToolGateway

__all__ = [
    "Verdict",
    "Evidence",
    "Source",
    "MockToolGateway",
    "ToolGateway",
    "list_strategies",
    "verify",
]

__version__ = "0.1.0"


def list_strategies() -> list[str]:
    """Return available verification strategy names."""
    from rvrb_verify.strategies import list_strategies as _ls

    return _ls()


def verify(
    claim_text: str,
    strategy: str = "fact-check",
    *,
    search_provider=None,
    judge_provider=None,
    tool_gateway: ToolGateway | None = None,
) -> Verdict:
    """Verify a claim using the given strategy.

    Parameters
    ----------
    claim_text : str
        The claim to verify.
    strategy : str
        Strategy name from ``list_strategies()``. Default: ``"fact-check"``.
    search_provider : ModelProvider | None
        Provider for the search phase. If None, tries ``n3rverberage.providers``
        as default, falling back to a basic Qwen/DashScope provider.
    judge_provider : ModelProvider | None
        Provider for the judge phase. Same fallback as search_provider.
    tool_gateway : ToolGateway | None
        Tool executor. Default: ``MockToolGateway()``.

    Returns
    -------
    Verdict
    """
    from rvrb_verify.engine import VerificationEngine
    from rvrb_verify.strategies import REGISTRY

    strategy_obj = REGISTRY.get(strategy)
    if strategy_obj is None:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Available: {', '.join(list_strategies())}"
        )

    provider = _resolve_provider()

    sp = search_provider or provider
    jp = judge_provider or provider

    engine = VerificationEngine(
        search_provider=sp,
        judge_provider=jp,
        tool_gateway=tool_gateway or MockToolGateway(),
    )
    return engine.verify(claim_text, strategy_obj)


def _resolve_provider():
    """Resolve a default provider, trying n3rverberage first."""
    try:
        from n3rverberage.providers import get_provider

        return get_provider()
    except ImportError:
        pass

    # Fallback: basic DashScope provider using openai client
    import os

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "No provider available. Install n3rverberage or set DASHSCOPE_API_KEY."
        )
    from rvrb_verify._default_provider import _DefaultProvider

    return _DefaultProvider(api_key=api_key)
