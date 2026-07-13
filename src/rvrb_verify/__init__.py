"""Claim verification engine for reverberage."""

from __future__ import annotations

from rvrb_verify.models import Evidence, Source, Verdict
from rvrb_verify.provider import ModelProvider, get_provider, DEFAULT_MODEL, DEFAULT_BASE_URL
from rvrb_verify.tools import MockToolGateway, ToolGateway

__all__ = [
    "Verdict",
    "Evidence",
    "Source",
    "ModelProvider",
    "get_provider",
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
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
    model: str | None = None,
    provider: str | None = None,
    search_provider: ModelProvider | None = None,
    judge_provider: ModelProvider | None = None,
    tool_gateway: ToolGateway | None = None,
) -> Verdict:
    """Verify a claim using the given strategy.

    Parameters
    ----------
    claim_text : str
        The claim to verify.
    strategy : str
        Strategy name from ``list_strategies()``. Default: ``"fact-check"``.
    model : str | None
        Override model ID for both phases. Overrides strategy-level defaults.
    provider : str | None
        Provider name (qwen, openai, local).  Overrides
        ``N3RVERBERAGE_PROVIDER`` env var.
    search_provider : ModelProvider | None
        Provider for the search phase. If None, resolves from ``model`` and
        ``provider`` params with the strategy's search model as fallback.
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
            f"Unknown strategy: {strategy!r}. Available: {', '.join(list_strategies())}"
        )

    # Resolve per-phase providers from strategy's model preferences
    if search_provider is None:
        search_model = model or getattr(strategy_obj, "model_search", None) or DEFAULT_MODEL
        search_provider = get_provider(model=search_model, provider=provider)

    if judge_provider is None:
        judge_model = model or getattr(strategy_obj, "model_judge", None) or DEFAULT_MODEL
        judge_provider = get_provider(model=judge_model, provider=provider)

    engine = VerificationEngine(
        search_provider=search_provider,
        judge_provider=judge_provider,
        tool_gateway=tool_gateway or MockToolGateway(),
    )
    return engine.verify(claim_text, strategy_obj)
