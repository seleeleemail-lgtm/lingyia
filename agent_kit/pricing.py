"""Model pricing table.

USD per 1 million tokens (prompt, completion). Cached prompt tokens are billed
at ``cached_rate * prompt_rate`` — by default 10% of the prompt rate, which
matches OpenAI and Anthropic prompt caching at the time of writing.

Prices change frequently. Override at runtime via ``register_pricing()`` or
pass an explicit pricing dict to ``estimate_cost``.
"""
from __future__ import annotations

from typing import Mapping, Optional


# USD per 1M tokens: (prompt_rate, completion_rate)
_PRICING: dict[str, tuple[float, float]] = {
    # --- OpenAI ---
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "o1": (15.0, 60.0),
    "o3-mini": (1.1, 4.4),

    # --- Anthropic ---
    "claude-opus-4-7": (15.0, 75.0),
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),

    # --- SiliconFlow hosted (USD est., RMB rates converted) ---
    "deepseek-ai/DeepSeek-V4-Flash": (0.05, 0.50),
    "deepseek-ai/DeepSeek-V3": (0.10, 0.80),
    "Qwen/Qwen2.5-72B-Instruct": (0.20, 0.60),
    "Pro/zai-org/GLM-5.1": (0.30, 1.20),
    "Pro/moonshotai/Kimi-K2.6": (0.50, 2.00),

    # --- MiniMax ---
    "MiniMax-M2": (0.50, 2.00),
    "MiniMax-M2.7-highspeed": (0.20, 0.80),
}

_CACHED_TOKEN_DISCOUNT = 0.1  # cached prompts billed at 10% of prompt rate


def register_pricing(model_id: str, prompt_rate: float, completion_rate: float) -> None:
    """Add or override pricing for a model. Rates are USD per 1M tokens."""
    _PRICING[model_id] = (prompt_rate, completion_rate)


def get_pricing(model_id: str) -> Optional[tuple[float, float]]:
    return _PRICING.get(model_id)


def estimate_cost(
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    *,
    pricing: Optional[Mapping[str, tuple[float, float]]] = None,
    cached_discount: float = _CACHED_TOKEN_DISCOUNT,
) -> float:
    """Compute USD cost for one model call.

    Returns 0.0 if the model is unknown — never raises, never blocks.
    """
    table = pricing or _PRICING
    rates = table.get(model_id)
    if rates is None:
        return 0.0
    prompt_rate, completion_rate = rates
    fresh_prompt = max(0, prompt_tokens - cached_tokens)
    cost = (
        (fresh_prompt / 1_000_000.0) * prompt_rate
        + (cached_tokens / 1_000_000.0) * (prompt_rate * cached_discount)
        + (completion_tokens / 1_000_000.0) * completion_rate
    )
    return round(cost, 6)
