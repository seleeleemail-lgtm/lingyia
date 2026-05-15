"""MiniMax Model adapter.

MiniMax exposes an OpenAI-compatible endpoint at
``https://api.minimaxi.com/v1`` (mainland) and supports tool calling on its
MiniMax-M2 / M2.5 / M2.7 series.

Tested defaults: ``MiniMax-M2`` for general use; ``MiniMax-M2.7-highspeed``
for latency-sensitive paths.
"""
from __future__ import annotations

from ._openai_base import OpenAICompatibleModel


class MiniMaxModel(OpenAICompatibleModel):
    DEFAULT_BASE_URL = "https://api.minimaxi.com/v1"
    DEFAULT_MODEL = "MiniMax-M2"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=kwargs.pop("base_url", self.DEFAULT_BASE_URL),
            model=model,
            **kwargs,
        )
