"""SiliconFlow (硅基流动) Model adapter.

SiliconFlow exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint at
``https://api.siliconflow.cn/v1`` and supports function/tool calling for
DeepSeek-V3, Qwen2.5/Qwen3, GLM-4, and other modern open models hosted on its
platform.

Tested defaults: ``Qwen/Qwen2.5-72B-Instruct`` for general agent work,
``deepseek-ai/DeepSeek-V3`` for stronger reasoning. Set ``model`` to anything
listed at https://siliconflow.cn/models.
"""
from __future__ import annotations

from ._openai_base import OpenAICompatibleModel


class SiliconFlowModel(OpenAICompatibleModel):
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=kwargs.pop("base_url", self.DEFAULT_BASE_URL),
            model=model,
            **kwargs,
        )
