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

from lingyia_core.blocks import BlockKind
from lingyia_core.capability import ModelCapabilities

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

    @property
    def capabilities(self) -> ModelCapabilities:
        """Per-model context windows."""
        context_map = {
            "Pro/zai-org/GLM-5.1": 205_000,
            "Pro/moonshotai/Kimi-K2.6": 200_000,
            "deepseek-ai/DeepSeek-V4-Flash": 128_000,
        }
        return ModelCapabilities(
            model_id=self.model,
            accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
            supports_streaming=True,
            supports_parallel_tools=True,
            supports_json_schema=True,
            max_context_tokens=context_map.get(self.model, 32_000),
            max_output_tokens=8192,
        )
