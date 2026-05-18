"""OpenAI Model adapter."""
from __future__ import annotations

from lingyia_core.blocks import BlockKind
from lingyia_core.capability import ModelCapabilities

from ._openai_base import OpenAICompatibleModel


class OpenAIModel(OpenAICompatibleModel):
    """OpenAI ChatGPT API.

    Default model is ``gpt-4o-mini`` for cost-controlled experimentation;
    override via the ``model`` kwarg.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, **kwargs):
        super().__init__(
            api_key=api_key,
            base_url=kwargs.pop("base_url", self.DEFAULT_BASE_URL),
            model=model,
            **kwargs,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Map known OpenAI model_id → capabilities."""
        if self.model.startswith("gpt-4o") or "vision" in self.model:
            accepts = frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT, BlockKind.IMAGE})
        else:
            accepts = frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT})
        return ModelCapabilities(
            model_id=self.model,
            accepts=accepts,
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
            supports_streaming=True,
            supports_parallel_tools=True,
            supports_json_schema=True,
            supports_strict_schema=True,
            supports_prompt_caching=True,
            max_context_tokens=128_000,
            max_output_tokens=16_384,
        )
