"""OpenAI Model adapter."""
from __future__ import annotations

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
