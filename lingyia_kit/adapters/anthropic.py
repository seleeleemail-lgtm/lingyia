"""Anthropic (Claude) Model adapter for v0.2.

v0.2 changes from v0.1:
- Reads ``RunState.messages`` directly. The legacy logic that reconstructed
  multi-turn assistant ``tool_use`` history from ``state.observations`` is
  gone — messages are the source of truth.
- Native ContentBlock ↔ Anthropic API block mapping. ``text``, ``tool_use``,
  ``tool_result``, ``image``, and ``thinking`` are 1:1 with our union.
- ``ThinkingBlock`` is first-class via ``capabilities.emits`` and is parsed
  back from response content blocks.
- SYSTEM-role messages are extracted into payload["system"] (Anthropic puts
  the system prompt in a top-level field, not in messages).

Implementation notes:
- Pure ``httpx``. Matches the "zero hard deps beyond httpx" philosophy used
  by the OpenAI-compatible base; drops the ``anthropic`` SDK dependency.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import httpx

from lingyia_core import Decision, RunState
from lingyia_core.blocks import (
    BlockKind,
    ImageBlock,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    block_to_dict,
)
from lingyia_core.capability import ModelCapabilities
from lingyia_core.message import Message
from lingyia_core.state import DecisionKind, ModelUsage


ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicModel:
    """Claude Messages-API adapter (v0.2 native blocks)."""

    DEFAULT_MODEL = "claude-sonnet-4-5"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        base_url: str = ANTHROPIC_API_BASE,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.base_url = base_url.rstrip("/")
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Claude capabilities. THINKING in emits (extended thinking models);
        IMAGE in accepts (vision is native to claude-3+).
        """
        return ModelCapabilities(
            model_id=self.model,
            accepts=frozenset({
                BlockKind.TEXT,
                BlockKind.TOOL_USE,
                BlockKind.TOOL_RESULT,
                BlockKind.IMAGE,
            }),
            emits=frozenset({
                BlockKind.TEXT,
                BlockKind.TOOL_USE,
                BlockKind.THINKING,
            }),
            supports_streaming=True,
            supports_parallel_tools=True,
            supports_json_schema=False,
            supports_strict_schema=False,
            supports_prompt_caching=True,
            max_context_tokens=200_000,
            max_output_tokens=8192,
        )

    async def adecide(
        self,
        context: Mapping[str, Any],
        state: RunState,
        tools: Sequence[Any],
    ) -> Decision:
        anthropic_messages = self._build_anthropic_messages(state.messages)
        system_text = self._extract_system_prompt(state.messages)
        tool_schemas = self._build_tool_schemas(tools)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_text:
            payload["system"] = system_text
        if tool_schemas:
            payload["tools"] = tool_schemas
        # Caller-supplied overrides win (e.g. thinking config, top_p, ...).
        payload.update(self.extra_body)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
            **self.extra_headers,
        }

        response = await self._client.post(
            f"{self.base_url}/messages",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    # Message construction -----------------------------------------------

    def _build_anthropic_messages(
        self,
        messages: list[Message],
    ) -> list[dict[str, Any]]:
        """Translate v0.2 ``RunState.messages`` to Anthropic API messages list.

        SYSTEM-role messages are extracted separately via
        :meth:`_extract_system_prompt`; Anthropic carries them in
        ``payload["system"]`` rather than in the messages array.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                continue  # handled by _extract_system_prompt

            api_role = msg.role.value  # "user" or "assistant"
            api_content: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    api_content.append({"type": "text", "text": block.text})

                elif isinstance(block, ToolUseBlock):
                    api_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input),
                    })

                elif isinstance(block, ToolResultBlock):
                    if isinstance(block.content, str):
                        tr_content: Any = block.content
                    else:
                        tr_content = [
                            self._block_to_anthropic_dict(b) for b in block.content
                        ]
                    api_content.append({
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": tr_content,
                        "is_error": block.is_error,
                    })

                elif isinstance(block, ImageBlock):
                    src = block.source
                    if src.url:
                        api_content.append({
                            "type": "image",
                            "source": {"type": "url", "url": src.url},
                        })
                    else:
                        api_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": src.media_type,
                                "data": src.data,
                            },
                        })

                elif isinstance(block, ThinkingBlock):
                    api_content.append({
                        "type": "thinking",
                        "thinking": block.thinking,
                        "signature": block.signature,
                    })
                # Unknown block kinds are dropped; the capability policy
                # should have rejected them upstream.

            if api_content:  # skip empty messages
                out.append({"role": api_role, "content": api_content})
        return out

    def _extract_system_prompt(self, messages: list[Message]) -> str:
        """Concatenate SYSTEM-role TextBlocks into a single Anthropic system
        prompt. Non-text blocks in system messages are ignored — Anthropic's
        ``system`` field is plain text.
        """
        sys_parts = [
            b.text
            for m in messages
            if m.role == Role.SYSTEM
            for b in m.content
            if isinstance(b, TextBlock)
        ]
        return "\n".join(sys_parts)

    def _block_to_anthropic_dict(self, block: Any) -> dict[str, Any]:
        """Render a nested ContentBlock (inside a ToolResultBlock) into the
        Anthropic API shape. lingyia ContentBlock type names already align
        with Anthropic's (text/tool_use/tool_result/image/thinking), so we
        can delegate to ``block_to_dict`` for the default mapping.
        """
        return block_to_dict(block)

    def _build_tool_schemas(self, tools: Sequence[Any]) -> list[dict[str, Any]]:
        """Build Anthropic ``tools`` array. Each entry is
        ``{name, description, input_schema}``.
        """
        schemas: list[dict[str, Any]] = []
        for t in tools:
            input_schema = dict(t.input_schema) if t.input_schema else {
                "type": "object",
                "properties": {},
            }
            schemas.append({
                "name": t.name,
                "description": t.description,
                "input_schema": input_schema,
            })
        return schemas

    # Response parsing ---------------------------------------------------

    def _parse_response(self, data: Mapping[str, Any]) -> Decision:
        """Parse Anthropic Messages API JSON response into a v0.2 Decision.

        Response ``content`` is already a list of typed blocks (text /
        tool_use / thinking). We map them straight to ContentBlocks.
        """
        usage = self._parse_usage(data.get("usage"))

        content_blocks: list[Any] = []
        has_tool_use = False
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                content_blocks.append(TextBlock(text=block.get("text", "") or ""))
            elif btype == "tool_use":
                has_tool_use = True
                content_blocks.append(ToolUseBlock(
                    id=block.get("id", "") or "",
                    name=block.get("name", "") or "",
                    input=dict(block.get("input") or {}),
                ))
            elif btype == "thinking":
                content_blocks.append(ThinkingBlock(
                    thinking=block.get("thinking", "") or "",
                    signature=block.get("signature", "") or "",
                ))
            # Unknown response block types are ignored; new Anthropic blocks
            # should be added explicitly here as the contract evolves.

        if has_tool_use:
            return Decision(
                kind=DecisionKind.CALL_TOOL,
                content=tuple(content_blocks),
                usage=usage,
            )

        return Decision(
            kind=DecisionKind.FINAL_ANSWER,
            content=tuple(content_blocks) if content_blocks else (TextBlock(text=""),),
            usage=usage,
        )

    def _parse_usage(self, u: Any) -> ModelUsage:
        """Anthropic usage dict shape:
            input_tokens, output_tokens, cache_read_input_tokens,
            cache_creation_input_tokens.

        ``cache_read`` is tracked as ``cached_tokens`` (discounted). ``cache_creation``
        is billed at the regular prompt rate, so we fold it into ``prompt_tokens``.
        """
        if not u:
            return ModelUsage(model_id=self.model)
        if not isinstance(u, Mapping):
            return ModelUsage(model_id=self.model)

        input_tokens = int(u.get("input_tokens") or 0)
        output_tokens = int(u.get("output_tokens") or 0)
        cache_read = int(u.get("cache_read_input_tokens") or 0)
        cache_create = int(u.get("cache_creation_input_tokens") or 0)
        return ModelUsage(
            prompt_tokens=input_tokens + cache_create + cache_read,
            completion_tokens=output_tokens,
            cached_tokens=cache_read,
            model_id=self.model,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
