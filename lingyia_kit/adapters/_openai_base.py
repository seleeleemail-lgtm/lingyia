"""OpenAI-compatible Model adapter base.

Targets the ``/v1/chat/completions`` endpoint with OpenAI-style tool-calling.
Works for OpenAI itself, SiliconFlow, MiniMax, OpenRouter, vLLM, llama.cpp,
LM Studio, and any other provider that conforms to the same shape.

Implementation notes:
- Pure ``httpx`` client. We deliberately do not depend on the ``openai`` SDK,
  so the only required wire-level dependency is HTTP+JSON.
- v0.2 contract: conversation history is read directly from
  ``RunState.messages`` (Anthropic-style ContentBlocks). Tool results live
  inside user-role messages as ``ToolResultBlock`` and are translated to
  OpenAI's separate ``role="tool"`` messages here.
- Parallel tool calls in a single iteration are emitted as one assistant
  turn carrying multiple ``tool_calls`` followed by N tool messages,
  matching the OpenAI protocol.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

import httpx

from lingyia_core import Decision, RunState
from lingyia_core.blocks import (
    BlockKind,
    ImageBlock,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    TruncationBlock,
    block_to_dict,
)
from lingyia_core.capability import ModelCapabilities
from lingyia_core.state import DecisionKind, ModelUsage


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI agent. Use the available tools to accomplish the user's goal. "
    "When the goal is fully accomplished, respond with a final answer instead of calling a tool."
)


def _truncation_to_text(b: TruncationBlock) -> str:
    """Spec §8: flatten TruncationBlock to text at the provider boundary."""
    return f"[earlier {b.count} messages omitted]"


class OpenAICompatibleModel:
    """Base adapter for any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Conservative defaults for any OpenAI-compatible endpoint.

        Subclasses (OpenAIModel, MiniMax, SiliconFlow) override this to
        declare vision support and per-provider context window sizes. The
        base class advertises the common safe minimum: text + function
        calling, no vision, no thinking.
        """
        return ModelCapabilities(
            model_id=self.model,
            accepts=frozenset({
                BlockKind.TEXT,
                BlockKind.TOOL_USE,
                BlockKind.TOOL_RESULT,
            }),
            emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
            supports_streaming=True,
            supports_parallel_tools=True,
            supports_json_schema=False,
            supports_strict_schema=False,
            supports_prompt_caching=False,
            max_context_tokens=8192,
            max_output_tokens=4096,
        )

    async def adecide(
        self,
        context: Mapping[str, Any],
        state: RunState,
        tools: Sequence[Any],
    ) -> Decision:
        messages = self._build_messages(context, state)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        tool_schemas = self._build_tool_schemas(tools)
        if tool_schemas:
            payload["tools"] = tool_schemas
            payload["tool_choice"] = "auto"
        # Caller-supplied overrides win (e.g. response_format, top_p, ...).
        payload.update(self.extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        response = await self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    # Message construction -----------------------------------------------

    def _build_messages(
        self,
        context: Mapping[str, Any],
        state: RunState,
    ) -> list[dict[str, Any]]:
        """v0.2: read state.messages, translate to OpenAI chat completion format.

        Per spec §8.1, TruncationBlock is flattened to text at the wire
        boundary. For each message we walk content in declaration order:
        runs of non-truncation blocks are projected by the role-specific
        logic; each TruncationBlock is emitted as its own system wire entry
        between those runs. This preserves block-declaration order so
        mixed-content messages (e.g. system instructions followed by a
        truncation marker) do not silently drop either side.
        """
        out: list[dict[str, Any]] = []
        for msg in state.messages:
            # Partition this message's content into runs separated by
            # TruncationBlocks. Each run is projected by the role-specific
            # helper; each TruncationBlock becomes a standalone system entry.
            run: list[Any] = []
            for block in msg.content:
                if isinstance(block, TruncationBlock):
                    if run:
                        out.extend(self._project_role_blocks(msg.role, run))
                        run = []
                    out.append({"role": "system", "content": _truncation_to_text(block)})
                else:
                    run.append(block)
            if run:
                out.extend(self._project_role_blocks(msg.role, run))
        return out

    def _project_role_blocks(
        self,
        role: Role,
        blocks: list[Any],
    ) -> list[dict[str, Any]]:
        """Project a single role's run of non-TruncationBlock content to wire entries.

        This is the original ``_build_messages`` per-role logic, extracted so
        ``_build_messages`` can splice TruncationBlock flattening between runs
        without losing any handling for TextBlock / ToolUseBlock /
        ToolResultBlock / ImageBlock.
        """
        out: list[dict[str, Any]] = []

        if role == Role.SYSTEM:
            text = "".join(b.text for b in blocks if isinstance(b, TextBlock))
            out.append({"role": "system", "content": text})

        elif role == Role.USER:
            # User messages may contain TextBlock OR ToolResultBlock
            # (Anthropic style). OpenAI requires tool results as separate
            # role="tool" messages, so we split them out here.
            text_parts: list[Any] = []
            for block in blocks:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ToolResultBlock):
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": self._flatten_tool_result(block.content),
                    })
                elif isinstance(block, ImageBlock):
                    text_parts.append(self._image_to_openai_part(block))
            if text_parts:
                if all(isinstance(p, str) for p in text_parts):
                    out.append({"role": "user", "content": "\n".join(text_parts)})
                else:
                    # Has images — structured content array
                    structured: list[dict[str, Any]] = []
                    for p in text_parts:
                        if isinstance(p, str):
                            structured.append({"type": "text", "text": p})
                        else:
                            structured.append(p)
                    out.append({"role": "user", "content": structured})

        elif role == Role.ASSISTANT:
            text = "".join(b.text for b in blocks if isinstance(b, TextBlock))
            tool_uses = [b for b in blocks if isinstance(b, ToolUseBlock)]
            entry: dict[str, Any] = {"role": "assistant"}
            entry["content"] = text or None
            if tool_uses:
                entry["tool_calls"] = [
                    {
                        "id": tu.id,
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(dict(tu.input), ensure_ascii=False),
                        },
                    }
                    for tu in tool_uses
                ]
            out.append(entry)

        return out

    def _flatten_tool_result(self, content: Any) -> str:
        """OpenAI tool message content is a string; flatten nested blocks if any."""
        if isinstance(content, str):
            return content
        text_parts: list[str] = []
        json_fallback: list[str] = []
        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            else:
                json_fallback.append(json.dumps(block_to_dict(block), ensure_ascii=False))
        return "\n".join(text_parts + json_fallback)

    def _image_to_openai_part(self, block: ImageBlock) -> dict[str, Any]:
        """Convert ImageBlock to OpenAI vision content part."""
        if block.source.url:
            return {"type": "image_url", "image_url": {"url": block.source.url}}
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{block.source.media_type};base64,{block.source.data}",
            },
        }

    def _build_tool_schemas(self, tools: Sequence[Any]) -> list[dict[str, Any]]:
        schemas = []
        for t in tools:
            input_schema = dict(t.input_schema) if t.input_schema else {
                "type": "object",
                "properties": {},
            }
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": input_schema,
                },
            })
        return schemas

    # Response parsing ---------------------------------------------------

    def _parse_response(self, data: Mapping[str, Any]) -> Decision:
        """Parse OpenAI chat completion response into a v0.2 Decision."""
        choices = data.get("choices") or []
        if not choices:
            return Decision.final_answer("")
        msg = choices[0].get("message") or {}
        usage = self._parse_usage(data)

        text_content = msg.get("content") or ""
        tool_calls_raw = msg.get("tool_calls") or []

        content_blocks: list[Any] = []
        if text_content:
            content_blocks.append(TextBlock(text=text_content))

        if tool_calls_raw:
            for tc in tool_calls_raw:
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        # Some smaller models emit malformed JSON; degrade
                        # gracefully so the loop sees a real failure instead
                        # of crashing.
                        args = {"_raw": raw_args}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                content_blocks.append(ToolUseBlock(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    input=args,
                ))
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

    def _parse_usage(self, body: Mapping[str, Any]) -> ModelUsage:
        u = body.get("usage") or {}
        details = u.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or 0)
        return ModelUsage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            cached_tokens=cached,
            model_id=self.model,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
