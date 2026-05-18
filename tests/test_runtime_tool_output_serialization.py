"""Regression tests for tool output serialization (codex P2 runtime.py:595).

Spec §8 rules for ToolResultBlock content:
- str output → kept as str
- tuple[ContentBlock, ...] → preserved (rich tool content)
- dict / list / other JSON-serializable → json.dumps(..., ensure_ascii=False)

Before the fix, dicts were stringified via str() (Python repr), which
breaks JSON parsing for any LLM that tries to read the tool result.
Tuples of ContentBlocks were also flattened to str() instead of being
passed through to the ToolResultBlock.
"""
from __future__ import annotations

import asyncio
import json

from lingyia_core import (
    Decision,
    Harness,
    Runtime,
    Tool,
    ToolResult,
)
from lingyia_core.blocks import (
    BlockKind,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from lingyia_core.capability import ModelCapabilities


_CAPS = ModelCapabilities(
    model_id="fake",
    accepts=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE, BlockKind.TOOL_RESULT}),
    emits=frozenset({BlockKind.TEXT, BlockKind.TOOL_USE}),
)


class _ScriptedModel:
    capabilities = _CAPS

    def __init__(self, *decisions: Decision) -> None:
        self._decisions = list(decisions)

    async def adecide(self, ctx, state, tools):
        if not self._decisions:
            raise AssertionError("scripted model exhausted")
        return self._decisions.pop(0)


def _find_tool_result(state, tool_use_id):
    for msg in state.messages:
        if msg.role != Role.USER:
            continue
        for b in msg.content:
            if isinstance(b, ToolResultBlock) and b.tool_use_id == tool_use_id:
                return b
    raise AssertionError(f"no ToolResultBlock for {tool_use_id}")


def _run(tool: Tool, output) -> tuple:
    """Drive runtime through one tool call returning `output`, return final state."""
    tool_use = ToolUseBlock(id="call_1", name=tool.name, input={})
    call = Decision.call_tools([tool_use])
    fin = Decision.final_answer("done")
    rt = Runtime.dev(model=_ScriptedModel(call, fin), max_iterations=4)
    harness = Harness(tools=[tool])
    result = asyncio.run(rt.arun(harness, "go"))
    return result, _find_tool_result(result.state, "call_1")


def test_str_output_passthrough():
    def handler(args, ctx):
        return ToolResult(tool_name="t", ok=True, output="hello world")
    tool = Tool.from_sync(name="t", description="t", handler=handler)
    _, block = _run(tool, "hello world")
    assert block.content == "hello world"


def test_dict_output_json_serialized_not_repr():
    """Codex P2: dicts must become JSON, not Python repr."""
    def handler(args, ctx):
        return ToolResult(
            tool_name="t",
            ok=True,
            output={"weather": "sunny", "temp_c": 22, "city": "上海"},
        )
    tool = Tool.from_sync(name="t", description="t", handler=handler)
    _, block = _run(tool, {})
    # The content must round-trip through json.loads — Python repr cannot.
    parsed = json.loads(block.content)
    assert parsed == {"weather": "sunny", "temp_c": 22, "city": "上海"}
    # Ensure non-ASCII chars were preserved unescaped.
    assert "上海" in block.content


def test_list_output_json_serialized():
    def handler(args, ctx):
        return ToolResult(tool_name="t", ok=True, output=[1, 2, "three"])
    tool = Tool.from_sync(name="t", description="t", handler=handler)
    _, block = _run(tool, [])
    assert json.loads(block.content) == [1, 2, "three"]


def test_tuple_of_content_blocks_passthrough():
    """Tool authors who want to return rich content (e.g. text + image)
    do so by returning a tuple of ContentBlocks. The runtime must preserve
    the tuple in ToolResultBlock.content instead of stringifying it.
    """
    rich = (TextBlock(text="part1"), TextBlock(text="part2"))

    def handler(args, ctx):
        return ToolResult(tool_name="t", ok=True, output=rich)
    tool = Tool.from_sync(name="t", description="t", handler=handler)
    _, block = _run(tool, rich)
    assert isinstance(block.content, tuple)
    assert len(block.content) == 2
    assert all(isinstance(b, TextBlock) for b in block.content)


def test_error_path_falls_back_to_error_string():
    """Failed tool: ToolResultBlock content is the error string."""
    def handler(args, ctx):
        return ToolResult(tool_name="t", ok=False, error="boom")
    tool = Tool.from_sync(name="t", description="t", handler=handler)
    _, block = _run(tool, None)
    assert block.is_error is True
    assert block.content == "boom"


def test_none_ok_output_becomes_empty_string():
    def handler(args, ctx):
        return ToolResult(tool_name="t", ok=True, output=None)
    tool = Tool.from_sync(name="t", description="t", handler=handler)
    _, block = _run(tool, None)
    assert block.content == ""
    assert block.is_error is False
