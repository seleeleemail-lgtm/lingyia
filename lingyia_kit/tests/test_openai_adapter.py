"""OpenAI adapter tests — v0.2 contract."""
import json

from lingyia_core import RunState
from lingyia_core.state import DecisionKind
from lingyia_core.blocks import (
    Role,
    BlockKind,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)
from lingyia_core.message import Message
from lingyia_kit.adapters.openai import OpenAIModel
from lingyia_kit.adapters._openai_base import OpenAICompatibleModel


def test_build_messages_simple_user_text():
    m = OpenAICompatibleModel(api_key="x", base_url="http://x", model="gpt-4")
    state = RunState(messages=[
        Message(role=Role.USER, content=(TextBlock(text="hi"),)),
    ])
    built = m._build_messages({}, state)
    assert built == [{"role": "user", "content": "hi"}]


def test_build_messages_system_message_included():
    m = OpenAICompatibleModel(api_key="x", base_url="http://x", model="gpt-4")
    state = RunState(messages=[
        Message(role=Role.SYSTEM, content=(TextBlock(text="be helpful"),)),
        Message(role=Role.USER, content=(TextBlock(text="hi"),)),
    ])
    built = m._build_messages({}, state)
    assert built[0]["role"] == "system"
    assert built[0]["content"] == "be helpful"
    assert built[1]["role"] == "user"


def test_build_messages_assistant_with_tool_use():
    m = OpenAICompatibleModel(api_key="x", base_url="http://x", model="gpt-4")
    state = RunState(messages=[
        Message(role=Role.USER, content=(TextBlock(text="search"),)),
        Message(role=Role.ASSISTANT, content=(
            TextBlock(text="searching..."),
            ToolUseBlock(id="t1", name="search", input={"q": "x"}),
        )),
    ])
    built = m._build_messages({}, state)
    assert built[1]["role"] == "assistant"
    assert built[1]["content"] == "searching..."
    assert built[1]["tool_calls"][0]["function"]["name"] == "search"
    parsed_args = json.loads(built[1]["tool_calls"][0]["function"]["arguments"])
    assert parsed_args == {"q": "x"}


def test_build_messages_tool_result_becomes_tool_role():
    """ToolResultBlock in USER role -> OpenAI role='tool' separate message."""
    m = OpenAICompatibleModel(api_key="x", base_url="http://x", model="gpt-4")
    state = RunState(messages=[
        Message(role=Role.USER, content=(
            ToolResultBlock(tool_use_id="t1", content="sunny"),
        )),
    ])
    built = m._build_messages({}, state)
    assert built[0]["role"] == "tool"
    assert built[0]["tool_call_id"] == "t1"
    assert built[0]["content"] == "sunny"


def test_parse_response_final_answer():
    m = OpenAICompatibleModel(api_key="x", base_url="http://x", model="gpt-4")
    resp = {
        "choices": [{"message": {"content": "done", "tool_calls": []}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    decision = m._parse_response(resp)
    assert decision.kind == DecisionKind.FINAL_ANSWER
    assert decision.text == "done"


def test_parse_response_tool_calls():
    m = OpenAICompatibleModel(api_key="x", base_url="http://x", model="gpt-4")
    resp = {
        "choices": [{
            "message": {
                "content": "checking",
                "tool_calls": [{
                    "id": "t1",
                    "function": {"name": "weather", "arguments": '{"city":"shanghai"}'},
                }],
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    decision = m._parse_response(resp)
    assert decision.kind == DecisionKind.CALL_TOOL
    assert decision.tool_calls[0].name == "weather"
    assert decision.tool_calls[0].input == {"city": "shanghai"}


def test_capabilities_gpt4o_includes_image():
    m = OpenAIModel(api_key="x", model="gpt-4o")
    assert BlockKind.IMAGE in m.capabilities.accepts


def test_capabilities_gpt35_text_only():
    m = OpenAIModel(api_key="x", model="gpt-3.5-turbo")
    assert BlockKind.IMAGE not in m.capabilities.accepts
