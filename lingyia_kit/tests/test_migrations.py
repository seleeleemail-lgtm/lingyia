"""Migration tests: v0.1 → v0.2 state dict."""
import pytest

from lingyia_kit.migrations.v01_to_v02 import migrate_state_dict
from lingyia_core import RunState
from lingyia_core.blocks import Role


def test_rejects_non_v01_input():
    with pytest.raises(ValueError, match="schema_version == 1"):
        migrate_state_dict({"schema_version": 2})


def test_goal_becomes_user_text_message():
    v01 = {
        "schema_version": 1,
        "goal": "Find the bug",
        "run_id": "r1",
        "observations": [],
        "feedback": [],
    }
    v02 = migrate_state_dict(v01)
    assert v02["schema_version"] == 2
    assert v02["messages"][0]["role"] == "user"
    assert v02["messages"][0]["content"][0]["type"] == "text"
    assert v02["messages"][0]["content"][0]["text"] == "Find the bug"


def test_observation_becomes_tool_use_plus_tool_result_pair():
    v01 = {
        "schema_version": 1,
        "goal": "Search docs",
        "run_id": "r1",
        "observations": [{
            "tool_call": {"id": "t1", "name": "search", "input": {"q": "x"}},
            "result": "found 3 hits",
        }],
        "feedback": [],
    }
    v02 = migrate_state_dict(v01)
    # First msg = goal as user text, then assistant tool_use, then user tool_result
    assert v02["messages"][1]["role"] == "assistant"
    assert v02["messages"][1]["content"][0]["type"] == "tool_use"
    assert v02["messages"][1]["content"][0]["id"] == "t1"
    assert v02["messages"][1]["content"][0]["name"] == "search"
    assert v02["messages"][2]["role"] == "user"
    assert v02["messages"][2]["content"][0]["type"] == "tool_result"
    assert v02["messages"][2]["content"][0]["tool_use_id"] == "t1"
    assert v02["messages"][2]["content"][0]["content"] == "found 3 hits"


def test_feedback_becomes_user_text():
    v01 = {
        "schema_version": 1,
        "goal": "ok",
        "run_id": "r1",
        "observations": [],
        "feedback": ["please retry", "be more careful"],
    }
    v02 = migrate_state_dict(v01)
    # 1 goal msg + 2 feedback msgs
    assert len(v02["messages"]) == 3
    assert v02["messages"][1]["content"][0]["text"] == "please retry"
    assert v02["messages"][2]["content"][0]["text"] == "be more careful"


def test_migrated_dict_is_loadable_by_v02_runstate():
    """Round-trip: v0.1 dict → migrate → RunState.from_dict succeeds."""
    v01 = {
        "schema_version": 1,
        "goal": "Plan trip",
        "run_id": "r-trip",
        "iteration": 3,
        "observations": [{
            "tool_call": {"id": "t-w", "name": "weather", "input": {"city": "Tokyo"}},
            "result": "sunny",
        }],
        "feedback": ["faster please"],
        "metadata": {"user": "alice"},
        "trace": [],
        "interrupt": None,
    }
    v02_dict = migrate_state_dict(v01)
    state = RunState.from_dict(v02_dict)
    assert state.schema_version == 2
    assert state.run_id == "r-trip"
    assert state.iteration == 3
    assert state.metadata == {"user": "alice"}
    # 1 goal user msg + 1 assistant tool_use + 1 user tool_result + 1 user feedback = 4
    assert len(state.messages) == 4
    assert state.messages[0].role == Role.USER
    assert state.messages[1].role == Role.ASSISTANT
    assert state.messages[2].role == Role.USER
    assert state.messages[3].role == Role.USER
