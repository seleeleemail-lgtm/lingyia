"""Regression tests for v0.1 → v0.2 migration (codex P2 v01_to_v02.py:48, :52).

Two codex findings:

[:48] Migration corrupts non-tool observations and common v0.1 payload shapes.
Before: every Observation became a fake assistant tool_use + user tool_result
pair, regardless of the observation's ``kind``. ``Observation(kind="text",
payload="...")`` was migrated as a phantom tool call. ``payload={"call_id":
..., "output": ...}`` shapes were ignored.

[:52] Migration could create duplicate tool_use_ids.
Before: if a v0.1 snapshot already had two observations with the same id
(possible in older runs that didn't guarantee uniqueness), the migration
preserved them — producing ambiguous tool_result cross-references.
"""
from __future__ import annotations

import re

from lingyia_kit.migrations.v01_to_v02 import migrate_state_dict


# ---------------------------------------------------------------------------
# kind="text" observation (P2 :48 part 1)
# ---------------------------------------------------------------------------


def test_text_observation_becomes_assistant_text_block():
    """A v0.1 observation with kind=="text" represents plain reasoning, not
    a tool call. It MUST migrate to an assistant TextBlock — not a fake
    tool_use/tool_result pair.
    """
    v01 = {
        "schema_version": 1,
        "goal": "ok",
        "run_id": "r",
        "observations": [{
            "kind": "text",
            "payload": "I considered this and decided to skip",
        }],
        "feedback": [],
    }
    v02 = migrate_state_dict(v01)
    # First msg is the goal; the second should be a single assistant TextBlock,
    # NOT a tool_use.
    assert v02["messages"][1]["role"] == "assistant"
    assert len(v02["messages"][1]["content"]) == 1
    assert v02["messages"][1]["content"][0]["type"] == "text"
    assert v02["messages"][1]["content"][0]["text"] == (
        "I considered this and decided to skip"
    )
    # No spurious tool_result message inserted.
    assert all(
        block["type"] != "tool_result"
        for m in v02["messages"]
        for block in m["content"]
    )


# ---------------------------------------------------------------------------
# kind="tool" with payload-nested tool fields (P2 :48 part 2)
# ---------------------------------------------------------------------------


def test_tool_observation_with_payload_call_id_output_shape():
    """Some v0.1 runs stored tool fields under ``payload`` instead of
    top-level. Both shapes must migrate correctly.

    payload shape: ``{"call_id": "...", "output": "...", "name": "...", "input": {...}}``
    """
    v01 = {
        "schema_version": 1,
        "goal": "search docs",
        "run_id": "r",
        "observations": [{
            "kind": "tool",
            "payload": {
                "call_id": "t-from-payload",
                "name": "search",
                "input": {"q": "lingyia"},
                "output": "found 5 hits",
            },
        }],
        "feedback": [],
    }
    v02 = migrate_state_dict(v01)
    # Find the assistant tool_use message.
    use = next(
        b for m in v02["messages"] for b in m["content"]
        if b["type"] == "tool_use"
    )
    assert use["id"] == "t-from-payload"
    assert use["name"] == "search"
    assert use["input"] == {"q": "lingyia"}
    # And the user tool_result message.
    res = next(
        b for m in v02["messages"] for b in m["content"]
        if b["type"] == "tool_result"
    )
    assert res["tool_use_id"] == "t-from-payload"
    assert res["content"] == "found 5 hits"


def test_tool_observation_top_level_fields_still_supported():
    """The original happy-path shape (tool_call dict + result string) must
    keep working — backwards compat in the migration itself.
    """
    v01 = {
        "schema_version": 1,
        "goal": "ok",
        "run_id": "r",
        "observations": [{
            "tool_call": {"id": "t1", "name": "search", "input": {"q": "x"}},
            "result": "found",
        }],
        "feedback": [],
    }
    v02 = migrate_state_dict(v01)
    use = next(b for m in v02["messages"] for b in m["content"] if b["type"] == "tool_use")
    res = next(b for m in v02["messages"] for b in m["content"] if b["type"] == "tool_result")
    assert use["id"] == "t1"
    assert use["name"] == "search"
    assert res["tool_use_id"] == "t1"
    assert res["content"] == "found"


def test_observation_with_unknown_kind_does_not_crash():
    """Robustness: an unrecognized observation kind should produce a
    plain assistant text observation rather than crash mid-migration.
    """
    v01 = {
        "schema_version": 1,
        "goal": "ok",
        "run_id": "r",
        "observations": [{"kind": "weird", "payload": "junk"}],
        "feedback": [],
    }
    # Should not raise.
    v02 = migrate_state_dict(v01)
    # No tool_use/tool_result faked.
    types = [b["type"] for m in v02["messages"] for b in m["content"]]
    assert "tool_use" not in types
    assert "tool_result" not in types


# ---------------------------------------------------------------------------
# duplicate id collision handling (P2 :52)
# ---------------------------------------------------------------------------


def test_duplicate_tool_use_ids_are_disambiguated():
    """Two observations with the same id MUST end up with distinct ids in
    the migrated transcript so tool_result cross-refs are unambiguous.
    The mapping between old and new ids must keep the tool_use/tool_result
    pairs together.
    """
    v01 = {
        "schema_version": 1,
        "goal": "ok",
        "run_id": "r",
        "observations": [
            {
                "tool_call": {"id": "dup", "name": "search", "input": {"q": "a"}},
                "result": "first",
            },
            {
                "tool_call": {"id": "dup", "name": "search", "input": {"q": "b"}},
                "result": "second",
            },
        ],
        "feedback": [],
    }
    v02 = migrate_state_dict(v01)

    # Collect tool_use ids in order, tool_result ids in order.
    use_ids = [
        b["id"]
        for m in v02["messages"] for b in m["content"]
        if b["type"] == "tool_use"
    ]
    result_pairs = [
        (b["tool_use_id"], b["content"])
        for m in v02["messages"] for b in m["content"]
        if b["type"] == "tool_result"
    ]

    # Two tool_use blocks with distinct ids.
    assert len(use_ids) == 2
    assert len(set(use_ids)) == 2, f"duplicate tool_use ids leaked: {use_ids}"

    # Each tool_result must reference one of the new ids and preserve order.
    assert len(result_pairs) == 2
    assert result_pairs[0][0] == use_ids[0]
    assert result_pairs[0][1] == "first"
    assert result_pairs[1][0] == use_ids[1]
    assert result_pairs[1][1] == "second"
