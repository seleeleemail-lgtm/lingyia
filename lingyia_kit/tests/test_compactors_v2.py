"""Regression tests for TokenAwareCompactor v2 (codex P2 token_aware.py:117 + :133).

Two distinct bugs were called out:

1. Truncation markers used Role.SYSTEM and were preserved across rounds.
   Repeated compaction can stack many "[earlier N truncated]" markers,
   and a SYSTEM-only over-budget state never terminates (compaction
   removes nothing but still claims to have compacted).

2. _adjust_for_tool_pairing only handled "kept ToolResultBlock with no
   kept ToolUseBlock" by orphan-dropping. It did NOT keep tool_use +
   tool_result(s) as an atomic group; a tool_use orphaned by truncation
   on the OTHER side (kept tool_use, dropped tool_result) was left
   dangling, and the cut boundary could split a group.

This module pins the post-fix behavior:
- compaction is by atomic transcript "groups" (assistant tool_use turn +
  the following user tool_result(s));
- the truncation marker is replaced (not stacked) on each round;
- a state where the system messages alone exceed budget exits cleanly
  (no infinite truncation-marker accumulation).
"""
from __future__ import annotations

import pytest

from lingyia_core import RunState
from lingyia_core.blocks import Role, TextBlock, ToolResultBlock, ToolUseBlock, TruncationBlock
from lingyia_core.message import Message
from lingyia_kit.compactors.token_aware import (
    TokenAwareCompactor,
    char_div4_estimator,
)


def _u(text: str) -> Message:
    return Message(role=Role.USER, content=(TextBlock(text=text),))


def _a(text: str) -> Message:
    return Message(role=Role.ASSISTANT, content=(TextBlock(text=text),))


def _a_tool(call_id: str, name: str = "f", text: str = "") -> Message:
    blocks: list = []
    if text:
        blocks.append(TextBlock(text=text))
    blocks.append(ToolUseBlock(id=call_id, name=name, input={}))
    return Message(role=Role.ASSISTANT, content=tuple(blocks))


def _u_tool_result(call_id: str, content: str = "ok") -> Message:
    return Message(role=Role.USER, content=(ToolResultBlock(tool_use_id=call_id, content=content),))


# ---------------------------------------------------------------------------
# Marker hygiene (codex P2 :117)
# ---------------------------------------------------------------------------


def test_repeated_compaction_does_not_stack_truncation_markers():
    """Compact, append more chatter, compact again. Marker count stays at 1."""
    c = TokenAwareCompactor(
        max_tokens=80, keep_last_turns=1, token_estimator=char_div4_estimator
    )
    # Build enough content to force compaction on each round.
    msgs = []
    for i in range(20):
        msgs.append(_u(f"user msg {i} " * 8))
        msgs.append(_a(f"assistant msg {i} " * 8))
    state = RunState(messages=msgs, run_id="x")
    assert c.should_compact(state)

    s1 = c.compact(state)
    marker_count_1 = sum(
        1 for m in s1.messages
        if m.role == Role.SYSTEM
        and any(isinstance(b, TruncationBlock) for b in m.content)
    )
    assert marker_count_1 == 1

    # Simulate further chatter and recompact.
    for i in range(20, 40):
        s1.messages.append(_u(f"user msg {i} " * 8))
        s1.messages.append(_a(f"assistant msg {i} " * 8))
    assert c.should_compact(s1)
    s2 = c.compact(s1)

    marker_count_2 = sum(
        1 for m in s2.messages
        if m.role == Role.SYSTEM
        and any(isinstance(b, TruncationBlock) for b in m.content)
    )
    assert marker_count_2 == 1, "truncation markers stacked across rounds"


def test_system_alone_over_budget_terminates_cleanly():
    """If the original system prompt alone exceeds budget, compaction must
    return a state without spinning or accumulating markers.

    This is the 'SYSTEM-only over budget' case: there's nothing to drop,
    so the compactor should make a finite decision (return state unchanged
    OR keep system + emit a single marker — either is acceptable; what
    must NOT happen is an infinite recursion or an accumulating marker).
    """
    c = TokenAwareCompactor(
        max_tokens=10, keep_last_turns=1, token_estimator=char_div4_estimator
    )
    big_system = Message(
        role=Role.SYSTEM,
        content=(TextBlock(text="x" * 1000),),
    )
    state = RunState(messages=[big_system], run_id="x")
    assert c.should_compact(state)

    s1 = c.compact(state)
    # Must not have added many markers
    marker_count = sum(
        1 for m in s1.messages
        if m.role == Role.SYSTEM
        and any(isinstance(b, TextBlock) and "truncated" in b.text for b in m.content)
    )
    assert marker_count <= 1

    # Compaction is idempotent in the SYSTEM-only case: repeat shouldn't
    # change the message count.
    s2 = c.compact(s1)
    assert len(s2.messages) == len(s1.messages)


def test_truncation_marker_uses_dedicated_metadata():
    """The marker must be distinguishable from a real system prompt so
    subsequent compactions can replace it instead of treating it as
    untouchable system context.

    We check via the Message instance role/content shape produced by the
    compactor — the exact predicate is documented on the compactor.
    """
    c = TokenAwareCompactor(
        max_tokens=80, keep_last_turns=1, token_estimator=char_div4_estimator
    )
    msgs = []
    for i in range(20):
        msgs.append(_u(f"user msg {i} " * 8))
        msgs.append(_a(f"assistant msg {i} " * 8))
    state = RunState(messages=msgs, run_id="x")
    s1 = c.compact(state)

    # The marker must be detectable via the compactor's own predicate.
    markers = [m for m in s1.messages if TokenAwareCompactor.is_truncation_marker(m)]
    assert len(markers) == 1


# ---------------------------------------------------------------------------
# Atomic tool-group keep/drop (codex P2 :133)
# ---------------------------------------------------------------------------


def test_tool_group_kept_or_dropped_atomically_with_kept_tool_use():
    """If the compactor would otherwise keep a tool_use turn, it must keep
    the following tool_result turn(s) too. A dangling assistant tool_use
    with no matching user tool_result is a protocol violation for every
    model.
    """
    # Layout: a lot of chatter, then one tool group, then short tail.
    msgs = []
    for i in range(10):
        msgs.append(_u(f"old chatter {i} " * 8))
        msgs.append(_a(f"old reply {i} " * 8))
    msgs.append(_a_tool("call_1", name="search", text="looking..."))
    msgs.append(_u_tool_result("call_1", content="found"))
    msgs.append(_a("here is the answer"))
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(
        max_tokens=80, keep_last_turns=2, token_estimator=char_div4_estimator
    )
    assert c.should_compact(state)
    new_state = c.compact(state)

    has_tool_use = any(
        isinstance(b, ToolUseBlock) and b.id == "call_1"
        for m in new_state.messages for b in m.content
    )
    has_tool_result = any(
        isinstance(b, ToolResultBlock) and b.tool_use_id == "call_1"
        for m in new_state.messages for b in m.content
    )
    # The whole group is kept or dropped together — never split.
    assert has_tool_use == has_tool_result


def test_dangling_tool_use_at_end_of_state_dropped_or_kept_with_result():
    """Codex finding: ``_adjust_for_tool_pairing`` did not handle a kept
    dangling assistant ``tool_use`` with no paired result.

    Layout: many old chatter messages, then a final assistant ``tool_use``
    whose tool_result does NOT exist yet (e.g. the run was checkpointed
    between issuing the tool call and the tool executing). The compactor
    keeps recent messages including this dangling tool_use. It should
    either keep it (because it's at the tail and represents pending state)
    OR drop it — but never leave any kept tool_use whose paired
    tool_result is in the kept tail too if there is no result at all.

    The atomic-group rule means: a tool_use without a matching tool_result
    is its own group; the compactor may keep or drop it as one unit, but
    must not interpret it as a partial broken group.
    """
    msgs = []
    for i in range(15):
        msgs.append(_u(f"chatter {i} " * 6))
        msgs.append(_a(f"reply {i} " * 6))
    # Trailing dangling tool_use: agent issued the call but the run was
    # interrupted before the tool result came back.
    msgs.append(_a_tool("call_pending", name="search"))
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(
        max_tokens=80, keep_last_turns=1, token_estimator=char_div4_estimator
    )
    new_state = c.compact(state)

    use_ids = {
        b.id
        for m in new_state.messages for b in m.content
        if isinstance(b, ToolUseBlock)
    }
    result_ids = {
        b.tool_use_id
        for m in new_state.messages for b in m.content
        if isinstance(b, ToolResultBlock)
    }
    # If we kept the pending tool_use, there can never be a result for it
    # (it's an interrupt-state). So we either dropped it entirely or kept
    # it standalone; the invariant is: no broken pairs in either direction.
    if "call_pending" in use_ids:
        # It's the only one; result-side must be empty (it never happened).
        assert result_ids == set()
    else:
        # Or it was dropped — also fine.
        assert "call_pending" not in result_ids


def test_tool_group_with_dangling_tool_use_at_cut_boundary_dropped():
    """The 'dangling assistant tool_use' case the codex review called out:
    the cut boundary falls in the middle of a group (kept tool_use, dropped
    tool_result). The compactor must back the cut up so the whole group
    is dropped — not leave a dangling assistant tool_use.
    """
    msgs = []
    # Old chatter + tool group inside dropped region:
    msgs.append(_u("very old"))
    msgs.append(_a_tool("call_old", name="f"))
    msgs.append(_u_tool_result("call_old"))
    # Then a long stretch of chatter that pushes us over budget and where
    # the natural cut point lands between the new tool_use and its result.
    for i in range(5):
        msgs.append(_u(f"chatter {i} " * 6))
        msgs.append(_a(f"reply {i} " * 6))
    msgs.append(_a_tool("call_new", name="f"))
    msgs.append(_u_tool_result("call_new"))
    msgs.append(_a("done"))
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(
        max_tokens=80, keep_last_turns=1, token_estimator=char_div4_estimator
    )
    new_state = c.compact(state)

    # Check no dangling tool_use:
    use_ids = {
        b.id
        for m in new_state.messages for b in m.content
        if isinstance(b, ToolUseBlock)
    }
    result_ids = {
        b.tool_use_id
        for m in new_state.messages for b in m.content
        if isinstance(b, ToolResultBlock)
    }
    # Every kept tool_use must have a kept tool_result.
    assert use_ids.issubset(result_ids), (
        f"dangling tool_use ids: {use_ids - result_ids}"
    )
    # And the reverse: every kept tool_result must have a kept tool_use.
    assert result_ids.issubset(use_ids), (
        f"orphan tool_result ids: {result_ids - use_ids}"
    )


# ---------------------------------------------------------------------------
# Budget-driven partition (codex verify P2 :275)
# ---------------------------------------------------------------------------


def _estimate_state_tokens(state: RunState, estimator) -> int:
    from lingyia_kit.compactors.token_aware import _estimate_message_tokens
    return sum(_estimate_message_tokens(m, estimator) for m in state.messages)


def test_compaction_respects_token_budget():
    """Codex verify P2 (token_aware.py:275): _partition_groups used to drop
    by message count (keep_last_turns / keep_recent_messages) instead of
    token budget. After ``compact()`` the total token count could still
    exceed ``max_tokens`` — making ``should_compact()`` keep returning True
    and (if the caller loops) burning iterations on no-progress compactions.

    With the budget-driven fix, a single compact() call brings the state
    at or below max_tokens (the kept tail + system + marker all fit) so
    long as the soft floor is reachable within budget. This test sizes
    the soft floor to be smaller than the budget so we're outside the
    irreducible-floor regime.
    """
    # Large history of ordinary chatter (≈30 tokens per message) and a
    # budget that allows the soft floor (2 messages ≈ 60 tokens) plus
    # a comfortable margin for the truncation marker (≈18 tokens).
    msgs = []
    for i in range(40):
        msgs.append(_u(f"user msg {i} " * 12))  # ~36 chars → ~9 tokens
        msgs.append(_a(f"assistant msg {i} " * 12))
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(
        max_tokens=200,
        keep_last_turns=1,
        keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )
    assert c.should_compact(state)

    new_state = c.compact(state)
    new_total = _estimate_state_tokens(new_state, char_div4_estimator)
    assert new_total <= c.max_tokens, (
        f"compaction did not bring state within budget: "
        f"new_total={new_total} > max_tokens={c.max_tokens}"
    )
    assert not c.should_compact(new_state), (
        "should_compact() still returns True after compact() — caller would "
        "loop forever in pathological setups"
    )


def test_system_alone_over_budget_is_idempotent():
    """If SYSTEM messages alone exceed the budget, compact() must be
    idempotent: calling it again yields the same state (no marker
    accumulation, no loss of system prompt, no oscillation).

    should_compact() may legitimately stay True in this irreducible-floor
    case — the compactor documents that it cannot reduce below SYSTEM+tail.
    What it MUST NOT do is mutate state across repeated calls.
    """
    big_system = Message(
        role=Role.SYSTEM,
        content=(TextBlock(text="x" * 4000),),
    )
    # Add a couple of non-system messages so the tail is non-empty.
    state = RunState(
        messages=[big_system, _u("hi"), _a("hello")],
        run_id="x",
    )
    c = TokenAwareCompactor(
        max_tokens=10, keep_last_turns=1, token_estimator=char_div4_estimator
    )
    assert c.should_compact(state)

    s1 = c.compact(state)
    s2 = c.compact(s1)
    s3 = c.compact(s2)

    # Idempotence: messages are identical (by role + text) across rounds.
    def _shape(st):
        return [
            (m.role, tuple(getattr(b, "text", str(type(b).__name__)) for b in m.content))
            for m in st.messages
        ]
    assert _shape(s1) == _shape(s2) == _shape(s3), (
        "compact() is not idempotent on the SYSTEM-alone-over-budget floor; "
        f"s1={_shape(s1)} s2={_shape(s2)} s3={_shape(s3)}"
    )


def test_dropped_groups_inserts_single_marker_not_multiple():
    """Compact then compact again on an even larger transcript. Across N
    rounds there must be at most ONE truncation marker in the final state
    (the latest), never accumulating one per round.

    Distinct from test_repeated_compaction_does_not_stack_truncation_markers
    above by stressing the count-based marker placement under budget
    pressure (large dropped span, small kept tail).
    """
    msgs = []
    for i in range(60):
        msgs.append(_u(f"u{i} " * 10))
        msgs.append(_a(f"a{i} " * 10))
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(
        max_tokens=40,
        keep_last_turns=1,
        keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )
    s1 = c.compact(state)
    # Add more chatter, compact again.
    for i in range(60, 120):
        s1.messages.append(_u(f"u{i} " * 10))
        s1.messages.append(_a(f"a{i} " * 10))
    s2 = c.compact(s1)
    # And a third time.
    for i in range(120, 180):
        s2.messages.append(_u(f"u{i} " * 10))
        s2.messages.append(_a(f"a{i} " * 10))
    s3 = c.compact(s2)

    marker_count = sum(
        1 for m in s3.messages
        if TokenAwareCompactor.is_truncation_marker(m)
    )
    assert marker_count == 1, (
        f"truncation markers accumulated across rounds: {marker_count}"
    )


# ---------------------------------------------------------------------------
# Codex round 3 P2 — marker cost + idempotence preservation
# ---------------------------------------------------------------------------


def test_compaction_actually_lands_under_budget():
    """Codex round 3 P2 (token_aware.py:196): the budget partition must
    account for the truncation marker's own token cost.

    Reproducer: 120 small messages, max_tokens=100, default char/4
    estimator. Pre-fix, compact() returned a state at 118 tokens — over
    budget — because the marker (~25 tokens) was inserted AFTER partition.

    Post-fix, the kept tail + system must leave room for the marker, so
    the final state fits under max_tokens (allowing a small overhead
    for marker token-estimate jitter — the marker text itself is ~25
    char/4 tokens).
    """
    msgs = []
    for _ in range(60):
        msgs.append(_u("x" * 20))
        msgs.append(_a("y" * 20))
    c = TokenAwareCompactor(
        max_tokens=100,
        keep_last_turns=1,
        keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )
    state = RunState(messages=msgs, run_id="x")
    s1 = c.compact(state)

    from lingyia_kit.compactors.token_aware import _estimate_message_tokens
    total = sum(_estimate_message_tokens(m, char_div4_estimator) for m in s1.messages)

    # Post-fix invariant: compaction lands under budget on its first call.
    # should_compact() reports False on the result of compact().
    assert total <= c.max_tokens, (
        f"compact() left state above budget: total={total}, "
        f"max_tokens={c.max_tokens}; marker cost was not accounted for"
    )
    assert not c.should_compact(s1), (
        f"compact() result still flags should_compact=True (total={total})"
    )


def test_existing_marker_preserved_when_no_new_drops():
    """Codex round 3 P2 (token_aware.py:170): if the input state already
    carries a truncation marker (recording that history was previously
    truncated), a subsequent compact() that drops zero new groups must
    PRESERVE the existing marker. Otherwise the audit trail of past
    truncation is silently lost on every compact() roundtrip.

    Reproducer: an over-budget SYSTEM plus modest tail; compact() makes
    the state "kept floor" sized; a second compact() on that result must
    return an identical message sequence (idempotent), including the
    marker.
    """
    big = Message(role=Role.SYSTEM, content=(TextBlock(text="x" * 4000),))
    msgs = [big]
    for i in range(10):
        msgs.append(_u(str(i)))
        msgs.append(_a(str(i)))
    c = TokenAwareCompactor(
        max_tokens=10,
        keep_last_turns=1,
        keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )
    state = RunState(messages=msgs, run_id="x")

    s1 = c.compact(state)
    s2 = c.compact(s1)

    # Marker present in s1 (history WAS dropped on the first compact).
    s1_markers = [m for m in s1.messages if TokenAwareCompactor.is_truncation_marker(m)]
    assert len(s1_markers) == 1, "first compact should have dropped history and emitted a marker"

    # And it must survive into s2 even though s2's compact dropped 0 new groups.
    s2_markers = [m for m in s2.messages if TokenAwareCompactor.is_truncation_marker(m)]
    assert len(s2_markers) == 1, (
        "existing marker was dropped on second compact when no new groups dropped; "
        "audit trail of prior truncation is lost"
    )

    # Stronger invariant: s1 and s2 are message-equal (true idempotence).
    def _shape(st):
        return [
            (m.role, tuple(getattr(b, "text", str(type(b).__name__)) for b in m.content))
            for m in st.messages
        ]
    assert _shape(s1) == _shape(s2), (
        f"compact() not idempotent across rounds: s1={_shape(s1)} vs s2={_shape(s2)}"
    )


# ---------------------------------------------------------------------------
# Codex round 4 P2 — cumulative marker count + non-monotonic estimator
# ---------------------------------------------------------------------------


def test_marker_count_accumulates_across_compactions():
    """Codex round 4 P2 (token_aware.py:243): the truncation marker's
    dropped-count must be CUMULATIVE across compact() rounds.

    Reproducer: compact once (drops N messages → marker count N). Append
    more chatter, compact again (drops another M messages). The new
    marker must read N+M, not M. Previously the count was per-round so
    the audit trail of total lost history was destroyed each round.
    """
    def _msg(role, text):
        return Message(role=role, content=(TextBlock(text=text),))

    c = TokenAwareCompactor(
        max_tokens=50,
        keep_last_turns=1,
        keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )
    state = RunState(
        messages=[_msg(Role.USER, "x" * 20) for _ in range(12)],
        run_id="x",
    )
    s1 = c.compact(state)

    s1_markers = [
        m.content[0] for m in s1.messages
        if TokenAwareCompactor.is_truncation_marker(m)
    ]
    assert len(s1_markers) == 1
    # Spec §7.1: marker content is TruncationBlock; read count directly.
    assert isinstance(s1_markers[0], TruncationBlock)
    s1_count = s1_markers[0].count
    assert s1_count > 0, "first round must have dropped at least one msg"

    # Append more chatter and compact again.
    s1.messages.extend([_msg(Role.USER, "y" * 20) for _ in range(6)])
    s2 = c.compact(s1)
    s2_markers = [
        m.content[0] for m in s2.messages
        if TokenAwareCompactor.is_truncation_marker(m)
    ]
    assert len(s2_markers) == 1
    assert isinstance(s2_markers[0], TruncationBlock)
    s2_count = s2_markers[0].count

    # Cumulative invariant: s2's count strictly exceeds s1's count
    # (this round dropped additional messages).
    assert s2_count > s1_count, (
        f"marker count did not accumulate: round1={s1_count}, round2={s2_count}"
    )


def test_non_monotonic_estimator_is_best_effort_per_contract():
    """v0.2-α: ``TokenEstimator`` requires monotonicity (see type alias
    docstring). Built-in ``char_div4_estimator`` and ``tiktoken_estimator``
    satisfy that contract. Custom estimators that violate it may cause
    the marker-reservation 2-pass to produce a slightly over-budget
    compacted state — this is documented as best-effort, NOT a crash and
    NOT an infinite loop.

    Reproducer: an estimator that maps marker text for count=6 to 10
    tokens while marker text for counts 5 and 10 cost 1 and 0
    respectively. With the simplified 2-pass scheme the compactor still
    terminates and produces a result; the result may exceed the budget.
    This test pins finite termination and successful return; it does NOT
    assert the result fits, because the estimator violates the contract.
    """
    import re as _re
    from lingyia_kit.compactors.token_aware import _estimate_message_tokens

    def est(text: str) -> int:
        if text.startswith("[lingyia:compactor-marker]"):
            m = _re.search(r"earlier (\d+) messages", text)
            if m:
                n = int(m.group(1))
                return {10: 0, 5: 1, 6: 10}.get(n, 0)
            return 0
        return 1

    def msg():
        return Message(role=Role.USER, content=(TextBlock(text="x"),))

    c = TokenAwareCompactor(
        max_tokens=5,
        keep_last_turns=0,
        keep_recent_messages=0,
        token_estimator=est,
    )
    state = RunState(messages=[msg() for _ in range(10)], run_id="x")
    assert c.should_compact(state)

    # The contract claim under test: compact() terminates and returns a
    # state. The compacted state may or may not be within budget — that's
    # the documented best-effort behavior under a non-monotonic estimator.
    s = c.compact(state)
    total = sum(_estimate_message_tokens(m, est) for m in s.messages)
    # Total must be a finite integer; the run must not have looped or
    # crashed. We deliberately do NOT assert ``total <= max_tokens``
    # because the estimator violates monotonicity.
    assert isinstance(total, int)


def test_compactor_emits_truncation_block_not_text():
    """Spec §7.1: new compaction emits Message(SYSTEM, (TruncationBlock,)), not text marker."""
    from lingyia_core import RunState, Role, Message, TextBlock, TruncationBlock
    from lingyia_kit.compactors.token_aware import TokenAwareCompactor, char_div4_estimator

    msgs = [Message(role=Role.USER, content=(TextBlock(text="x" * 50),)) for _ in range(40)]
    state = RunState(messages=msgs, run_id="x")

    c = TokenAwareCompactor(
        max_tokens=50, keep_last_turns=1, keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )
    new_state = c.compact(state)

    markers = [m for m in new_state.messages if TokenAwareCompactor.is_truncation_marker(m)]
    assert len(markers) == 1, f"Expected exactly one marker, got {len(markers)}"

    marker_msg = markers[0]
    assert marker_msg.role == Role.SYSTEM
    assert len(marker_msg.content) == 1
    assert isinstance(marker_msg.content[0], TruncationBlock)
    assert marker_msg.content[0].count > 0
    assert not isinstance(marker_msg.content[0], TextBlock)


def test_is_truncation_marker_tightened_predicate():
    """Spec §7.2 (codex P2.5): predicate requires SYSTEM role AND single-block
    TruncationBlock tuple. Other shapes return False."""
    from lingyia_core import Role, Message, TextBlock, TruncationBlock
    from lingyia_kit.compactors.token_aware import TokenAwareCompactor

    canonical = Message(role=Role.SYSTEM, content=(TruncationBlock(count=3),))
    assert TokenAwareCompactor.is_truncation_marker(canonical) is True

    user_with_block = Message(role=Role.USER, content=(TruncationBlock(count=3),))
    assert TokenAwareCompactor.is_truncation_marker(user_with_block) is False

    mixed = Message(role=Role.SYSTEM, content=(TextBlock(text="hi"), TruncationBlock(count=3)))
    assert TokenAwareCompactor.is_truncation_marker(mixed) is False

    plain = Message(role=Role.SYSTEM, content=(TextBlock(text="System instructions"),))
    assert TokenAwareCompactor.is_truncation_marker(plain) is False


def test_compactor_reads_legacy_text_marker_with_deprecation():
    """Spec §7.3 backward-compat (codex P2.11): legacy text-encoded marker is
    detected on load, a DeprecationWarning fires, and count is parsed correctly."""
    import warnings
    from lingyia_core import Role, Message, TextBlock
    from lingyia_kit.compactors.token_aware import TokenAwareCompactor

    legacy_marker = Message(
        role=Role.SYSTEM,
        content=(TextBlock(text="[lingyia:compactor-marker] earlier 17 messages truncated"),),
    )

    # Detection: predicate matches legacy form
    assert TokenAwareCompactor.is_truncation_marker(legacy_marker) is True

    # Parse: count = 17 AND DeprecationWarning fires
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        count = TokenAwareCompactor._parse_marker_count(legacy_marker)
        assert count == 17
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) >= 1, "Expected at least one DeprecationWarning"
        msg = str(deprecations[0].message).lower()
        assert "legacy" in msg or "alpha" in msg or "v0.2.0" in msg


def test_compactor_legacy_marker_rewritten_on_next_compact():
    """Spec §11.1: state with legacy text marker + new drops → output contains
    TruncationBlock with cumulative count (legacy count + new drops)."""
    from lingyia_core import RunState, Role, Message, TextBlock, TruncationBlock
    from lingyia_kit.compactors.token_aware import TokenAwareCompactor, char_div4_estimator

    legacy_marker = Message(
        role=Role.SYSTEM,
        content=(TextBlock(text="[lingyia:compactor-marker] earlier 5 messages truncated"),),
    )
    new_msgs = [Message(role=Role.USER, content=(TextBlock(text="y" * 30),)) for _ in range(10)]
    state = RunState(messages=[legacy_marker] + new_msgs, run_id="legacy")

    c = TokenAwareCompactor(
        max_tokens=20, keep_last_turns=1, keep_recent_messages=2,
        token_estimator=char_div4_estimator,
    )

    new_state = c.compact(state)

    markers = [m for m in new_state.messages if TokenAwareCompactor.is_truncation_marker(m)]
    assert len(markers) == 1

    marker_block = markers[0].content[0]
    assert isinstance(marker_block, TruncationBlock)
    # Cumulative: legacy 5 + newly dropped (>=1 since compaction did drop messages)
    assert marker_block.count >= 6, f"Cumulative count should be >= 6 (legacy 5 + drops), got {marker_block.count}"

    # No legacy text marker remains
    legacy_texts = [
        m for m in new_state.messages
        if any(isinstance(b, TextBlock) and b.text.startswith("[lingyia:compactor-marker]") for b in m.content)
    ]
    assert len(legacy_texts) == 0, "Legacy text marker should be replaced after compact()"
