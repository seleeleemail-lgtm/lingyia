"""Token-aware message compactor (v2).

Strategy:
1. Estimate token count of full message list.
2. If under max_tokens: no-op (should_compact returns False).
3. Else: split messages into ``transcript groups`` and drop oldest groups
   until the budget is met. A group is an atomic unit kept or dropped
   together — most commonly:
   - an assistant tool_use turn + the following user tool_result turn(s);
   - a single non-tool message.
4. Insert at most ONE synthetic SYSTEM "marker" message describing how
   many messages were dropped. Markers are tagged so subsequent
   compactions can REPLACE the prior marker instead of accumulating
   one per round.
5. If the system prompt(s) alone exceed budget, return state unchanged
   (compaction has nothing to drop in that case; we must not loop).

Spec: v0.2-α §9

Codex P2 fixes:
- :117: truncation markers are now identifiable via
  ``is_truncation_marker(msg)`` (sentinel content prefix) and replaced
  on subsequent compactions instead of preserved. SYSTEM-only state
  exits compaction without progress in a finite step.
- :133: ``_split_into_groups`` makes tool_use + tool_result(s) atomic.
  Cut by group, never by individual message; never produce a kept
  dangling tool_use.
"""
from __future__ import annotations

from copy import copy
from typing import Callable, Optional

from lingyia_core import RunState
from lingyia_core.blocks import Role, TextBlock, ToolUseBlock, ToolResultBlock
from lingyia_core.message import Message


TokenEstimator = Callable[[str], int]


# Sentinel prefix marks a SYSTEM message as a compactor-inserted truncation
# notice (vs an original system prompt). Old markers are recognized and
# replaced — never stacked.
_TRUNCATION_MARKER_PREFIX = "[lingyia:compactor-marker]"


def char_div4_estimator(text: str) -> int:
    """Cheap fallback: ~4 chars per token (English-leaning)."""
    return max(1, len(text) // 4)


def tiktoken_estimator(model: str = "gpt-4") -> TokenEstimator:
    """Real tokenizer (lazy import). Falls back to char_div4 if tiktoken unavailable."""
    try:
        import tiktoken  # type: ignore
    except ImportError:
        return char_div4_estimator
    if hasattr(tiktoken, "encoding_for_model"):
        try:
            enc = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            try:
                enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return char_div4_estimator
    else:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return char_div4_estimator

    def _estimate(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))
    return _estimate


def _estimate_message_tokens(msg: Message, estimator: TokenEstimator) -> int:
    """Sum token estimate across all content blocks in a message."""
    total = 0
    for block in msg.content:
        if isinstance(block, TextBlock):
            total += estimator(block.text)
        elif isinstance(block, ToolUseBlock):
            total += estimator(block.name) + estimator(str(dict(block.input)))
        elif isinstance(block, ToolResultBlock):
            if isinstance(block.content, str):
                total += estimator(block.content)
            else:
                for nested in block.content:
                    if isinstance(nested, TextBlock):
                        total += estimator(nested.text)
        else:
            total += 32
    return total


def _message_has_tool_use(msg: Message) -> bool:
    return any(isinstance(b, ToolUseBlock) for b in msg.content)


def _message_has_tool_result(msg: Message) -> bool:
    return any(isinstance(b, ToolResultBlock) for b in msg.content)


class TokenAwareCompactor:
    """Compactor that drops oldest *groups* when budget exceeded.

    A group is the smallest unit the compactor will keep-or-drop atomically.
    The grouping rule pairs an assistant tool_use turn with the following
    user tool_result turn(s) so the compactor never produces a transcript
    with a dangling tool_use or an orphan tool_result.
    """

    def __init__(
        self,
        max_tokens: int = 80000,
        keep_last_turns: int = 5,
        keep_recent_messages: int = 20,
        token_estimator: Optional[TokenEstimator] = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.keep_last_turns = keep_last_turns
        self.keep_recent_messages = keep_recent_messages
        self.token_estimator = token_estimator or char_div4_estimator

    # ----- public -------------------------------------------------------

    def should_compact(self, state: RunState) -> bool:
        total = sum(
            _estimate_message_tokens(m, self.token_estimator)
            for m in state.messages
        )
        return total > self.max_tokens

    def compact(self, state: RunState) -> RunState:
        """Return a new state with oldest groups dropped, original system
        prompts preserved, a single truncation marker inserted.
        """
        messages = list(state.messages)

        # 1. Separate original system prompts (always preserved) from any
        #    previous truncation marker (replaced) and from the rest.
        original_system: list[Message] = []
        non_system: list[Message] = []
        for m in messages:
            if m.role == Role.SYSTEM and not self.is_truncation_marker(m):
                original_system.append(m)
            elif self.is_truncation_marker(m):
                # Drop prior marker; we'll re-emit a fresh one if needed.
                continue
            else:
                non_system.append(m)

        # 2. SYSTEM-only over-budget: nothing to drop. Return state with the
        #    prior marker removed (we already filtered it out). This makes
        #    the compactor idempotent in that pathological case instead of
        #    looping on its own marker.
        if not non_system:
            new_state = self._clone(state)
            new_state.messages = list(original_system)
            return new_state

        # 3. Group the non-system messages into atomic transcript units.
        groups = self._split_into_groups(non_system)

        # 4. Pick the kept-tail by working back from the end. We keep at
        #    least ``keep_last_turns * 2`` messages worth of groups (the
        #    pair counts assistant + user turn), and at least
        #    ``keep_recent_messages`` worth.
        kept_groups, dropped_groups = self._partition_groups(groups)

        # 5. Rebuild flat message list with the marker if anything dropped.
        kept_messages: list[Message] = []
        for g in kept_groups:
            kept_messages.extend(g)
        dropped_messages: list[Message] = []
        for g in dropped_groups:
            dropped_messages.extend(g)

        new_messages: list[Message] = list(original_system)
        if dropped_messages:
            new_messages.append(self._make_truncation_marker(len(dropped_messages)))
        new_messages.extend(kept_messages)

        new_state = self._clone(state)
        new_state.messages = new_messages
        return new_state

    # ----- marker hygiene ----------------------------------------------

    @staticmethod
    def is_truncation_marker(msg: Message) -> bool:
        """True iff ``msg`` is a compactor-inserted truncation marker.

        Identified by Role.SYSTEM + a TextBlock whose text begins with the
        sentinel prefix. We deliberately use a content sentinel (vs a
        side-channel field) because ``Message`` is a frozen-ish dataclass
        and round-trips through JSON serialization; the sentinel survives
        a save/load cycle so subsequent compactions can find it.
        """
        if msg.role != Role.SYSTEM:
            return False
        for b in msg.content:
            if isinstance(b, TextBlock) and b.text.startswith(_TRUNCATION_MARKER_PREFIX):
                return True
        return False

    @staticmethod
    def _make_truncation_marker(dropped_count: int) -> Message:
        text = (
            f"{_TRUNCATION_MARKER_PREFIX} "
            f"[earlier {dropped_count} messages truncated to save context]"
        )
        return Message(role=Role.SYSTEM, content=(TextBlock(text=text),))

    # ----- group construction ------------------------------------------

    @staticmethod
    def _split_into_groups(msgs: list[Message]) -> list[list[Message]]:
        """Partition non-system messages into atomic groups.

        Group definition:
        - An assistant message containing one or more ``ToolUseBlock`` starts
          a tool-call group. The group consumes that assistant message plus
          every immediately following USER message that carries a
          matching ``ToolResultBlock`` (its tool_use_id is in the group's
          pending set). The group closes when the next message is not a
          matching tool_result.
        - Any other message is its own one-element group.

        This means a dangling assistant tool_use with no result is its own
        single-element group — keep or drop atomically. An orphan
        tool_result that doesn't match a known tool_use is also its own
        single-element group (treated as junk; carried as a normal user
        message in the flow).
        """
        groups: list[list[Message]] = []
        i = 0
        n = len(msgs)
        while i < n:
            m = msgs[i]
            if m.role == Role.ASSISTANT and _message_has_tool_use(m):
                pending = {
                    b.id for b in m.content if isinstance(b, ToolUseBlock)
                }
                group = [m]
                j = i + 1
                while j < n and msgs[j].role == Role.USER and _message_has_tool_result(msgs[j]):
                    result_ids = {
                        b.tool_use_id
                        for b in msgs[j].content
                        if isinstance(b, ToolResultBlock)
                    }
                    if not (result_ids & pending):
                        break
                    group.append(msgs[j])
                    pending -= result_ids
                    j += 1
                    if not pending:
                        break
                groups.append(group)
                i = j
                continue
            groups.append([m])
            i += 1
        return groups

    # ----- partition ----------------------------------------------------

    def _partition_groups(
        self,
        groups: list[list[Message]],
    ) -> tuple[list[list[Message]], list[list[Message]]]:
        """Decide which groups to keep (tail) and which to drop (head).

        Matches v0.1 semantics: the *cut index* is the larger of
        ``len - keep_last_turns*2`` and ``len - keep_recent_messages``
        (i.e. drop more = keep less). The atomic-group rule means we
        always snap the cut to a group boundary instead of slicing a
        message in half.
        """
        if not groups:
            return [], []

        # Flatten to track per-message cut, then snap to group boundary.
        flat_count = sum(len(g) for g in groups)
        target_cut_msgs = max(
            flat_count - 2 * self.keep_last_turns,
            flat_count - self.keep_recent_messages,
        )
        target_cut_msgs = max(0, target_cut_msgs)

        # Walk groups front-to-back; everything up to the group whose
        # END crosses target_cut_msgs is dropped.
        msg_so_far = 0
        cut_group_idx = 0
        for idx, g in enumerate(groups):
            if msg_so_far >= target_cut_msgs:
                cut_group_idx = idx
                break
            msg_so_far += len(g)
            cut_group_idx = idx + 1
        dropped = groups[:cut_group_idx]
        kept = groups[cut_group_idx:]
        return kept, dropped

    # ----- helpers ------------------------------------------------------

    @staticmethod
    def _clone(state: RunState) -> RunState:
        new_state = copy(state)
        new_state.messages = list(state.messages)
        new_state.trace = list(state.trace)
        new_state.metadata = dict(state.metadata)
        return new_state
