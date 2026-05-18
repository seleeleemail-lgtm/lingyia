"""Token-aware message compactor.

Strategy:
1. Estimate token count of full message list.
2. If under max_tokens: no-op (should_compact returns False).
3. Else: preserve SYSTEM messages, keep last N user-assistant turn pairs,
   drop oldest first, insert synthetic SYSTEM marker for truncated span.
4. Tool_use + tool_result must stay adjacent (dropped together).

Spec: v0.2-α §9
"""
from __future__ import annotations

from copy import copy
from typing import Callable, Optional

from lingyia_core import RunState
from lingyia_core.blocks import Role, TextBlock, ToolUseBlock, ToolResultBlock
from lingyia_core.message import Message


TokenEstimator = Callable[[str], int]


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


class TokenAwareCompactor:
    """Compactor that drops oldest messages when budget exceeded."""

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

    def should_compact(self, state: RunState) -> bool:
        total = sum(_estimate_message_tokens(m, self.token_estimator) for m in state.messages)
        return total > self.max_tokens

    def compact(self, state: RunState) -> RunState:
        """Return new state with oldest messages dropped, system preserved, marker inserted."""
        messages = list(state.messages)

        # 1. Separate system messages (always preserved at head)
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        # 2. Determine "keep tail" — last N turn pairs OR keep_recent_messages cap
        tail_keep_idx = max(0, len(non_system) - 2 * self.keep_last_turns)
        tail_keep_idx = max(tail_keep_idx, len(non_system) - self.keep_recent_messages)

        # 3. Ensure tool_use / tool_result adjacency
        tail_keep_idx = self._adjust_for_tool_pairing(non_system, tail_keep_idx)

        dropped = non_system[:tail_keep_idx]
        kept_tail = non_system[tail_keep_idx:]

        # 4. Build truncation marker if anything was dropped
        new_messages = list(system_msgs)
        if dropped:
            marker = Message(
                role=Role.SYSTEM,
                content=(TextBlock(text=f"[earlier {len(dropped)} messages truncated to save context]"),),
            )
            new_messages.append(marker)
        new_messages.extend(kept_tail)

        # 5. Return new state (don't mutate input)
        new_state = copy(state)
        new_state.messages = new_messages
        new_state.trace = list(state.trace)
        new_state.metadata = dict(state.metadata)
        return new_state

    def _adjust_for_tool_pairing(self, msgs: list[Message], cut_idx: int) -> int:
        """If cut_idx would orphan a tool_result, push cut forward to drop it too."""
        kept_use_ids: set[str] = set()
        for m in msgs[cut_idx:]:
            for b in m.content:
                if isinstance(b, ToolUseBlock):
                    kept_use_ids.add(b.id)
        # Check kept tail for orphan tool_results
        for idx, m in enumerate(msgs[cut_idx:], start=cut_idx):
            for b in m.content:
                if isinstance(b, ToolResultBlock):
                    if b.tool_use_id not in kept_use_ids:
                        # Orphan! Push cut forward past this message
                        new_cut = idx + 1
                        if new_cut < len(msgs):
                            return self._adjust_for_tool_pairing(msgs, new_cut)
                        return new_cut
        return cut_idx
