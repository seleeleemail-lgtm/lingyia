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

import re
import warnings
from copy import copy
from typing import Callable, Optional

from lingyia_core import RunState
from lingyia_core.blocks import Role, TextBlock, ToolUseBlock, ToolResultBlock, TruncationBlock
from lingyia_core.message import Message


# Legacy v0.2.0-alpha text-encoded marker constants (transitional support;
# targeted removal in v0.4 per spec §11.2). Detection is additive — both
# the new TruncationBlock format and the legacy text form are recognized
# by ``is_truncation_marker``. Each legacy parse emits a DeprecationWarning.
_LEGACY_MARKER_PREFIX = "[lingyia:compactor-marker]"
_LEGACY_MARKER_COUNT_RE = re.compile(r"earlier (\d+) messages")


TokenEstimator = Callable[[str], int]
"""A function that estimates the token cost of a text string.

Contract: must return a non-negative integer. The estimator MUST be
monotonic with respect to text content — specifically, longer text
should never estimate to fewer tokens than shorter text. Non-monotonic
estimators may cause the TokenAwareCompactor's budget reservation to
produce over-budget compacted states. Built-in char_div4_estimator
and tiktoken_estimator satisfy this contract.
"""


# Sentinel prefix marks a SYSTEM message as a compactor-inserted truncation
# notice (vs an original system prompt). Old markers are recognized and
# replaced — never stacked.
_TRUNCATION_MARKER_PREFIX = "[lingyia:compactor-marker]"

# Regex used to recover the prior marker's cumulative count so a fresh
# marker can accumulate (codex round 4 P2: token_aware.py:243). Must match
# the literal text emitted by ``_make_truncation_marker``. If parsing
# fails (corrupted / hand-edited marker), callers default to 0 — do NOT
# crash on a bad marker.
_MARKER_COUNT_RE = re.compile(r"earlier (\d+) messages truncated")


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
        """Construct a token-aware compactor.

        ``keep_last_turns`` and ``keep_recent_messages`` are **soft floors**:
        they bound the minimum size of the kept tail, regardless of budget.
        The cut still drops oldest groups until total estimated tokens
        ≤ ``max_tokens`` (or until the floor is hit). If the floor itself
        exceeds budget, the compactor returns the floor unchanged — and
        ``should_compact`` may still report True ("irreducible system+tail
        floor"). This is documented as a non-recoverable condition.
        """
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

        Compaction is **budget-driven** (codex verify P2: token_aware.py:275):
        oldest atomic groups are dropped one at a time until total estimated
        tokens ≤ ``max_tokens`` OR the keep_last_turns / keep_recent_messages
        soft floor is reached. The earlier count-based partition could leave
        state above budget after a single compact() call, making
        should_compact() stay True forever in pathological setups.

        Codex round 3 P2 fixes:
        - :196 The truncation marker's own token cost is now included in the
          partition budget. Without this, the marker (~25 tokens at char/4)
          would push state over budget on its first compact() call.
        - :170 An existing truncation marker in the input is **preserved**
          across rounds when this call drops zero new groups. The marker is
          the audit record that history was already truncated; it must
          survive subsequent compactions until/unless replaced by a fresher
          one. (When new groups ARE dropped this round, the fresh marker
          subsumes the old; we still emit only one marker.)

        Codex round 4 P2 fix:
        - :243 The fresh marker's count is now CUMULATIVE: prior_count +
          newly_dropped_count. The prior marker's count is recovered by
          regex from its text. Previously the new marker overwrote the old
          count, silently losing the audit trail across rounds. Corrupted
          markers parse to 0 (no crash) so the accumulator resets cleanly
          rather than taking the run down.

        v0.2-α simplification:
        - The marker-budget reservation uses a 2-pass scheme assuming a
          monotonic ``TokenEstimator`` (see type alias docstring): pass 1
          estimates the marker cost as if we drop nothing new; pass 2
          re-partitions with the actual marker cost for that drop count.
          Built-in ``char_div4_estimator`` and ``tiktoken_estimator`` are
          monotonic so this converges. Non-monotonic estimators may
          produce a slightly over-budget result — documented as a
          best-effort property of the contract, not a crash.
        """
        messages = list(state.messages)

        # 1. Separate original system prompts (always preserved) from any
        #    previous truncation marker (carried forward conditionally) and
        #    from the rest.
        original_system: list[Message] = []
        prior_marker: Optional[Message] = None
        non_system: list[Message] = []
        for m in messages:
            if m.role == Role.SYSTEM and not self.is_truncation_marker(m):
                original_system.append(m)
            elif self.is_truncation_marker(m):
                # Keep the most recent prior marker so we can re-emit it if
                # this call drops no new groups (preserves audit trail).
                prior_marker = m
                continue
            else:
                non_system.append(m)

        # 2. SYSTEM-only over-budget: nothing to drop. Return state with the
        #    prior marker preserved (if it existed). This makes the
        #    compactor idempotent in that pathological case instead of
        #    looping on its own marker.
        if not non_system:
            new_state = self._clone(state)
            new_messages = list(original_system)
            if prior_marker is not None:
                new_messages.append(prior_marker)
            new_state.messages = new_messages
            return new_state

        # 3. Group the non-system messages into atomic transcript units.
        groups = self._split_into_groups(non_system)

        # 4. Estimate the system "floor" we must always keep, then drop
        #    oldest groups until SYSTEM + kept_tail + MARKER fits under
        #    max_tokens (or we hit the soft floor enforced by
        #    keep_last_turns / keep_recent_messages).
        system_tokens = sum(
            _estimate_message_tokens(m, self.token_estimator)
            for m in original_system
        )

        # Marker reservation: 2-pass under the monotonic TokenEstimator
        # contract (see type alias docstring). Pass 1 estimates the
        # marker as if no new groups are dropped this round; pass 2
        # repartitions using the actual marker cost for the drop count
        # that pass 1 produced. Built-in estimators are monotonic so
        # this converges in 2 passes. Non-monotonic estimators may
        # leave the result slightly over budget (documented limitation).
        prior_count = self._parse_marker_count(prior_marker)

        def _marker_cost(extra_drops: int) -> int:
            # Spec §4.1: TruncationBlock.__post_init__ rejects count <= 0.
            # Guard the estimation path so a hypothetical "no drops yet"
            # call (prior_count=0, extra_drops=0) doesn't construct an
            # invalid block during budget reservation.
            count = prior_count + extra_drops
            if count <= 0:
                return 0
            return _estimate_message_tokens(
                self._make_truncation_marker(count),
                self.token_estimator,
            )

        # Pass 1: reserve for the prior cumulative count (no new drops).
        marker_reserve = _marker_cost(0)
        kept_groups, dropped_groups = self._partition_groups(
            groups,
            system_tokens=system_tokens,
            marker_tokens=marker_reserve,
        )
        dropped_message_count = sum(len(g) for g in dropped_groups)

        # Pass 2: if the actual marker for this drop count costs more
        # than the pass-1 reservation, repartition once with the true
        # cost. For monotonic estimators a single repartition is enough.
        if dropped_message_count > 0:
            actual_marker = _marker_cost(dropped_message_count)
            if actual_marker > marker_reserve:
                marker_reserve = actual_marker
                kept_groups, dropped_groups = self._partition_groups(
                    groups,
                    system_tokens=system_tokens,
                    marker_tokens=marker_reserve,
                )
                dropped_message_count = sum(len(g) for g in dropped_groups)

        # 5. Rebuild flat message list with the marker if anything dropped
        #    OR if a prior marker existed and must be carried forward.
        kept_messages: list[Message] = []
        for g in kept_groups:
            kept_messages.extend(g)

        new_messages: list[Message] = list(original_system)
        if dropped_message_count > 0:
            # Cumulative count = prior marker's recorded count + this
            # round's new drops (codex round 4 P2: token_aware.py:243).
            # If the prior marker text was corrupted/edited and parsing
            # failed, prior_count defaults to 0 — we don't crash, we just
            # restart the counter from this round's drops.
            cumulative_count = prior_count + dropped_message_count
            new_messages.append(
                self._make_truncation_marker(cumulative_count)
            )
        elif prior_marker is not None:
            # No new drops this round — preserve the prior marker verbatim
            # so the audit trail of past truncation isn't silently lost.
            new_messages.append(prior_marker)
        new_messages.extend(kept_messages)

        new_state = self._clone(state)
        new_state.messages = new_messages
        return new_state

    # ----- marker hygiene ----------------------------------------------

    @staticmethod
    def is_truncation_marker(msg: Message) -> bool:
        """Per spec §7.2: detect both canonical new-format markers and legacy
        v0.2.0-alpha text-encoded markers (with DeprecationWarning side effect
        in _parse_marker_count). Targeted removal: v0.4.x.
        """
        if msg.role != Role.SYSTEM:
            return False
        # New format: SYSTEM + single TruncationBlock content
        if len(msg.content) == 1 and isinstance(msg.content[0], TruncationBlock):
            return True
        # Legacy text-encoded marker (transitional)
        for b in msg.content:
            if isinstance(b, TextBlock) and b.text.startswith(_LEGACY_MARKER_PREFIX):
                return True
        return False

    @staticmethod
    def _make_truncation_marker(count: int) -> Message:
        """Per spec §7.1: emit Message(SYSTEM, (TruncationBlock(count),)).
        No longer text-encoded. Caller MUST ensure count > 0 (enforced by
        TruncationBlock.__post_init__)."""
        return Message(
            role=Role.SYSTEM,
            content=(TruncationBlock(count=count),),
        )

    @staticmethod
    def _parse_marker_count(msg: Optional[Message]) -> int:
        """Spec §7.3: parse marker count.

        Prefer TruncationBlock.count; fall back to legacy text regex with
        DeprecationWarning. For mixed markers (legacy text + TruncationBlock —
        only possible from corrupted snapshots), return the MAX count from
        either source to avoid silently losing the larger legacy count
        (codex post-impl P2.2). DeprecationWarning still fires when legacy
        text is present regardless of which source wins.

        Returns 0 if no parseable marker present.
        """
        if msg is None:
            return 0

        candidates: list[int] = []
        has_legacy_text = False

        # Canonical: TruncationBlock count
        for b in msg.content:
            if isinstance(b, TruncationBlock):
                candidates.append(b.count)

        # Legacy text marker fallback
        for b in msg.content:
            if isinstance(b, TextBlock):
                m = _LEGACY_MARKER_COUNT_RE.search(b.text)
                if m:
                    has_legacy_text = True
                    candidates.append(int(m.group(1)))

        # Emit DeprecationWarning if legacy text was found, regardless of which
        # source wins (mixed corrupted snapshot still warrants the heads-up).
        if has_legacy_text:
            warnings.warn(
                "Loaded a v0.2.0-alpha text-encoded truncation marker. "
                "Next compact() will rewrite to TruncationBlock. "
                "Legacy text marker support will be removed in v0.4.",
                DeprecationWarning,
                stacklevel=2,
            )

        return max(candidates) if candidates else 0

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
        system_tokens: int = 0,
        marker_tokens: int = 0,
    ) -> tuple[list[list[Message]], list[list[Message]]]:
        """Decide which groups to keep (tail) and which to drop (head).

        Budget-driven (codex verify P2): we drop oldest groups one at a
        time until ``system_tokens + marker_tokens + kept_group_tokens
        ≤ max_tokens``, snapping at atomic group boundaries. The soft
        floor enforced by ``keep_last_turns`` and ``keep_recent_messages``
        bounds the minimum size of the kept tail.

        ``marker_tokens`` is the caller's reservation for the truncation
        marker that will be inserted between system and kept-tail when
        any group is dropped (codex round 3 P2: token_aware.py:196). Pass
        0 if no marker will be emitted.

        Floor semantics match the v0.1 contract: the kept-tail floor is
        ``min(2 * keep_last_turns, keep_recent_messages)`` messages — the
        MORE PERMISSIVE of the two constraints (whichever allows cutting
        more). This is the same as the prior count-based
        ``max(flat - 2*kl, flat - kr)`` cut formula, re-expressed as a
        floor instead of a cut bound. Translated to groups, we keep at
        least the trailing N groups whose total message count first
        meets that floor.

        If respecting the floor leaves us still over budget, we keep the
        floor anyway — the caller is in the irreducible-floor regime
        documented on ``compact()``.
        """
        if not groups:
            return [], []

        # Pre-compute token cost of each group so we can scan cheaply.
        group_tokens = [
            sum(_estimate_message_tokens(m, self.token_estimator) for m in g)
            for g in groups
        ]

        # ----- soft floor: minimum kept-group count -------------------
        # The floor in *messages* is the smaller of the two constraints
        # (whichever allows MORE truncation, mirroring v0.1 ``max(cut1,
        # cut2)`` which picks the larger cut = smaller kept).
        floor_in_msgs = min(2 * self.keep_last_turns, self.keep_recent_messages)
        floor_in_msgs = max(0, floor_in_msgs)

        # Convert the message-floor to a group-floor by walking back from
        # the tail until the flattened message count reaches the floor.
        flat_lens = [len(g) for g in groups]
        msgs_walked = 0
        min_kept = 0
        for k in range(1, len(groups) + 1):
            msgs_walked += flat_lens[-k]
            min_kept = k
            if msgs_walked >= floor_in_msgs:
                break
        if floor_in_msgs == 0:
            min_kept = 0
        min_kept = min(min_kept, len(groups))

        # ----- budget-driven cut ---------------------------------------
        # Walk from the tail forward; accumulate kept-tail tokens. Stop
        # adding once tokens + system + marker would exceed budget, but
        # never shrink below ``min_kept`` groups.
        budget = max(0, self.max_tokens - system_tokens - marker_tokens)
        kept_count = 0
        kept_tokens = 0
        for k in range(1, len(groups) + 1):
            cost = group_tokens[-k]
            # Always honor the soft floor: we must accept at least
            # ``min_kept`` groups regardless of cost.
            if k <= min_kept:
                kept_count = k
                kept_tokens += cost
                continue
            # Past the floor: only accept this group if it still fits.
            if kept_tokens + cost > budget:
                break
            kept_count = k
            kept_tokens += cost

        # Edge case: if even ``min_kept`` overshoots the budget, kept_count
        # may have been set to min_kept already — we accept that ("kept
        # tail is the floor") rather than dropping into it, because
        # dropping below the floor would lose the recent context every
        # caller needs.

        cut_group_idx = len(groups) - kept_count
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
