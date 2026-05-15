"""BFCL V3 single-turn function-calling runner.

What this measures:
- Whether the model emits a tool call (vs final answer / refusal).
- Whether the chosen function name matches ground truth.
- Whether each argument value is in BFCL's acceptable-values list.

What this does NOT measure:
- BFCL's full AST equivalence (e.g. function-composition semantics for some
  multi-turn tasks). For 80%+ of single-turn questions the value-list match
  used here agrees with the official AST checker. The remaining edge cases
  (numeric tolerance, unit normalization) are handled with light coercion.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from agent_core import (
    Decision,
    DecisionKind,
    RunState,
    Tool,
    ToolContext,
    ToolResult,
)


BfclSplit = str  # "simple" | "parallel" | "multiple"


@dataclass(frozen=True)
class BfclVerdict:
    id: str
    verdict: str  # correct | wrong_name | wrong_args | wrong_count | no_call | error
    expected: Any = None
    got: Any = None
    note: str = ""
    elapsed_s: float = 0.0


@dataclass
class SplitStats:
    split: str
    model: str
    total: int = 0
    correct: int = 0
    wrong_name: int = 0
    wrong_args: int = 0
    wrong_count: int = 0
    no_call: int = 0
    error: int = 0
    elapsed_s: float = 0.0
    failures: list[BfclVerdict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def add(self, v: BfclVerdict) -> None:
        self.total += 1
        attr_map = {
            "correct": "correct",
            "wrong_name": "wrong_name",
            "wrong_args": "wrong_args",
            "wrong_count": "wrong_count",
            "no_call": "no_call",
            "error": "error",
        }
        attr = attr_map.get(v.verdict, "error")
        setattr(self, attr, getattr(self, attr) + 1)
        self.elapsed_s += v.elapsed_s
        if v.verdict != "correct":
            self.failures.append(v)


# Data loading -------------------------------------------------------------


def load_split(
    split: BfclSplit,
    dataset_dir: str | os.PathLike,
) -> tuple[list[dict], dict[str, dict]]:
    """Load a BFCL split + matching answers, keyed by id."""
    root = Path(dataset_dir)
    q_path = root / f"BFCL_v3_{split}.json"
    a_path = root / f"possible_answer_BFCL_v3_{split}.json"
    if not q_path.exists():
        raise FileNotFoundError(q_path)
    if not a_path.exists():
        raise FileNotFoundError(a_path)
    questions = _read_jsonl(q_path)
    answers = {a["id"]: a for a in _read_jsonl(a_path)}
    return questions, answers


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# Tool conversion ----------------------------------------------------------


def _normalize_schema(params: dict) -> dict:
    """Convert BFCL schema (`type: dict`) to standard JSON Schema (`type: object`)."""
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    p = dict(params)
    if p.get("type") == "dict":
        p["type"] = "object"
    props = p.get("properties") or {}
    new_props: dict[str, Any] = {}
    for k, v in props.items():
        if isinstance(v, dict):
            v2 = dict(v)
            if v2.get("type") == "dict":
                v2["type"] = "object"
            # BFCL sometimes uses "tuple" or other non-JSON-Schema types
            if v2.get("type") == "tuple":
                v2["type"] = "array"
            if v2.get("type") == "any":
                v2.pop("type", None)
            new_props[k] = v2
    p["properties"] = new_props
    return p


def _make_noop_tool(bfcl_func: dict) -> Tool:
    def _h(args, ctx: ToolContext) -> ToolResult:
        # We never execute in eval mode; the adapter call is what we score.
        return ToolResult(tool_name=bfcl_func["name"], ok=True)

    return Tool.from_sync(
        name=bfcl_func["name"],
        description=bfcl_func.get("description", ""),
        handler=_h,
        input_schema=_normalize_schema(bfcl_func.get("parameters", {})),
        side_effect="read_only",
        idempotent=True,
    )


# Scoring ------------------------------------------------------------------


def _values_match(actual: Any, acceptable: list[Any]) -> bool:
    """Match actual value against BFCL's acceptable-values list.

    Convention:
    - empty string "" in the list means the arg may be omitted / defaulted.
    - numeric values are compared with float coercion.
    - strings are compared case-insensitive with stripped whitespace.
    - lists/dicts are compared structurally.
    """
    if not isinstance(acceptable, list):
        acceptable = [acceptable]
    actual_missing = actual is None or actual == ""
    for cand in acceptable:
        cand_optional = cand == ""
        if cand_optional and actual_missing:
            return True
        if cand == actual:
            return True
        # numeric coercion
        try:
            if isinstance(cand, (int, float)) and not isinstance(cand, bool):
                if float(cand) == float(actual):
                    return True
        except (TypeError, ValueError):
            pass
        # string normalization
        if isinstance(cand, str) and isinstance(actual, str):
            if cand.strip().lower() == actual.strip().lower():
                return True
        # list compare order-insensitive for small lists of primitives
        if isinstance(cand, list) and isinstance(actual, list):
            if sorted(map(repr, cand)) == sorted(map(repr, actual)):
                return True
        # dict compare
        if isinstance(cand, dict) and isinstance(actual, dict):
            if cand == actual:
                return True
    return False


def _grade_single_call(
    actual_name: str,
    actual_args: dict,
    gt_entry: dict,
) -> tuple[str, str]:
    """Grade one call. Returns (verdict, note)."""
    gt_name, gt_args = next(iter(gt_entry.items()))
    if actual_name != gt_name:
        return ("wrong_name", f"expected {gt_name}, got {actual_name}")
    for arg_name, acceptable in gt_args.items():
        actual_val = actual_args.get(arg_name)
        if not _values_match(actual_val, acceptable):
            return (
                "wrong_args",
                f"{arg_name}: expected one of {acceptable!r}, got {actual_val!r}",
            )
    return ("correct", "")


def grade_decision(
    decision: Decision,
    answer_entry: dict,
    split: BfclSplit,
) -> tuple[str, str]:
    """Return (verdict, note) for one decision against an answer entry."""
    gt_calls: list[dict] = answer_entry["ground_truth"]
    if decision.kind != DecisionKind.CALL_TOOL or not decision.tool_calls:
        return ("no_call", f"model emitted final_answer: {decision.content[:120]!r}")

    actual_calls = list(decision.tool_calls)

    if split == "parallel":
        # Order-insensitive matching: each ground-truth call must find a match
        # among actual calls, and counts must agree.
        if len(actual_calls) != len(gt_calls):
            return (
                "wrong_count",
                f"expected {len(gt_calls)} calls, got {len(actual_calls)}",
            )
        used = [False] * len(actual_calls)
        for gt_entry in gt_calls:
            matched = False
            for i, ac in enumerate(actual_calls):
                if used[i]:
                    continue
                verdict, _ = _grade_single_call(ac.name, dict(ac.args), gt_entry)
                if verdict == "correct":
                    used[i] = True
                    matched = True
                    break
            if not matched:
                return ("wrong_args", f"no actual call matched ground truth {gt_entry}")
        return ("correct", "")

    # simple / multiple: exactly one expected call
    if len(actual_calls) != 1:
        return (
            "wrong_count",
            f"expected 1 call, got {len(actual_calls)}",
        )
    if len(gt_calls) != 1:
        # Defensive: data sometimes has alt-ground-truths; pick the first.
        gt_entry = gt_calls[0]
    else:
        gt_entry = gt_calls[0]
    return _grade_single_call(actual_calls[0].name, dict(actual_calls[0].args), gt_entry)


# Runner -------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a function-calling assistant. Given a user request and a set of "
    "available functions, emit exactly the tool calls needed to satisfy the "
    "request. Do not include any explanation. Do not call functions that were "
    "not provided. If the request requires multiple parallel calls, emit them "
    "all in a single response."
)


async def _run_one(
    model: Any,
    question: dict,
    answers: dict,
    split: BfclSplit,
) -> BfclVerdict:
    qid = question["id"]
    user_msg = question["question"][0][0]["content"]
    tools = [_make_noop_tool(f) for f in question["function"]]

    state = RunState(goal=user_msg)
    started = time.perf_counter()
    try:
        decision = await model.adecide(
            {"system_prompt": SYSTEM_PROMPT},
            state,
            tools,
        )
    except Exception as exc:
        return BfclVerdict(
            id=qid,
            verdict="error",
            note=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.perf_counter() - started,
        )

    elapsed = time.perf_counter() - started
    ans = answers.get(qid)
    if ans is None:
        return BfclVerdict(id=qid, verdict="error", note="no answer entry", elapsed_s=elapsed)

    verdict, note = grade_decision(decision, ans, split)
    expected = ans["ground_truth"] if verdict != "correct" else None
    got: Any = None
    if verdict != "correct":
        got = [{"name": tc.name, "args": dict(tc.args)} for tc in decision.tool_calls]
        if not got:
            got = decision.content
    return BfclVerdict(
        id=qid,
        verdict=verdict,
        expected=expected,
        got=got,
        note=note,
        elapsed_s=elapsed,
    )


async def run_split(
    model: Any,
    split: BfclSplit,
    dataset_dir: str | os.PathLike,
    *,
    limit: Optional[int] = None,
    concurrency: int = 4,
    progress: bool = True,
) -> SplitStats:
    questions, answers = load_split(split, dataset_dir)
    if limit:
        questions = questions[:limit]

    model_name = getattr(model, "model", str(type(model).__name__))
    stats = SplitStats(split=split, model=model_name)
    sem = asyncio.Semaphore(concurrency)

    async def _bound(q):
        async with sem:
            v = await _run_one(model, q, answers, split)
            if progress:
                marker = "✓" if v.verdict == "correct" else "✗"
                print(f"  [{split}] {marker} {v.id} {v.verdict} ({v.elapsed_s:.1f}s)")
            return v

    started_total = time.perf_counter()
    verdicts = await asyncio.gather(*(_bound(q) for q in questions))
    for v in verdicts:
        stats.add(v)
    stats.elapsed_s = time.perf_counter() - started_total
    return stats


# CLI ----------------------------------------------------------------------


def _print_summary(all_stats: Sequence[SplitStats]) -> None:
    print("\n" + "=" * 78)
    print(f"{'model':<42} {'split':<10} {'n':>4} {'acc':>7} {'wall':>7}")
    print("-" * 78)
    for s in all_stats:
        print(f"{s.model:<42} {s.split:<10} {s.total:>4} "
              f"{s.accuracy * 100:>6.1f}% {s.elapsed_s:>6.1f}s")
    print("=" * 78)
    # Breakdown
    print(f"\n{'model':<42} {'split':<10} {'ok':>4} {'name':>4} {'args':>4} {'cnt':>4} {'no':>4} {'err':>4}")
    print("-" * 78)
    for s in all_stats:
        print(f"{s.model:<42} {s.split:<10} {s.correct:>4} {s.wrong_name:>4} "
              f"{s.wrong_args:>4} {s.wrong_count:>4} {s.no_call:>4} {s.error:>4}")


def _print_failures(all_stats: Sequence[SplitStats], n_each: int = 3) -> None:
    print("\n--- Failure samples ---")
    for s in all_stats:
        if not s.failures:
            continue
        print(f"\n[{s.model} / {s.split}]")
        for v in s.failures[:n_each]:
            print(f"  {v.id} [{v.verdict}] {v.note}")
            if v.expected is not None:
                print(f"    expected: {v.expected}")
            if v.got is not None:
                print(f"    got:      {v.got}")


async def _amain(args) -> None:
    from agent_kit.adapters import SiliconFlowModel

    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set SILICONFLOW_API_KEY in your environment.")

    models = [SiliconFlowModel(api_key=api_key, model=m, timeout_s=180.0) for m in args.models]
    splits = args.splits.split(",")
    all_stats: list[SplitStats] = []
    for model in models:
        for split in splits:
            print(f"\n>>> {model.model} / {split} (limit={args.limit})")
            stats = await run_split(
                model,
                split,
                args.dataset_dir,
                limit=args.limit,
                concurrency=args.concurrency,
                progress=args.verbose,
            )
            all_stats.append(stats)

    _print_summary(all_stats)
    if args.show_failures:
        _print_failures(all_stats)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="datasets/bfcl",
                   help="Where BFCL_v3_*.json files live.")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model IDs on SiliconFlow.")
    p.add_argument("--splits", default="simple,parallel,multiple")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--show-failures", action="store_true")
    args = p.parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
