"""Minimal end-to-end example: list + read + summarize.

Usage:

    SILICONFLOW_API_KEY=... python -m lingyia_kit.examples.hello_agent \\
        --provider siliconflow --model deepseek-ai/DeepSeek-V4-Flash

    ANTHROPIC_API_KEY=... python -m lingyia_kit.examples.hello_agent \\
        --provider anthropic --model claude-sonnet-4-5

Without --model the provider's built-in default is used.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from lingyia_core import Runtime
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_kit import (
    AnthropicModel,
    MiniMaxModel,
    OpenAIModel,
    SiliconFlowModel,
    list_dir_tool,
    read_file_tool,
    react_harness,
)


PROVIDERS = {
    "siliconflow": ("SILICONFLOW_API_KEY", SiliconFlowModel),
    "minimax": ("MINIMAX_API_KEY", MiniMaxModel),
    "openai": ("OPENAI_API_KEY", OpenAIModel),
    "anthropic": ("ANTHROPIC_API_KEY", AnthropicModel),
}

DEFAULT_GOAL = (
    "List the top-level entries in /Users/yuchen/Documents/leagl, "
    "then read lingyia_core/__init__.py and tell me the public API in one paragraph."
)


async def run_once(provider: str, model_name: str | None, goal: str, max_iter: int) -> dict:
    env_var, cls = PROVIDERS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"Set {env_var} in your environment.")

    kwargs: dict = {"api_key": api_key}
    if model_name:
        kwargs["model"] = model_name
    model = cls(**kwargs)

    runtime = Runtime.dev(model=model, max_iterations=max_iter)
    runtime.telemetry = NoopTelemetry()

    harness = react_harness(
        tools=[
            list_dir_tool(root="/Users/yuchen/Documents/leagl"),
            read_file_tool(root="/Users/yuchen/Documents/leagl"),
        ],
    )

    started = time.perf_counter()
    result = await runtime.arun(harness, goal=goal)
    elapsed = time.perf_counter() - started

    tool_calls = sum(
        1 for o in result.state.observations if o.kind == "tool_result"
    )
    failed = sum(
        1 for o in result.state.observations
        if o.kind == "tool_result" and not o.payload.get("ok", True)
    )
    return {
        "provider": provider,
        "model": model_name or getattr(cls, "DEFAULT_MODEL", "default"),
        "status": result.status.value,
        "iterations": result.state.iteration,
        "tool_calls": tool_calls,
        "tool_failures": failed,
        "elapsed_s": round(elapsed, 2),
        "summary": result.summary,
        "reason": result.reason,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a hello-world agent against a provider.")
    p.add_argument("--provider", default="siliconflow", choices=list(PROVIDERS))
    p.add_argument("--model", default=None, help="Override the provider's default model.")
    p.add_argument("--goal", default=DEFAULT_GOAL)
    p.add_argument("--max-iter", type=int, default=8)
    return p.parse_args(argv)


async def main_async(args: argparse.Namespace) -> None:
    out = await run_once(args.provider, args.model, args.goal, args.max_iter)
    print("\n=== RESULT ===")
    for k in ("provider", "model", "status", "iterations", "tool_calls", "tool_failures", "elapsed_s"):
        print(f"{k}: {out[k]}")
    print("---")
    if out["summary"]:
        print(out["summary"])
    if out["reason"]:
        print(f"(reason: {out['reason']})")


if __name__ == "__main__":
    asyncio.run(main_async(parse_args(sys.argv[1:])))
