"""Minimal end-to-end example: read a file then summarize it.

Run with one of:

    SILICONFLOW_API_KEY=sk-... python -m agent_kit.examples.hello_agent siliconflow
    MINIMAX_API_KEY=...        python -m agent_kit.examples.hello_agent minimax
    OPENAI_API_KEY=sk-...      python -m agent_kit.examples.hello_agent openai
    ANTHROPIC_API_KEY=sk-...   python -m agent_kit.examples.hello_agent anthropic
"""
from __future__ import annotations

import asyncio
import os
import sys

from agent_core import Runtime
from agent_core.defaults.telemetry import StdoutTelemetry
from agent_kit import (
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


async def main(provider: str = "siliconflow") -> None:
    env_var, cls = PROVIDERS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        print(f"Set {env_var} in your environment.")
        sys.exit(1)

    model = cls(api_key=api_key)
    runtime = Runtime.dev(model=model, max_iterations=6)
    # Quiet the dev telemetry; flip to StdoutTelemetry() to inspect events.
    runtime.telemetry = StdoutTelemetry()

    harness = react_harness(
        tools=[
            list_dir_tool(root="/Users/yuchen/Documents/leagl"),
            read_file_tool(root="/Users/yuchen/Documents/leagl"),
        ],
    )

    goal = (
        "List the top-level entries in /Users/yuchen/Documents/leagl, "
        "then read agent_core/__init__.py and tell me the public API in one paragraph."
    )

    result = await runtime.arun(harness, goal=goal)
    print("\n=== RESULT ===")
    print(f"status:  {result.status.value}")
    print(f"summary: {result.summary}")
    print(f"iterations: {result.state.iteration}")
    print(f"observations: {len(result.state.observations)}")


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "siliconflow"
    if provider not in PROVIDERS:
        print(f"Unknown provider: {provider}. Pick one of {list(PROVIDERS)}.")
        sys.exit(1)
    asyncio.run(main(provider))
