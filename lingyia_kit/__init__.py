"""lingyia_kit: L2 composition layer.

Sits on top of :mod:`lingyia_core` and provides:
- Provider adapters (OpenAI, SiliconFlow, MiniMax, Anthropic)
- Generic tools (filesystem, HTTP, shell)
- Agent patterns (ReAct)
- Pre-wired examples

Stable contract: anything in here implements one of the lingyia_core protocols
(Model, Tool, Compactor, TelemetrySink, Checkpointer). Anything that does not
belongs in an L3 application package.
"""
from lingyia_kit.adapters import (
    AnthropicModel,
    MiniMaxModel,
    OpenAICompatibleModel,
    OpenAIModel,
    SiliconFlowModel,
)
from lingyia_kit.patterns import react_harness
from lingyia_kit.tools import (
    fetch_url_tool,
    list_dir_tool,
    read_file_tool,
    run_shell_tool,
    write_file_tool,
)

__all__ = [
    # Adapters
    "AnthropicModel",
    "MiniMaxModel",
    "OpenAICompatibleModel",
    "OpenAIModel",
    "SiliconFlowModel",
    # Patterns
    "react_harness",
    # Tools
    "fetch_url_tool",
    "list_dir_tool",
    "read_file_tool",
    "run_shell_tool",
    "write_file_tool",
]
