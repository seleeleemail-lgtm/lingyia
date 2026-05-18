"""Agent core: domain-agnostic Runtime + pluggable Harness.

Example:

    from lingyia_core import Runtime, Harness, Tool, Decision

    runtime = Runtime.dev(model=...)
    harness = Harness(tools=[Tool.from_async(...)], validator=...)
    result = runtime.run(harness, goal="...")
"""
from .core import (
    Checkpointer,
    Compactor,
    Decision,
    DecisionKind,
    GuardResult,
    Harness,
    Interrupt,
    InterruptReason,
    Model,
    Observation,
    RetryPolicy,
    RunResult,
    RunState,
    RunStatus,
    Runtime,
    TelemetryEvent,
    TelemetrySink,
    Tool,
    ToolCall,
    ToolContext,
    ToolExecutor,
    ToolResult,
    ToolSchema,
    ValidationResult,
)

# v0.2 message/content abstractions
from .blocks import (
    AudioBlock,
    AudioSource,
    BlockKind,
    ContentBlock,
    ImageBlock,
    ImageSource,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UnknownBlockTypeError,
    block_from_dict,
    block_to_dict,
)
from .capability import (
    CapabilityMismatchError,
    CapabilityPolicy,
    FailFastCapabilityPolicy,
    ModelCapabilities,
)
from .message import Message

__all__ = [
    "Checkpointer",
    "Compactor",
    "Decision",
    "DecisionKind",
    "GuardResult",
    "Harness",
    "Interrupt",
    "InterruptReason",
    "Model",
    "Observation",
    "RetryPolicy",
    "RunResult",
    "RunState",
    "RunStatus",
    "Runtime",
    "TelemetryEvent",
    "TelemetrySink",
    "Tool",
    "ToolCall",
    "ToolContext",
    "ToolExecutor",
    "ToolResult",
    "ToolSchema",
    "ValidationResult",
    # v0.2 message/content abstractions
    "AudioBlock",
    "AudioSource",
    "BlockKind",
    "ContentBlock",
    "ImageBlock",
    "ImageSource",
    "Role",
    "TextBlock",
    "ThinkingBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "UnknownBlockTypeError",
    "block_from_dict",
    "block_to_dict",
    "CapabilityMismatchError",
    "CapabilityPolicy",
    "FailFastCapabilityPolicy",
    "ModelCapabilities",
    "Message",
]
