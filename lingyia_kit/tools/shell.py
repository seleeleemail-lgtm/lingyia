"""Shell tool (restricted).

The default configuration disables shell execution. Callers must explicitly
opt in by passing ``enabled=True`` AND a non-empty allow list of commands.
The agent never runs arbitrary shell.
"""
from __future__ import annotations

import asyncio
import shlex
from typing import Iterable, Mapping, Optional

from lingyia_core import Tool, ToolContext, ToolResult


def run_shell_tool(
    *,
    enabled: bool = False,
    allowed_commands: Optional[Iterable[str]] = None,
    timeout_s: float = 30.0,
    max_output_bytes: int = 200_000,
) -> Tool:
    """Build a ``run_shell`` tool. Disabled by default."""

    allowed = set(allowed_commands or [])

    async def _handler(args: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        if not enabled:
            return ToolResult(
                tool_name="run_shell",
                ok=False,
                error="run_shell is disabled in this runtime",
            )
        cmd = args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return ToolResult(tool_name="run_shell", ok=False, error="missing 'command'")
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            return ToolResult(tool_name="run_shell", ok=False, error=f"unparseable: {e}")
        if not parts:
            return ToolResult(tool_name="run_shell", ok=False, error="empty command")
        if allowed and parts[0] not in allowed:
            return ToolResult(
                tool_name="run_shell",
                ok=False,
                error=f"command '{parts[0]}' not in allow list",
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    tool_name="run_shell",
                    ok=False,
                    error=f"timeout after {timeout_s}s",
                )
            out = stdout[:max_output_bytes].decode("utf-8", errors="replace")
            err = stderr[:max_output_bytes].decode("utf-8", errors="replace")
            return ToolResult(
                tool_name="run_shell",
                ok=proc.returncode == 0,
                output={"stdout": out, "stderr": err, "exit_code": proc.returncode},
                error="" if proc.returncode == 0 else f"exit code {proc.returncode}",
            )
        except Exception as e:
            return ToolResult(tool_name="run_shell", ok=False, error=str(e))

    return Tool.from_async(
        name="run_shell",
        description="Run a shell command from a restricted allow list and return stdout/stderr.",
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The full shell command line, e.g. 'ls -la /tmp'.",
                },
            },
            "required": ["command"],
        },
        timeout_s=timeout_s,
        side_effect="external_io",
        idempotent=False,
    )
