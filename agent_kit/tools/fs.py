"""Filesystem tools.

Side-effect classification:
- ``read_file``: read_only, idempotent
- ``write_file``: write, NOT idempotent (overwrites)
- ``list_dir``: read_only, idempotent
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from agent_core import Tool, ToolContext, ToolResult


def read_file_tool(root: str | None = None, max_bytes: int = 1_000_000) -> Tool:
    """Build a ``read_file`` tool optionally rooted at ``root`` for safety."""

    root_path = Path(root).resolve() if root else None

    def _handler(args: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return ToolResult(tool_name="read_file", ok=False, error="missing 'path'")
        target = Path(path).expanduser().resolve()
        if root_path and root_path not in target.parents and target != root_path:
            return ToolResult(
                tool_name="read_file",
                ok=False,
                error=f"path '{target}' is outside the allowed root '{root_path}'",
            )
        try:
            size = target.stat().st_size
            if size > max_bytes:
                return ToolResult(
                    tool_name="read_file",
                    ok=False,
                    error=f"file too large ({size} bytes, max {max_bytes})",
                )
            text = target.read_text(encoding="utf-8", errors="replace")
            return ToolResult(tool_name="read_file", ok=True, output=text)
        except FileNotFoundError:
            return ToolResult(tool_name="read_file", ok=False, error=f"not found: {target}")
        except Exception as e:
            return ToolResult(tool_name="read_file", ok=False, error=str(e))

    return Tool.from_sync(
        name="read_file",
        description="Read the UTF-8 text contents of a file from disk.",
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or user-relative path to a text file.",
                },
            },
            "required": ["path"],
        },
        side_effect="read_only",
        idempotent=True,
    )


def write_file_tool(root: str | None = None) -> Tool:
    """Build a ``write_file`` tool optionally rooted at ``root`` for safety."""

    root_path = Path(root).resolve() if root else None

    def _handler(args: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        path = args.get("path")
        content = args.get("content")
        if not isinstance(path, str) or not path:
            return ToolResult(tool_name="write_file", ok=False, error="missing 'path'")
        if not isinstance(content, str):
            return ToolResult(tool_name="write_file", ok=False, error="missing 'content' (string)")
        target = Path(path).expanduser().resolve()
        if root_path and root_path not in target.parents and target != root_path:
            return ToolResult(
                tool_name="write_file",
                ok=False,
                error=f"path '{target}' is outside the allowed root '{root_path}'",
            )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult(
                tool_name="write_file",
                ok=True,
                output={"path": str(target), "bytes": len(content.encode("utf-8"))},
            )
        except Exception as e:
            return ToolResult(tool_name="write_file", ok=False, error=str(e))

    return Tool.from_sync(
        name="write_file",
        description="Write UTF-8 text to a file, creating parent directories. Overwrites if exists.",
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Target file path."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
        side_effect="write",
        idempotent=False,
    )


def list_dir_tool(root: str | None = None) -> Tool:
    root_path = Path(root).resolve() if root else None

    def _handler(args: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        path = args.get("path", ".")
        if not isinstance(path, str):
            return ToolResult(tool_name="list_dir", ok=False, error="'path' must be a string")
        target = Path(path).expanduser().resolve()
        if root_path and root_path not in target.parents and target != root_path:
            return ToolResult(
                tool_name="list_dir",
                ok=False,
                error=f"path '{target}' is outside the allowed root '{root_path}'",
            )
        try:
            entries = []
            for entry in sorted(os.listdir(target)):
                full = target / entry
                entries.append({
                    "name": entry,
                    "is_dir": full.is_dir(),
                    "size": full.stat().st_size if full.is_file() else 0,
                })
            return ToolResult(tool_name="list_dir", ok=True, output=entries)
        except Exception as e:
            return ToolResult(tool_name="list_dir", ok=False, error=str(e))

    return Tool.from_sync(
        name="list_dir",
        description="List entries in a directory.",
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to list. Defaults to current dir."},
            },
        },
        side_effect="read_only",
        idempotent=True,
    )
