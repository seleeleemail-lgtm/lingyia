"""Tests for the generic tool implementations."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from lingyia_core import ToolContext
from lingyia_kit.tools import (
    list_dir_tool,
    read_file_tool,
    run_shell_tool,
    write_file_tool,
)


def _ctx() -> ToolContext:
    return ToolContext(run_id="test", iteration=0, messages=(), metadata={})


def _run_sync(tool, args):
    """Tools can be sync or async; this normalizes to a single call path."""
    if tool.is_async:
        return asyncio.run(tool.handler(args, _ctx()))
    return tool.handler(args, _ctx())


class FilesystemToolTests(unittest.TestCase):
    def test_read_write_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "hello.txt")
            wt = write_file_tool(root=d)
            rt = read_file_tool(root=d)
            wr = _run_sync(wt, {"path": path, "content": "hi 你好"})
            self.assertTrue(wr.ok)
            self.assertEqual(wr.output["bytes"], len("hi 你好".encode("utf-8")))
            rr = _run_sync(rt, {"path": path})
            self.assertTrue(rr.ok)
            self.assertEqual(rr.output, "hi 你好")

    def test_root_sandbox_blocks_escape(self):
        with tempfile.TemporaryDirectory() as d:
            rt = read_file_tool(root=d)
            r = _run_sync(rt, {"path": "/etc/passwd"})
            self.assertFalse(r.ok)
            self.assertIn("outside", r.error)

    def test_list_dir_returns_entries(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.txt").write_text("x")
            (Path(d) / "sub").mkdir()
            lt = list_dir_tool(root=d)
            r = _run_sync(lt, {"path": d})
            self.assertTrue(r.ok)
            names = {e["name"] for e in r.output}
            self.assertIn("a.txt", names)
            self.assertIn("sub", names)

    def test_read_missing_file_returns_error(self):
        with tempfile.TemporaryDirectory() as d:
            rt = read_file_tool(root=d)
            r = _run_sync(rt, {"path": str(Path(d) / "nope")})
            self.assertFalse(r.ok)
            self.assertIn("not found", r.error)


class ShellToolTests(unittest.TestCase):
    def test_disabled_by_default(self):
        st = run_shell_tool()
        r = _run_sync(st, {"command": "echo hi"})
        self.assertFalse(r.ok)
        self.assertIn("disabled", r.error)

    def test_allow_list_blocks_unknown(self):
        st = run_shell_tool(enabled=True, allowed_commands=["echo"])
        r = _run_sync(st, {"command": "rm -rf /"})
        self.assertFalse(r.ok)
        self.assertIn("not in allow list", r.error)

    def test_allowed_command_runs(self):
        st = run_shell_tool(enabled=True, allowed_commands=["echo"])
        r = _run_sync(st, {"command": "echo hi"})
        self.assertTrue(r.ok)
        self.assertIn("hi", r.output["stdout"])


if __name__ == "__main__":
    unittest.main()
