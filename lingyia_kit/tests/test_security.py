"""Tests for secret backends, PII redaction, and tool permission gating."""
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from lingyia_core import (
    Decision,
    Harness,
    Role,
    Runtime,
    Tool,
    ToolResult,
    ToolResultBlock,
    ToolUseBlock,
    ValidationResult,
)
from lingyia_core.defaults.telemetry import NoopTelemetry
from lingyia_kit.redaction import RegexRedactor
from lingyia_kit.secrets import (
    ChainSecretBackend,
    DotenvSecretBackend,
    EnvSecretBackend,
    InMemorySecretBackend,
    SecretNotFoundError,
)


# Secrets ----------------------------------------------------------------


class SecretBackendTests(unittest.TestCase):
    def test_env_backend_reads_from_environ(self):
        os.environ["TEST_K1"] = "v1"
        try:
            self.assertEqual(EnvSecretBackend().get("TEST_K1"), "v1")
        finally:
            del os.environ["TEST_K1"]

    def test_env_backend_missing_raises(self):
        with self.assertRaises(SecretNotFoundError):
            EnvSecretBackend().get("UNLIKELY_TO_EXIST_KEY_xxx")

    def test_env_backend_prefix(self):
        os.environ["AGENT_K2"] = "v2"
        try:
            be = EnvSecretBackend(prefix="AGENT_")
            self.assertEqual(be.get("K2"), "v2")
        finally:
            del os.environ["AGENT_K2"]

    def test_dotenv_backend(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text(
                "# comment\n"
                "API_KEY=plain\n"
                'QUOTED="double"\n'
                "SINGLEQ='single'\n"
                "EMPTY=\n"
            )
            be = DotenvSecretBackend(path)
            self.assertEqual(be.get("API_KEY"), "plain")
            self.assertEqual(be.get("QUOTED"), "double")
            self.assertEqual(be.get("SINGLEQ"), "single")
            self.assertEqual(be.get("EMPTY"), "")
            with self.assertRaises(SecretNotFoundError):
                be.get("MISSING")

    def test_chain_first_match_wins(self):
        primary = InMemorySecretBackend({"X": "from_primary"})
        secondary = InMemorySecretBackend({"X": "from_secondary", "Y": "only_secondary"})
        chain = ChainSecretBackend(primary, secondary)
        self.assertEqual(chain.get("X"), "from_primary")
        self.assertEqual(chain.get("Y"), "only_secondary")
        with self.assertRaises(SecretNotFoundError):
            chain.get("Z")

    def test_chain_requires_at_least_one(self):
        with self.assertRaises(ValueError):
            ChainSecretBackend()


# Redaction --------------------------------------------------------------


class RegexRedactorTests(unittest.TestCase):
    def test_redacts_common_pii(self):
        r = RegexRedactor()
        text = (
            "Contact alice@example.com or call 13912345678 "
            "or 415-555-0100. Card: 4242 4242 4242 4242. "
            "Key: sk-abcdefghijklmnopqrstuvwxyz12345."
        )
        out = r.redact_text(text)
        self.assertIn("[REDACTED_EMAIL]", out)
        self.assertIn("[REDACTED_PHONE]", out)
        self.assertIn("[REDACTED_CARD]", out)
        self.assertIn("[REDACTED_KEY]", out)
        self.assertNotIn("alice@example.com", out)
        self.assertNotIn("13912345678", out)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz12345", out)

    def test_sensitive_keys_get_masked_regardless(self):
        r = RegexRedactor(replace_in_keys={"password", "api_key"})
        payload = {"password": "anything", "api_key": "still-masked", "name": "alice"}
        out = r(payload)
        self.assertEqual(out["password"], "[REDACTED]")
        self.assertEqual(out["api_key"], "[REDACTED]")
        self.assertEqual(out["name"], "alice")

    def test_nested_dict_and_list(self):
        r = RegexRedactor()
        payload = {
            "user": {"email": "alice@example.com"},
            "contacts": ["b@b.com", {"phone": "13912345678"}],
        }
        out = r(payload)
        self.assertEqual(out["user"]["email"], "[REDACTED_EMAIL]")
        self.assertEqual(out["contacts"][0], "[REDACTED_EMAIL]")
        self.assertEqual(out["contacts"][1]["phone"], "[REDACTED_PHONE]")


# Tool permission --------------------------------------------------------


def _noop_handler(args, ctx):
    return ToolResult(tool_name="noop", ok=True, output="ran")


def _harness_with_restricted_tool(granted: frozenset) -> Harness:
    tool = Tool.from_sync(
        name="dangerous",
        description="sends an email",
        handler=_noop_handler,
        required_permissions=frozenset({"network.write"}),
    )
    return Harness(
        tools=[tool],
        granted_permissions=granted,
        validator=lambda s: ValidationResult(),
    )


class _CallToolModel:
    async def adecide(self, ctx, state, tools):
        return Decision.call_tools([
            ToolUseBlock(id="call-1", name="dangerous", input={}),
        ])


def _last_tool_result(result) -> ToolResultBlock:
    """Find the most recent ToolResultBlock in state.messages."""
    for msg in reversed(result.state.messages):
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                return block
    raise AssertionError("no ToolResultBlock in transcript")


class ToolPermissionTests(unittest.TestCase):
    def test_default_wildcard_allows_everything(self):
        # default Harness has frozenset({"*"})
        harness = _harness_with_restricted_tool(frozenset({"*"}))

        rt = Runtime.dev(model=_CallToolModel(), max_iterations=2)
        rt.telemetry = NoopTelemetry()
        result = asyncio.run(rt.arun(harness, "do it"))
        # Tool ran — find a successful ToolResultBlock
        ok_results = [
            b for m in result.state.messages
            for b in m.content
            if isinstance(b, ToolResultBlock) and not b.is_error
        ]
        self.assertTrue(ok_results, "no successful tool result found")

    def test_missing_permission_blocks_tool(self):
        # Grant only read perm; tool needs network.write
        harness = _harness_with_restricted_tool(frozenset({"fs.read"}))

        rt = Runtime.dev(model=_CallToolModel(), max_iterations=2)
        rt.telemetry = NoopTelemetry()
        result = asyncio.run(rt.arun(harness, "do it"))
        last = _last_tool_result(result)
        self.assertTrue(last.is_error)
        self.assertIn("permission denied", last.content)
        self.assertIn("network.write", last.content)

    def test_granted_permission_allows_tool(self):
        harness = _harness_with_restricted_tool(
            frozenset({"network.write", "fs.read"})
        )
        rt = Runtime.dev(model=_CallToolModel(), max_iterations=2)
        rt.telemetry = NoopTelemetry()
        result = asyncio.run(rt.arun(harness, "do it"))
        last = _last_tool_result(result)
        self.assertFalse(last.is_error)


if __name__ == "__main__":
    unittest.main()
