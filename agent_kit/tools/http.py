"""HTTP tools (read-only fetch)."""
from __future__ import annotations

from typing import Mapping, Optional

import httpx

from agent_core import Tool, ToolContext, ToolResult


def fetch_url_tool(
    *,
    max_bytes: int = 2_000_000,
    timeout_s: float = 30.0,
    allow_hosts: Optional[set[str]] = None,
) -> Tool:
    """Build a ``fetch_url`` tool that fetches a URL via HTTP GET.

    ``allow_hosts``, if set, restricts what the agent can request. Use this in
    production to prevent SSRF or unintended traffic. Leave ``None`` to allow
    any public URL (suitable for dev/sandbox only).
    """

    async def _handler(args: Mapping[str, object], ctx: ToolContext) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url:
            return ToolResult(tool_name="fetch_url", ok=False, error="missing 'url'")
        if allow_hosts is not None:
            try:
                from urllib.parse import urlparse
                host = urlparse(url).hostname or ""
            except Exception as e:
                return ToolResult(tool_name="fetch_url", ok=False, error=f"bad url: {e}")
            if host not in allow_hosts:
                return ToolResult(
                    tool_name="fetch_url",
                    ok=False,
                    error=f"host '{host}' not in allow list",
                )
        try:
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as c:
                r = await c.get(url)
            body = r.text
            if len(body.encode("utf-8")) > max_bytes:
                body = body[: max_bytes // 2]
                truncated = True
            else:
                truncated = False
            return ToolResult(
                tool_name="fetch_url",
                ok=True,
                output={
                    "status_code": r.status_code,
                    "content_type": r.headers.get("content-type", ""),
                    "body": body,
                    "truncated": truncated,
                },
            )
        except Exception as e:
            return ToolResult(tool_name="fetch_url", ok=False, error=str(e))

    return Tool.from_async(
        name="fetch_url",
        description="Fetch a URL via HTTP GET and return the response body.",
        handler=_handler,
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch (http or https)."},
            },
            "required": ["url"],
        },
        timeout_s=timeout_s,
        side_effect="external_io",
        idempotent=True,
    )
