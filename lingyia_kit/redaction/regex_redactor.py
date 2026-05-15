"""Regex-based PII redactor.

Use as the ``redactor`` argument of a telemetry sink to mask PII before it
leaves the process. Default patterns cover common formats; pass ``patterns``
to override or extend.

Limitations:
- Best-effort only. Adversarial inputs can bypass any regex.
- For regulated data (PHI, payment) combine with structured field
  redaction at the source, not just on the way out.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional


# (regex, replacement) pairs. Order matters because some patterns overlap.
_DEFAULT_PATTERNS: dict[str, tuple[str, str]] = {
    "email": (
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED_EMAIL]",
    ),
    "credit_card": (
        r"\b(?:\d[ -]?){13,19}\b",
        "[REDACTED_CARD]",
    ),
    "id_card_cn": (
        r"\b\d{17}[\dXx]\b",
        "[REDACTED_ID]",
    ),
    "phone_cn": (
        r"\b1[3-9]\d{9}\b",
        "[REDACTED_PHONE]",
    ),
    "phone_us": (
        r"\b\d{3}[ -]?\d{3}[ -]?\d{4}\b",
        "[REDACTED_PHONE]",
    ),
    "api_key_sk": (
        r"\bsk-[A-Za-z0-9]{20,}\b",
        "[REDACTED_KEY]",
    ),
}


class RegexRedactor:
    def __init__(
        self,
        patterns: Optional[Mapping[str, tuple[str, str]]] = None,
        *,
        replace_in_keys: Optional[set[str]] = None,
    ) -> None:
        """
        Args:
            patterns: name -> (regex, replacement). Defaults to a builtin set.
            replace_in_keys: dict keys whose values should always be masked
                regardless of content (e.g. {"api_key", "password"}).
        """
        self._patterns = patterns or _DEFAULT_PATTERNS
        self._compiled = [
            (re.compile(pat), repl) for pat, repl in self._patterns.values()
        ]
        self._sensitive_keys = {k.lower() for k in (replace_in_keys or set())}

    def redact_text(self, text: str) -> str:
        out = text
        for pattern, repl in self._compiled:
            out = pattern.sub(repl, out)
        return out

    def __call__(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._redact_dict(payload)

    def _redact_dict(self, d: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(k, str) and k.lower() in self._sensitive_keys:
                out[k] = "[REDACTED]"
            else:
                out[k] = self._redact_value(v)
        return out

    def _redact_value(self, v: Any) -> Any:
        if isinstance(v, str):
            return self.redact_text(v)
        if isinstance(v, dict):
            return self._redact_dict(v)
        if isinstance(v, (list, tuple)):
            return type(v)(self._redact_value(x) for x in v)
        return v
