"""Secret backends.

Production deployments should NEVER hardcode API keys. This module provides
a uniform ``SecretBackend`` protocol with three batteries-included backends:

- ``EnvSecretBackend``: read from process environment (12-factor friendly).
- ``DotenvSecretBackend``: read from a .env file (dev / local).
- ``ChainSecretBackend``: try multiple backends in order (typical: env + .env).

Future backends (Vault, AWS Secrets Manager, GCP Secret Manager) implement
the same protocol and slot in without code changes elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Protocol, runtime_checkable


class SecretNotFoundError(KeyError):
    """Raised when a backend can't find the requested secret."""


@runtime_checkable
class SecretBackend(Protocol):
    def get(self, key: str) -> str: ...


class InMemorySecretBackend:
    """Dictionary-backed backend, mainly for tests."""

    def __init__(self, data: Optional[Mapping[str, str]] = None) -> None:
        self._data: dict[str, str] = dict(data or {})

    def get(self, key: str) -> str:
        if key not in self._data:
            raise SecretNotFoundError(key)
        return self._data[key]

    def set(self, key: str, value: str) -> None:
        self._data[key] = value


class EnvSecretBackend:
    """Read secrets from ``os.environ``. Optional prefix gets prepended."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def get(self, key: str) -> str:
        full = f"{self.prefix}{key}" if self.prefix else key
        value = os.environ.get(full)
        if value is None:
            raise SecretNotFoundError(full)
        return value


class DotenvSecretBackend:
    """Read secrets from a .env file. Loaded once at construction time."""

    def __init__(self, path: str | Path = ".env") -> None:
        self._path = Path(path)
        self._data: dict[str, str] = self._load(self._path)

    @staticmethod
    def _load(path: Path) -> dict[str, str]:
        data: dict[str, str] = {}
        if not path.exists():
            return data
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            data[key] = value
        return data

    def get(self, key: str) -> str:
        if key not in self._data:
            raise SecretNotFoundError(key)
        return self._data[key]


class ChainSecretBackend:
    """Try each backend in order; first hit wins."""

    def __init__(self, *backends: SecretBackend) -> None:
        if not backends:
            raise ValueError("ChainSecretBackend requires at least one backend")
        self.backends = backends

    def get(self, key: str) -> str:
        for backend in self.backends:
            try:
                return backend.get(key)
            except SecretNotFoundError:
                continue
        raise SecretNotFoundError(key)
