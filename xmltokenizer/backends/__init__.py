"""Optional backends for the standalone CLI / convenience orchestrator.

xmltokenizer is primarily a library — most consumers (notably flexipipe)
already produce CoNLL-U themselves and use the library API directly. The
backends in this package are conveniences for the standalone CLI case.

The single concrete backend is `ExternalCommandBackend`, which runs any
external command that consumes plaintext on stdin and emits CoNLL-U on
stdout. Two convenience constructors are provided:

- `flexipipe_backend()`  — the **default** for the standalone CLI.
  Configurable per the deployment's flexipipe invocation.
- `udpipe1_backend(model_path)` — for running UDPipe v1 directly without
  flexipipe (useful for testing or scripts that don't have flexipipe
  available).

The user is free to construct their own `ExternalCommandBackend` for any
other pipeline that follows the same stdin/stdout contract.
"""

from __future__ import annotations

from typing import Any

from .base import (
    BackendError,
    ExternalCommandBackend,
    TokenizerBackend,
    flexipipe_backend,
    udpipe1_backend,
)
from .naive import NaiveBackend


def get(name: str, **kwargs: Any) -> TokenizerBackend:
    """Construct a backend by short name. Used by the CLI."""
    if name == "naive":
        return NaiveBackend(**kwargs)
    if name == "flexipipe":
        return flexipipe_backend(**kwargs)
    if name == "udpipe1-local":
        return udpipe1_backend(**kwargs)
    if name == "command":
        return ExternalCommandBackend(**kwargs)
    raise ValueError(
        f"unknown backend {name!r}. "
        f"Known names: naive (default), flexipipe, udpipe1-local, command."
    )


__all__ = [
    "TokenizerBackend",
    "ExternalCommandBackend",
    "NaiveBackend",
    "BackendError",
    "flexipipe_backend",
    "udpipe1_backend",
    "get",
]
