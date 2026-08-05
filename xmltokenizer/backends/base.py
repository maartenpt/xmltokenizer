"""Backend protocol and a generic external-command backend.

xmltokenizer is a *library*. Backends here are conveniences for the
standalone CLI — flexipipe (the default) and the integration test path
using local UDPipe v1. Production callers like flexipipe never go
through this module; they run their own pipeline and call the library
API directly.
"""

from __future__ import annotations

import os
import shutil
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TokenizerBackend(Protocol):
    """A callable that turns plaintext into CoNLL-U.

    Implementations must be safe to call sequentially from a single
    thread. Concurrency is the orchestrator's responsibility.
    """

    #: Soft limit on plaintext length per `tokenize()` call. 0 = no limit.
    max_chunk_chars: int

    def tokenize(self, plaintext: str) -> str:
        ...


class BackendError(RuntimeError):
    """Raised when the backend command fails irrecoverably."""


@dataclass
class ExternalCommandBackend:
    """Generic backend that pipes plaintext through an external command.

    The command is expected to consume plaintext on **stdin** and emit
    CoNLL-U on **stdout**. Most NLP toolchains can be invoked this way
    (flexipipe, udpipe, custom shells, etc.).

    Parameters
    ----------
    argv : list[str]
        The full command line, e.g. ``["flexipipe", "--task=tokenize", ...]``
        or ``["udpipe", "model.udpipe", "--tokenize", "--tag", "--parse"]``.
    max_chunk_chars : int
        Backend size limit hint for the orchestrator. 0 means "no limit"
        (use for local processes like UDPipe v1). Set explicitly when
        wrapping a REST-backed flexipipe with a known payload cap.
    timeout_seconds : float, optional
        Per-call timeout. None = no timeout.
    extra_env : dict[str, str], optional
        Environment variables to merge into os.environ for the child.
    """

    argv: list[str]
    max_chunk_chars: int = 0
    timeout_seconds: Optional[float] = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("argv must not be empty")
        # Verify the binary exists. We allow either a PATH lookup or an
        # absolute / relative path that exists on disk.
        binary = self.argv[0]
        if shutil.which(binary) is None and not os.path.exists(binary):
            raise BackendError(
                f"command {binary!r} not found on PATH and not an existing "
                f"file. Adjust `argv` or set the binary's full path."
            )

    def tokenize(self, plaintext: str) -> str:
        env = None
        if self.extra_env:
            env = {**os.environ, **self.extra_env}
        try:
            proc = subprocess.run(
                self.argv,
                input=plaintext.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            raise BackendError(
                f"command {shlex.join(self.argv)!r} timed out after "
                f"{self.timeout_seconds}s"
            ) from e
        except FileNotFoundError as e:
            raise BackendError(
                f"could not exec {self.argv[0]!r}: {e}"
            ) from e
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise BackendError(
                f"{shlex.join(self.argv)!r} exited with status "
                f"{proc.returncode}: {stderr.strip()}"
            )
        return proc.stdout.decode("utf-8")


# ---------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------


def flexipipe_backend(
    *,
    command: str = "flexipipe",
    tasks: str = "tokenize,tag,parse",
    language: Optional[str] = None,
    extra_args: tuple[str, ...] = (),
    timeout_seconds: Optional[float] = None,
    max_chunk_chars: int = 0,
) -> ExternalCommandBackend:
    """The **default** backend for the standalone CLI.

    Builds an ``ExternalCommandBackend`` invoking ``flexipipe`` with a
    typical task list. The exact flag names may need adjusting per the
    deployed flexipipe version — override via `command` and `extra_args`.

    The expectation is that flexipipe reads plaintext on stdin and emits
    CoNLL-U on stdout. If your flexipipe build uses different I/O, fall
    back to constructing ``ExternalCommandBackend`` manually.

  `language` is passed as ``--language=…`` when set. Flexipipe requires
    this for raw plaintext input (no ``# language =`` header on stdin).
    """
    argv = [command, f"--tasks={tasks}"]
    if language:
        argv.append(f"--language={language}")
    argv.extend(extra_args)
    return ExternalCommandBackend(
        argv=argv,
        max_chunk_chars=max_chunk_chars,
        timeout_seconds=timeout_seconds,
    )


def udpipe1_backend(
    model_path: str,
    *,
    udpipe_binary: str = "udpipe",
    extra_args: tuple[str, ...] = ("--tokenize", "--tag", "--parse"),
    timeout_seconds: Optional[float] = None,
) -> ExternalCommandBackend:
    """Convenience for running UDPipe v1 locally (testing path).

    Not the default — flexipipe is. Useful when flexipipe is not
    installed or you want to exercise the pipeline against a local
    UDPipe model file directly.
    """
    if not os.path.exists(model_path):
        raise BackendError(f"UDPipe model not found: {model_path}")
    argv = [udpipe_binary, model_path, *extra_args]
    return ExternalCommandBackend(
        argv=argv,
        max_chunk_chars=0,  # local UDPipe v1 has no hard size limit
        timeout_seconds=timeout_seconds,
    )
