"""Deprecated stub for backward compatibility.

The dedicated `Udpipe1Local` class has been removed in favour of the
generic `ExternalCommandBackend` + `udpipe1_backend()` factory. Import
from `xmltokenizer.backends` instead:

    from xmltokenizer.backends import udpipe1_backend
    backend = udpipe1_backend("path/to/model.udpipe")

This module re-exports the factory under the old name for callers that
still import `Udpipe1Local` directly. It will be removed in a future
release.
"""

from .base import udpipe1_backend as Udpipe1Local  # noqa: F401

__all__ = ["Udpipe1Local"]
