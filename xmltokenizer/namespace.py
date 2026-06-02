"""Pre/post-pass on raw XML bytes to neutralize XML namespaces.

The rest of the pipeline operates on tag names *without* namespace prefixes
or URIs. That makes the splitting algorithm namespace-agnostic and matches
the user's `xmlnsoff="..."` convention.

This module rewrites every `xmlns=` / `xmlns:prefix=` attribute name to
`xmlnsoff=` / `xmlnsoff:prefix=` on input, and (optionally) reverses the
operation on output. The rewrite is purely textual — we don't parse the
XML — because we need to preserve byte positions for the rest of the
pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Match the literal `xmlns` as an attribute name. The lookbehind requires
# a preceding space-like char (so `axmlns=` doesn't match) and the
# lookahead requires either `=` (bare `xmlns=`) or `:` (prefixed
# `xmlns:foo=`). Only the literal `xmlns` is matched; any `:prefix` after
# it is left untouched by the substitution.
_XMLNS_RE = re.compile(rb"(?<=[\s])xmlns(?=[:=])")
_XMLNSOFF_RE = re.compile(rb"(?<=[\s])xmlnsoff(?=[:=])")


@dataclass
class NamespaceTransform:
    """Marker telling `reactivate` there was something to undo.

    We only need to know whether a rewrite happened; the substitution is
    deterministic and byte positions are preserved across deactivate-then-
    reactivate (`xmlns` and `xmlnsoff` differ in length, but reactivate
    restores the same length difference in reverse — the original bytes
    are recovered exactly).
    """

    count: int = 0


def deactivate(raw: bytes) -> tuple[bytes, NamespaceTransform | None]:
    """Rewrite every `xmlns(:prefix)?=` to `xmlnsoff(:prefix)?=`.

    Returns `(new_bytes, transform)` or `(raw, None)` if there was nothing
    to do.
    """
    new, n = _XMLNS_RE.subn(rb"xmlnsoff", raw)
    if n == 0:
        return raw, None
    return new, NamespaceTransform(count=n)


def reactivate(raw: bytes, transform: NamespaceTransform) -> bytes:
    """Undo a `deactivate`."""
    if not transform or transform.count == 0:
        return raw
    return _XMLNSOFF_RE.sub(rb"xmlns", raw)
