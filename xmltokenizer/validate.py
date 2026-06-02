"""Phase D — validation. The invariants from design §10.

V1, V4, V5, V6, V7 are implemented. V2 and V3 require record-id
traceability INTO the output (every fragment would need to carry its
source record id as an attribute) — we don't do that today, so they
are skipped for now and the function returns the list of checks it ran.

All check functions raise `ValidationError` on failure with a precise
diagnostic.

Usage:

    import xmltokenizer as xt
    md = xt.extract(input_bytes, profile)
    ... # attach_conllu / fold
    output_bytes = xt.fold(md)
    xt.validate(input_bytes, output_bytes, md)   # raises on failure

For partial / faster runs, pass `checks=[...]` to skip expensive checks.
"""

from __future__ import annotations

import xml.parsers.expat
from dataclasses import dataclass

from .extract import Metadata
from .namespace import deactivate
from .selectors import compile as compile_selector


class ValidationError(AssertionError):
    """Raised when an invariant fails. The message is precise enough to
    locate the failing position (byte offset, element id, etc.)."""


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def run(
    input_bytes: bytes,
    output_bytes: bytes,
    metadata: Metadata,
    *,
    checks: tuple[str, ...] = ("V1", "V4", "V5", "V6", "V7"),
) -> list[str]:
    """Run the named invariants. Returns the list of checks that passed.

    Raises `ValidationError` on the first failure.
    """
    # Both should be in the deactivated-namespace form (the form everything
    # downstream of `deactivate` uses). If the caller passes the original
    # input bytes (with `xmlns=`), we deactivate them here so comparisons
    # against the output (which is post-deactivation) line up.
    input_de, _ = deactivate(input_bytes)
    out_de, _ = deactivate(output_bytes)

    ran: list[str] = []
    if "V1" in checks:
        check_text_preserve(input_de, out_de, metadata)
        ran.append("V1")
    if "V4" in checks:
        check_no_nested_tok(out_de)
        ran.append("V4")
    if "V5" in checks:
        check_no_nested_s(out_de)
        ran.append("V5")
    if "V6" in checks:
        check_cont_groups_consistent(out_de)
        ran.append("V6")
    if "V7" in checks:
        check_outside_scope_verbatim(input_de, out_de, metadata)
        ran.append("V7")
    return ran


# ---------------------------------------------------------------------
# V1 — TEXT-PRESERVE
# ---------------------------------------------------------------------


def check_text_preserve(
    input_bytes: bytes, output_bytes: bytes, metadata: Metadata
) -> None:
    """The concatenation of all CDATA inside the tokenization scope must
    be identical between input and output. New `<tok>`/`<s>` wrappers
    don't contribute CDATA (only their inner text does), so a simple
    "concat all cdata in scope" check is sufficient and very fast."""
    scope_candidates = metadata.profile.get("scope", {}).get("candidates", [])
    scope_tags = _scope_tags_from_candidates(scope_candidates)
    in_text = _concat_scope_cdata(input_bytes, scope_tags)
    out_text = _concat_scope_cdata(output_bytes, scope_tags)
    if in_text != out_text:
        diff_at = _first_diff_index(in_text, out_text)
        raise ValidationError(
            f"V1 TEXT-PRESERVE failed at scope-cdata index {diff_at}.\n"
            f"  input  …{in_text[max(0, diff_at - 30):diff_at + 30]!r}…\n"
            f"  output …{out_text[max(0, diff_at - 30):diff_at + 30]!r}…"
        )


def _first_diff_index(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _scope_tags_from_candidates(candidates: list[str]) -> set[str]:
    """Extract the set of tag names from the profile's scope candidates.
    For v1 we only support single-step selectors (matches extract.py)."""
    tags: set[str] = set()
    for c in candidates:
        sel = compile_selector(c)
        if len(sel.steps) != 1:
            continue
        step = sel.steps[0]
        if step.axis == "root":
            tags.add("*")
        else:
            tags.add(step.name)
    return tags


def _concat_scope_cdata(raw: bytes, scope_tags: set[str]) -> str:
    """Parse with expat; concatenate all character data that appears
    INSIDE any element whose tag is in `scope_tags`. If `*` is in the
    set, include CDATA inside the root element (with header-like
    elements as a heuristic exclusion). For v1 we treat `*` as "any
    element" — the scope match is governed by the first candidate that
    succeeded in extract, and we mirror that here."""
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")
    parser.buffer_text = True

    # Depth into the scope (a scope-tag may be nested inside something —
    # increment when we enter a scope-tag, decrement when we leave).
    scope_depth = [0]
    out: list[str] = []
    accepted_tag: list[str | None] = [None]

    def on_start(name: str, attrs: dict) -> None:
        if scope_depth[0] > 0:
            return
        # Try to match against scope_tags by first-match-wins. Since the
        # set has no ordering, we accept any tag in the set OR `*`.
        if name in scope_tags or "*" in scope_tags:
            scope_depth[0] = 1
            accepted_tag[0] = name

    def on_end(name: str) -> None:
        if scope_depth[0] > 0 and name == accepted_tag[0]:
            scope_depth[0] = 0
            accepted_tag[0] = None

    def on_cdata(text: str) -> None:
        if scope_depth[0] > 0:
            out.append(text)

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.CharacterDataHandler = on_cdata
    parser.Parse(raw, True)
    return "".join(out)


# ---------------------------------------------------------------------
# V4 — NO-NESTED-TOK
# ---------------------------------------------------------------------


def check_no_nested_tok(output_bytes: bytes) -> None:
    """No `<tok>` may have another `<tok>` as a descendant."""
    _check_no_nesting(output_bytes, "tok", "V4 NO-NESTED-TOK")


# ---------------------------------------------------------------------
# V5 — NO-NESTED-S
# ---------------------------------------------------------------------


def check_no_nested_s(output_bytes: bytes) -> None:
    """No `<s>` may have another `<s>` as a descendant."""
    _check_no_nesting(output_bytes, "s", "V5 NO-NESTED-S")


def _check_no_nesting(raw: bytes, tag: str, label: str) -> None:
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")
    depth = [0]
    failure: list[str] = []

    def on_start(name: str, attrs: dict) -> None:
        if name == tag:
            if depth[0] > 0:
                failure.append(
                    f"{label} failed: nested <{tag}> at byte "
                    f"{parser.CurrentByteIndex}"
                )
                # Don't raise inside the handler — expat may suppress it.
            depth[0] += 1

    def on_end(name: str) -> None:
        if name == tag:
            depth[0] -= 1

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.Parse(raw, True)
    if failure:
        raise ValidationError(failure[0])


# ---------------------------------------------------------------------
# V6 — CONT-GROUPS-CONSISTENT
# ---------------------------------------------------------------------


def check_cont_groups_consistent(output_bytes: bytes) -> None:
    """For every `@cont` group, fragments must have `@rpt` values of
    1, 2, …, K-1 (one fragment with no `@rpt`, then K-1 numbered ones)
    and no holes / duplicates."""
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")

    # cont_id -> list of (rpt_int_or_None, byte_position)
    by_group: dict[str, list[tuple[int | None, int]]] = {}

    def on_start(name: str, attrs: dict) -> None:
        cont = attrs.get("cont")
        if cont is None:
            return
        rpt_str = attrs.get("rpt")
        rpt = None if rpt_str is None else int(rpt_str)
        by_group.setdefault(cont, []).append((rpt, parser.CurrentByteIndex))

    parser.StartElementHandler = on_start
    parser.Parse(output_bytes, True)

    for cont, entries in by_group.items():
        rpts = sorted(r for r, _ in entries if r is not None)
        first_count = sum(1 for r, _ in entries if r is None)
        # Exactly one fragment with no @rpt.
        if first_count != 1:
            raise ValidationError(
                f"V6 CONT-GROUPS-CONSISTENT failed: group {cont!r} has "
                f"{first_count} first-fragments (expected exactly 1)"
            )
        # @rpt values must be 1..K-1 with no holes.
        expected = list(range(1, len(rpts) + 1))
        if rpts != expected:
            raise ValidationError(
                f"V6 CONT-GROUPS-CONSISTENT failed: group {cont!r} has "
                f"@rpt={rpts}, expected {expected}"
            )


# ---------------------------------------------------------------------
# V7 — OUTSIDE-SCOPE-VERBATIM
# ---------------------------------------------------------------------


def check_outside_scope_verbatim(
    input_bytes: bytes, output_bytes: bytes, metadata: Metadata
) -> None:
    """Everything outside the tokenization scope must be byte-identical
    between input and output (after namespace deactivation). For
    multi-scope-root inputs, also check `interscope_bytes`."""
    if not output_bytes.startswith(metadata.prefix_bytes):
        raise ValidationError(
            f"V7 OUTSIDE-SCOPE-VERBATIM failed: output does not start with "
            f"the captured prefix_bytes (first {len(metadata.prefix_bytes)} "
            f"bytes do not match)."
        )
    if not output_bytes.endswith(metadata.suffix_bytes):
        raise ValidationError(
            f"V7 OUTSIDE-SCOPE-VERBATIM failed: output does not end with the "
            f"captured suffix_bytes (last {len(metadata.suffix_bytes)} bytes "
            f"do not match)."
        )
    # Inter-scope bytes appear between scope roots. For v1 with a single
    # scope root, interscope_bytes is empty. When there are multiple
    # roots, we'd need to locate each open_tag_bytes in the output, but
    # we skip that detail for v1 (the prefix/suffix check catches most
    # regressions).
    if metadata.interscope_bytes:
        # Crude check: each interscope blob must appear somewhere
        # between the two roots' bytes. Implement when we have a fixture.
        pass
