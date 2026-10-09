"""Phase D — validation. The invariants from design §10.

V1, V4, V5, V6, V7, V8, V9, V10 are implemented. V2 and V3 require record-id
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
    checks: tuple[str, ...] = ("V1", "V4", "V5", "V6", "V7", "V8", "V9", "V10"),
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
    if "V8" in checks:
        check_structure_preserve(input_de, out_de, metadata)
        ran.append("V8")
    if "V9" in checks:
        check_tok_atomic(out_de, structural=structural_tags(metadata.profile))
        ran.append("V9")
    if "V10" in checks and not metadata.profile.get("sentences_may_cross_barriers", False):
        check_s_within_block(out_de, metadata.profile)
        ran.append("V10")
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


# ---------------------------------------------------------------------
# V8 — STRUCTURE-PRESERVE
# ---------------------------------------------------------------------

# Attributes fold adds to source elements it splits into fragments.
_FRAGMENT_ATTRS = frozenset({"cont", "rpt"})


def _inserted_tags(metadata: Metadata) -> set[str]:
    """Tags of the elements tokenization inserts (tok, dtok, s, ...)."""
    tags = {"tok", "dtok", "s"}
    for root in metadata.scope_roots:
        if root.udpipe_layer is not None:
            tags.update(r.tag for r in root.udpipe_layer.records)
    return tags


def _structure_events(raw: bytes, drop_tags: set[str]) -> list[tuple]:
    """Start/end/text events of `raw` without the `drop_tags` elements (their
    content kept), with split fragments (`@rpt`) merged back into one element."""
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")
    parser.buffer_text = True
    parser.ordered_attributes = False
    events: list[tuple] = []

    def on_start(name: str, attrs: dict) -> None:
        if name in drop_tags:
            return
        if "rpt" in attrs and events:
            # glue the fragment onto its predecessor, also across whitespace
            # (edge whitespace may have been moved between the fragments)
            if events[-1] == ("end", name):
                events.pop()
                return
            if (len(events) >= 2 and events[-1][0] == "text" and not events[-1][1].strip()
                    and events[-2] == ("end", name)):
                ws = events.pop()
                events.pop()
                events.append(ws)
                return
        clean = tuple(sorted((k, v) for k, v in attrs.items() if k not in _FRAGMENT_ATTRS))
        events.append(("start", name, clean, parser.CurrentByteIndex))

    def on_end(name: str) -> None:
        if name in drop_tags:
            return
        events.append(("end", name))

    def on_cdata(text: str) -> None:
        if events and events[-1][0] == "text":
            events[-1] = ("text", events[-1][1] + text)
        else:
            events.append(("text", text))

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.CharacterDataHandler = on_cdata
    parser.Parse(raw, True)
    return events


def _hoist_whitespace(events: list[tuple]) -> list[tuple]:
    """Canonical whitespace placement: whitespace at the inner edge of an
    element moves just outside it (fold may do the same, see
    fold._hoist_edge_whitespace), so V8 compares structure against content.
    One pass: leading whitespace of a text is pushed before the run of start
    tags right in front of it, trailing whitespace after the run of end tags
    right behind it."""
    out: list[tuple] = []

    def emit_text(t: str) -> None:
        if not t:
            return
        if out and out[-1][0] == "text":
            out[-1] = ("text", out[-1][1] + t)
        else:
            out.append(("text", t))

    i, n = 0, len(events)
    while i < n:
        ev = events[i]
        if ev[0] == "text":
            emit_text(ev[1])
            i += 1
            continue
        kind = ev[0]
        j = i
        while j < n and events[j][0] == kind:
            j += 1
        run = events[i:j]
        if kind == "start" and j < n and events[j][0] == "text":
            t = events[j][1]
            lead = t[: len(t) - len(t.lstrip())]
            emit_text(lead)
            out.extend(run)
            emit_text(t[len(lead):])
            i = j + 1
        elif kind == "end" and out and out[-1][0] == "text":
            t = out[-1][1]
            trail = t[len(t.rstrip()):]
            if trail:
                out[-1] = ("text", t[: len(t) - len(trail)])
                if not out[-1][1]:
                    out.pop()
            out.extend(run)
            emit_text(trail)
            i = j
        else:
            out.extend(run)
            i = j
    return out


def _hoist_fully(events: list[tuple]) -> list[tuple]:
    """`_hoist_whitespace` until nothing moves (one pass per nesting level)."""
    while True:
        nxt = _hoist_whitespace(events)
        if nxt == events:
            return nxt
        events = nxt


def _event_key(ev: tuple) -> tuple:
    return ev[:3] if ev[0] == "start" else ev


def check_structure_preserve(
    input_bytes: bytes, output_bytes: bytes, metadata: Metadata
) -> None:
    """With the inserted elements removed (and split fragments merged), the
    output must have the input's elements in the input's order relative to
    the text: no source element may move (e.g. an <lb/> between </p> and <p>
    pulled into the next <p>)."""
    drop = _inserted_tags(metadata)
    a = _hoist_fully(_structure_events(input_bytes, drop))
    b = _hoist_fully(_structure_events(output_bytes, drop))
    n = min(len(a), len(b))
    for i in range(n + 1):
        ea = a[i] if i < len(a) else None
        eb = b[i] if i < len(b) else None
        if ea is None and eb is None:
            return
        if ea is not None and eb is not None and _event_key(ea) == _event_key(eb):
            continue
        at = eb[3] if eb is not None and eb[0] == "start" else None

        def show(ev):
            if ev is None:
                return "end of document"
            if ev[0] == "start":
                return f"<{ev[1]}{''.join(f' {k}={v!r}' for k, v in ev[2])}>"
            if ev[0] == "end":
                return f"</{ev[1]}>"
            return f"text {ev[1][:40]!r}"

        ctx = "".join(show(e) for e in a[max(0, i - 3):i])
        raise ValidationError(
            f"V8 STRUCTURE-PRESERVE failed at structure event {i}"
            + (f" (output byte {at})" if at is not None else "")
            + f": after {ctx!r} the input has {show(ea)}, the output {show(eb)}."
        )


# ---------------------------------------------------------------------
# V9 — TOK-ATOMIC
# ---------------------------------------------------------------------


def structural_tags(profile: dict) -> set[str]:
    """Elements a token can never cross or contain (as in fold._nest_rank)."""
    tags: set[str] = set()
    for key in ("barrier_elements", "unsplittable", "token_break_elements",
                "chunk_boundary_elements"):
        tags.update(profile.get(key, []) or [])
    return tags


def check_tok_atomic(
    output_bytes: bytes,
    tags: tuple[str, ...] = ("tok", "dtok"),
    structural: frozenset[str] | set[str] = frozenset(),
) -> None:
    """Tokens are the atomic units of NLP: never split into fragments, and never
    around a structural element (a <p> inside a <tok> is always wrong). A source
    element crossing a token boundary splits instead (see fold._lift_atomic)."""
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")
    failure: list[str] = []
    in_tok = [0]

    def on_start(name: str, attrs: dict) -> None:
        if failure:
            return
        if name in tags:
            if "rpt" in attrs:
                failure.append(
                    f"V9 TOK-ATOMIC failed: <{name}> split into fragments at byte "
                    f"{parser.CurrentByteIndex} (rpt={attrs['rpt']!r})"
                )
            in_tok[0] += 1
        elif in_tok[0] and name in structural:
            failure.append(
                f"V9 TOK-ATOMIC failed: structural <{name}> inside a token at byte "
                f"{parser.CurrentByteIndex}"
            )

    def on_end(name: str) -> None:
        if name in tags:
            in_tok[0] -= 1

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.Parse(output_bytes, True)
    if failure:
        raise ValidationError(failure[0])


# ---------------------------------------------------------------------
# V10 — S-WITHIN-BLOCK
# ---------------------------------------------------------------------


def check_s_within_block(output_bytes: bytes, profile: dict) -> None:
    """A sentence never crosses a barrier element (<p>, <head>, ...: the
    profile's barrier_elements / unsplittable) unless the profile sets
    `sentences_may_cross_barriers`. Checked on the sentence's tokens: those
    inside a wrapping <s>, or listed in an anchor <s/>'s @corresp."""
    blocks = set(profile.get("barrier_elements", []) or []) | set(profile.get("unsplittable", []) or [])
    s_attrs = profile.get("s_attrs", {}) or {}
    tok_attrs = profile.get("tok_attrs", {}) or {}
    tok_id_attr = tok_attrs.get("id_attr", "xml:id")
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")
    block_stack: list[int] = []
    block_counter = [0]
    tok_block: dict[str, int] = {}
    s_open: list[tuple[str, set[int]]] = []   # wrapping <s>: (label, blocks seen)
    anchors: list[tuple[str, list[str]]] = []  # anchor <s/>: (label, token ids)
    failure: list[str] = []

    def current_block() -> int:
        return block_stack[-1] if block_stack else 0

    def on_start(name: str, attrs: dict) -> None:
        if name in blocks:
            block_counter[0] += 1
            block_stack.append(block_counter[0])
        elif name == "tok":
            tid = attrs.get(tok_id_attr) or attrs.get("id") or attrs.get("xml:id")
            if tid:
                tok_block[tid] = current_block()
            for _, seen in s_open:
                seen.add(current_block())
        elif name == "s":
            label = attrs.get(s_attrs.get("id_attr", "xml:id")) or attrs.get("id") or attrs.get("n", "?")
            if attrs.get("corresp"):
                anchors.append((label, [r.lstrip("#") for r in attrs["corresp"].split()]))
            s_open.append((label, set()))

    def on_end(name: str) -> None:
        if name in blocks and block_stack:
            block_stack.pop()
        elif name == "s" and s_open:
            label, seen = s_open.pop()
            if len(seen) > 1 and not failure:
                failure.append(f"V10 S-WITHIN-BLOCK failed: <s> {label!r} contains a block boundary")

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.Parse(output_bytes, True)
    for label, refs in anchors:
        seen = {tok_block[r] for r in refs if r in tok_block}
        if len(seen) > 1 and not failure:
            failure.append(
                f"V10 S-WITHIN-BLOCK failed: sentence {label!r} has tokens in "
                f"{len(seen)} different blocks"
            )
    if failure:
        raise ValidationError(failure[0])
