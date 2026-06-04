"""Phase C — fold: emit XML from plaintext + standoff layers.

This is the **v1** folder. It implements the no-op round-trip path:
given a Metadata with one or more ScopeRoots whose xml-layer is populated
and whose udpipe_layer is None, it emits a document byte-identical to the
input (after namespace deactivation).

Splitting, sentence wrapping, MWT `<dtok>`, continuation groups — none of
that yet. That's step 6 of the implementation order.
"""

from __future__ import annotations

import xml.sax.saxutils
from dataclasses import dataclass, field
from typing import Optional

from .extract import Metadata, ScopeRoot
from .records import Layer, Record


def run(metadata: Metadata) -> bytes:
    """Emit the complete output XML, byte-for-byte:

        prefix_bytes
        + for each scope root:  open_tag_bytes + folded_content + close_tag_bytes
                                 + interscope_bytes (if not last)
        + suffix_bytes
    """
    parts: list[bytes] = [metadata.prefix_bytes]
    profile = getattr(metadata, "profile", None) or {}
    for i, root in enumerate(metadata.scope_roots):
        parts.append(root.open_tag_bytes)
        parts.append(fold_scope(root, profile).encode("utf-8"))
        parts.append(root.close_tag_bytes)
        if i < len(metadata.scope_roots) - 1:
            parts.append(metadata.interscope_bytes[i])
    parts.append(metadata.suffix_bytes)
    return b"".join(parts)


def fold_scope(root: ScopeRoot, profile: Optional[dict] = None) -> str:
    """The generic merge-and-emit algorithm. Operates on the scope root's
    fold_plaintext and all its layers (currently: xml_layer, optionally
    udpipe_layer).

    Uses the rebalance algorithm from design doc §7.5: at each event-offset
    boundary, compute the desired open stack and reconcile against the
    current one by closing inside-out and opening outside-in.
    """
    profile = profile or {}
    tok_inner_anchors: set[str] = set(profile.get("tok_inner_anchors", []))

    layers: list[Layer] = [root.xml_layer]
    if root.udpipe_layer is not None:
        layers.append(root.udpipe_layer)

    # Gather all element records in deterministic order; assign each a
    # sequence number for tiebreaking at coincident offsets.
    all_records: list[Record] = []
    for layer in layers:
        all_records.extend(layer.records)
    sequence: dict[str, int] = {rec.id: i for i, rec in enumerate(all_records)}

    # Pre-pass: split detection + anchor-fallback materialization.
    states, anchor_fallbacks = _precompute_states(all_records)
    if anchor_fallbacks:
        all_records = all_records + anchor_fallbacks
        for fb in anchor_fallbacks:
            sequence[fb.id] = len(sequence)

    # Build a set of all unique offsets where things happen, AND pre-index
    # element records by their start and end offsets so we can maintain
    # the active set incrementally (O(N log N) total instead of O(N²)).
    offsets: set[int] = set()
    opens_at: dict[int, list[Record]] = {}
    closes_at: dict[int, list[Record]] = {}
    for rec in all_records:
        if rec.kind == "element":
            assert rec.start is not None and rec.end is not None
            if states[rec.id].use_anchor_fallback:
                continue
            offsets.add(rec.start)
            offsets.add(rec.end)
            opens_at.setdefault(rec.start, []).append(rec)
            closes_at.setdefault(rec.end, []).append(rec)
        else:  # anchor or verbatim
            assert rec.offset is not None
            offsets.add(rec.offset)
    sorted_offsets = sorted(offsets)

    plaintext = root.fold_plaintext
    out: list[str] = []
    cursor = 0
    current_stack: list[Record] = []

    # Index anchors and verbatims by offset for quick lookup.
    point_records: dict[int, list[Record]] = {}
    for rec in all_records:
        if rec.kind in ("anchor", "verbatim"):
            assert rec.offset is not None
            point_records.setdefault(rec.offset, []).append(rec)
    # Sort point records at each offset by priority (DESC) then sequence.
    for offset, recs in point_records.items():
        recs.sort(key=lambda r: (-r.priority, sequence[r.id]))

    # Maintain the set of "active at offset+ε" element records
    # incrementally — at each offset, remove those that close here and
    # add those that open here. This avoids the O(N) scan-all-records
    # per offset that became the dominant cost on PressMint (~9k records).
    active: list[Record] = []

    for offset in sorted_offsets:
        # Maintain the active set incrementally: remove closes, add opens.
        for rec in closes_at.get(offset, []):
            active.remove(rec)
        for rec in opens_at.get(offset, []):
            active.append(rec)

        # Emit plaintext from cursor to offset. CDATA from the source
        # was DECODED by expat (e.g. `&amp;` became a literal `&`), so
        # we must re-escape the structural characters here — otherwise
        # the output XML is malformed. See design doc EC-25.
        if offset > cursor:
            out.append(_escape_cdata(plaintext[cursor:offset]))
            cursor = offset

        # Identify any atomic-policy element (i.e., a <tok>) opening or
        # closing at this offset — used by the tok-edge anchor rules.
        atomic_opens_here = [
            r for r in opens_at.get(offset, []) if r.policy == "atomic"
        ]
        atomic_closes_here = [
            r for r in closes_at.get(offset, []) if r.policy == "atomic"
        ]

        # Phase 1: anchors/verbatims that must fire INSIDE a currently
        # open element. Three reasons:
        #   (a) explicit `wrap_inside` reference,
        #   (b) `parent` that's about to close at this offset (implicit
        #       wrap_inside for natural-XML empty children at the end of
        #       their parent's content — e.g. <dtok> at the end of <tok>),
        #   (c) anchor tag in `tok_inner_anchors` AND a tok is closing at
        #       this offset → the anchor lands inside the closing tok
        #       (e.g. <gap/> at tok.end).
        # Also classify "drift-outside" anchors: when a tok OPENS at this
        # offset and the anchor's tag is NOT in tok_inner_anchors, the
        # anchor fires BETWEEN closes and opens (so e.g. <lb/> at tok.start
        # sits BEFORE the <tok>, not inside it).
        deferred: list[Record] = []
        drift_outside: list[Record] = []
        for rec in point_records.get(offset, []):
            target = rec.wrap_inside
            if target is None and rec.parent:
                parent_idx = _find_in_stack(current_stack, rec.parent)
                if parent_idx is not None:
                    parent_rec = current_stack[parent_idx]
                    if parent_rec.end == offset:
                        target = rec.parent
            if (
                target is None
                and rec.kind == "anchor"
                and rec.tag in tok_inner_anchors
                and atomic_closes_here
            ):
                for tok in atomic_closes_here:
                    if _find_in_stack(current_stack, tok.id) is not None:
                        target = tok.id
                        break
            target_idx = _find_in_stack(current_stack, target)
            if target_idx is not None:
                while len(current_stack) > target_idx + 1:
                    closed = current_stack.pop()
                    out.append(_format_close(closed))
                if rec.kind == "anchor":
                    out.append(_format_anchor(rec, root))
                elif rec.kind == "verbatim":
                    out.append(_format_verbatim(rec))
                continue
            # Drift-outside candidates: ordinary xml-layer anchors at a
            # tok edge AND synthetic anchor-fallback records (their id
            # ends with "--fb") — the latter REPRESENT an `<s>` and must
            # always sit immediately before the sentence's first token,
            # never inside it.
            is_anchor_fallback = (
                rec.kind == "anchor" and rec.id.endswith("--fb")
            )
            if (
                rec.kind == "anchor"
                and atomic_opens_here
                and rec.tag not in tok_inner_anchors
                and not rec.wrap_inside
                and not rec.join_group
                and (rec.layer == "xml" or is_anchor_fallback)
            ):
                drift_outside.append(rec)
                continue
            deferred.append(rec)

        # Phase 2: rebalance — sort the current active set in nest order
        # and reconcile with current_stack. Split into closes-phase and
        # opens-phase so the drift-outside anchors can fire between.
        desired = sorted(
            active, key=lambda r: (r.start, -r.priority, sequence[r.id])
        )
        # Closes-phase.
        lcp = 0
        while (
            lcp < len(current_stack)
            and lcp < len(desired)
            and current_stack[lcp].id == desired[lcp].id
        ):
            lcp += 1
        while len(current_stack) > lcp:
            closed = current_stack.pop()
            out.append(_format_close(closed))
        # Opens-phase. Each drift-outside anchor fires JUST BEFORE the
        # first element it should land outside of. Two rules combine:
        #   - Always before any INSERTED (non-xml-layer) open — so the
        #     anchor sits outside `<tok>`, `<s>`, etc.
        #   - For source-XML opens at the same offset, interleave by
        #     `source_open_order`: an anchor with a smaller
        #     source_open_order fires BEFORE the source-XML element
        #     (preserves the source `<lb/><p>...` order); a larger
        #     source_open_order fires AFTER (preserves `<p><lb/>...`).
        drift_outside.sort(key=lambda r: r.source_open_order or 0)
        drift_iter = iter(drift_outside)
        next_drift = next(drift_iter, None)

        def _drift_should_fire_before(elem: Record) -> bool:
            if next_drift is None:
                return False
            if elem.layer != "xml":
                return True
            if next_drift.source_open_order is None or elem.source_open_order is None:
                return False
            return next_drift.source_open_order < elem.source_open_order

        for i in range(lcp, len(desired)):
            rec = desired[i]
            while _drift_should_fire_before(rec):
                out.append(_format_anchor(next_drift, root))
                next_drift = next(drift_iter, None)
            out.append(_format_open(rec, states, root))
            states[rec.id].fragments_emitted += 1
            current_stack.append(rec)
            if deferred:
                fired = []
                for d in deferred:
                    if d.wrap_inside == rec.id:
                        if d.kind == "anchor":
                            out.append(_format_anchor(d, root))
                        elif d.kind == "verbatim":
                            out.append(_format_verbatim(d))
                        fired.append(d)
                for f in fired:
                    deferred.remove(f)
        # Any drift anchors left after all opens — they fire here
        # (e.g. when there are no inserted opens at the offset, they
        # land between closes and the next plaintext chunk).
        while next_drift is not None:
            out.append(_format_anchor(next_drift, root))
            next_drift = next(drift_iter, None)

        # Phase 3: anchors and verbatims that didn't fire during phases 1
        # or 2 — they sit inside whatever the now-current stack provides.
        for rec in deferred:
            if rec.kind == "anchor":
                out.append(_format_anchor(rec, root))
            elif rec.kind == "verbatim":
                out.append(_format_verbatim(rec))

    # Trailing plaintext after the last event.
    if cursor < len(plaintext):
        out.append(_escape_cdata(plaintext[cursor:]))

    if current_stack:  # pragma: no cover
        raise RuntimeError(
            f"fold ended with open elements: {[r.id for r in current_stack]}"
        )

    return "".join(out)


# ---------------------------------------------------------------------
# Per-element fold-time state
# ---------------------------------------------------------------------


@dataclass
class _ElementState:
    """Per-element fold-time bookkeeping.

    Keyed by record.id in the `states` dict.
    """

    record: Record
    will_split: bool = False
    cont_id: Optional[str] = None      # set only if will_split
    fragments_emitted: int = 0
    # Set when an element with policy="split-with-anchor-fallback" would
    # have to be split. In that case the fold emits an empty
    # <tag .../> anchor at the element's start offset instead of wrapping.
    use_anchor_fallback: bool = False
    # True iff this element's range contains any non-self record's
    # offset/start strictly between (start, end). Used by fold to
    # promote a tok's private `_xt_form` attribute to a real `@form`
    # only when the tok actually wraps inline XML.
    has_inner_xml: bool = False


def _precompute_states(
    records: list[Record],
) -> tuple[dict[str, _ElementState], list[Record]]:
    """Determine which element records will be split, assign cont ids,
    and generate phantom anchor records for `split-with-anchor-fallback`
    elements that can't wrap cleanly.

    Returns (states, anchor_fallbacks):
      - states: dict[record.id] -> _ElementState
      - anchor_fallbacks: list of phantom anchor records to emit
        at the start offsets of would-be-wrapping elements.

    Overlap detection: an element E will be "split" if there is some
    OTHER record A (typically an atomic one such as a `<tok>`) whose
    range overlaps E's range with neither containing the other.
    For v1 we only consider atomic records as causing splits.

    Policy semantics enforced here:
      - "split" → element gets cont id, will be emitted in fragments.
      - "split-with-anchor-fallback" → element gets `use_anchor_fallback=True`,
        will be emitted as a single empty anchor at its start. Sentences
        use this so they're never split.
      - "atomic" → not split-eligible; if two atomics straddle, that's a
        hard error elsewhere.
    """
    import bisect
    states: dict[str, _ElementState] = {
        rec.id: _ElementState(record=rec) for rec in records if rec.kind == "element"
    }
    all_elements = [r for r in records if r.kind == "element"]
    anchor_fallbacks: list[Record] = []

    # Element E will be SPLIT during emission iff some other element F is
    # OUTER to E (lower sort key in the desired_stack ordering) AND F
    # ends strictly inside E's range — F leaves the stack while E is
    # still active. The rebalance algorithm must then close E to close F
    # (LIFO) and reopen E afterwards, producing fragments.
    #
    # Naive check is O(N²); we narrow with a sorted-by-start index so
    # each E only checks elements that could plausibly be outer.
    seq_for: dict[str, int] = {rec.id: i for i, rec in enumerate(all_elements)}

    def _sort_key(rec: Record) -> tuple:
        return (rec.start, -rec.priority, seq_for[rec.id])

    by_start = sorted(all_elements, key=lambda r: r.start)
    starts = [r.start for r in by_start]

    # Pre-compute `has_inner_xml` for every element record. We look at
    # the offsets of all OTHER records (element starts, anchor/verbatim
    # offsets) and ask whether any falls strictly inside (rec.start,
    # rec.end). Used by fold to promote `_xt_form` → `@form` on toks
    # that wrap inline XML.
    other_offsets: list[tuple[int, str]] = []
    for r in records:
        if r.kind == "element":
            assert r.start is not None
            other_offsets.append((r.start, r.id))
        else:
            assert r.offset is not None
            other_offsets.append((r.offset, r.id))
    other_offsets.sort()
    sorted_off_only = [o for o, _ in other_offsets]
    for rec in records:
        if rec.kind != "element":
            continue
        # Find offsets strictly inside (rec.start, rec.end).
        lo = bisect.bisect_right(sorted_off_only, rec.start)
        hi = bisect.bisect_left(sorted_off_only, rec.end)
        for i in range(lo, hi):
            other_id = other_offsets[i][1]
            if other_id != rec.id:
                states[rec.id].has_inner_xml = True
                break

    cont_counter = 0
    for rec in records:
        if rec.kind != "element" or rec.policy == "atomic":
            continue
        e_start, e_end = rec.start, rec.end
        e_key = _sort_key(rec)
        # Candidates: any element with start <= e_start. Everything else
        # starts AFTER rec and so cannot be outer.
        hi = bisect.bisect_right(starts, e_start)
        would_split = False
        for i in range(hi):
            other = by_start[i]
            if other.id == rec.id:
                continue
            # Must overlap rec's interval (still active when rec starts).
            if other.end <= e_start:
                continue
            # Must end before rec — otherwise it contains rec, no split.
            if other.end >= e_end:
                continue
            # Must be outer to rec by sort key (not inner).
            if _sort_key(other) >= e_key:
                continue
            would_split = True
            break
        if not would_split:
            continue

        if rec.policy == "split-with-anchor-fallback":
            # Sentences NEVER split — demote to a single empty anchor at
            # the sentence's start offset. The fold's drift-outside path
            # then ensures the anchor fires immediately BEFORE the first
            # token of the sentence.
            states[rec.id].use_anchor_fallback = True
            # Build @corresp listing the xml:ids of every atomic token
            # in this sentence's range. The user requirement: a phantom
            # `<s/>` must always identify which tokens it heads.
            tok_refs: list[str] = []
            for other in all_elements:
                if other.policy != "atomic":
                    continue
                if other.start < rec.start or other.end > rec.end:
                    continue
                xml_id = other.attrs.get("xml:id")
                if xml_id:
                    tok_refs.append(f"#{xml_id}")
            fb_attrs = dict(rec.attrs)
            if tok_refs:
                fb_attrs["corresp"] = " ".join(tok_refs)
            anchor_fallbacks.append(
                Record(
                    kind="anchor",
                    layer=rec.layer,
                    id=f"{rec.id}--fb",
                    tag=rec.tag,
                    attrs=fb_attrs,
                    offset=rec.start,
                    priority=rec.priority,
                    parent=rec.parent,
                    depth=rec.depth,
                )
            )
        else:
            # Default: split with cont id.
            states[rec.id].will_split = True
            cont_counter += 1
            states[rec.id].cont_id = f"g{cont_counter}"
    return states, anchor_fallbacks


# ---------------------------------------------------------------------
# Desired-stack computation
# ---------------------------------------------------------------------


def _compute_desired_stack(
    records: list[Record], offset: int, sequence: dict[str, int],
    states: Optional[dict[str, _ElementState]] = None,
) -> list[Record]:
    """Return the list of records that should be open at offset + ε,
    in nest order (outermost first)."""
    active: list[Record] = []
    for rec in records:
        if rec.kind != "element":
            continue
        if states is not None:
            st = states.get(rec.id)
            if st and st.use_anchor_fallback:
                # This element is being emitted as an anchor; never open
                # it on the stack.
                continue
        if rec.start <= offset < rec.end:
            active.append(rec)
        elif rec.start == rec.end == offset:
            # Zero-length element (rare). Don't include as "active" at
            # offset+ε because at the boundary it's already past.
            pass
    # Sort by (start ASC, -priority, sequence) so:
    #   - earlier-starting elements are outermost
    #   - at the same start, higher priority is outermost
    #   - at the same (start, priority), document order wins
    active.sort(key=lambda r: (r.start, -r.priority, sequence[r.id]))
    return active


# ---------------------------------------------------------------------
# Rebalance algorithm
# ---------------------------------------------------------------------


def _find_in_stack(stack: list[Record], target_id: Optional[str]) -> Optional[int]:
    """Return the index of the record with id == target_id, or None."""
    if not target_id:
        return None
    for i, rec in enumerate(stack):
        if rec.id == target_id:
            return i
    return None


def _rebalance(
    current: list[Record],
    desired: list[Record],
    out: list[str],
    states: dict[str, _ElementState],
    root: ScopeRoot,
    sequence: dict[str, int],
) -> None:
    """Reconcile `current` (mutated) to match `desired` by closing inside-
    out, then opening outside-in. Emits to `out`."""
    _rebalance_with_wrap_inside(current, desired, [], out, states, root, sequence)


def _format_verbatim(rec: Record) -> str:
    """Emit a verbatim record. Like `_format_anchor`, it honors the
    join-region whitespace fields so an absorbed `<pc>-</pc>` brings its
    original surrounding whitespace back on fold."""
    body = rec.raw_xml or ""
    pre = rec.pre_whitespace or ""
    post = rec.post_whitespace or ""
    join_prefix = ""
    if rec.join_position == 0 and rec.join_group:
        join_prefix = rec.join_prefix_strip + rec.join_prefix_whitespace
    return pre + join_prefix + body + post


def _rebalance_with_wrap_inside(
    current: list[Record],
    desired: list[Record],
    deferred: list[Record],
    out: list[str],
    states: dict[str, _ElementState],
    root: ScopeRoot,
    sequence: dict[str, int],
) -> None:
    """Same as `_rebalance`, but also fires any deferred records whose
    `wrap_inside` names a freshly-opened element, immediately after that
    element is opened. Deferred records that fire here are removed from
    the `deferred` list (mutated)."""
    # Longest common prefix of current and desired by record id.
    lcp = 0
    while (
        lcp < len(current)
        and lcp < len(desired)
        and current[lcp].id == desired[lcp].id
    ):
        lcp += 1
    # Close in reverse.
    while len(current) > lcp:
        rec = current.pop()
        out.append(_format_close(rec))
    # Open the rest of desired, firing wrap_inside records right after
    # their target opens.
    for i in range(lcp, len(desired)):
        rec = desired[i]
        out.append(_format_open(rec, states, root))
        states[rec.id].fragments_emitted += 1
        current.append(rec)
        if deferred:
            fired = []
            for d in deferred:
                if d.wrap_inside == rec.id:
                    if d.kind == "anchor":
                        out.append(_format_anchor(d, root))
                    elif d.kind == "verbatim":
                        out.append(_format_verbatim(d))
                    fired.append(d)
            for f in fired:
                deferred.remove(f)


# ---------------------------------------------------------------------
# Tag formatting
# ---------------------------------------------------------------------


def _escape_cdata(text: str) -> str:
    """Re-escape XML structural characters in decoded CDATA.

    Expat resolves entity references (`&amp;` → `&`, `&lt;` → `<` etc.)
    when it hands character data to our `CharacterDataHandler`, so
    `fold_plaintext` and the join-region whitespace fields contain
    LITERAL `&`, `<`, `>`. When we splice those back into the output
    XML we have to re-escape — otherwise the output is malformed (an
    `&` in the source would generate an unparseable output). Per design
    doc EC-25, named entities are not preserved as entities; they
    round-trip as their resolved characters, but the output still has
    to be well-formed XML.
    """
    if "&" not in text and "<" not in text and ">" not in text:
        return text
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Attributes that XML requires to be unique per document. On continuation
# fragments (rpt > 0) we must NOT re-emit them — otherwise the output
# fails xml:id uniqueness validation. The @rpt + @cont pair identifies
# fragments unambiguously, so the first fragment keeps the id and later
# fragments rely on @cont/@rpt.
_UNIQUE_ID_ATTRS = frozenset({"xml:id", "id"})


def _strip_unique_ids(attrs: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in attrs.items() if k not in _UNIQUE_ID_ATTRS}


def _xt_promote_attrs(
    attrs: dict, has_inner_xml: bool
) -> dict:
    """Resolve private `_xt_*` attributes:

    - `_xt_form`: promoted to `@form` iff the element wraps inline XML
      (so the surface form is preserved when the inner text differs from
      it — e.g. in joins and MWT). Otherwise dropped.
    """
    if "_xt_form" not in attrs:
        return attrs
    out = {}
    for k, v in attrs.items():
        if k == "_xt_form":
            if has_inner_xml:
                out["form"] = v
            # else: drop it
        else:
            out[k] = v
    return out


def _format_open(
    rec: Record, states: dict[str, _ElementState], root: ScopeRoot
) -> str:
    state = states.get(rec.id)
    fragments_before = state.fragments_emitted if state else 0
    will_split = state.will_split if state else False
    cont_id = state.cont_id if state else None
    has_inner_xml = state.has_inner_xml if state else False

    attrs = _xt_promote_attrs(rec.attrs, has_inner_xml)

    if fragments_before == 0:
        # First fragment.
        if not will_split:
            # Single-fragment emission. Prefer raw bytes from the source
            # for byte-exact attribute round-trip — BUT only when there
            # are no private `_xt_*` attributes to resolve (otherwise we
            # have to reformat to surface @form etc.).
            if "_xt_form" not in rec.attrs:
                raw = root.raw_open_bytes_by_id.get(rec.id)
                if raw is not None:
                    return raw.decode("utf-8")
            return f"<{rec.tag}{_format_attrs(attrs)}>"
        # First fragment of a split element: add @cont but no @rpt.
        extra = {"cont": cont_id} if cont_id is not None else {}
        return f"<{rec.tag}{_format_attrs(attrs, extra)}>"
    else:
        # 2nd+ fragment: @rpt + @cont (rpt = how many fragments before this).
        # Strip any unique-id attributes (xml:id, id) — those belong to the
        # first fragment only; the @cont/@rpt pair identifies subsequent
        # ones.
        attrs = _strip_unique_ids(attrs)
        extra = {"rpt": str(fragments_before)}
        if cont_id is not None:
            extra["cont"] = cont_id
        return f"<{rec.tag}{_format_attrs(attrs, extra)}>"


def _format_close(rec: Record) -> str:
    return f"</{rec.tag}>"


def _format_anchor(rec: Record, root: ScopeRoot) -> str:
    # Special pseudo-tags from extract.py
    if rec.tag == "#comment":
        return f"<!--{rec.attrs.get('text', '')}-->"
    if rec.tag == "#pi":
        target = rec.attrs.get("target", "")
        data = rec.attrs.get("data", "")
        body = f"{target} {data}".rstrip()
        return f"<?{body}?>"
    # Body of the anchor — prefer raw bytes from the source if captured.
    raw = root.raw_empty_bytes_by_id.get(rec.id)
    if raw is not None:
        body = raw.decode("utf-8")
    else:
        body = f"<{rec.tag}{_format_attrs(rec.attrs)}/>"
    # If this anchor is part of a truncation join region, restore the
    # whitespace and the optional hyphen that the extractor stripped from
    # fold_plaintext. See design §5.5.4. Order:
    #   pre_whitespace  +  (join_prefix_strip + join_prefix_whitespace, only on the
    #   first participant) + body + post_whitespace
    pre = rec.pre_whitespace or ""
    post = rec.post_whitespace or ""
    join_prefix = ""
    if rec.join_position == 0 and rec.join_group:
        join_prefix = rec.join_prefix_strip + rec.join_prefix_whitespace
    return pre + join_prefix + body + post


def _format_attrs(attrs: dict, extra: Optional[dict] = None) -> str:
    """Format attribute list. If `extra` is given, those attributes are
    appended *after* the originals so existing attributes appear in their
    source order."""
    items = list(attrs.items())
    if extra:
        items.extend(extra.items())
    if not items:
        return ""
    parts = []
    for k, v in items:
        parts.append(f' {k}="{xml.sax.saxutils.escape(str(v), {chr(34): "&quot;"})}"')
    return "".join(parts)
