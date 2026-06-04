"""Phase A — extract: XML bytes -> (fold_plaintext, xml-layer records, metadata).

This is the **v1** extractor. It implements the no-op round-trip path
(structural elements, anchors for empty elements, basic CDATA handling)
and is the foundation that joins, prefer-one-child-of, and excluded
subtrees will hang off of in subsequent steps.

Expat is used directly (not ElementTree) so we can capture exact byte
positions of every tag and CDATA section — essential for byte-identical
round-trip of the prefix/suffix and for verbatim re-emission.
"""

from __future__ import annotations

import xml.parsers.expat
from dataclasses import dataclass, field
from typing import Optional

from .records import Layer, Record
from .selectors import compile as compile_selector


# ---------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------


@dataclass
class ScopeRoot:
    """One tokenization root (e.g. one ``<text>`` element).

    The fields ``nlp_plaintext``, ``nlp_to_fold``, and ``udpipe_layer`` are
    filled in by later phases; Phase A leaves them at their defaults.

    ``raw_open_bytes_by_id`` and ``raw_empty_bytes_by_id`` carry the exact
    source bytes of each element's open tag (or self-closing tag for an
    empty element). The folder uses them when the element is emitted as a
    single fragment, so attribute formatting (multi-line whitespace, single-
    vs. double-quotes, etc.) round-trips byte-identically. For split
    elements the folder must reformat to inject ``@rpt``/``@cont``.
    """

    open_tag_bytes: bytes
    close_tag_bytes: bytes
    fold_plaintext: str
    xml_layer: Layer
    raw_open_bytes_by_id: dict[str, bytes] = field(default_factory=dict)
    raw_empty_bytes_by_id: dict[str, bytes] = field(default_factory=dict)
    nlp_plaintext: str = ""
    nlp_to_fold: list[int] = field(default_factory=list)
    udpipe_layer: Optional[Layer] = None


@dataclass
class Metadata:
    source_path: str
    scope_roots: list[ScopeRoot]
    profile_name: str
    profile: dict
    prefix_bytes: bytes
    suffix_bytes: bytes
    # Between scope roots i and i+1 (length = len(scope_roots) - 1).
    interscope_bytes: list[bytes] = field(default_factory=list)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


class AlreadyTokenizedError(ValueError):
    """Raised when the input already contains `<tok>` or `<s>` elements
    inside the tokenization scope. xmltokenizer is not idempotent in v1;
    the caller should detokenize (strip the existing `<tok>` and `<s>`
    wrappers, keeping their inner content) before re-running. A future
    `--retokenize` flag may automate that pre-pass."""


def run(raw: bytes, profile: dict, source_path: str = "<bytes>") -> Metadata:
    """Parse `raw` with expat, find scope roots, build metadata."""
    events = _parse_events(raw)
    scope_tag_match = _build_scope_tag_matcher(profile)
    scope_event_spans = _identify_scope_roots(events, scope_tag_match)

    if not scope_event_spans:
        candidates = profile.get("scope", {}).get("candidates", [])
        raise ValueError(
            f"no scope root matched in {source_path}; tried selectors {candidates}"
        )

    # Idempotency check: refuse already-tokenized input. Walk each scope
    # root's event range and flag any `<tok>` or `<s>` element. (Empty
    # placeholder `<s/>` anchors — from the anchor-fallback path — also
    # count; the caller should still detokenize before re-running.)
    _refuse_if_already_tokenized(events, scope_event_spans, source_path)

    # Pre-pass: identify children of preferring parents that lose the
    # preferred-branch decision. Loser children get skipped during scope
    # building and become verbatim records.
    skip_ranges, verbatim_payloads = _compute_preference_decisions(
        events, profile, raw
    )

    scope_roots: list[ScopeRoot] = []
    for s_idx, e_idx in scope_event_spans:
        root = _build_scope_root(
            raw, events, s_idx, e_idx, profile,
            skip_ranges, verbatim_payloads,
        )
        scope_roots.append(root)

    # Prefix is bytes from 0 up to the start of the first scope root's open tag.
    first_open_event = events[scope_event_spans[0][0]]
    prefix_bytes = raw[: first_open_event.byte_start]

    # Suffix is bytes from after the last scope root's close tag to end of doc.
    last_close_event = events[scope_event_spans[-1][1]]
    suffix_bytes = raw[last_close_event.byte_end:]

    # Interscope bytes: between scope root i's close end and scope root i+1's
    # open start.
    interscope_bytes: list[bytes] = []
    for i in range(len(scope_event_spans) - 1):
        a_close_end = events[scope_event_spans[i][1]].byte_end
        b_open_start = events[scope_event_spans[i + 1][0]].byte_start
        interscope_bytes.append(raw[a_close_end:b_open_start])

    return Metadata(
        source_path=source_path,
        scope_roots=scope_roots,
        profile_name=profile.get("name", "default"),
        profile=profile,
        prefix_bytes=prefix_bytes,
        suffix_bytes=suffix_bytes,
        interscope_bytes=interscope_bytes,
    )


# ---------------------------------------------------------------------
# Internals — event parsing
# ---------------------------------------------------------------------


@dataclass
class _Event:
    kind: str  # "start" | "end" | "cdata" | "comment" | "pi"
    byte_start: int
    byte_end: int
    # start: (tag, attrs_list)   attrs_list preserves source order as a list of (k, v).
    # end:   tag
    # cdata: text
    # comment: text
    # pi: (target, data)
    data: object
    # Empty element? Set True for events that are <tag .../> (self-closing).
    # An empty element produces ONE event of kind="start" with empty=True
    # (we synthesize an end-event paired to it in identification, but we
    # do not emit it twice in the event list).
    empty: bool = False


def _parse_events(raw: bytes) -> list[_Event]:
    parser = xml.parsers.expat.ParserCreate(encoding="UTF-8")
    parser.ordered_attributes = True  # we want source order
    # Don't merge adjacent text chunks — we want each CDATA call separately.
    parser.buffer_text = True
    parser.XmlDeclHandler = lambda *a, **k: None  # ignore the <?xml ...?>

    events: list[_Event] = []
    # One-shot flag: True means the next EndElement call is the synthetic
    # one paired with a self-closing StartElement; consume it silently.
    pending_synth_end: list[bool] = [False]

    def on_start(name: str, attrs_flat: list) -> None:
        # attrs_flat is [k1, v1, k2, v2, ...] because ordered_attributes=True
        attrs = list(zip(attrs_flat[0::2], attrs_flat[1::2]))
        start = parser.CurrentByteIndex
        end = _find_tag_end(raw, start)
        is_empty = raw[end - 2 : end - 1] == b"/"  # <foo .../>
        events.append(_Event("start", start, end, (name, attrs), empty=is_empty))
        if is_empty:
            pending_synth_end[0] = True

    def on_end(name: str) -> None:
        # For an empty element <foo/>, expat fires StartElement then
        # immediately EndElement. We collapsed the empty case into a
        # single "start" event with empty=True; the synthetic
        # EndElement must be consumed without emitting an "end" event.
        #
        # We can't use byte-position comparison because when a real
        # `</tag>` immediately follows an empty `<foo/>` there are zero
        # bytes between them — expat reports the same CurrentByteIndex
        # for both calls. Use a one-shot flag instead.
        if pending_synth_end[0]:
            pending_synth_end[0] = False
            return
        start = parser.CurrentByteIndex
        end = start + len(b"</") + len(name.encode("utf-8")) + 1  # "</tag>"
        events.append(_Event("end", start, end, name))

    def on_cdata(text: str) -> None:
        start = parser.CurrentByteIndex
        end = start + len(text.encode("utf-8"))
        events.append(_Event("cdata", start, end, text))

    def on_comment(text: str) -> None:
        start = parser.CurrentByteIndex
        # `<!--` + text + `-->`  (the comment delimiters are 4 + 3 bytes)
        end = start + 4 + len(text.encode("utf-8")) + 3
        events.append(_Event("comment", start, end, text))

    def on_pi(target: str, data: str) -> None:
        start = parser.CurrentByteIndex
        # Scan forward for `?>`
        idx = raw.find(b"?>", start)
        if idx == -1:
            raise ValueError(f"unterminated processing instruction at byte {start}")
        end = idx + 2
        events.append(_Event("pi", start, end, (target, data)))

    parser.StartElementHandler = on_start
    parser.EndElementHandler = on_end
    parser.CharacterDataHandler = on_cdata
    parser.CommentHandler = on_comment
    parser.ProcessingInstructionHandler = on_pi

    parser.Parse(raw, True)
    return events


def _find_tag_end(raw: bytes, start: int) -> int:
    """Find the byte position just after the `>` that closes the tag
    that starts with `<` at `start`. Handles attribute-quoted `>`s."""
    assert raw[start : start + 1] == b"<", f"expected `<` at byte {start}"
    i = start + 1
    n = len(raw)
    in_quote: Optional[bytes] = None
    while i < n:
        c = raw[i : i + 1]
        if in_quote is not None:
            if c == in_quote:
                in_quote = None
        else:
            if c == b'"' or c == b"'":
                in_quote = c
            elif c == b">":
                return i + 1
        i += 1
    raise ValueError(f"unterminated tag starting at byte {start}")


# ---------------------------------------------------------------------
# Internals — scope root identification
# ---------------------------------------------------------------------


def _build_scope_tag_matcher(profile: dict) -> "callable":
    """Return a function f(tag, ancestor_tags) -> bool.

    v1 supports only `//name`, `/name`, `/*` (single-step selectors).
    Predicate and child-step support comes when we need it.
    """
    candidates = profile.get("scope", {}).get("candidates", [])
    compiled = [compile_selector(c) for c in candidates]
    for sel in compiled:
        if len(sel.steps) != 1:
            raise NotImplementedError(
                f"v1 extractor only supports single-step selectors; got {sel.source!r}"
            )

    def matches(tag: str, ancestor_tags: list[str]) -> Optional[int]:
        """Return the index of the first candidate that matches this element,
        or None if no candidate matches."""
        for idx, sel in enumerate(compiled):
            step = sel.steps[0]
            if step.axis == "root":
                # `/*` — must be the document root, i.e. no ancestors.
                if not ancestor_tags:
                    return idx
            elif step.axis == "descendant_or_self":
                # `//name` — element tag must equal step.name.
                if tag == step.name:
                    return idx
            elif step.axis == "child_of_root":
                # `/name` — element must be the document root and named.
                if not ancestor_tags and tag == step.name:
                    return idx
        return None

    return matches


# ---------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------


_TOKENIZED_TAGS = frozenset({"tok", "s", "dtok"})


def _refuse_if_already_tokenized(
    events: list[_Event], scope_spans: list[tuple[int, int]], source_path: str
) -> None:
    """Raise AlreadyTokenizedError if any `<tok>`, `<s>`, or `<dtok>`
    element is found inside any scope root."""
    for s_idx, e_idx in scope_spans:
        for ev in events[s_idx + 1 : e_idx]:
            if ev.kind == "start" and ev.data[0] in _TOKENIZED_TAGS:
                tag = ev.data[0]
                raise AlreadyTokenizedError(
                    f"{source_path}: input already contains <{tag}> element "
                    f"at byte {ev.byte_start} inside the tokenization scope. "
                    f"xmltokenizer is not idempotent. Detokenize first "
                    f"(strip existing <tok>/<s>/<dtok> wrappers, keeping "
                    f"their inner content) before re-running."
                )


# ---------------------------------------------------------------------
# prefer_one_child_of pre-pass (design §5.3)
# ---------------------------------------------------------------------


def _compute_preference_decisions(
    events: list[_Event], profile: dict, raw: bytes
) -> tuple[dict[int, int], dict[int, bytes]]:
    """For each element whose tag matches a `prefer_one_child_of` rule,
    identify which direct child wins. Return (skip_ranges, verbatim_payloads):
      - skip_ranges[child_start_event_idx] = child_end_event_idx (inclusive)
      - verbatim_payloads[child_start_event_idx] = raw XML bytes to emit
    Loser empty elements have child_start_event_idx == child_end_event_idx.
    """
    rules: dict[str, dict] = {
        rule["parent"]: rule
        for rule in profile.get("prefer_one_child_of", [])
    }
    if not rules:
        return {}, {}

    # children_by_parent[parent_start_event_idx] = list of (child_start, child_end)
    children_by_parent: dict[int, list[tuple[int, int]]] = {}
    stack: list[tuple[int, list[tuple[int, int]]]] = []
    for i, ev in enumerate(events):
        if ev.kind == "start":
            child_list: list[tuple[int, int]] = []
            children_by_parent[i] = child_list
            if ev.empty:
                # Empty element is a child of the parent on top of stack.
                if stack:
                    stack[-1][1].append((i, i))
            else:
                stack.append((i, child_list))
        elif ev.kind == "end":
            start_i, _ = stack.pop()
            if stack:
                stack[-1][1].append((start_i, i))

    skip_ranges: dict[int, int] = {}
    verbatim_payloads: dict[int, bytes] = {}
    for parent_start, child_list in children_by_parent.items():
        ev = events[parent_start]
        if ev.kind != "start" or ev.empty:
            continue
        rule = rules.get(ev.data[0])
        if rule is None:
            continue
        winner = _pick_preferred_child(child_list, events, rule)
        for cs, ce in child_list:
            if winner is not None and (cs, ce) == winner:
                continue
            skip_ranges[cs] = ce
            verbatim_payloads[cs] = raw[events[cs].byte_start : events[ce].byte_end]
    return skip_ranges, verbatim_payloads


def _pick_preferred_child(
    child_list: list[tuple[int, int]],
    events: list[_Event],
    rule: dict,
) -> tuple[int, int] | None:
    """Apply the preference rule, returning the (start, end) of the winner
    or None if `fallback="drop_all"` and no child matches."""
    prefer = rule.get("prefer", [])
    fallback = rule.get("fallback", "first_child")
    for pref_tag in prefer:
        for cs, ce in child_list:
            if events[cs].data[0] == pref_tag:
                return (cs, ce)
    if fallback == "drop_all":
        return None
    # first_child
    if child_list:
        return child_list[0]
    return None


def _identify_scope_roots(
    events: list[_Event], matches
) -> list[tuple[int, int]]:
    """Walk the event stream and return (start_event_idx, end_event_idx) for
    each scope-root element, using the first-matching-candidate-wins rule.

    'First-match-wins' is applied *per the candidate index*: we pick all
    scope roots that match the candidate with the lowest index (so the
    `//text` candidate wins over `//body` even if both would match).
    """
    # First pass: find every element that matches *some* candidate.
    # Each entry: (cand_idx, start_evt_idx, end_evt_idx).
    matches_found: list[tuple[int, int, int]] = []
    ancestor_stack: list[str] = []
    # Stack entries: (event_idx, tag, cand_idx_or_None).
    open_stack: list[tuple[int, str, Optional[int]]] = []

    for i, ev in enumerate(events):
        if ev.kind == "start":
            tag, _attrs = ev.data
            cand_idx = matches(tag, ancestor_stack)
            if ev.empty:
                if cand_idx is not None:
                    matches_found.append((cand_idx, i, i))
            else:
                open_stack.append((i, tag, cand_idx))
                ancestor_stack.append(tag)
        elif ev.kind == "end":
            tag = ev.data
            if ancestor_stack:
                ancestor_stack.pop()
            if open_stack:
                start_evt_idx, start_tag, cand_idx = open_stack.pop()
                if start_tag != tag:
                    raise ValueError(
                        f"tag mismatch closing {tag!r}: open stack had {start_tag!r}"
                    )
                if cand_idx is not None:
                    matches_found.append((cand_idx, start_evt_idx, i))

    # Apply "lowest cand_idx wins"
    if not matches_found:
        return []
    winner = min(m[0] for m in matches_found)
    # Re-emit in document order (matches_found grew as elements *closed*).
    out = sorted(
        ((s, e) for (c, s, e) in matches_found if c == winner),
        key=lambda x: x[0],
    )
    return out


# ---------------------------------------------------------------------
# Internals — building one ScopeRoot
# ---------------------------------------------------------------------


def _build_scope_root(
    raw: bytes, events: list[_Event], start_idx: int, end_idx: int,
    profile: dict,
    skip_ranges: dict[int, int] | None = None,
    verbatim_payloads: dict[int, bytes] | None = None,
) -> ScopeRoot:
    """Build a ScopeRoot for the element whose start event is events[start_idx]
    and whose end event is events[end_idx]."""
    open_ev = events[start_idx]
    close_ev = events[end_idx]

    open_tag_bytes = raw[open_ev.byte_start : open_ev.byte_end]
    close_tag_bytes = raw[close_ev.byte_start : close_ev.byte_end]

    skip_ranges = skip_ranges or {}
    verbatim_payloads = verbatim_payloads or {}

    # Build fold_plaintext and xml-layer records by walking descendants,
    # skipping over loser-child ranges and emitting verbatim records for
    # them.
    builder = _ScopeBuilder(raw, profile)
    i = start_idx + 1
    while i < end_idx:
        if i in skip_ranges:
            payload = verbatim_payloads[i]
            ev = events[i]
            tag = ev.data[0] if ev.kind == "start" else "#skip"
            builder.feed_verbatim_skip(ev, tag, payload)
            i = skip_ranges[i] + 1
            continue
        builder.feed(events[i])
        i += 1
    builder.finish()

    return ScopeRoot(
        open_tag_bytes=open_tag_bytes,
        close_tag_bytes=close_tag_bytes,
        fold_plaintext=builder.fold_plaintext,
        xml_layer=Layer(name="xml", records=builder.records),
        raw_open_bytes_by_id=builder.raw_open_bytes_by_id,
        raw_empty_bytes_by_id=builder.raw_empty_bytes_by_id,
    )


class _ScopeBuilder:
    """Builds fold_plaintext and xml-layer records for one scope root.

    v1: handles element open/close (including empty elements as anchors),
    CDATA, comments (as anchors with tag='#comment'), and processing
    instructions (as anchors with tag='#pi').

    Does NOT yet handle:
        - hard-exclusion subtrees (records as element for now; will become
          verbatim in a later step)
        - prefer-one-child-of preferred-branch logic
        - truncation join regions
    """

    def __init__(self, raw: bytes, profile: dict) -> None:
        self.raw = raw
        self.profile = profile
        self.fold_plaintext_parts: list[str] = []
        self.fold_offset = 0
        self.records: list[Record] = []
        self.next_id = 1
        # Stack of currently-open element records-in-progress:
        # each entry is (record_id, tag, attrs, start_offset, depth, raw_open_bytes, source_open_order)
        self.stack: list[tuple[str, str, list, int, int, bytes, int]] = []
        # Monotonic counter incremented at each START event (empty or
        # non-empty). The value is recorded on the resulting Record as
        # `source_open_order`, giving the folder a way to preserve
        # source order between an anchor at offset X and an element
        # opening at the same offset X.
        self.source_open_counter = 0
        # Raw bytes of each element's open tag (or self-closing tag), keyed
        # by record id. Used by fold.py to round-trip attribute whitespace
        # exactly for elements that are emitted as a single fragment.
        self.raw_open_bytes_by_id: dict[str, bytes] = {}
        self.raw_empty_bytes_by_id: dict[str, bytes] = {}
        # --- truncation join state (see design §5.5) ---
        # `in_join` is True while we are scanning forward through a join
        # region. The first participant is the trigger anchor itself.
        self.in_join: bool = False
        self.join_counter: int = 0
        self.current_join_id: str = ""
        self.position_in_join: int = 0
        # Whitespace accumulated since the most recently emitted participant
        # (will become its `post_whitespace` when the next participant or
        # end-of-join arrives).
        self.pending_whitespace: str = ""
        # Index of the last join participant's record in self.records — so
        # we can mutate its post_whitespace when we discover what came after.
        self.last_join_record_idx: int = -1
        # The most recent fold offset where an element open or close
        # happened. Strips during a join must not cross this position —
        # otherwise they would eat characters from a different element's
        # content (see the <pc>-</pc> case in PressMint).
        self.last_boundary_offset: int = 0
        # Cached config from profile.
        ba = profile.get("break_attribute", {}) or {}
        self.break_bearers: set[str] = set(ba.get("bearers", []))
        self.break_no_value: str = ba.get("no_value", "no")
        self.break_default_value: str = ba.get("default_value", "yes")
        self.truncation_strip_chars: tuple[str, ...] = tuple(
            profile.get("truncation_strip_chars", [])
        )
        self.join_absorb_elements: set[str] = set(
            profile.get("join_absorb_elements", [])
        )
        # Tags whose element record should nest INSIDE a coincident
        # <tok> (priority below atomic tok's 50, so the sort puts them
        # inner).
        self.tok_inner_elements: set[str] = set(
            profile.get("tok_inner_elements", [])
        )

    @property
    def fold_plaintext(self) -> str:
        return "".join(self.fold_plaintext_parts)

    def _new_id(self) -> str:
        rid = f"xml-{self.next_id}"
        self.next_id += 1
        return rid

    def _depth(self) -> int:
        return len(self.stack)

    def _parent_id(self) -> Optional[str]:
        return self.stack[-1][0] if self.stack else None

    def feed(self, ev: _Event) -> None:
        # First, deal with the join state machine: if we're scanning a
        # join region, classify the event accordingly.
        if self.in_join:
            if ev.kind == "cdata":
                if ev.data.isspace():
                    # Accumulate whitespace; will be attached to the
                    # previous participant's `post_whitespace`.
                    self.pending_whitespace += ev.data
                    return
                else:
                    # Non-whitespace CDATA ends the join — but any LEADING
                    # whitespace in this CDATA still belongs to the join
                    # (becomes the previous participant's `post_whitespace`).
                    # Split the cdata at the first non-ws char.
                    text = ev.data
                    i = 0
                    while i < len(text) and text[i].isspace():
                        i += 1
                    if i > 0:
                        self.pending_whitespace += text[:i]
                    self._close_join()
                    # Replace the event with the trimmed remainder for the
                    # normal CDATA path below.
                    ev = _Event(
                        kind="cdata",
                        byte_start=ev.byte_start + i,
                        byte_end=ev.byte_end,
                        data=text[i:],
                    )
                    # fall through to normal processing
            elif ev.kind == "start":
                tag, attrs_list = ev.data
                if ev.empty:
                    # Empty element: continues the join as another
                    # participant.
                    self._add_join_anchor(ev)
                    return
                else:
                    # Non-empty element opens during a join — terminate.
                    # Per design §5.5.2, this is unusual; log via assert
                    # (we treat as join-end).
                    self._close_join()
                    # fall through
            elif ev.kind == "end":
                # The enclosing element is closing; terminate the join.
                self._close_join()
                # fall through
            else:
                # Comment / PI inside a join: terminate for safety.
                self._close_join()
                # fall through

        if ev.kind == "cdata":
            self.fold_plaintext_parts.append(ev.data)
            self.fold_offset += len(ev.data)
        elif ev.kind == "start":
            tag, attrs_list = ev.data
            raw_tag_bytes = self.raw[ev.byte_start : ev.byte_end]
            if ev.empty:
                # Check whether this anchor triggers a join.
                if self._is_join_trigger(tag, attrs_list):
                    self._start_join(ev)
                    return
                # Otherwise: regular anchor record.
                self.source_open_counter += 1
                rid = self._new_id()
                self.records.append(
                    Record(
                        kind="anchor",
                        layer="xml",
                        id=rid,
                        tag=tag,
                        attrs=dict(attrs_list),
                        offset=self.fold_offset,
                        priority=200,
                        parent=self._parent_id(),
                        depth=self._depth(),
                        source_open_order=self.source_open_counter,
                    )
                )
                self.raw_empty_bytes_by_id[rid] = raw_tag_bytes
                self.last_boundary_offset = self.fold_offset
            else:
                self.source_open_counter += 1
                rid = self._new_id()
                self.stack.append(
                    (rid, tag, attrs_list, self.fold_offset, self._depth(),
                     raw_tag_bytes, self.source_open_counter)
                )
                self.raw_open_bytes_by_id[rid] = raw_tag_bytes
                self.last_boundary_offset = self.fold_offset
        elif ev.kind == "end":
            rid, tag, attrs_list, start_off, depth, _raw_open, soo = self.stack.pop()
            # Default xml-element priority is 100. Override down to 30
            # for tags configured to nest INSIDE a coincident <tok>; the
            # rebalance's (start, -priority, seq) sort then naturally
            # places them as inner children of the tok rather than as
            # outer wrappers around it.
            elem_priority = 30 if tag in self.tok_inner_elements else 100
            self.records.append(
                Record(
                    kind="element",
                    layer="xml",
                    id=rid,
                    tag=tag,
                    attrs=dict(attrs_list),
                    start=start_off,
                    end=self.fold_offset,
                    priority=elem_priority,
                    parent=self.stack[-1][0] if self.stack else None,
                    depth=depth,
                    source_open_order=soo,
                )
            )
            self.last_boundary_offset = self.fold_offset
        elif ev.kind == "comment":
            rid = self._new_id()
            self.records.append(
                Record(
                    kind="anchor",
                    layer="xml",
                    id=rid,
                    tag="#comment",
                    attrs={"text": ev.data},
                    offset=self.fold_offset,
                    priority=200,
                    parent=self._parent_id(),
                    depth=self._depth(),
                )
            )
        elif ev.kind == "pi":
            target, data = ev.data
            rid = self._new_id()
            self.records.append(
                Record(
                    kind="anchor",
                    layer="xml",
                    id=rid,
                    tag="#pi",
                    attrs={"target": target, "data": data},
                    offset=self.fold_offset,
                    priority=200,
                    parent=self._parent_id(),
                    depth=self._depth(),
                )
            )
        else:  # pragma: no cover
            raise AssertionError(f"unknown event kind {ev.kind!r}")

    def feed_verbatim_skip(self, ev: "_Event", tag: str, raw_xml: bytes) -> None:
        """Emit a verbatim record at the current fold offset, without
        adding the skipped element's CDATA to fold_plaintext.

        Used by the prefer_one_child_of pre-pass to drop loser branches
        into the standoff as opaque raw-XML chunks. The verbatim's
        `wrap_inside` is set to the currently-open enclosing element id
        so the folder will place it correctly inside its parent (e.g.
        inside `<choice>` rather than as a sibling).
        """
        # Close any pending join when we hit a verbatim skip.
        if self.in_join:
            self._close_join()
        parent_id = self._parent_id()
        rid = self._new_id()
        self.records.append(
            Record(
                kind="verbatim",
                layer="xml",
                id=rid,
                tag=tag,
                attrs={},
                offset=self.fold_offset,
                raw_xml=raw_xml.decode("utf-8"),
                priority=200,
                parent=parent_id,
                depth=self._depth(),
                wrap_inside=parent_id,
            )
        )
        self.last_boundary_offset = self.fold_offset

    # ----- truncation join helpers ----------------------------------

    def _is_join_trigger(self, tag: str, attrs_list: list) -> bool:
        """Return True iff this empty element should open a join region."""
        if tag not in self.break_bearers:
            return False
        attrs = dict(attrs_list)
        break_val = attrs.get("break")
        if break_val == self.break_no_value:
            return True
        # Heuristic / default-break-no: not implemented for v1 — TEI default
        # is "yes" and we require explicit `break="no"` to trigger.
        return False

    def _strip_hyphen_and_trailing_ws(self) -> tuple[str, str]:
        """Pop the trailing whitespace and (if present) a final hyphen-like
        char from fold_plaintext_parts. Updates fold_offset accordingly.

        Will not strip past `last_boundary_offset` — if doing so would eat
        characters that belong to an element that has already opened or
        closed, the strip stops at the boundary (which is the safe choice
        for the corpus-PressMint case where the hyphen lives inside
        `<pc>-</pc>` rather than as bare CDATA next to the trigger).

        Returns (hyphen_char_or_empty, trailing_whitespace).
        """
        text = "".join(self.fold_plaintext_parts)
        boundary = self.last_boundary_offset
        # Strip trailing whitespace, but not past the boundary.
        ws_chars: list[str] = []
        while len(text) > boundary and text[-1].isspace():
            ws_chars.append(text[-1])
            text = text[:-1]
        trailing_ws = "".join(reversed(ws_chars))
        # Strip a final hyphen-like char if (a) we still have chars past
        # the boundary, AND (b) the last char is in the strip set.
        prefix_strip = ""
        if (
            len(text) > boundary
            and self.truncation_strip_chars
            and text[-1] in self.truncation_strip_chars
        ):
            prefix_strip = text[-1]
            text = text[:-1]
        self.fold_plaintext_parts = [text] if text else []
        self.fold_offset = len(text)
        return prefix_strip, trailing_ws

    def _try_absorb_pc_like(self) -> Optional[dict]:
        """If the just-stripped position lands on the close of an
        absorbable element whose content is all hyphen-like chars
        (per profile's `join_absorb_elements` + `truncation_strip_chars`),
        absorb it: remove its content from fold_plaintext, drop the
        record, and return a payload dict the caller turns into a
        verbatim join participant."""
        if not self.join_absorb_elements:
            return None
        if self.fold_offset != self.last_boundary_offset:
            return None
        if not self.records or self.records[-1].kind != "element":
            return None
        elem = self.records[-1]
        if elem.tag not in self.join_absorb_elements:
            return None
        if elem.end != self.fold_offset:
            return None
        full_text = "".join(self.fold_plaintext_parts)
        content = full_text[elem.start : elem.end]
        if not content or not self.truncation_strip_chars:
            return None
        if not all(c in self.truncation_strip_chars for c in content):
            return None

        # Build the verbatim payload BEFORE mutating state.
        open_bytes = self.raw_open_bytes_by_id.get(elem.id, b"")
        content_bytes = content.encode("utf-8")
        close_bytes = b"</" + elem.tag.encode("utf-8") + b">"
        raw_xml = (open_bytes + content_bytes + close_bytes).decode("utf-8")

        # Mutate: strip content, drop element record, rewind boundary.
        self.fold_plaintext_parts = [full_text[: elem.start]] if elem.start else []
        self.fold_offset = elem.start
        self.records.pop()
        self.raw_open_bytes_by_id.pop(elem.id, None)
        self.last_boundary_offset = elem.start

        return {"tag": elem.tag, "raw_xml": raw_xml}

    def _start_join(self, ev: "_Event") -> None:
        """Begin a join region. `ev` is the trigger anchor (e.g. <lb break="no"/>)."""
        prefix_strip, prefix_ws = self._strip_hyphen_and_trailing_ws()

        # Try to absorb a `<pc>`-style element immediately before the
        # join point. When we do, its CONTENT (the hyphen char) is gone
        # from fold_plaintext, so NLP sees the clean joined form. The
        # element survives as a verbatim join participant.
        absorbed = self._try_absorb_pc_like()

        self.join_counter += 1
        self.current_join_id = f"j{self.join_counter}"
        self.in_join = True
        self.position_in_join = 0
        self.pending_whitespace = ""

        parent_id = self._parent_id()
        depth = self._depth()

        # If we absorbed an element, it becomes participant 0 (a
        # verbatim that re-emits the original <pc>-</pc> bytes). The
        # whitespace `prefix_ws` we already captured was BETWEEN the
        # absorbed element and the trigger, so it becomes the absorbed
        # element's post_whitespace. `prefix_strip` (the hyphen) — if we
        # somehow caught one before absorbing — also rides on this
        # verbatim's `join_prefix_strip`.
        if absorbed:
            rid = self._new_id()
            self.records.append(
                Record(
                    kind="verbatim",
                    layer="xml",
                    id=rid,
                    tag=absorbed["tag"],
                    attrs={},
                    offset=self.fold_offset,
                    raw_xml=absorbed["raw_xml"],
                    priority=200,
                    parent=parent_id,
                    depth=depth,
                    join_group=self.current_join_id,
                    join_position=0,
                    join_prefix_strip=prefix_strip,
                    join_prefix_whitespace="",
                    post_whitespace=prefix_ws,
                )
            )
            self.last_join_record_idx = len(self.records) - 1
            self.position_in_join = 1
            # Reset for the trigger participant: it carries no prefix.
            prefix_strip = ""
            prefix_ws = ""

        tag, attrs_list = ev.data
        raw_tag_bytes = self.raw[ev.byte_start : ev.byte_end]
        rid = self._new_id()
        rec = Record(
            kind="anchor",
            layer="xml",
            id=rid,
            tag=tag,
            attrs=dict(attrs_list),
            offset=self.fold_offset,
            priority=200,
            parent=parent_id,
            depth=depth,
            join_group=self.current_join_id,
            join_position=self.position_in_join,
            join_prefix_strip=prefix_strip,
            join_prefix_whitespace=prefix_ws,
        )
        self.records.append(rec)
        self.raw_empty_bytes_by_id[rid] = raw_tag_bytes
        self.last_join_record_idx = len(self.records) - 1
        self.position_in_join += 1

    def _add_join_anchor(self, ev: "_Event") -> None:
        """Add a non-trigger empty element as the next join participant."""
        # Flush pending whitespace onto the PREVIOUS participant's post.
        if self.pending_whitespace:
            prev = self.records[self.last_join_record_idx]
            prev.post_whitespace = prev.post_whitespace + self.pending_whitespace
            self.pending_whitespace = ""

        tag, attrs_list = ev.data
        raw_tag_bytes = self.raw[ev.byte_start : ev.byte_end]
        rid = self._new_id()
        rec = Record(
            kind="anchor",
            layer="xml",
            id=rid,
            tag=tag,
            attrs=dict(attrs_list),
            offset=self.fold_offset,
            priority=200,
            parent=self._parent_id(),
            depth=self._depth(),
            join_group=self.current_join_id,
            join_position=self.position_in_join,
        )
        self.records.append(rec)
        self.raw_empty_bytes_by_id[rid] = raw_tag_bytes
        self.last_join_record_idx = len(self.records) - 1
        self.position_in_join += 1

    def _close_join(self) -> None:
        """End the current join region. Flushes pending whitespace onto the
        last participant's `post_whitespace`."""
        if not self.in_join:
            return
        if self.pending_whitespace:
            last = self.records[self.last_join_record_idx]
            last.post_whitespace = last.post_whitespace + self.pending_whitespace
            self.pending_whitespace = ""
        self.in_join = False
        self.current_join_id = ""
        self.position_in_join = 0
        self.last_join_record_idx = -1

    def finish(self) -> None:
        if self.stack:
            unfinished = ", ".join(s[1] for s in self.stack)
            raise ValueError(
                f"scope ended with unclosed elements: {unfinished}"
            )
        # Sort records by start offset (or anchor offset) for stable downstream
        # processing. Records were emitted in close order, so element records
        # appear out of document order. Re-sort by (start_or_offset, depth).
        def _sort_key(r: Record) -> tuple[int, int]:
            pos = r.start if r.kind == "element" else r.offset
            return (pos or 0, r.depth)

        self.records.sort(key=_sort_key)
