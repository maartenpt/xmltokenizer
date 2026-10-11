"""Span layers: new elements around existing ones, named by id.

The general case of folding a layer into XML that already has its tokens: each
span names its first and last element by id (typically ``<tok>``s, or anything
else with an id), and the folder puts one new element around them, nested and
split by the same rules as every other layer (dev/01-design.md §7.4). Typical
uses: constituency trees (``<node>`` over tokens), named entities, error or
discourse annotation from a standoff source.

    from xmltokenizer import extract, fold, load_profile, Span, attach_spans

    metadata = extract(raw, load_profile("teitok"))
    root = metadata.scope_roots[0]
    report = attach_spans(root, [
        Span("w-2", "w-2", tag="node", attrs={"cat": "NP"}, id="n2", parent="n1"),
        Span("w-2", "w-5", tag="node", attrs={"cat": "IP"}, id="n1"),
    ], profile=metadata.profile)
    out = fold(metadata)

Rules:

* A span over part of an atomic element (an id inside a ``<tok>``, e.g. a
  ``<dtok>`` of a contraction) is widened to that whole element — a ``<tok>`` is
  never split — and, with ``mark_widened``, gets that attribute set to ``"1"``.
* Spans of one layer nest or stay apart: a span that would partly overlap one
  added before it (neighbours both widened onto one contraction) is left out and
  reported with the crossing ones.
* A span that crosses a structural source element (the profile's barrier,
  unsplittable, token-break and chunk-boundary elements: ``<p>``, ``<s>``, …) or,
  with ``on_cross="skip"`` (the default), any other source element, is left out
  and reported; ``on_cross="split"`` lets the folder split it like any inline
  element (continuations with ``@rpt``/``@cont``).
* Spans nest by their extent; spans with the same extent nest by ``parent``
  (the enclosing span's id): the folder then keeps a tree order such as
  ``NP > N`` over one word.
* Unknown ids are reported, never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .records import POLICY_SPLIT, Layer, Record


@dataclass
class Span:
    """One new element: from the element with id ``start_id`` to the one with ``end_id``."""

    start_id: str
    end_id: str
    tag: str = "span"
    attrs: dict[str, str] = field(default_factory=dict)
    id: Optional[str] = None
    parent: Optional[str] = None


@dataclass
class SpanReport:
    """What attach_spans did with each span (by span id, or ``start_id..end_id``)."""

    added: list[str] = field(default_factory=list)
    widened: list[str] = field(default_factory=list)
    skipped_crossing: list[str] = field(default_factory=list)
    skipped_unknown: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{len(self.added)} added ({len(self.widened)} widened to a whole token), "
                f"{len(self.skipped_crossing)} left out (crossing), "
                f"{len(self.skipped_unknown)} left out (unknown id)")


def _source_id(rec: Record) -> Optional[str]:
    return rec.attrs.get("id") or rec.attrs.get("xml:id")


def build_span_layer(
    root,
    spans: Iterable[Span],
    *,
    profile: Optional[dict] = None,
    layer: str = "spans",
    atomic_tags: Iterable[str] = ("tok",),
    mark_widened: Optional[str] = None,
    on_cross: str = "skip",
    priority_base: int = 90,
) -> tuple[Layer, SpanReport]:
    """A Layer of element records for ``spans`` over ``root``'s source XML, and a report.

    ``root`` is a ScopeRoot from ``extract``; ids are looked up among its source
    elements. ``priority_base`` places a new element of the same extent as a source
    element: below a source element's 100 (a tree over a whole ``<s>`` goes inside
    it), above a ``<tok>``'s 50 (a node over one word wraps the token); deeper spans
    get a lower priority (never below 51) so nested spans of equal extent keep their
    order.
    """
    if on_cross not in ("skip", "split"):
        raise ValueError(f"on_cross must be 'skip' or 'split', not {on_cross!r}")
    atomic = set(atomic_tags)
    all_src = root.xml_layer.records
    src = [r for r in all_src if r.kind == "element"]
    by_rec_id = {r.id: r for r in all_src}
    # ids of every source record: elements, and empty elements kept as anchors
    # (a <dtok/> inside its <tok> is one; its parent is the tok)
    by_src_id: dict[str, Record] = {}
    for r in all_src:
        sid = _source_id(r)
        if sid and sid not in by_src_id:
            by_src_id[sid] = r
    structural: set[str] = set()
    if profile:
        from .fold import _structural_tags
        structural = _structural_tags(profile)
    # elements a span may not cross: structural ones always, inline ones too with on_cross=skip
    barriers = [r for r in src if r.start < r.end and r.tag not in atomic
                and (r.tag in structural or on_cross == "skip")]

    def resolve(sid: str) -> tuple[Optional[Record], bool]:
        rec = by_src_id.get(sid)
        if rec is None:
            return None, False
        if rec.tag in atomic and rec.kind == "element":
            return rec, False
        p = rec.parent
        while p is not None and p in by_rec_id:
            anc = by_rec_id[p]
            if anc.tag in atomic:
                return anc, True
            p = anc.parent
        if rec.kind != "element":
            # an empty element outside any token: the span starts or ends at its point
            return Record(kind="element", layer=rec.layer, id=rec.id, tag=rec.tag,
                          start=rec.offset, end=rec.offset), False
        return rec, False

    report = SpanReport()
    spans = list(spans)
    span_ids = {s.id for s in spans if s.id}
    parent_of = {s.id: s.parent for s in spans if s.id}

    def depth(s: Span) -> int:
        d, p, seen = 0, s.parent, set()
        while p and p in span_ids and p not in seen:
            seen.add(p)
            d += 1
            p = parent_of.get(p)
        return d

    records: list[Record] = []
    kept: set[str] = set()
    for n, s in enumerate(spans):
        label = s.id or f"{s.start_id}..{s.end_id}"
        a, wa = resolve(s.start_id)
        b, wb = resolve(s.end_id)
        if a is None or b is None:
            report.skipped_unknown.append(label)
            continue
        start, end = min(a.start, b.start), max(a.end, b.end)
        if any((e.start < start < e.end < end) or (start < e.start < end < e.end) for e in barriers):
            report.skipped_crossing.append(label)
            continue
        # spans of this layer nest or are apart: widening can make two neighbours share a
        # token (one ends in a contraction's first part, the next starts in its second);
        # the later one is left out rather than split
        if any((k.start < start < k.end < end) or (start < k.start < end < k.end) for k in records):
            report.skipped_crossing.append(label)
            continue
        attrs = dict(s.attrs)
        if (wa or wb):
            report.widened.append(label)
            if mark_widened:
                attrs[mark_widened] = "1"
        rid = f"{layer}-{s.id or n}"
        records.append(Record(
            kind="element", layer=layer, id=rid, tag=s.tag, attrs=attrs,
            priority=max(51, priority_base - depth(s)), policy=POLICY_SPLIT,
            parent=f"{layer}-{s.parent}" if s.parent and s.parent in span_ids else None,
            depth=depth(s), start=start, end=end,
        ))
        kept.add(rid)
        report.added.append(label)
    # a parent left out: the child keeps its place by extent alone
    for r in records:
        if r.parent and r.parent not in kept:
            r.parent = None
    return Layer(name=layer, records=records, default_priority=priority_base), report


def attach_spans(root, spans: Iterable[Span], **kwargs) -> SpanReport:
    """build_span_layer and add the layer to ``root`` (folded with the others by ``fold``)."""
    lay, report = build_span_layer(root, spans, **kwargs)
    root.extra_layers.append(lay)
    return report
