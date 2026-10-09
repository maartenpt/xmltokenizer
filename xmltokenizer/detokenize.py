"""Detokenize: strip an existing tokenization, keep a TSV record of it.

Removes the tokenization elements (`<tok>`, `<dtok>`, `<s>` by default) inside
the scope element (`<text>`), keeping their inner content, and merges the
fragments a tokenization split elements into (`<hi cont="g1">…</hi><hi rpt="1"
cont="g1">…</hi>` → one `<hi>`). Every other byte of the input is kept as it was:
the work is done on the byte ranges of the parsed tags, never with regex.

Everything that is removed is returned as rows (one per sentence, token and
sub-token) with all its attributes and its character offsets into the text of
the scope, so the tokenization can be inspected, compared or restored later.

Not touched: TEI `<w>`/`<pc>` (a TEI tokenization is converted, not stripped),
and fragments that cannot be merged (another element sits between them).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Optional

from .extract import _Event, _parse_events

DEFAULT_TAGS = ("tok", "dtok", "s")
FRAGMENT_ATTRS = ("rpt", "cont")


@dataclass
class DetokenizeResult:
    xml: bytes
    rows: list[dict] = field(default_factory=list)
    text: str = ""            # the text of the scope(s), as the offsets count it
    merged_fragments: int = 0
    kept_fragments: int = 0   # split elements that could not be merged


def _attr(attrs: list[tuple[str, str]], *names: str) -> str:
    for k, v in attrs:
        if k in names:
            return v
    return ""


def _attr_byte_span(tag: bytes, name: str) -> Optional[tuple[int, int]]:
    """Byte span of ` name="value"` (with its leading whitespace) inside one start
    tag, found by scanning the tag's attributes (names, `=`, quoted values)."""
    i = 1
    n = len(tag)
    while i < n and tag[i:i + 1] not in (b" ", b"\t", b"\n", b"\r", b"/", b">"):
        i += 1  # element name
    while i < n:
        ws_start = i
        while i < n and tag[i:i + 1] in (b" ", b"\t", b"\n", b"\r"):
            i += 1
        if i >= n or tag[i:i + 1] in (b"/", b">"):
            return None
        name_start = i
        while i < n and tag[i:i + 1] not in (b"=", b" ", b"\t", b"\n", b"\r"):
            i += 1
        attr_name = tag[name_start:i].decode("utf-8")
        while i < n and tag[i:i + 1] != b"=":
            i += 1
        i += 1
        while i < n and tag[i:i + 1] in (b" ", b"\t", b"\n", b"\r"):
            i += 1
        quote = tag[i:i + 1]
        i += 1
        while i < n and tag[i:i + 1] != quote:
            i += 1
        i += 1
        if attr_name == name:
            return ws_start, i
    return None


def detokenize(
    raw: bytes,
    *,
    tags: tuple[str, ...] = DEFAULT_TAGS,
    scope: str = "text",
    merge_fragments: bool = True,
) -> DetokenizeResult:
    """Strip `tags` inside every `scope` element of `raw` (whole document when
    there is no `scope` element). See the module docstring."""
    events = _parse_events(raw)
    has_scope = any(e.kind == "start" and e.data[0] == scope for e in events)
    tagset = set(tags)

    delete: list[tuple[int, int]] = []           # byte ranges to drop
    rewrite: dict[int, bytes] = {}                # event idx -> new start-tag bytes
    rows: list[dict] = []
    text_parts: list[str] = []
    offset = 0
    scope_depth = 0
    # open elements: (tag, kind, payload); kind "strip" (payload: row or None for a
    # token fragment), "keep" (payload: @cont), "scope"
    stack: list[tuple[str, str, object]] = []
    s_rows: list[dict] = []
    tok_rows: list[dict] = []
    last_tok: Optional[dict] = None
    dtok_depth = 0
    dtok_start = 0
    # the last event that stays in the output, ignoring whitespace-only text:
    # ("end", tag, cont, event) right after a kept close, else (kind, ...)
    last_kept: tuple = ("none",)
    first_fragment: dict[str, int] = {}           # cont id -> event idx
    fragments_seen: dict[str, int] = {}           # cont id -> continuation fragments
    fragments_merged: dict[str, int] = {}
    merged = 0

    def in_scope() -> bool:
        return scope_depth > 0 or not has_scope

    for idx, ev in enumerate(events):
        if dtok_depth:  # inside a non-empty <dtok>: dropped whole, content recorded
            if ev.kind == "start" and not ev.empty:
                dtok_depth += 1
            elif ev.kind == "end":
                dtok_depth -= 1
                if dtok_depth == 0:
                    delete.append((dtok_start, ev.byte_end))
                    rows[-1]["content"] = raw[dtok_start:ev.byte_end].decode("utf-8")
            continue

        if ev.kind == "start":
            tag, attrs = ev.data
            if tag == scope:
                if not ev.empty:
                    scope_depth += 1
                    stack.append((tag, "scope", None))
                last_kept = ("start",)
                continue
            if in_scope() and tag in tagset:
                delete.append((ev.byte_start, ev.byte_end))
                row: Optional[dict] = {"kind": tag, "id": _attr(attrs, "xml:id", "id"),
                                       "start": offset, "end": offset, "attrs": dict(attrs)}
                if tag == "tok" and _attr(attrs, "rpt") and last_tok is not None:
                    last_tok["fragments"] = last_tok.get("fragments", 1) + 1
                    row = None  # continuation of the previous token
                elif tag in ("tok", "dtok"):
                    row["s"] = s_rows[-1]["id"] if s_rows else ""
                    if tag == "dtok":
                        row["tok"] = tok_rows[-1]["id"] if tok_rows else ""
                    rows.append(row)
                else:
                    rows.append(row)
                if tag == "dtok" and not ev.empty:
                    delete.pop()
                    dtok_depth, dtok_start = 1, ev.byte_start
                    continue
                if not ev.empty:
                    stack.append((tag, "strip", row if row is not None else last_tok))
                    if tag == "s":
                        s_rows.append(row)
                    elif tag == "tok":
                        tok = row if row is not None else last_tok
                        tok_rows.append(tok)
                        last_tok = tok
                continue
            rpt = _attr(attrs, "rpt")
            cont = _attr(attrs, "cont")
            if (merge_fragments and in_scope() and rpt and last_kept[0] == "end"
                    and last_kept[1] == tag and (not cont or last_kept[2] == cont)):
                # glue onto the previous fragment: drop its close and this open
                prev_end = last_kept[3]
                delete.append((prev_end.byte_start, prev_end.byte_end))
                delete.append((ev.byte_start, ev.byte_end))
                merged += 1
                if cont:
                    fragments_seen[cont] = fragments_seen.get(cont, 0) + 1
                    fragments_merged[cont] = fragments_merged.get(cont, 0) + 1
                stack.append((tag, "keep", cont))
                last_kept = ("start",)
                continue
            if cont and not rpt:
                first_fragment[cont] = idx
            elif cont and rpt:
                fragments_seen[cont] = fragments_seen.get(cont, 0) + 1
            if not ev.empty:
                stack.append((tag, "keep", cont))
            last_kept = ("start",)
        elif ev.kind == "end":
            tag, kind, payload = stack.pop()
            if kind == "scope":
                scope_depth -= 1
                last_kept = ("end", tag, "", ev)
            elif kind == "strip":
                delete.append((ev.byte_start, ev.byte_end))
                if isinstance(payload, dict):
                    payload["end"] = offset
                if tag == "s" and s_rows:
                    s_rows.pop()
                elif tag == "tok" and tok_rows:
                    tok_rows.pop()
            else:
                last_kept = ("end", tag, payload, ev)
        elif ev.kind == "cdata":
            if in_scope():
                text_parts.append(ev.data)
                offset += len(ev.data)
            if ev.data.strip():
                last_kept = ("text",)
        else:
            last_kept = (ev.kind,)

    # a group merged completely is one element again: its first fragment loses
    # @cont (a partly merged group keeps it, as do its remaining fragments)
    for cont, fi in first_fragment.items():
        if fragments_seen.get(cont) and fragments_merged.get(cont) == fragments_seen[cont]:
            fev = events[fi]
            tag_bytes = raw[fev.byte_start:fev.byte_end]
            span = _attr_byte_span(tag_bytes, "cont")
            if span:
                rewrite[fi] = tag_bytes[:span[0]] + tag_bytes[span[1]:]

    # anchor <s/>s: their span is that of the tokens they list
    by_id = {r["id"]: r for r in rows if r["kind"] == "tok" and r["id"]}
    for r in rows:
        if r["kind"] == "s" and r["start"] == r["end"] and r["attrs"].get("corresp"):
            toks = [by_id[x.lstrip("#")] for x in r["attrs"]["corresp"].split()
                    if x.lstrip("#") in by_id]
            if toks:
                r["start"] = min(t["start"] for t in toks)
                r["end"] = max(t["end"] for t in toks)
                for t in toks:
                    t["s"] = t["s"] or r["id"]

    edits = sorted(
        [(b, e, b"") for b, e in delete]
        + [(events[i].byte_start, events[i].byte_end, nb) for i, nb in rewrite.items()],
        key=lambda x: (x[0], -x[1]),
    )
    out = bytearray()
    pos = 0
    for b, e, nb in edits:
        if b < pos:
            continue  # inside a range already dropped
        out += raw[pos:b]
        out += nb
        pos = e
    out += raw[pos:]

    text = "".join(text_parts)
    for r in rows:
        r["text"] = text[r["start"]:r["end"]]
    kept_frag = sum(1 for e in _parse_events(bytes(out))
                    if e.kind == "start" and _attr(e.data[1], "rpt"))
    return DetokenizeResult(bytes(out), rows, text, merged, kept_frag)


def rows_to_tsv(rows: list[dict]) -> str:
    """One row per removed element. Fixed columns first, then every attribute
    that occurs (id/xml:id are in `id`). Tabs and newlines are escaped."""
    fixed = ["kind", "id", "s", "tok", "start", "end", "text", "fragments", "content"]
    skip = {"id", "xml:id"}
    extra: list[str] = []
    for r in rows:
        for k in r["attrs"]:
            if k not in skip and k not in extra:
                extra.append(k)

    def clean(v: object) -> str:
        return str(v).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")

    buf = io.StringIO()
    w = csv.writer(buf, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar=None,
                   lineterminator="\n", quotechar=None)
    w.writerow(fixed + extra)
    for r in rows:
        w.writerow([clean(r.get(c, "")) for c in fixed]
                   + [clean(r["attrs"].get(k, "")) for k in extra])
    return buf.getvalue()
