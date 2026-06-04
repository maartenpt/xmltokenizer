"""Core data types and JSONL I/O for the standoff layer architecture.

Every annotation source — the original XML structure (`xml` layer), UDPipe
tokens and sentences (`udpipe` layer), future NER (`nametag` layer), etc. —
uses the same Record schema described in dev/01-design.md §5.6.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Optional

# Three kinds of standoff records:
#   element  — has a start/end span; will emit <tag>...</tag> on fold
#   anchor   — has a single offset; will emit <tag/> on fold
#   verbatim — has a single offset and a raw_xml payload; pasted on fold
Kind = Literal["element", "anchor", "verbatim"]


# Policy values control how the folder treats this record when another
# record's boundary overlaps it. See dev/01-design.md §7.4.
POLICY_SPLIT = "split"
POLICY_ATOMIC = "atomic"
POLICY_SPLIT_WITH_ANCHOR_FALLBACK = "split-with-anchor-fallback"

VALID_POLICIES = frozenset({
    POLICY_SPLIT,
    POLICY_ATOMIC,
    POLICY_SPLIT_WITH_ANCHOR_FALLBACK,
})


@dataclass
class Record:
    """One standoff record. All layers use this schema.

    Fields are union-of-all-kinds; the kind field selects which subset is
    meaningful. The JSONL serializer drops fields that are None or empty
    so the on-disk format stays compact and readable.
    """

    # --- universal ---
    kind: Kind
    layer: str
    id: str
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    priority: int = 100
    policy: str = POLICY_SPLIT
    parent: Optional[str] = None
    depth: int = 0

    # --- element-only ---
    start: Optional[int] = None
    end: Optional[int] = None

    # --- anchor-only ---
    offset: Optional[int] = None
    wrap_inside: Optional[str] = None
    join_group: Optional[str] = None
    join_position: Optional[int] = None
    pre_whitespace: str = ""
    post_whitespace: str = ""
    join_prefix_strip: str = ""
    join_prefix_whitespace: str = ""

    # --- verbatim-only ---
    raw_xml: Optional[str] = None

    # --- source-order tie-breaker (populated by extract) ---
    # Monotonic counter incremented on every start event in the source
    # XML (both empty elements and non-empty opens). Used by the folder
    # to decide whether an anchor at the same offset as an element open
    # should fire before or after it — preserving the original source
    # order between source-XML constructs without depending on the more
    # fragile element-close ordering that `Layer.records` uses.
    source_open_order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "element":
            if self.start is None or self.end is None:
                raise ValueError(
                    f"element record {self.id!r} missing start/end"
                )
            if self.start > self.end:
                raise ValueError(
                    f"element record {self.id!r} has start > end "
                    f"({self.start} > {self.end})"
                )
        elif self.kind == "anchor":
            if self.offset is None:
                raise ValueError(
                    f"anchor record {self.id!r} missing offset"
                )
        elif self.kind == "verbatim":
            if self.offset is None or self.raw_xml is None:
                raise ValueError(
                    f"verbatim record {self.id!r} missing offset or raw_xml"
                )
        else:
            raise ValueError(f"unknown record kind {self.kind!r}")
        if self.policy not in VALID_POLICIES:
            raise ValueError(
                f"record {self.id!r} has unknown policy {self.policy!r}"
            )


@dataclass
class Layer:
    """A named JSONL stream of standoff records over a single plaintext."""

    name: str
    records: list[Record] = field(default_factory=list)
    default_policy: str = POLICY_SPLIT
    default_priority: int = 100


# =====================================================================
# JSONL I/O
# =====================================================================

# Fields that are union-of-all-kinds; serializer skips defaults to keep
# on-disk lines compact.
_DEFAULTS_TO_DROP = {
    "attrs": {},
    "parent": None,
    "depth": 0,
    "start": None,
    "end": None,
    "offset": None,
    "wrap_inside": None,
    "join_group": None,
    "join_position": None,
    "pre_whitespace": "",
    "post_whitespace": "",
    "join_prefix_strip": "",
    "join_prefix_whitespace": "",
    "raw_xml": None,
    "source_open_order": None,
}


def record_to_dict(rec: Record) -> dict:
    """Convert a Record to a dict suitable for json.dumps; drop defaults."""
    d = asdict(rec)
    for k, default in _DEFAULTS_TO_DROP.items():
        if d.get(k) == default:
            d.pop(k, None)
    return d


def dict_to_record(d: dict) -> Record:
    """Reverse of record_to_dict; supply defaults for missing keys."""
    # Defensive: a stray None is treated as missing.
    kwargs = {k: v for k, v in d.items() if v is not None or k in ("offset", "start", "end")}
    return Record(**kwargs)


def write_jsonl(path: str, records: Iterable[Record]) -> None:
    """Write records to a JSONL file (one record per line, UTF-8, \\n line endings)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(record_to_dict(rec), ensure_ascii=False))
            f.write("\n")


def read_jsonl(path: str) -> list[Record]:
    """Read a JSONL file produced by write_jsonl."""
    out: list[Record] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no}: bad JSON ({e})"
                ) from e
            out.append(dict_to_record(d))
    return out
