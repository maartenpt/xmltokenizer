"""Phase A.5 — build `nlp_plaintext` and `nlp_to_fold` from `fold_plaintext`.

See dev/01-design.md §5.8.

`fold_plaintext` is the byte-exact character data extracted from the XML
scope. We send a DERIVED string (`nlp_plaintext`) to the NLP backend with
`\\n\\n` barriers inserted at points where a sentence MUST NOT span — end
of `<cell>`, `</p>`, around `<note>` etc. UDPipe (and any sentence
segmenter) treats blank-line-separated blocks as separate paragraphs, so
a sentence physically cannot cross a barrier.

The `nlp_to_fold[]` array maps each NLP-plaintext position back to its
fold-plaintext position (or `-1` for inserted barrier characters), so
the udpipe-layer builder can translate token spans back to fold
coordinates.
"""

from __future__ import annotations

from typing import Iterable

from .extract import ScopeRoot
from .records import Record


def build(root: ScopeRoot, profile: dict) -> None:
    """Mutates `root`: sets `nlp_plaintext` and `nlp_to_fold`.

    Two tiers of barrier are inserted (design §5.8 + Session 5 addition):

    **Sentence barriers (`\\n\\n`)** — prevent sentence span across:
    - `profile.unsplittable` ∪ `profile.barrier_elements` element closes.
    - The offset of every excluded verbatim record (e.g. `<note>`) when
      `profile.barrier_around_verbatim` is true.

    **Token barriers (single space)** — prevent token span across:
    - `profile.token_break_elements` element closes (TEI elements where
      encoders frequently omit whitespace between siblings, e.g.
      `<cell>`, `<item>`, `<l>`, `<row>`).
    - Anchors whose tag is in `profile.break_attribute.bearers` AND whose
      `@break` is NOT the `no_value` (so they really do end a token):
      `<lb/>`, `<cb/>`, `<pb/>`. The `break="no"` case is handled by the
      join scanner and does not insert anything.

    If a single offset has both a sentence and token insertion, the
    sentence barrier wins (`\\n\\n` is also a token break).
    """
    insertions = _collect_insertions(root.xml_layer.records, profile)
    nlp_plaintext, nlp_to_fold = _splice(root.fold_plaintext, insertions)
    root.nlp_plaintext = nlp_plaintext
    root.nlp_to_fold = nlp_to_fold


def _collect_insertions(
    records: Iterable[Record], profile: dict
) -> list[tuple[int, str]]:
    """Return a sorted list of `(fold_offset, text_to_insert)` records.

    If multiple insertions land on the same offset, only the longest
    (i.e. the strongest barrier — `\\n\\n` over ` `) is kept.
    """
    sentence_tags: set[str] = set(profile.get("unsplittable", [])) | set(
        profile.get("barrier_elements", [])
    )
    token_break_element_tags: set[str] = set(
        profile.get("token_break_elements", [])
    )
    around_verbatim = bool(profile.get("barrier_around_verbatim", False))
    break_attr = profile.get("break_attribute", {}) or {}
    break_bearers: set[str] = set(break_attr.get("bearers", []))
    break_no_value: str = break_attr.get("no_value", "no")

    # offset -> longest insertion text so far.
    by_offset: dict[int, str] = {}

    def add(offset: int, text: str) -> None:
        existing = by_offset.get(offset, "")
        if len(text) > len(existing):
            by_offset[offset] = text

    for rec in records:
        if rec.kind == "element":
            if rec.tag in sentence_tags:
                assert rec.end is not None
                add(rec.end, "\n\n")
            elif rec.tag in token_break_element_tags:
                assert rec.end is not None
                add(rec.end, " ")
        elif rec.kind == "anchor":
            assert rec.offset is not None
            if rec.tag in break_bearers:
                # `lb`/`cb`/`pb` with `break="no"` was already consumed
                # by the join scanner and arrives here only when it
                # actually breaks a token.
                break_val = rec.attrs.get("break")
                if break_val != break_no_value:
                    add(rec.offset, " ")
        elif rec.kind == "verbatim":
            # Barrier only around verbatims that represent EXCLUDED
            # subtrees. Skip if:
            #   - wrap_inside is set → the verbatim is a preferred-branch
            #     loser, NOT an excluded note; sibling content shouldn't
            #     break sentences.
            #   - join_group is set → the verbatim was absorbed into a
            #     truncation join (e.g. <pc>-</pc>); it sits inside the
            #     resulting <tok>, so a barrier here would split the
            #     joined word.
            if (
                around_verbatim
                and rec.wrap_inside is None
                and not rec.join_group
            ):
                assert rec.offset is not None
                add(rec.offset, "\n\n")

    return sorted(by_offset.items())


def _splice(
    fold: str, insertions: list[tuple[int, str]]
) -> tuple[str, list[int]]:
    """Walk `fold` and copy chars into `nlp`, emitting the configured
    insertion text at each offset. Returns (nlp_plaintext, nlp_to_fold).
    Inserted characters get `-1` in the map."""
    nlp_chars: list[str] = []
    nlp_to_fold: list[int] = []
    ins_idx = 0
    n_inserts = len(insertions)

    def fire_inserts_at(off: int) -> None:
        nonlocal ins_idx
        while ins_idx < n_inserts and insertions[ins_idx][0] == off:
            text = insertions[ins_idx][1]
            nlp_chars.append(text)
            nlp_to_fold.extend([-1] * len(text))
            ins_idx += 1

    for i, ch in enumerate(fold):
        fire_inserts_at(i)
        nlp_chars.append(ch)
        nlp_to_fold.append(i)
    fire_inserts_at(len(fold))

    return "".join(nlp_chars), nlp_to_fold


# ---------------------------------------------------------------------
# Coordinate translation helpers
# ---------------------------------------------------------------------


def nlp_to_fold_offset(root: ScopeRoot, nlp_offset: int) -> int:
    """Translate an nlp_plaintext offset to a fold_plaintext offset.

    Raises ValueError if the offset maps to an inserted barrier char
    (which means a token landed on a barrier — EC-07a, a hard error).
    """
    if not root.nlp_to_fold:
        # nlp_plaintext.build() was never called — pass-through.
        return nlp_offset
    if nlp_offset < 0 or nlp_offset >= len(root.nlp_to_fold):
        raise ValueError(
            f"nlp_offset {nlp_offset} out of range "
            f"(len={len(root.nlp_to_fold)})"
        )
    fold = root.nlp_to_fold[nlp_offset]
    if fold < 0:
        raise ValueError(
            f"nlp_offset {nlp_offset} maps to an inserted barrier char "
            f"(EC-07a — a token landed on \\n\\n, alignment broken)"
        )
    return fold


def fold_to_nlp_offset(root: ScopeRoot, fold_offset: int) -> int:
    """Translate a fold_plaintext offset to an nlp_plaintext offset.

    Returns the FIRST nlp position whose fold value equals `fold_offset`.
    If barriers were inserted at this fold position, the returned nlp
    offset points to the position AFTER the barrier (i.e. the actual
    character, not the inserted `\\n`).
    """
    if not root.nlp_to_fold:
        return fold_offset
    # Linear scan is fine for the sizes we deal with; a binary search
    # would be faster but more code. Add when profiling demands it.
    for i, v in enumerate(root.nlp_to_fold):
        if v == fold_offset:
            return i
    raise ValueError(
        f"fold_offset {fold_offset} has no matching nlp position"
    )
