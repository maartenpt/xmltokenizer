"""CoNLL-U parser, MWT-aware alignment, and udpipe-layer builder.

See dev/01-design.md §6.4 / §6.5 / §6.6.

The pipeline:
  CoNLL-U text  -- parse() -->  list[CSent]
  list[CSent] + plaintext  -- align_to_plaintext() -->  list[AlignedToken]
  list[AlignedToken] + profile  -- build_udpipe_layer() -->  Layer

Phase B's job is to call backend.tokenize() to get the CoNLL-U text, then
run those three steps. We separate them so each is independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .records import Layer, Record


# =====================================================================
# CoNLL-U parsing
# =====================================================================


@dataclass
class CTok:
    """One CoNLL-U token line."""

    id: str           # "3", "3-4" (mwt head), or "3.1" (empty token)
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: str
    deprel: str
    deps: str
    misc: str
    is_mwt_head: bool = False
    mwt_member_ids: list[str] = field(default_factory=list)


@dataclass
class CSent:
    sent_id: str
    tokens: list[CTok]


class CoNLLUParseError(ValueError):
    pass


def parse(text: str) -> list[CSent]:
    """Parse CoNLL-U text into a list of sentences.

    Robust to UTF-8 BOM, CRLF line endings, and skipping enhanced-only
    empty tokens (ids like "3.1"). MWT range lines ("3-4") are kept and
    annotated with `is_mwt_head=True` and `mwt_member_ids=["3", "4"]`.
    """
    if text.startswith("﻿"):
        text = text[1:]
    sentences: list[CSent] = []
    current_tokens: list[CTok] = []
    current_sent_id: Optional[str] = None
    sentence_index = 0

    def flush() -> None:
        nonlocal current_tokens, current_sent_id, sentence_index
        if not current_tokens:
            return
        sentence_index += 1
        sid = current_sent_id if current_sent_id is not None else str(sentence_index)
        sentences.append(CSent(sent_id=sid, tokens=current_tokens))
        current_tokens = []
        current_sent_id = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            flush()
            continue
        if line.startswith("#"):
            # Recognise "# sent_id = X" specifically.
            stripped = line.lstrip("#").strip()
            if stripped.startswith("sent_id"):
                _, _, val = stripped.partition("=")
                current_sent_id = val.strip()
            continue

        fields = line.split("\t")
        if len(fields) < 10:
            raise CoNLLUParseError(
                f"malformed line (need ≥10 tab-separated fields): {line!r}"
            )
        tok_id = fields[0]
        is_mwt = "-" in tok_id
        is_empty = "." in tok_id  # enhanced UD empty tokens
        if is_empty:
            # We don't use empty tokens in v1. Skip.
            continue
        tok = CTok(
            id=tok_id, form=fields[1], lemma=fields[2], upos=fields[3],
            xpos=fields[4], feats=fields[5], head=fields[6],
            deprel=fields[7], deps=fields[8], misc=fields[9],
            is_mwt_head=is_mwt,
        )
        if is_mwt:
            a, _, b = tok_id.partition("-")
            try:
                tok.mwt_member_ids = [str(x) for x in range(int(a), int(b) + 1)]
            except ValueError as e:
                raise CoNLLUParseError(
                    f"malformed MWT range {tok_id!r}"
                ) from e
        current_tokens.append(tok)
    flush()
    return sentences


# =====================================================================
# Alignment: CoNLL-U tokens -> plaintext spans
# =====================================================================


@dataclass
class AlignedToken:
    """A CoNLL-U token mapped to plaintext offsets.

    For normal tokens: `nlp_start` < `nlp_end`, the substring matches the
    token's form.
    For MWT *sub-tokens*: `nlp_start == nlp_end == surface_end` — they are
    zero-width point references that will become `<dtok>` anchors in the
    udpipe layer.
    """

    sent_id: str
    ctok: CTok
    nlp_start: int
    nlp_end: int


class AlignmentError(ValueError):
    pass


def align_to_plaintext(
    sentences: list[CSent], plaintext: str, start_cursor: int = 0
) -> list[AlignedToken]:
    """Linear scan: consume whitespace, match each token's surface form,
    record offsets. Aborts with a precise diagnostic on mismatch.

    For MWT (e.g. French ``aux`` → ``à`` + ``les``): only the surface form
    is matched against plaintext; sub-tokens get zero-width spans at the
    surface end. The caller (build_udpipe_layer) turns those into
    ``<dtok>`` anchors.
    """
    aligned: list[AlignedToken] = []
    cursor = start_cursor
    n = len(plaintext)
    for sent in sentences:
        toks = sent.tokens
        i = 0
        while i < len(toks):
            tok = toks[i]
            if tok.is_mwt_head:
                cursor = _skip_ws(plaintext, cursor)
                start, end = _match_form(plaintext, cursor, tok.form)
                aligned.append(AlignedToken(sent.sent_id, tok, start, end))
                cursor = end
                # Pair up the sub-tokens as point references.
                sub_count = len(tok.mwt_member_ids)
                for j in range(sub_count):
                    idx = i + 1 + j
                    if idx >= len(toks):
                        raise AlignmentError(
                            f"MWT {tok.id!r} declared {sub_count} sub-tokens "
                            f"but only {len(toks) - i - 1} follow"
                        )
                    sub = toks[idx]
                    aligned.append(AlignedToken(sent.sent_id, sub, end, end))
                i += 1 + sub_count
            else:
                cursor = _skip_ws(plaintext, cursor)
                start, end = _match_form(plaintext, cursor, tok.form)
                aligned.append(AlignedToken(sent.sent_id, tok, start, end))
                cursor = end
                i += 1
    return aligned


def _skip_ws(plaintext: str, cursor: int) -> int:
    n = len(plaintext)
    while cursor < n and plaintext[cursor].isspace():
        cursor += 1
    return cursor


def _match_form(plaintext: str, cursor: int, form: str) -> tuple[int, int]:
    """Strict form match: plaintext[cursor:cursor+len(form)] must equal
    form exactly. Mismatches abort with a precise diagnostic — they
    surface real upstream tokenizer bugs immediately rather than
    silently masking them with workarounds."""
    end = cursor + len(form)
    if plaintext[cursor:end] != form:
        ctx_lo = max(0, cursor - 30)
        ctx_hi = min(len(plaintext), cursor + len(form) + 30)
        raise AlignmentError(
            f"alignment failed at offset {cursor}: "
            f"expected {form!r}, got {plaintext[cursor:end]!r} "
            f"(context: {plaintext[ctx_lo:ctx_hi]!r})"
        )
    return cursor, end


# =====================================================================
# Building the udpipe layer
# =====================================================================


# Map of CoNLL-U field name → attribute key as emitted on <tok>/<dtok>.
# The profile may list "ohead" to mean "head renamed to ohead".
_FIELD_TO_CTOK_ATTR = {
    "lemma": "lemma",
    "upos": "upos",
    "xpos": "xpos",
    "feats": "feats",
    "head": "head",
    "ohead": "head",  # rename
    "deprel": "deprel",
    "deps": "deps",
    "misc": "misc",
}


def build_udpipe_layer(
    aligned: list[AlignedToken],
    profile: dict,
    global_w_counter: list[int],
    global_s_counter: list[int],
    nlp_to_fold: Optional[list[int]] = None,
) -> Layer:
    """Convert aligned tokens into the udpipe standoff layer.

    `global_w_counter` and `global_s_counter` are single-element lists used
    as mutable counters so a multi-scope or multi-chunk run shares one
    monotonic numbering across all calls. Pass `[0]` to start fresh.

    `nlp_to_fold`: if given, every offset in `aligned` is treated as an
    nlp_plaintext offset and translated to fold_plaintext coordinates. If
    None, offsets are assumed to already be in fold space (which is the
    case when no dual-plaintext step has been done).
    """
    tok_attrs_cfg = profile.get("tok_attrs", {})
    s_attrs_cfg = profile.get("s_attrs", {})
    tok_id_template = tok_attrs_cfg.get("id_template", "w-{n}")
    tok_id_attr = tok_attrs_cfg.get("id_attr", "xml:id")
    s_id_template = s_attrs_cfg.get("id_template", "s-{n}")
    s_id_attr = s_attrs_cfg.get("id_attr", "xml:id")
    fields_to_emit = tok_attrs_cfg.get(
        "fields_to_emit", ["lemma", "upos", "xpos", "feats", "ohead", "deprel"]
    )

    def to_fold(off: int) -> int:
        if nlp_to_fold is None:
            return off
        if off < 0 or off >= len(nlp_to_fold):
            raise AlignmentError(f"offset {off} out of range for nlp_to_fold")
        v = nlp_to_fold[off]
        if v < 0:
            raise AlignmentError(
                f"offset {off} maps to inserted char (-1) — token alignment "
                f"broken (EC-07a)"
            )
        return v

    records: list[Record] = []
    # Group aligned tokens by sent_id while preserving order.
    sent_buckets: dict[str, list[AlignedToken]] = {}
    sent_order: list[str] = []
    for at in aligned:
        if at.sent_id not in sent_buckets:
            sent_buckets[at.sent_id] = []
            sent_order.append(at.sent_id)
        sent_buckets[at.sent_id].append(at)

    for sent_id in sent_order:
        sent_aligned = sent_buckets[sent_id]
        # Sentence span = from first non-zero-width token's start to last
        # non-zero-width token's end. (MWT sub-tokens are zero-width.)
        substantive = [
            at for at in sent_aligned if at.nlp_start < at.nlp_end
        ]
        if not substantive:
            continue
        sent_nlp_start = substantive[0].nlp_start
        sent_nlp_end = substantive[-1].nlp_end
        sent_fold_start = to_fold(sent_nlp_start)
        # `to_fold(end-1)+1` gives the exclusive-end in fold coords without
        # tripping the inserted-char check that to_fold(end) might trip if
        # end == len(nlp_plaintext) sits on a barrier.
        sent_fold_end = to_fold(sent_nlp_end - 1) + 1

        # Build per-token records.
        token_records: list[Record] = []
        i = 0
        while i < len(sent_aligned):
            at = sent_aligned[i]
            ctok = at.ctok
            if ctok.is_mwt_head:
                global_w_counter[0] += 1
                n = global_w_counter[0]
                surface_tok_id = tok_id_template.format(n=n)
                surface_record_id = f"udpipe-w-{n}"
                fold_start = to_fold(at.nlp_start)
                fold_end = to_fold(at.nlp_end - 1) + 1
                # Surface <tok>: id + ord + (conditional) form. The
                # private `_xt_form` is promoted to `@form` by the
                # folder only when the token has inline XML inside.
                surface_attrs = {
                    tok_id_attr: surface_tok_id,
                    "ord": ctok.id,
                    "_xt_form": ctok.form,
                }
                token_records.append(
                    Record(
                        kind="element",
                        layer="udpipe",
                        id=surface_record_id,
                        tag="tok",
                        attrs=surface_attrs,
                        start=fold_start,
                        end=fold_end,
                        priority=50,
                        policy="atomic",
                    )
                )
                # Sub-tokens as <dtok> anchors with wrap_inside.
                sub_count = len(ctok.mwt_member_ids)
                for j in range(sub_count):
                    sub_at = sent_aligned[i + 1 + j]
                    sub_attrs: dict[str, str] = {"ord": sub_at.ctok.id}
                    sub_attrs.update(
                        _build_tok_attrs(
                            sub_at.ctok, fields_to_emit, include_form=True
                        )
                    )
                    token_records.append(
                        Record(
                            kind="anchor",
                            layer="udpipe",
                            id=f"{surface_record_id}-d{j + 1}",
                            tag="dtok",
                            attrs=sub_attrs,
                            offset=fold_end,
                            wrap_inside=surface_record_id,
                            priority=40,
                        )
                    )
                i += 1 + sub_count
            else:
                global_w_counter[0] += 1
                n = global_w_counter[0]
                tok_id = tok_id_template.format(n=n)
                fold_start = to_fold(at.nlp_start)
                fold_end = to_fold(at.nlp_end - 1) + 1
                # Order of attrs: xml:id, ord, _xt_form (private; promoted
                # to @form by fold when the tok contains inline XML),
                # then the morphology fields (lemma/upos/...).
                attrs = {
                    tok_id_attr: tok_id,
                    "ord": ctok.id,
                    "_xt_form": ctok.form,
                }
                attrs.update(_build_tok_attrs(ctok, fields_to_emit))
                token_records.append(
                    Record(
                        kind="element",
                        layer="udpipe",
                        id=f"udpipe-w-{n}",
                        tag="tok",
                        attrs=attrs,
                        start=fold_start,
                        end=fold_end,
                        priority=50,
                        policy="atomic",
                    )
                )
                i += 1

        # Sentence record + its tokens, in that order.
        global_s_counter[0] += 1
        sn = global_s_counter[0]
        s_xml_id = s_id_template.format(n=sn)
        s_attrs = {s_id_attr: s_xml_id, "n": str(sent_id)}
        records.append(
            Record(
                kind="element",
                layer="udpipe",
                id=f"udpipe-s-{sn}",
                tag="s",
                attrs=s_attrs,
                start=sent_fold_start,
                end=sent_fold_end,
                priority=60,
                # Sentences NEVER split: if structure inside would force
                # a split, the folder emits an empty <s xml:id=".."/>
                # anchor at the sentence's start position instead.
                policy="split-with-anchor-fallback",
            )
        )
        records.extend(token_records)

    return Layer(name="udpipe", records=records, default_priority=50)


def _build_tok_attrs(
    ctok: CTok, fields_to_emit: list[str], include_form: bool = False
) -> dict[str, str]:
    """Build the attrs dict for a <tok> or <dtok>, skipping CoNLL-U '_' values."""
    out: dict[str, str] = {}
    if include_form:
        out["form"] = ctok.form
    for fld in fields_to_emit:
        ctok_field = _FIELD_TO_CTOK_ATTR.get(fld)
        if ctok_field is None:
            continue
        v = getattr(ctok, ctok_field, "")
        if v and v != "_":
            out[fld] = v
    return out
