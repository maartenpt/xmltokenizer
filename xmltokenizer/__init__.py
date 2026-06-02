"""xmltokenizer — a library for inline tokenization of TEI/XML.

**This is a library.** It is consumed by flexipipe (and any other tool
that wants TEI-aware tokenization). Flexipipe is a local Python process
that already provides the NLP pipeline producing CoNLL-U; it imports
this package directly. xmltokenizer's job is the XML side: extract
plaintext + standoff from XML, then fold CoNLL-U token spans back into
the original XML, honoring TEI's edge cases (splittable elements,
atomic tokens, truncation joins, preferred-branch rules, MWT, etc.).

## Library API (the production contract)

The integration in flexipipe (or any caller) looks like:

    import xmltokenizer as xt

    profile = xt.load_profile("tei")
    metadata = xt.extract(xml_bytes, profile)

    w_counter = [0]   # mutable counters — kept across all scope roots so
    s_counter = [0]   # token/sentence ids are globally unique.

    for root in metadata.scope_roots:
        xt.build_nlp_plaintext(root, profile)        # Phase A.5
        plaintext = root.nlp_plaintext               # feed this to NLP

        conllu_text = run_flexipipe_pipeline(plaintext)

        xt.attach_conllu(root, conllu_text,
                         profile=profile,
                         w_counter=w_counter,
                         s_counter=s_counter)

    output_xml_bytes = xt.fold(metadata)

The caller controls the NLP pipeline entirely. xmltokenizer does not
call any subprocess or network endpoint by itself — every byte that
leaves the library originates from the input XML or from the CoNLL-U
the caller passed in.

## Standalone CLI (convenience only)

A thin `python -m xmltokenizer ...` CLI exists for ad-hoc smoke testing
when you don't want to integrate via Python. It shells out to an
external command to produce CoNLL-U. **Production callers should NOT
use the CLI** — they should call this library directly from Python.
"""

from .extract import AlreadyTokenizedError, Metadata, ScopeRoot, run as extract
from .fold import run as fold
from .conllu import (
    AlignmentError,
    CoNLLUParseError,
    align_to_plaintext,
    build_udpipe_layer,
    parse as parse_conllu,
)
from .namespace import deactivate, reactivate
from .nlp_plaintext import build as build_nlp_plaintext
from .profile import load_profile, default_profile
from .records import Layer, Record
from .validate import ValidationError, run as validate

__version__ = "0.0.1"


def attach_conllu(
    root: ScopeRoot,
    conllu_text: str,
    *,
    profile: dict,
    w_counter: list[int],
    s_counter: list[int],
) -> Layer:
    """Convenience: parse `conllu_text`, align tokens, build & attach the
    udpipe layer.

    If `root.nlp_to_fold` is non-empty (because `build_nlp_plaintext` was
    called), the CoNLL-U is assumed to have been produced from
    `root.nlp_plaintext` and offsets are translated to fold coordinates.
    Otherwise we align against `root.fold_plaintext` directly.

    `w_counter` and `s_counter` are single-element mutable lists used to
    keep global token/sentence numbering across multiple scope roots (and
    across multiple chunks, when chunking is implemented). Pass `[0]` to
    start fresh.
    """
    sentences = parse_conllu(conllu_text)
    if root.nlp_to_fold:
        base = root.nlp_plaintext
        nlp_to_fold = root.nlp_to_fold
    else:
        base = root.fold_plaintext
        nlp_to_fold = None
    aligned = align_to_plaintext(sentences, base)
    layer = build_udpipe_layer(
        aligned, profile, w_counter, s_counter, nlp_to_fold=nlp_to_fold
    )
    root.udpipe_layer = layer
    return layer


__all__ = [
    "Metadata",
    "ScopeRoot",
    "Layer",
    "Record",
    "extract",
    "fold",
    "attach_conllu",
    "build_nlp_plaintext",
    "parse_conllu",
    "align_to_plaintext",
    "build_udpipe_layer",
    "deactivate",
    "reactivate",
    "load_profile",
    "default_profile",
    "AlignmentError",
    "AlreadyTokenizedError",
    "CoNLLUParseError",
    "validate",
    "ValidationError",
    "__version__",
]
