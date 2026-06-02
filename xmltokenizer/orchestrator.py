"""Standalone-CLI convenience: one-shot extract → backend → fold.

xmltokenizer is primarily a *library*. The intended consumer (flexipipe
and similar) does not use this orchestrator — it calls `extract`,
`attach_conllu`, and `fold` directly so it can interpose its own NLP
pipeline. The orchestrator here exists for the standalone CLI / smoke
test path where a single external command produces the CoNLL-U.

For library integration see the top-level ``xmltokenizer`` module
docstring.
"""

from __future__ import annotations

from . import attach_conllu, build_nlp_plaintext, extract, fold, load_profile
from .backends.base import TokenizerBackend
from .namespace import deactivate, reactivate


def run(
    raw: bytes,
    backend: TokenizerBackend,
    profile_name: str = "tei",
    *,
    source_path: str = "<bytes>",
    restore_xmlns: bool = False,
) -> bytes:
    """End-to-end: bytes in, bytes out.

    `backend` is any object satisfying `TokenizerBackend` — typically a
    `flexipipe_backend()` (the default) but you can plug in anything
    that pipes plaintext through stdin and returns CoNLL-U on stdout.
    """
    profile = load_profile(profile_name)
    raw_de, ns_transform = deactivate(raw)
    metadata = extract(raw_de, profile, source_path=source_path)

    w_counter = [0]
    s_counter = [0]
    for root in metadata.scope_roots:
        if not root.fold_plaintext.strip():
            continue  # EC-18: empty scope
        # Phase A.5: build the NLP plaintext with barrier insertions.
        build_nlp_plaintext(root, profile)
        conllu_text = backend.tokenize(root.nlp_plaintext)
        attach_conllu(
            root, conllu_text,
            profile=profile, w_counter=w_counter, s_counter=s_counter,
        )

    out = fold(metadata)
    if restore_xmlns and ns_transform is not None:
        out = reactivate(out, ns_transform)
    return out
