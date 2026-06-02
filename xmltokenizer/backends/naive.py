"""Naive whitespace + punctuation tokenizer — pure Python, zero deps.

This backend exists so xmltokenizer can run standalone with no external
tool installed. Tokenization is naive but well-behaved:

- Sentence boundaries come from blank lines (`\\n\\n`) in the input.
  This dovetails with `nlp_plaintext` whose barriers are already
  blank-line insertions at unsplittable / barrier-element boundaries —
  so even the naive backend respects the document structure.
- Within a sentence: split on whitespace into rough words, then peel off
  leading and trailing ASCII punctuation as separate tokens. So
  `"test."` becomes `["test", "."]` and `"(hello)"` becomes `["(",
  "hello", ")"]`.
- Lemma, UPOS, FEATS etc. are all left as `_` (CoNLL-U "unset") since
  the naive backend has no morphology.

For real corpus work you almost certainly want a real backend
(flexipipe / UDPipe / etc.) — this is the "just works without
installation" default for the CLI.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

# Punctuation we peel off the edges of whitespace-separated chunks.
# Hyphens and apostrophes inside a word are deliberately KEPT — that's
# what makes `it's`, `state-of-the-art`, and `Mr.` survive as single
# tokens up to a final period.
_EDGE_PUNCT = set(string.punctuation)
# Don't peel hyphens off the *start* or *end* of a word — they're
# typically intra-word (compounds, dash continuations, etc.). The user
# can still strip them with a real backend.
_EDGE_PUNCT_LEADING = _EDGE_PUNCT - {"-"}
_EDGE_PUNCT_TRAILING = _EDGE_PUNCT - {"-"}


@dataclass
class NaiveBackend:
    """A pure-Python whitespace-and-punctuation tokenizer.

    Implements the `TokenizerBackend` protocol from `backends.base`.
    """

    max_chunk_chars: int = 0   # no limit
    name: str = "naive"

    def tokenize(self, plaintext: str) -> str:
        lines: list[str] = []
        sent_id = 0
        # Sentence-segment on blank lines (one or more \n\n runs).
        for chunk in _split_paragraphs(plaintext):
            chunk = chunk.strip()
            if not chunk:
                continue
            # Within a paragraph we still do one sentence per paragraph
            # — naive segmentation. (A real backend may produce multiple
            # sentences per paragraph; that's fine, our fold doesn't
            # care.)
            sent_id += 1
            forms = _tokenize_sentence(chunk)
            if not forms:
                continue
            lines.append(f"# sent_id = {sent_id}")
            lines.append(f"# text = {' '.join(forms)}")
            for i, form in enumerate(forms, start=1):
                # CoNLL-U: ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
                lines.append(f"{i}\t{form}\t_\t_\t_\t_\t_\t_\t_\t_")
            lines.append("")
        return "\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _split_paragraphs(text: str) -> list[str]:
    """Split on any run of two or more newlines (with optional spaces in
    between). Returns the inter-paragraph chunks in order."""
    out: list[str] = []
    current: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        # Try to consume a blank-line separator.
        j = i
        nl_count = 0
        while j < n and (text[j] == "\n" or text[j] == "\r" or text[j] == " " or text[j] == "\t"):
            if text[j] == "\n":
                nl_count += 1
            j += 1
        if nl_count >= 2 and j > i:
            if current:
                out.append("".join(current))
                current = []
            i = j
            continue
        current.append(text[i])
        i += 1
    if current:
        out.append("".join(current))
    return out


def _tokenize_sentence(sentence: str) -> list[str]:
    """Whitespace-split, then peel ASCII punctuation off the edges of
    each chunk. Returns a flat list of token-form strings."""
    tokens: list[str] = []
    for chunk in sentence.split():
        tokens.extend(_peel_punct(chunk))
    return tokens


def _peel_punct(word: str) -> list[str]:
    """Split off leading and trailing punctuation as separate tokens."""
    leading: list[str] = []
    trailing: list[str] = []
    # Leading punctuation.
    while word and word[0] in _EDGE_PUNCT_LEADING:
        leading.append(word[0])
        word = word[1:]
    # Trailing punctuation.
    tail_buf: list[str] = []
    while word and word[-1] in _EDGE_PUNCT_TRAILING:
        tail_buf.append(word[-1])
        word = word[:-1]
    trailing = list(reversed(tail_buf))
    out = leading
    if word:
        out.append(word)
    out.extend(trailing)
    return out
