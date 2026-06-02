"""Tiny XPath-ish selector engine for the profile's scope candidates.

Supported micro-grammar (per dev/01-design.md §5.2.2):

    selector := '//' name                                # any descendant
              | '/' name                                  # root, must equal name
              | '/*'                                      # the root element itself
              | selector '/' name                         # child step
              | selector '[' '@' attr '=' '"' val '"' ']' # attribute predicate

Examples:
    //text
    //body
    /*
    /TEI/text
    //div[@type="article"]/content

The match domain is an `xml.etree.ElementTree` (or any object exposing
`.tag`, `.attrib`, `.iter`, and iteration over children). The returned
Element objects are exactly what ElementTree produced.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One step in a compiled selector.

    `axis` is one of:
        'descendant_or_self' for `//name`
        'child'              for `/name` and  bare `name` after another step
        'root'               for `/*`
    `predicates` is a tuple of (attr_name, attr_value) pairs.
    """

    axis: str
    name: str
    predicates: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Selector:
    steps: tuple[Step, ...]
    source: str

    def __repr__(self) -> str:
        return f"Selector({self.source!r})"


_TOKEN_RE = re.compile(
    r"""
    (?P<dslash>//)                 |   # //
    (?P<slash>/)                   |   # /
    (?P<star>\*)                   |   # *
    (?P<name>[A-Za-z_][\w.\-]*)    |   # element name
    (?P<lbracket>\[)               |   # [
    (?P<rbracket>\])               |   # ]
    (?P<at>@)                      |   # @
    (?P<eq>=)                      |   # =
    (?P<string>"[^"]*")            |   # "..."
    (?P<ws>\s+)                        # whitespace (ignored)
    """,
    re.VERBOSE,
)


class SelectorSyntaxError(ValueError):
    pass


def compile(expr: str) -> Selector:
    """Compile a selector string into a Selector object."""
    tokens = list(_tokenize(expr))
    parser = _Parser(tokens, expr)
    steps = parser.parse()
    return Selector(steps=tuple(steps), source=expr)


def _tokenize(expr: str) -> Iterable[tuple[str, str]]:
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise SelectorSyntaxError(
                f"unexpected character {expr[pos]!r} at position {pos} in {expr!r}"
            )
        kind = m.lastgroup
        text = m.group(0)
        pos = m.end()
        if kind == "ws":
            continue
        yield kind, text  # type: ignore[misc]


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]], source: str) -> None:
        self.tokens = tokens
        self.i = 0
        self.source = source

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _eat(self, kind: str) -> str:
        tok = self._peek()
        if tok is None or tok[0] != kind:
            got = tok[1] if tok else "<EOF>"
            raise SelectorSyntaxError(
                f"expected {kind} in {self.source!r}, got {got!r}"
            )
        self.i += 1
        return tok[1]

    def parse(self) -> list[Step]:
        steps: list[Step] = []
        first = self._peek()
        if first is None:
            raise SelectorSyntaxError(f"empty selector {self.source!r}")
        # First step must start with // or /.
        if first[0] == "dslash":
            self._eat("dslash")
            name = self._eat("name")
            steps.append(Step("descendant_or_self", name, self._predicates()))
        elif first[0] == "slash":
            self._eat("slash")
            if self._peek() and self._peek()[0] == "star":
                self._eat("star")
                steps.append(Step("root", "*", self._predicates()))
                return steps
            name = self._eat("name")
            steps.append(Step("child_of_root", name, self._predicates()))
        else:
            raise SelectorSyntaxError(
                f"selector must start with / or //; got {first[1]!r} in {self.source!r}"
            )
        # Subsequent steps: / name
        while self._peek() is not None:
            if self._peek()[0] == "slash":
                self._eat("slash")
                name = self._eat("name")
                steps.append(Step("child", name, self._predicates()))
            else:
                raise SelectorSyntaxError(
                    f"unexpected token {self._peek()[1]!r} in {self.source!r}"
                )
        return steps

    def _predicates(self) -> tuple[tuple[str, str], ...]:
        preds: list[tuple[str, str]] = []
        while self._peek() and self._peek()[0] == "lbracket":
            self._eat("lbracket")
            self._eat("at")
            attr = self._eat("name")
            self._eat("eq")
            val = self._eat("string")
            self._eat("rbracket")
            preds.append((attr, val[1:-1]))  # strip enclosing quotes
        return tuple(preds)


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------


def match(sel: Selector, root: ET.Element) -> list[ET.Element]:
    """Return all elements (in document order) matching the selector
    against the tree rooted at `root`."""
    if not sel.steps:
        return []
    current: list[ET.Element] = []
    first = sel.steps[0]
    if first.axis == "root":
        # `/*` — the root itself, optionally with predicates.
        if _passes_predicates(root, first.predicates):
            current = [root]
    elif first.axis == "child_of_root":
        # `/name` — root must equal name.
        if root.tag == first.name and _passes_predicates(root, first.predicates):
            current = [root]
    elif first.axis == "descendant_or_self":
        # `//name` — any descendant (including root if it matches).
        for el in root.iter(first.name):
            if _passes_predicates(el, first.predicates):
                current.append(el)
    else:  # pragma: no cover
        raise AssertionError(f"unknown first-step axis {first.axis!r}")

    for step in sel.steps[1:]:
        nxt: list[ET.Element] = []
        for el in current:
            for child in el:
                if child.tag == step.name and _passes_predicates(child, step.predicates):
                    nxt.append(child)
        current = nxt
    return current


def _passes_predicates(el: ET.Element, preds: tuple[tuple[str, str], ...]) -> bool:
    for attr, val in preds:
        if el.attrib.get(attr) != val:
            return False
    return True
