"""A tiny TOML reader covering the subset our profiles use.

Used only as a fallback when tomllib (Python 3.11 stdlib) and tomli (the
backport) are unavailable. The standard tomllib is *always* preferred when
present so we don't carry the maintenance cost of a real TOML implementation.

Subset supported (sufficient for the profile schema):
  - line comments starting with '#'
  - key = "string"         (escapes: \\n \\t \\\\ \\" \\u00ad)
  - key = 123
  - key = true / false
  - key = []  /  key = ["a", "b", 1, true]
  - [table.subtable]
  - [[array.of.tables]]
  - Whitespace + blank lines

Not supported (we deliberately don't use them in profiles):
  - multiline strings
  - dotted keys (a.b = c)
  - inline tables ( { ... } )
  - dates/times
  - hex/octal/binary integers, floats, +/-inf/nan
"""

from __future__ import annotations

from typing import Any, BinaryIO


def load(f: BinaryIO) -> dict:
    """tomllib-compatible entrypoint: take a binary file, return a dict."""
    return loads(f.read().decode("utf-8"))


def loads(text: str) -> dict:
    parser = _Parser(text)
    return parser.parse()


# =====================================================================
# Parser
# =====================================================================


class TOMLError(ValueError):
    pass


class _Parser:
    def __init__(self, text: str) -> None:
        self.lines = text.splitlines()
        self.line_no = 0

    def parse(self) -> dict:
        root: dict = {}
        current: dict = root
        # current path tracks the table we're filling for diagnostics
        current_path: list[str] = []

        for raw in self.lines:
            self.line_no += 1
            stripped = _strip_comment(raw).strip()
            if not stripped:
                continue

            if stripped.startswith("[["):
                if not stripped.endswith("]]"):
                    raise self._err("malformed array-of-tables header")
                path = stripped[2:-2].strip().split(".")
                current = self._aot(root, path)
                current_path = path
            elif stripped.startswith("["):
                if not stripped.endswith("]"):
                    raise self._err("malformed table header")
                path = stripped[1:-1].strip().split(".")
                current = self._table(root, path)
                current_path = path
            else:
                if "=" not in stripped:
                    raise self._err(f"expected '=' in: {stripped!r}")
                key, _, val = stripped.partition("=")
                key = key.strip()
                val = val.strip()
                if not key:
                    raise self._err("empty key")
                current[key] = self._parse_value(val)
        return root

    # --- helpers --------------------------------------------------

    def _err(self, msg: str) -> TOMLError:
        return TOMLError(f"line {self.line_no}: {msg}")

    def _table(self, root: dict, path: list[str]) -> dict:
        node: Any = root
        for p in path:
            if p not in node:
                node[p] = {}
            elif not isinstance(node[p], dict):
                raise self._err(f"path {'.'.join(path)} collides with non-table")
            node = node[p]
        return node

    def _aot(self, root: dict, path: list[str]) -> dict:
        parent = self._table(root, path[:-1]) if len(path) > 1 else root
        last = path[-1]
        if last not in parent:
            parent[last] = []
        elif not isinstance(parent[last], list):
            raise self._err(f"{last!r} already exists as non-list")
        new: dict = {}
        parent[last].append(new)
        return new

    def _parse_value(self, val: str) -> Any:
        if not val:
            raise self._err("empty value")
        if val[0] == '"':
            return _parse_string(val, self.line_no)
        if val[0] == "[":
            return self._parse_array(val)
        if val == "true":
            return True
        if val == "false":
            return False
        # numeric?
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            raise self._err(f"unrecognized value: {val!r}")

    def _parse_array(self, val: str) -> list:
        if not val.endswith("]"):
            raise self._err("array must end with ']' on the same line")
        inner = val[1:-1].strip()
        if not inner:
            return []
        items = _split_array_items(inner, self.line_no)
        return [self._parse_value(item) for item in items]


# =====================================================================
# Low-level string / array helpers
# =====================================================================


def _strip_comment(line: str) -> str:
    """Remove a '#' comment, respecting double-quoted strings."""
    out: list[str] = []
    in_str = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == '"' and (i == 0 or line[i - 1] != "\\"):
            in_str = not in_str
            out.append(c)
        elif c == "#" and not in_str:
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _parse_string(val: str, line_no: int) -> str:
    if len(val) < 2 or val[0] != '"' or val[-1] != '"':
        raise TOMLError(f"line {line_no}: malformed string {val!r}")
    inner = val[1:-1]
    out: list[str] = []
    i = 0
    while i < len(inner):
        c = inner[i]
        if c == "\\":
            if i + 1 >= len(inner):
                raise TOMLError(f"line {line_no}: trailing backslash in string")
            nxt = inner[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
            elif nxt == "t":
                out.append("\t")
                i += 2
            elif nxt == "\\":
                out.append("\\")
                i += 2
            elif nxt == '"':
                out.append('"')
                i += 2
            elif nxt == "u":
                if i + 6 > len(inner):
                    raise TOMLError(f"line {line_no}: bad \\u escape")
                out.append(chr(int(inner[i + 2 : i + 6], 16)))
                i += 6
            else:
                raise TOMLError(f"line {line_no}: bad escape \\{nxt}")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _split_array_items(inner: str, line_no: int) -> list[str]:
    """Split an array body on top-level commas, respecting strings."""
    items: list[str] = []
    buf: list[str] = []
    in_str = False
    for c in inner:
        if c == '"':
            in_str = not in_str
            buf.append(c)
        elif c == "," and not in_str:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return [it for it in items if it]
