"""Profile loading. TOML-based per dev/02-architecture.md §9.

A profile is a dict shaped like DEFAULTS below. Built-in profiles live in
profiles_builtin/ (one TOML per profile name). User-supplied profiles can
be passed by path. Profiles can `extends = "<parent-name>"` to inherit.
"""

from __future__ import annotations

import copy
from pathlib import Path

# Profile files are TOML. Prefer stdlib tomllib (3.11+); fall back to the
# tomli backport (3.10); otherwise use our in-tree mini-parser, which
# handles the subset the profile schema actually needs.
try:  # pragma: no cover
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        from . import _minitoml as tomllib  # type: ignore[no-redef]

BUILTIN_DIR = Path(__file__).parent / "profiles_builtin"


# The defaults the loader merges every profile on top of. This is the
# single source of truth for what keys a fully-resolved profile contains.
DEFAULTS: dict = {
    "name": "default",
    "scope": {
        "candidates": ["/*"],
        "preserve_verbatim_outside_scope": True,
        "scope_excludes_at_root": [],
    },
    "exclude_elements": [],
    "prefer_one_child_of": [],
    "break_attribute": {
        "bearers": [],
        "default_value": "yes",
        "no_value": "no",
    },
    # When a break-bearing anchor has no @break attribute: "yes" (TEI
    # default — token break), "no" (always join), or "heuristic" (join
    # only if preceded by a truncation_strip_chars character). See
    # dev/01-design.md §5.5.1 / §5.5.6.
    "default_break": "yes",
    "truncation_strip_chars": [],
    # Elements whose hyphen-like content is "absorbed" into a truncation
    # join. The element + its content are stripped from fold_plaintext
    # (so NLP sees a clean joined word); the original element is
    # re-emitted verbatim inside the resulting <tok>. Typical TEI: <pc>.
    "join_absorb_elements": [],
    # Elements (with content) that, when their span coincides with a
    # `<tok>` span, should wrap FROM INSIDE the tok rather than from
    # outside. Implementation: these tags receive priority 30 at extract
    # time (below tok's priority 50), so the natural rebalance sort
    # places them as a child of the tok. Default: empty (everything
    # wraps from outside). TEI default adds editorial / orthography
    # tags like <del>, <supplied>, <unclear>, <add>, <corr>.
    "tok_inner_elements": [],
    # Anchors (empty elements) whose tag is listed here will sit INSIDE
    # the `<tok>` whose edge they coincide with. Default behavior for
    # unlisted anchors is to drift OUTSIDE the tok at its edges
    # (`<lb/><tok>word</tok>` rather than `<tok><lb/>word</tok>`).
    "tok_inner_anchors": [],
    "unsplittable": [],
    "barrier_elements": [],
    "token_break_elements": [],
    "barrier_around_verbatim": False,
    "tok_attrs": {
        "fields_to_emit": ["lemma", "upos", "xpos", "feats", "head", "deprel"],
        "id_template": "w-{n}",
        "id_attr": "xml:id",
    },
    "s_attrs": {
        "id_template": "s-{n}",
        "id_attr": "xml:id",
    },
    "chunk_boundary_elements": ["p", "div", "lg", "head", "ab", "lem"],
    "restore_xmlns": False,
}


def default_profile() -> dict:
    """Return a deep copy of the defaults. Safe to mutate."""
    return copy.deepcopy(DEFAULTS)


def load_profile(name_or_path: str) -> dict:
    """Load a profile by built-in name (e.g. ``"tei"``) or by file path.

    Resolves ``extends = "<parent>"`` recursively, then merges everything
    over the defaults. The returned dict has every key from DEFAULTS.
    """
    profile = default_profile()
    _load_into(name_or_path, profile, seen=set())
    return profile


def _load_into(name_or_path: str, target: dict, seen: set[str]) -> None:
    key = name_or_path
    if key in seen:
        raise ValueError(f"profile inheritance cycle involving {key!r}")
    seen.add(key)
    path = _resolve_profile_path(name_or_path)
    with open(path, "rb") as f:
        data = tomllib.load(f)
    parent = data.pop("extends", None)
    if parent:
        _load_into(parent, target, seen)
    merge(target, data)


def _resolve_profile_path(name_or_path: str) -> Path:
    if name_or_path.endswith(".toml") or "/" in name_or_path or "\\" in name_or_path:
        p = Path(name_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        return p
    return BUILTIN_DIR / f"{name_or_path}.toml"


VALID_DEFAULT_BREAK = frozenset({"yes", "no", "heuristic"})


def parse_truncation_strip_chars(value: str) -> list[str]:
    """Parse a comma-separated list of truncation marker characters."""
    return [part for part in (p.strip() for p in value.split(",")) if part]


def apply_profile_overrides(profile: dict, **overrides) -> dict:
    """Apply optional caller/CLI overrides in-place. Returns ``profile``."""
    if (default_break := overrides.get("default_break")) is not None:
        if default_break not in VALID_DEFAULT_BREAK:
            raise ValueError(
                f"default_break must be one of {sorted(VALID_DEFAULT_BREAK)}, "
                f"got {default_break!r}"
            )
        profile["default_break"] = default_break
    if (strip_chars := overrides.get("truncation_strip_chars")) is not None:
        profile["truncation_strip_chars"] = list(strip_chars)
    return profile


def parse_profile_option(spec: str) -> tuple[list[str], str]:
    """Parse ``KEY=VALUE`` or ``nested.key=VALUE`` into path + raw value."""
    if "=" not in spec:
        raise ValueError(f"profile option must be KEY=VALUE, got {spec!r}")
    key, value = spec.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"profile option has empty key: {spec!r}")
    return [part for part in key.split(".") if part], value


def _defaults_type(key_path: list[str]) -> str | None:
    """Return a coarse type tag for a dotted path in DEFAULTS."""
    node: object = DEFAULTS
    for part in key_path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, list):
        return "list"
    if isinstance(node, bool):
        return "bool"
    if isinstance(node, int):
        return "int"
    if isinstance(node, dict):
        return "dict"
    return "str"


def _coerce_bool(raw: str) -> bool:
    low = raw.lower()
    if low in {"1", "true", "yes", "on"}:
        return True
    if low in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, got {raw!r}")


def _coerce_list(raw: str) -> list[str]:
    if "," in raw:
        return parse_truncation_strip_chars(raw)
    return [raw] if raw else []


def coerce_profile_value(key_path: list[str], raw: str) -> object:
    """Coerce a CLI string to the type implied by DEFAULTS."""
    if key_path[-1] == "default_break":
        if raw not in VALID_DEFAULT_BREAK:
            raise ValueError(
                f"default_break must be one of {sorted(VALID_DEFAULT_BREAK)}, "
                f"got {raw!r}"
            )
        return raw
    kind = _defaults_type(key_path)
    if kind == "list":
        return _coerce_list(raw)
    if kind == "bool":
        return _coerce_bool(raw)
    if kind == "int":
        return int(raw)
    return raw


def _get_profile_node(profile: dict, key_path: list[str]) -> object:
    node: object = profile
    for part in key_path:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(
                f"unknown profile key {'.'.join(key_path)!r}"
            )
        node = node[part]
    return node


def _set_profile_node(profile: dict, key_path: list[str], value: object) -> None:
    node = profile
    for part in key_path[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ValueError(
                f"unknown profile key {'.'.join(key_path)!r}"
            )
        node = node[part]
    leaf = key_path[-1]
    if leaf not in node:
        raise ValueError(
            f"unknown profile key {'.'.join(key_path)!r}"
        )
    node[leaf] = value


def apply_profile_options(profile: dict, options: list[str]) -> dict:
    """Apply ``KEY=VALUE`` overrides (repeatable on the CLI).

    Keys use dot notation for nested profile dicts
    (e.g. ``break_attribute.no_value=no``). List-typed keys accept a
    single value or a comma-separated list. Append to a list with the
    ``_add`` suffix (e.g. ``truncation_strip_chars_add=¬``).
    """
    for spec in options:
        key_path, raw = parse_profile_option(spec)
        leaf = key_path[-1]
        if leaf.endswith("_add"):
            base_path = key_path[:-1] + [leaf[:-4]]
            existing = _get_profile_node(profile, base_path)
            if not isinstance(existing, list):
                raise ValueError(
                    f"profile key {'.'.join(base_path)!r} is not a list"
                )
            existing.extend(_coerce_list(raw))
        else:
            _set_profile_node(
                profile, key_path, coerce_profile_value(key_path, raw)
            )
    return profile


def merge(parent: dict, child: dict) -> dict:
    """Merge ``child`` into ``parent`` in-place (recursive for dicts).

    Lists are replaced wholesale. To *append* to an inherited list, use a
    key with the suffix ``_add`` in the child (e.g. ``exclude_elements_add
    = ["fw"]``).
    """
    for k, v in child.items():
        if k.endswith("_add"):
            base_key = k[:-4]
            existing = parent.get(base_key, [])
            if not isinstance(existing, list):
                raise ValueError(
                    f"profile key {k!r}: target {base_key!r} is not a list"
                )
            parent[base_key] = list(existing) + list(v)
        elif isinstance(v, dict) and isinstance(parent.get(k), dict):
            merge(parent[k], v)
        else:
            parent[k] = v
    return parent
