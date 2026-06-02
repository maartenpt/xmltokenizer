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
