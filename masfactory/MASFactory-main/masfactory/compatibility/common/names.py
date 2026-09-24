from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from masfactory.utils.naming import is_valid_name

if TYPE_CHECKING:
    from masfactory.compatibility.common.blueprint import ExternalNode

_NON_ID_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_RESERVED_NODE_NAMES = frozenset({"entry", "exit"})


def slugify_node_name(raw: str, *, fallback: str) -> str:
    """Map an arbitrary external id to a MASFactory-valid node name."""
    s = raw.strip() if isinstance(raw, str) else str(raw)
    s = _NON_ID_CHARS.sub("_", s).strip("_")
    if not s:
        s = fallback
    if s and (not s[0].isalpha() and s[0] != "_"):
        s = f"n_{s}"
    if not is_valid_name(s):
        s = _NON_ID_CHARS.sub("_", s)
        if not s:
            s = fallback
        if s and (not s[0].isalpha() and s[0] != "_"):
            s = f"n_{s}"
    return s


def _kind_slug(kind: str) -> str:
    return slugify_node_name(kind.replace("-", "_"), fallback="node")


def _dify_name_base(ext: "ExternalNode") -> str:
    """Build a readable graph node name from Dify kind + title, falling back to id suffix."""
    kind_part = _kind_slug(ext.kind)
    label_part = slugify_node_name(ext.label, fallback="")
    if label_part and not re.fullmatch(r"n_\d+", label_part):
        return slugify_node_name(f"{kind_part}_{label_part}", fallback=kind_part)
    raw_id = str(ext.id)
    suffix = raw_id[-8:] if len(raw_id) > 8 else raw_id
    return slugify_node_name(f"{kind_part}_{suffix}", fallback=kind_part)


def uniquify_dify_node_names(nodes: Iterable["ExternalNode"]) -> dict[str, str]:
    """Map Dify node id -> MASFactory node name (prefer kind+title, else kind+id tail)."""
    used: set[str] = set()
    out: dict[str, str] = {}
    for i, ext in enumerate(nodes):
        base = _dify_name_base(ext)
        name = base
        idx = 1
        while name in used or name.lower() in _RESERVED_NODE_NAMES:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        out[str(ext.id)] = name
    return out


def uniquify_names(ids: Iterable[str]) -> dict[str, str]:
    """Return a mapping external_id -> unique graph-safe name."""
    unique = list(dict.fromkeys(ids))
    used: set[str] = set()
    out: dict[str, str] = {}
    for i, raw in enumerate(unique):
        base = slugify_node_name(raw, fallback=f"node_{i}")
        name = base
        idx = 1
        while name in used or name.lower() in _RESERVED_NODE_NAMES:
            name = f"{base}_{idx}"
            idx += 1
        used.add(name)
        out[raw] = name
    return out
