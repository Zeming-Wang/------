from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ENTRY_TOKEN = "__masf_compat_entry__"
EXIT_TOKEN = "__masf_compat_exit__"


@dataclass(frozen=True)
class ExternalNode:
    """One node discovered in an external workflow document."""

    id: str
    kind: str
    label: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class ExternalEdge:
    """One directed edge between external node ids."""

    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class GraphBlueprint:
    """Normalized DAG-ish description before materializing a MASFactory `Graph`."""

    nodes: tuple[ExternalNode, ...]
    edges: tuple[ExternalEdge, ...]
    metadata: dict[str, Any] | None = None
