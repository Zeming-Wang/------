from __future__ import annotations

from typing import Any

from masfactory.compatibility.errors import CompatibilityImportError


def discover_workflow_graph(doc: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Find the largest `(nodes, edges)` pair embedded in a nested mapping (LangFlow / Dify style exports)."""
    if not isinstance(doc, dict):
        raise CompatibilityImportError("Document root must be a JSON object / YAML mapping.")

    best: tuple[int, list[dict[str, Any]], list[dict[str, Any]]] | None = None
    stack: list[Any] = [doc]

    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            nodes = cur.get("nodes")
            edges = cur.get("edges")
            if isinstance(nodes, list) and isinstance(edges, list) and nodes:
                score = len(nodes)
                node_dicts = [n for n in nodes if isinstance(n, dict)]
                edge_dicts = [e for e in edges if isinstance(e, dict)]
                if not best or score > best[0]:
                    best = (score, node_dicts, edge_dicts)
            for v in cur.values():
                stack.append(v)
        elif isinstance(cur, list):
            for item in cur:
                stack.append(item)

    if best is None:
        raise CompatibilityImportError(
            "Could not find a non-empty `nodes` + `edges` structure. "
            "Expected shapes similar to LangFlow exports or Dify workflow DSL."
        )
    return best[1], best[2]


def coerce_edge_endpoints(edge: dict[str, Any]) -> tuple[str, str] | None:
    for a, b in (
        ("source", "target"),
        ("from", "to"),
        ("src", "dst"),
        ("source_node_id", "target_node_id"),
    ):
        if a in edge and b in edge:
            return str(edge[a]), str(edge[b])
    return None
