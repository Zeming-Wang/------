from __future__ import annotations

from typing import Any

from masfactory.compatibility.common.blueprint import ExternalEdge, ExternalNode, GraphBlueprint
from masfactory.compatibility.common.discover import coerce_edge_endpoints, discover_workflow_graph
from masfactory.compatibility.errors import CompatibilityImportError


def _dedupe_nodes(node_dicts: list[dict[str, Any]]) -> tuple[ExternalNode, ...]:
    seen: dict[str, ExternalNode] = {}
    for i, raw in enumerate(node_dicts):
        nid = raw.get("id")
        if nid is None:
            nid = raw.get("name")
        if nid is None:
            nid = f"anon_{i}"
        nid = str(nid)
        if nid in seen:
            continue
        kind = str(raw.get("type") or raw.get("kind") or "")
        data = raw.get("data")
        if not kind and isinstance(data, dict):
            kind = str(data.get("type") or data.get("name") or data.get("title") or "")
        label = str(raw.get("label") or raw.get("title") or kind or nid)
        seen[nid] = ExternalNode(id=nid, kind=kind, label=label, raw=dict(raw))
    if not seen:
        raise CompatibilityImportError("No usable node objects found under `nodes`.")
    return tuple(seen.values())


def _edges_from_raw(edge_dicts: list[dict[str, Any]]) -> tuple[ExternalEdge, ...]:
    out: list[ExternalEdge] = []
    for e in edge_dicts:
        pair = coerce_edge_endpoints(e)
        if pair:
            src, dst = pair
            out.append(ExternalEdge(source=src, target=dst, raw=dict(e)))
    return tuple(out)


def workflow_export_to_blueprint(doc: dict[str, Any]) -> GraphBlueprint:
    """Parse LangFlow / Dify style `{nodes: [...], edges: [...]}` exports."""
    nodes_raw, edges_raw = discover_workflow_graph(doc)
    nodes = _dedupe_nodes(nodes_raw)
    edges = _edges_from_raw(edges_raw)
    return GraphBlueprint(nodes=nodes, edges=edges)
