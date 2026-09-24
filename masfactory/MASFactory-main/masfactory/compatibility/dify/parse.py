from __future__ import annotations

from collections import defaultdict
from typing import Any

from masfactory.compatibility.common.blueprint import ExternalEdge, ExternalNode, GraphBlueprint
from masfactory.compatibility.dify.node_kinds import DIFY_END_NODE_KINDS, DIFY_ENTRY_NODE_KINDS
from masfactory.compatibility.errors import CompatibilityImportError

DIFY_START_KINDS = DIFY_ENTRY_NODE_KINDS
DIFY_END_KINDS = DIFY_END_NODE_KINDS


def is_dify_app_document(doc: dict[str, Any]) -> bool:
    """Return True when `doc` looks like a Dify app/workflow DSL export."""
    if doc.get("kind") != "app":
        return False
    workflow = doc.get("workflow")
    if not isinstance(workflow, dict):
        return False
    graph = workflow.get("graph")
    if not isinstance(graph, dict):
        return False
    nodes = graph.get("nodes")
    return isinstance(nodes, list)


def _parse_dify_node(raw: dict[str, Any], *, index: int) -> ExternalNode:
    nid = raw.get("id")
    if nid is None:
        raise CompatibilityImportError(f"Dify node at index {index} is missing `id`.")
    data = raw.get("data")
    if not isinstance(data, dict):
        data = {}
    kind = str(data.get("type") or raw.get("type") or "")
    label = str(data.get("title") or kind or nid)
    return ExternalNode(id=str(nid), kind=kind, label=label, raw=dict(raw))


def _extract_loop_subgraphs(
    nodes_raw: list[Any],
    edges_raw: list[Any] | None,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Build per-loop subgraphs from nodes tagged with `data.loop_id`."""
    by_loop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    child_ids: set[str] = set()
    for item in nodes_raw:
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        if not isinstance(data, dict):
            continue
        loop_id = data.get("loop_id")
        if not loop_id:
            continue
        lid = str(loop_id)
        by_loop[lid].append(dict(item))
        nid = item.get("id")
        if nid is not None:
            child_ids.add(str(nid))

    edge_list = edges_raw if isinstance(edges_raw, list) else []
    subgraphs: dict[str, dict[str, Any]] = {}
    for lid, children in by_loop.items():
        ids = {str(c.get("id")) for c in children if c.get("id") is not None}
        inner_edges = [
            dict(e)
            for e in edge_list
            if isinstance(e, dict)
            and str(e.get("source")) in ids
            and str(e.get("target")) in ids
        ]
        subgraphs[lid] = {"nodes": children, "edges": inner_edges}
    return subgraphs, child_ids


def _parse_dify_edge(raw: dict[str, Any]) -> ExternalEdge | None:
    source = raw.get("source")
    target = raw.get("target")
    if source is None or target is None:
        return None
    return ExternalEdge(
        source=str(source),
        target=str(target),
        source_handle=str(raw["sourceHandle"]) if raw.get("sourceHandle") is not None else None,
        target_handle=str(raw["targetHandle"]) if raw.get("targetHandle") is not None else None,
        raw=dict(raw),
    )


def dify_document_to_blueprint(doc: dict[str, Any]) -> GraphBlueprint:
    """Parse a Dify `kind: app` workflow DSL export into a `GraphBlueprint`."""
    if not is_dify_app_document(doc):
        raise CompatibilityImportError(
            "Not a Dify app workflow document. Expected `kind: app` with `workflow.graph.nodes`."
        )

    workflow = doc["workflow"]
    graph = workflow["graph"]
    nodes_raw = graph.get("nodes")
    edges_raw = graph.get("edges")

    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise CompatibilityImportError("Dify workflow graph must contain a non-empty `nodes` list.")

    nodes: list[ExternalNode] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(nodes_raw):
        if not isinstance(item, dict):
            continue
        node = _parse_dify_node(item, index=i)
        if node.id in seen_ids:
            continue
        seen_ids.add(node.id)
        nodes.append(node)

    if not nodes:
        raise CompatibilityImportError("Dify workflow graph contains no usable node objects.")

    edges: list[ExternalEdge] = []
    if isinstance(edges_raw, list):
        for item in edges_raw:
            if not isinstance(item, dict):
                continue
            edge = _parse_dify_edge(item)
            if edge is not None:
                edges.append(edge)

    app = doc.get("app")
    app_meta = dict(app) if isinstance(app, dict) else {}
    kind_counts: dict[str, int] = {}
    for node in nodes:
        kind_counts[node.kind] = kind_counts.get(node.kind, 0) + 1

    metadata: dict[str, Any] = {
        "dify": {
            "kind": doc.get("kind"),
            "version": doc.get("version"),
            "app_name": app_meta.get("name"),
            "app_mode": app_meta.get("mode"),
            "app": app_meta,
        },
        "start_node_ids": [n.id for n in nodes if n.kind in DIFY_START_KINDS],
        "end_node_ids": [n.id for n in nodes if n.kind in DIFY_END_KINDS],
        "node_kind_counts": kind_counts,
    }
    if isinstance(workflow.get("conversation_variables"), list):
        metadata["dify"]["conversation_variables"] = workflow["conversation_variables"]
    if isinstance(workflow.get("environment_variables"), list):
        metadata["dify"]["environment_variables"] = workflow["environment_variables"]
    if isinstance(workflow.get("iteration_graph"), dict):
        metadata["dify"]["iteration_graph"] = workflow["iteration_graph"]
    loop_subgraphs, loop_child_ids = _extract_loop_subgraphs(nodes_raw, edges_raw)
    if loop_subgraphs:
        metadata["dify"]["loop_subgraphs"] = loop_subgraphs
    if loop_child_ids:
        metadata["dify"]["container_child_node_ids"] = sorted(loop_child_ids)

    return GraphBlueprint(nodes=tuple(nodes), edges=tuple(edges), metadata=metadata)
