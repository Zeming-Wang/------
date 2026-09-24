from __future__ import annotations

from collections import defaultdict
from typing import Any

from masfactory.components.custom_node import CustomNode
from masfactory.components.graphs.graph import Graph

from masfactory.compatibility.common.blueprint import ENTRY_TOKEN, EXIT_TOKEN, GraphBlueprint
from masfactory.compatibility.common.names import uniquify_names
from masfactory.compatibility.errors import CompatibilityImportError


def _normalize_endpoint(token: str) -> str:
    t = token.strip()
    low = t.lower()
    if low in {"entry", "start", "__entry__", "__start__"}:
        return ENTRY_TOKEN
    if low in {"exit", "end", "__exit__", "__end__"}:
        return EXIT_TOKEN
    return t


def _resolve_edge_endpoint(token: str, id_set: set[str]) -> str:
    """Treat `entry`/`exit` as virtual ports only when they are not real external ids."""
    if token in id_set:
        return token
    return _normalize_endpoint(token)


def _dedupe_wiring_edges(
    edges: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Collapse parallel Dify edges that share the same resolved `(source, target)`.

    MASFactory forbids duplicate edges between the same node pair. Dify may emit
    several handles from one node into the same downstream node; we keep one wire
    and record the merged handles in compatibility metadata.
    """
    seen: set[tuple[str, str]] = set()
    wiring: list[tuple[str, str]] = []
    merged: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = {}
    for src, dst in edges:
        counts[(src, dst)] = counts.get((src, dst), 0) + 1
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        wiring.append((src, dst))
    for (src, dst), total in counts.items():
        if total > 1:
            merged.append({"source": src, "target": dst, "parallel_count": total})
    return wiring, merged


def blueprint_to_graph(
    blueprint: GraphBlueprint,
    *,
    graph_name: str,
    source: str,
) -> Graph:
    """Materialize a `Graph` using placeholder `CustomNode` instances (passthrough)."""
    if not blueprint.nodes:
        raise CompatibilityImportError("Blueprint contains no nodes.")

    node_ids = [n.id for n in blueprint.nodes]
    id_set = set(node_ids)

    resolved_edges: list[tuple[str, str]] = []
    for edge in blueprint.edges:
        a = _resolve_edge_endpoint(edge.source, id_set)
        b = _resolve_edge_endpoint(edge.target, id_set)
        resolved_edges.append((a, b))

    for src, dst in resolved_edges:
        if src != ENTRY_TOKEN and src not in id_set:
            raise CompatibilityImportError(f"Edge references unknown source node id: {src!r}")
        if dst != EXIT_TOKEN and dst not in id_set:
            raise CompatibilityImportError(f"Edge references unknown target node id: {dst!r}")

    id_to_graph_name = uniquify_names(node_ids)

    compatibility: dict[str, Any] = {
        "source": source,
        "external_nodes": {
            ext.id: {"kind": ext.kind, "label": ext.label} for ext in blueprint.nodes
        },
        "external_edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "source_handle": edge.source_handle,
                "target_handle": edge.target_handle,
            }
            for edge in blueprint.edges
        ],
    }
    if blueprint.metadata:
        compatibility.update(blueprint.metadata)
    attributes: dict[str, Any] = {"compatibility": compatibility}
    graph = Graph(name=graph_name, attributes=attributes)

    instances: dict[str, CustomNode] = {}
    for ext in blueprint.nodes:
        gname = id_to_graph_name[ext.id]
        meta = {"external_id": ext.id, "kind": ext.kind, "label": ext.label, "raw": ext.raw}
        node = graph.create_node(
            CustomNode,
            name=gname,
            attributes={"compatibility_node": meta},
        )
        instances[ext.id] = node

    incoming_internal: defaultdict[str, int] = defaultdict(int)
    outgoing_internal: defaultdict[str, int] = defaultdict(int)
    from_entry: set[str] = set()
    to_exit: set[str] = set()

    for src, dst in resolved_edges:
        if src == ENTRY_TOKEN and dst in id_set:
            from_entry.add(dst)
        elif dst == EXIT_TOKEN and src in id_set:
            to_exit.add(src)
        elif src in id_set and dst in id_set:
            incoming_internal[dst] += 1
            outgoing_internal[src] += 1

    to_create: list[tuple[str, str]] = list(resolved_edges)

    for nid in node_ids:
        if incoming_internal[nid] == 0 and nid not in from_entry:
            to_create.append((ENTRY_TOKEN, nid))
    for nid in node_ids:
        if outgoing_internal[nid] == 0 and nid not in to_exit:
            to_create.append((nid, EXIT_TOKEN))

    wiring_edges, merged_parallel = _dedupe_wiring_edges(to_create)
    if merged_parallel:
        compatibility["merged_parallel_edges"] = merged_parallel

    try:
        for src, dst in wiring_edges:
            if src == ENTRY_TOKEN:
                graph.edge_from_entry(instances[dst], keys={})
            elif dst == EXIT_TOKEN:
                graph.edge_to_exit(instances[src], keys={})
            else:
                graph.create_edge(instances[src], instances[dst], keys={})
    except ValueError as exc:
        raise CompatibilityImportError(
            "Failed to wire graph edges (cycle, duplicate edge, or invalid topology). "
            f"Original error: {exc}"
        ) from exc

    return graph
