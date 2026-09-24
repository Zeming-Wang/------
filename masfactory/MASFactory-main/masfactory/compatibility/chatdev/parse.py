from __future__ import annotations

from pathlib import Path
from typing import Any

from masfactory.compatibility.common.blueprint import ENTRY_TOKEN, EXIT_TOKEN, ExternalEdge, ExternalNode, GraphBlueprint
from masfactory.compatibility.common.io import load_document_auto
from masfactory.compatibility.common.conditions import is_always_true_condition
from masfactory.compatibility.errors import CompatibilityImportError

_MAX_SUBGRAPH_DEPTH = 8
MAJORITY_VOTE_NODE_ID = "__chatdev_majority__"


def _normalize_graph_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def is_chatdev_majority_voting_document(doc: dict[str, Any]) -> bool:
    graph = doc.get("graph")
    if not isinstance(graph, dict):
        return False
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False
    return graph.get("is_majority_voting") is True


def is_chatdev_workflow_document(doc: dict[str, Any]) -> bool:
    graph = doc.get("graph")
    if not isinstance(graph, dict):
        return False
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return False
    if graph.get("is_majority_voting") is True:
        return True
    edges = graph.get("edges")
    return isinstance(edges, list)


def is_chatdev_chain_document(doc: dict[str, Any]) -> bool:
    chain = doc.get("chain")
    return isinstance(chain, list) and bool(chain) and not is_chatdev_workflow_document(doc)


def _load_subgraph_file(path: str, base_dir: Path | None) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    if not p.is_file():
        raise CompatibilityImportError(f"ChatDev subgraph file not found: {path!r}")
    doc = load_document_auto(p)
    if not isinstance(doc, dict) or not is_chatdev_workflow_document(doc):
        raise CompatibilityImportError(f"Not a ChatDev workflow document: {p}")
    return doc


def _subgraph_path(node: dict[str, Any]) -> str | None:
    cfg = node.get("config") if isinstance(node.get("config"), dict) else {}
    inner = cfg.get("config") if isinstance(cfg.get("config"), dict) else {}
    path = inner.get("path")
    return path.strip() if isinstance(path, str) and path.strip() else None


def _condition_handle(cond: Any) -> str | None:
    if cond is None:
        return None
    return "true" if is_always_true_condition(cond) else "conditional"


class _SubgraphPortal:
    __slots__ = ("node_id", "start_ids", "end_ids")

    def __init__(self, node_id: str, start_ids: list[str], end_ids: list[str]) -> None:
        self.node_id = node_id
        self.start_ids = start_ids
        self.end_ids = end_ids


def _flatten_graph(
    graph: dict[str, Any],
    *,
    id_prefix: str,
    base_dir: Path | None,
    depth: int,
    portals: dict[str, _SubgraphPortal],
) -> tuple[list[ExternalNode], list[ExternalEdge], list[str], list[str]]:
    nodes_out: list[ExternalNode] = []
    edges_out: list[ExternalEdge] = []

    for item in graph.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        nid = item.get("id")
        if nid is None:
            continue
        kind = str(item.get("type") or "")
        ext_id = f"{id_prefix}{nid}"

        if kind == "subgraph":
            path = _subgraph_path(item)
            if path is None:
                nodes_out.append(
                    ExternalNode(id=ext_id, kind=kind, label=str(nid), raw=dict(item))
                )
                continue
            if depth >= _MAX_SUBGRAPH_DEPTH:
                raise CompatibilityImportError(
                    f"ChatDev subgraph nesting exceeds depth {_MAX_SUBGRAPH_DEPTH}."
                )
            sub_doc = _load_subgraph_file(path, base_dir)
            inner_nodes, inner_edges, inner_start, inner_end = _flatten_graph(
                sub_doc["graph"],
                id_prefix=f"{ext_id}__",
                base_dir=base_dir,
                depth=depth + 1,
                portals=portals,
            )
            nodes_out.extend(inner_nodes)
            edges_out.extend(inner_edges)
            portals[ext_id] = _SubgraphPortal(ext_id, inner_start, inner_end)
            continue

        nodes_out.append(
            ExternalNode(
                id=ext_id,
                kind=kind,
                label=str(nid),
                raw=dict(item),
            )
        )

    node_ids = {n.id for n in nodes_out}
    for item in graph.get("edges") or []:
        if not isinstance(item, dict):
            continue
        src = item.get("from")
        dst = item.get("to")
        if src is None or dst is None:
            continue
        src_id = f"{id_prefix}{src}"
        dst_id = f"{id_prefix}{dst}"
        edges_out.append(
            ExternalEdge(
                source=src_id,
                target=dst_id,
                source_handle=_condition_handle(item.get("condition")),
                raw=dict(item),
            )
        )

    start_ids = [f"{id_prefix}{s}" for s in _normalize_graph_id_list(graph.get("start"))]
    end_ids = [f"{id_prefix}{e}" for e in _normalize_graph_id_list(graph.get("end"))]
    start_ids = [s for s in start_ids if s in node_ids]
    end_ids = [e for e in end_ids if e in node_ids]
    return nodes_out, edges_out, start_ids, end_ids


def _rewire_subgraph_edges(
    edges: list[ExternalEdge],
    portals: dict[str, _SubgraphPortal],
    node_ids: set[str],
) -> list[ExternalEdge]:
    if not portals:
        return edges
    out: list[ExternalEdge] = []
    for edge in edges:
        src, dst = edge.source, edge.target
        portal_dst = portals.get(dst)
        portal_src = portals.get(src)
        if portal_dst is not None:
            wired = False
            for start in portal_dst.start_ids:
                if start in node_ids:
                    out.append(
                        ExternalEdge(
                            source=src,
                            target=start,
                            source_handle=edge.source_handle,
                            raw=edge.raw,
                        )
                    )
                    wired = True
            if wired:
                continue
        if portal_src is not None:
            wired = False
            for end in portal_src.end_ids:
                if end in node_ids:
                    out.append(
                        ExternalEdge(
                            source=end,
                            target=dst,
                            source_handle=edge.source_handle,
                            raw=edge.raw,
                        )
                    )
                    wired = True
            if wired:
                continue
        if src in node_ids and dst in node_ids:
            out.append(edge)
    return out


def chatdev_majority_voting_to_blueprint(doc: dict[str, Any]) -> GraphBlueprint:
    """Compile ChatDev `is_majority_voting: true` graphs (parallel agents, no `edges` block)."""
    if not is_chatdev_majority_voting_document(doc):
        raise CompatibilityImportError("Not a ChatDev majority-voting workflow document.")
    graph = doc["graph"]
    nodes: list[ExternalNode] = []
    for item in graph.get("nodes") or []:
        if not isinstance(item, dict):
            continue
        nid = item.get("id")
        if nid is None:
            continue
        nodes.append(
            ExternalNode(
                id=str(nid),
                kind=str(item.get("type") or "agent"),
                label=str(nid),
                raw=dict(item),
            )
        )
    if not nodes:
        raise CompatibilityImportError("Majority-voting workflow has no usable nodes.")

    start_ids = _normalize_graph_id_list(graph.get("start"))
    voter_ids = [n.id for n in nodes if n.id in start_ids] or [n.id for n in nodes]
    voter_set = set(voter_ids)

    nodes.append(
        ExternalNode(
            id=MAJORITY_VOTE_NODE_ID,
            kind="majority_vote",
            label="majority_vote",
            raw={"voter_ids": voter_ids},
        )
    )

    edges: list[ExternalEdge] = []
    for vid in voter_ids:
        edges.append(ExternalEdge(source=ENTRY_TOKEN, target=vid, source_handle="start"))
        edges.append(ExternalEdge(source=vid, target=MAJORITY_VOTE_NODE_ID, source_handle="vote"))

    end_ids = _normalize_graph_id_list(graph.get("end"))
    terminal = MAJORITY_VOTE_NODE_ID
    if len(end_ids) == 1 and end_ids[0] not in voter_set and end_ids[0] != MAJORITY_VOTE_NODE_ID:
        terminal = end_ids[0]
        edges.append(ExternalEdge(source=MAJORITY_VOTE_NODE_ID, target=terminal, source_handle="aggregate"))
    edges.append(ExternalEdge(source=terminal, target=EXIT_TOKEN, source_handle="end"))

    metadata: dict[str, Any] = {
        "chatdev": {
            "format": "majority_voting_yaml",
            "graph_id": graph.get("id"),
            "description": graph.get("description"),
            "vars": doc.get("vars") if isinstance(doc.get("vars"), dict) else {},
            "version": doc.get("version"),
            "voter_ids": voter_ids,
        },
        "node_kind_counts": {},
    }
    for n in nodes:
        metadata["node_kind_counts"][n.kind] = metadata["node_kind_counts"].get(n.kind, 0) + 1

    return GraphBlueprint(nodes=tuple(nodes), edges=tuple(edges), metadata=metadata)


def chatdev_workflow_to_blueprint(
    doc: dict[str, Any],
    *,
    base_path: Path | str | None = None,
    _depth: int = 0,
) -> GraphBlueprint:
    if is_chatdev_majority_voting_document(doc):
        return chatdev_majority_voting_to_blueprint(doc)
    if not is_chatdev_workflow_document(doc):
        raise CompatibilityImportError(
            "Not a ChatDev 2.0 workflow document. Expected top-level `graph.nodes` and `graph.edges`."
        )
    graph = doc["graph"]
    base_dir: Path | None = None
    if isinstance(base_path, Path):
        base_dir = base_path.parent if base_path.is_file() else base_path
    elif isinstance(base_path, str) and base_path:
        p = Path(base_path)
        base_dir = p.parent if p.suffix else p

    portals: dict[str, _SubgraphPortal] = {}
    nodes, edges, start_ids, end_ids = _flatten_graph(
        graph,
        id_prefix="",
        base_dir=base_dir,
        depth=_depth,
        portals=portals,
    )
    if not nodes:
        raise CompatibilityImportError("ChatDev workflow graph has no usable nodes.")

    node_ids = {n.id for n in nodes}
    edges = _rewire_subgraph_edges(edges, portals, node_ids)

    for sid in start_ids:
        if sid in node_ids:
            edges.append(ExternalEdge(source=ENTRY_TOKEN, target=sid, source_handle="start"))
    for eid in end_ids:
        if eid in node_ids:
            edges.append(ExternalEdge(source=eid, target=EXIT_TOKEN, source_handle="end"))
    for term in ("FINAL", "END"):
        if term in node_ids and not any(e.source == term and e.target == EXIT_TOKEN for e in edges):
            edges.append(ExternalEdge(source=term, target=EXIT_TOKEN, source_handle="end"))

    metadata: dict[str, Any] = {
        "chatdev": {
            "format": "workflow_yaml",
            "graph_id": graph.get("id"),
            "description": graph.get("description"),
            "vars": doc.get("vars") if isinstance(doc.get("vars"), dict) else {},
            "version": doc.get("version"),
            "expanded_subgraphs": list(portals.keys()),
        },
        "node_kind_counts": {},
    }
    for n in nodes:
        metadata["node_kind_counts"][n.kind] = metadata["node_kind_counts"].get(n.kind, 0) + 1

    return GraphBlueprint(nodes=tuple(nodes), edges=tuple(edges), metadata=metadata)


def chatdev_chain_to_blueprint(doc: dict[str, Any]) -> GraphBlueprint:
    """Build a linear blueprint from ChatDev ``ChatChainConfig.json`` style ``chain`` array."""
    chain = doc.get("chain")
    if not isinstance(chain, list) or not chain:
        raise CompatibilityImportError(
            "Expected a document with a non-empty list field `chain` (ChatDev ChatChainConfig shape)."
        )
    nodes: list[ExternalNode] = []
    edges: list[ExternalEdge] = []
    prev_id: str | None = None
    for i, phase in enumerate(chain):
        if not isinstance(phase, dict):
            continue
        pid = phase.get("phase")
        if pid is None:
            pid = f"phase_{i}"
        pid = str(pid)
        kind = str(phase.get("phaseType", ""))
        nodes.append(ExternalNode(id=pid, kind=kind, label=pid, raw=dict(phase)))
        if prev_id is not None:
            edges.append(ExternalEdge(source=prev_id, target=pid))
        prev_id = pid
    if not nodes:
        raise CompatibilityImportError("`chain` contained no usable phase objects.")
    return GraphBlueprint(
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={"chatdev": {"format": "chain_json"}},
    )


def chatdev_document_to_blueprint(
    doc: dict[str, Any],
    *,
    base_path: Path | str | None = None,
) -> GraphBlueprint:
    """Parse ChatDev v1 chain JSON or v2 workflow YAML into a blueprint."""
    if is_chatdev_majority_voting_document(doc):
        return chatdev_majority_voting_to_blueprint(doc)
    if is_chatdev_workflow_document(doc):
        return chatdev_workflow_to_blueprint(doc, base_path=base_path)
    if is_chatdev_chain_document(doc):
        return chatdev_chain_to_blueprint(doc)
    raise CompatibilityImportError(
        "Unrecognized ChatDev document. Expected `graph.nodes`/`graph.edges` (2.0) "
        "or non-empty `chain` (1.x ChatChainConfig)."
    )
