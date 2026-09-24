from __future__ import annotations

from collections import defaultdict
from typing import Any

from masfactory.compatibility.common.blueprint import ENTRY_TOKEN, EXIT_TOKEN, ExternalEdge, ExternalNode, GraphBlueprint
from masfactory.compatibility.common.discover import coerce_edge_endpoints, discover_workflow_graph
from masfactory.compatibility.errors import CompatibilityImportError
from masfactory.compatibility.langflow.node_kinds import (
    CHAT_INPUT_KINDS,
    CHAT_OUTPUT_KINDS,
    SKIP_LANGFLOW_NODE_KINDS,
)
from masfactory.compatibility.langflow.runtime import langflow_handle_name


def is_langflow_document(doc: dict[str, Any]) -> bool:
    try:
        nodes_raw, _ = discover_workflow_graph(doc)
    except CompatibilityImportError:
        return False
    for raw in nodes_raw:
        if _is_skipped_langflow_node(raw):
            continue
        kind = _langflow_kind(raw)
        if kind and kind not in SKIP_LANGFLOW_NODE_KINDS:
            return True
        if str(raw.get("type") or "") == "genericNode":
            return True
    return False


def _is_skipped_langflow_node(raw: dict[str, Any]) -> bool:
    top = str(raw.get("type") or "")
    if top in ("noteNode", "note"):
        return True
    data = raw.get("data")
    if isinstance(data, dict):
        kind = str(data.get("type") or "")
        if kind in ("note", "noteNode"):
            return True
    nid = str(raw.get("id") or "")
    if nid.startswith("undefined"):
        return True
    return False


def _langflow_kind(raw: dict[str, Any]) -> str:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    kind = str(data.get("type") or "").strip()
    if kind and kind not in SKIP_LANGFLOW_NODE_KINDS:
        return kind
    top = str(raw.get("type") or "").strip()
    if top and top not in SKIP_LANGFLOW_NODE_KINDS:
        return top
    return "unknown"


def _langflow_label(raw: dict[str, Any], *, kind: str, node_id: str) -> str:
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    for key in ("display_name", "title", "name"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    node = data.get("node") if isinstance(data.get("node"), dict) else {}
    val = node.get("display_name")
    if isinstance(val, str) and val.strip():
        return val.strip()
    return kind or node_id


def _edges_from_langflow(edge_dicts: list[dict[str, Any]], *, valid_ids: set[str]) -> tuple[ExternalEdge, ...]:
    out: list[ExternalEdge] = []
    for e in edge_dicts:
        pair = coerce_edge_endpoints(e)
        if not pair:
            continue
        src, dst = pair
        if src not in valid_ids or dst not in valid_ids:
            continue
        data = e.get("data") if isinstance(e.get("data"), dict) else {}
        src_handle = langflow_handle_name(e.get("sourceHandle") or data.get("sourceHandle"))
        dst_handle = langflow_handle_name(e.get("targetHandle") or data.get("targetHandle"))
        out.append(
            ExternalEdge(
                source=src,
                target=dst,
                source_handle=src_handle,
                target_handle=dst_handle,
                raw=dict(e),
            )
        )
    return tuple(out)


def langflow_document_to_blueprint(doc: dict[str, Any]) -> GraphBlueprint:
    """Parse a Langflow React Flow export into a normalized blueprint."""
    nodes_raw, edges_raw = discover_workflow_graph(doc)
    seen: dict[str, ExternalNode] = {}
    for i, raw in enumerate(nodes_raw):
        if _is_skipped_langflow_node(raw):
            continue
        nid = raw.get("id")
        if nid is None:
            nid = f"anon_{i}"
        nid = str(nid)
        if nid in seen:
            continue
        kind = _langflow_kind(raw)
        label = _langflow_label(raw, kind=kind, node_id=nid)
        seen[nid] = ExternalNode(id=nid, kind=kind, label=label, raw=dict(raw))

    if not seen:
        raise CompatibilityImportError("Langflow export contains no executable nodes (only notes?).")

    valid_ids = set(seen)
    edges = list(_edges_from_langflow(edges_raw, valid_ids=valid_ids))

    incoming: defaultdict[str, int] = defaultdict(int)
    outgoing: defaultdict[str, int] = defaultdict(int)
    for edge in edges:
        incoming[edge.target] += 1
        outgoing[edge.source] += 1

    entry_ids: set[str] = set()
    exit_ids: set[str] = set()
    for nid, ext in seen.items():
        if ext.kind in CHAT_INPUT_KINDS:
            entry_ids.add(nid)
        if ext.kind in CHAT_OUTPUT_KINDS:
            exit_ids.add(nid)
        if incoming[nid] == 0:
            entry_ids.add(nid)
        if outgoing[nid] == 0:
            exit_ids.add(nid)

    edge_pairs = {(e.source, e.target) for e in edges}
    for nid in entry_ids:
        if (ENTRY_TOKEN, nid) not in edge_pairs:
            edges.append(ExternalEdge(source=ENTRY_TOKEN, target=nid))
    for nid in exit_ids:
        if (nid, EXIT_TOKEN) not in edge_pairs:
            edges.append(ExternalEdge(source=nid, target=EXIT_TOKEN))

    return GraphBlueprint(
        nodes=tuple(seen.values()),
        edges=tuple(edges),
        metadata={"langflow": {"format": "react_flow_json"}},
    )
