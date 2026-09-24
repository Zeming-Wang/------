from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from masfactory.compatibility.common.blueprint import (
    ENTRY_TOKEN,
    EXIT_TOKEN,
    ExternalEdge,
    ExternalNode,
    GraphBlueprint,
)
from masfactory.compatibility.common.conditions import is_always_true_condition, needs_logic_switch
from masfactory.compatibility.common.loops import LoopPlan, nodes_in_loops, plan_loop_regions
from masfactory.compatibility.common.names import slugify_node_name, uniquify_dify_node_names, uniquify_names
from masfactory.compatibility.common.wire_loops import build_loop_aware_wire_pairs
from masfactory.compatibility.dify.node_kinds import CONTAINER_MARKER_KINDS, DIFY_UI_NODE_KINDS

_SWITCH_KINDS = frozenset({"if-else", "question-classifier"})
_LOOP_KINDS = frozenset({"loop", "iteration", "LoopComponent"})
_SUBGRAPH_KINDS = frozenset({"subgraph"})
_SKIP_KINDS = frozenset(CONTAINER_MARKER_KINDS | DIFY_UI_NODE_KINDS)


def _is_dify_blueprint(blueprint: GraphBlueprint) -> bool:
    meta = blueprint.metadata or {}
    return isinstance(meta.get("dify"), dict)


def _dify_skip_ids(blueprint: GraphBlueprint) -> frozenset[str]:
    dify = (blueprint.metadata or {}).get("dify") or {}
    raw = dify.get("container_child_node_ids")
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(str(x) for x in raw)


def _dify_loop_subgraphs(blueprint: GraphBlueprint) -> dict[str, dict[str, Any]]:
    dify = (blueprint.metadata or {}).get("dify") or {}
    raw = dify.get("loop_subgraphs")
    return dict(raw) if isinstance(raw, dict) else {}


def _build_id_map(blueprint: GraphBlueprint, *, active_nodes: list[ExternalNode]) -> dict[str, str]:
    if _is_dify_blueprint(blueprint):
        return uniquify_dify_node_names(active_nodes)
    return uniquify_names(n.id for n in active_nodes)


def _infer_v4_type(ext: ExternalNode) -> str:
    kind = ext.kind or ""
    if kind in _SWITCH_KINDS:
        return "Switch"
    if kind in _LOOP_KINDS:
        return "Loop"
    if kind in _SUBGRAPH_KINDS:
        return "Subgraph"
    return "Action"


def _compat_agent_name(ext: ExternalNode) -> str:
    kind = slugify_node_name(ext.kind.replace("-", "_"), fallback="node")
    return f"compat_{kind}"


def _action_node(
    ext: ExternalNode,
    *,
    node_id: str,
    scope: str,
    v4_type: str = "Action",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "id": node_id,
        "label": ext.label or ext.kind or node_id,
        "type": v4_type,
        "scope": scope,
        "compat_external_id": ext.id,
        "compat_kind": ext.kind,
    }
    if v4_type == "Action":
        node["agent"] = _compat_agent_name(ext)
        node["tools_allowed"] = []
        node["input_fields"] = []
        node["output_fields"] = []
    if v4_type == "Loop":
        node.setdefault("max_iterations", 20)
        node.setdefault("terminate_condition", "Imported loop (compatibility preview).")
    if extra:
        node.update(extra)
    return node


def _edge_condition(edge: ExternalEdge) -> str | None:
    raw = edge.raw if isinstance(edge.raw, dict) else {}
    for key in ("condition", "source_handle", "label"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip() and not is_always_true_condition(val):
            return val.strip()
    if edge.source_handle and not is_always_true_condition(edge.source_handle):
        return str(edge.source_handle).strip()
    return None


def _wire_pairs_to_v4_edges(
    wire_pairs: list[tuple[str, str]],
    *,
    id_to_name: dict[str, str],
    route_nodes: dict[str, str] | None = None,
    edges_by_pair: dict[tuple[str, str], ExternalEdge] | None = None,
) -> list[dict[str, Any]]:
    route_nodes = route_nodes or {}
    edges_by_pair = edges_by_pair or {}
    out: list[dict[str, Any]] = []
    for src, dst in wire_pairs:
        if src == ENTRY_TOKEN:
            v4_src = "ENTRY"
        else:
            v4_src = route_nodes.get(src, id_to_name.get(src, slugify_node_name(src, fallback="node")))
        if dst == EXIT_TOKEN:
            v4_dst = "EXIT"
        else:
            v4_dst = id_to_name.get(dst, slugify_node_name(dst, fallback="node"))

        edge: dict[str, Any] = {"source": v4_src, "target": v4_dst}
        ext_edge = edges_by_pair.get((src, dst))
        if ext_edge is not None:
            cond = _edge_condition(ext_edge)
            if cond:
                edge["condition"] = cond
        out.append(edge)
    return out


def _index_edges(edges: tuple[ExternalEdge, ...]) -> dict[tuple[str, str], ExternalEdge]:
    out: dict[tuple[str, str], ExternalEdge] = {}
    for edge in edges:
        out[(edge.source, edge.target)] = edge
    return out


def _build_scope_subgraph(
    member_ids: set[str],
    edges: list[ExternalEdge],
    *,
    nodes_by_id: dict[str, ExternalNode],
    id_to_name: dict[str, str],
    loop: bool,
    scope: str,
) -> dict[str, Any]:
    """Build a v4 `sub_graph` for Loop (CONTROLLER/TERMINATE) or Subgraph (ENTRY/EXIT)."""
    scoped_nodes: list[dict[str, Any]] = []
    for ext_id in sorted(member_ids, key=lambda x: (len(x), x)):
        ext = nodes_by_id.get(ext_id)
        if ext is None or ext.kind in _SKIP_KINDS:
            continue
        name = id_to_name.get(ext_id, slugify_node_name(ext_id, fallback="node"))
        v4_type = _infer_v4_type(ext)
        scoped_nodes.append(_action_node(ext, node_id=name, scope=scope, v4_type=v4_type))

    out_by_source: dict[str, list[ExternalEdge]] = defaultdict(list)
    internal: list[ExternalEdge] = []
    for edge in edges:
        if edge.source in member_ids and edge.target in member_ids:
            out_by_source[edge.source].append(edge)
            internal.append(edge)

    route_nodes: dict[str, str] = {}
    for src, out_edges in out_by_source.items():
        if needs_logic_switch(out_edges):
            route_nodes[src] = slugify_node_name(f"{id_to_name.get(src, src)}__route", fallback="route")

    scoped_edges: list[dict[str, Any]] = []
    entry_src = "CONTROLLER" if loop else "ENTRY"
    exit_dst = "TERMINATE" if loop else "EXIT"

    for src, out_edges in out_by_source.items():
        if src in route_nodes:
            route_name = route_nodes[src]
            scoped_edges.append({"source": id_to_name[src], "target": route_name})
            for edge in out_edges:
                if is_always_true_condition((edge.raw or {}).get("condition")) and is_always_true_condition(
                    edge.source_handle
                ):
                    scoped_edges.append({"source": route_name, "target": id_to_name[edge.target]})
                    continue
                item: dict[str, Any] = {
                    "source": route_name,
                    "target": id_to_name[edge.target],
                }
                cond = _edge_condition(edge)
                if cond:
                    item["condition"] = cond
                scoped_edges.append(item)
            continue
        for edge in out_edges:
            scoped_edges.append(
                {
                    "source": id_to_name[edge.source],
                    "target": id_to_name[edge.target],
                }
            )

    internal_targets = {e.target for e in internal}
    entry_members = [m for m in member_ids if m not in internal_targets]
    if not entry_members:
        entry_members = list(member_ids)

    for entry in entry_members[:1]:
        scoped_edges.insert(0, {"source": entry_src, "target": id_to_name[entry]})

    for member in member_ids:
        has_out = any(e.source == member for e in internal)
        if not has_out:
            scoped_edges.append({"source": id_to_name[member], "target": exit_dst})

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for edge in scoped_edges:
        key = (edge["source"], edge["target"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)

    return {"nodes": scoped_nodes, "edges": deduped}


def _loop_node_from_plan(
    plan: LoopPlan,
    *,
    nodes_by_id: dict[str, ExternalNode],
    id_to_name: dict[str, str],
) -> dict[str, Any]:
    loop_name = slugify_node_name(plan.loop_id, fallback="loop")
    scope = f"root/{loop_name}"
    sub = _build_scope_subgraph(
        set(plan.member_ids),
        list(plan.internal_edges),
        nodes_by_id=nodes_by_id,
        id_to_name=id_to_name,
        loop=True,
        scope=scope,
    )
    return _action_node(
        ExternalNode(id=plan.loop_id, kind="loop", label=loop_name, raw={}),
        node_id=loop_name,
        scope="root",
        v4_type="Loop",
        extra={
            "max_iterations": plan.max_iterations,
            "terminate_condition": f"Imported loop (max {plan.max_iterations} iterations).",
            "sub_graph": sub,
        },
    )


def _dify_inner_subgraph(
    loop_id: str,
    inner: dict[str, Any],
    *,
    nodes_by_id: dict[str, ExternalNode],
    id_to_name: dict[str, str],
) -> dict[str, Any]:
    child_ids: set[str] = set()
    for raw in inner.get("nodes") or []:
        if isinstance(raw, dict) and raw.get("id") is not None:
            child_ids.add(str(raw["id"]))

    member_ids = {cid for cid in child_ids if cid in nodes_by_id}
    if not member_ids:
        member_ids = child_ids

    internal_edges: list[ExternalEdge] = []
    for raw in inner.get("edges") or []:
        if not isinstance(raw, dict):
            continue
        src, dst = raw.get("source"), raw.get("target")
        if src is None or dst is None:
            continue
        if str(src) not in member_ids or str(dst) not in member_ids:
            continue
        internal_edges.append(
            ExternalEdge(
                source=str(src),
                target=str(dst),
                source_handle=str(raw["sourceHandle"]) if raw.get("sourceHandle") is not None else None,
                target_handle=str(raw["targetHandle"]) if raw.get("targetHandle") is not None else None,
                raw=dict(raw),
            )
        )

    loop_name = id_to_name.get(loop_id, slugify_node_name(loop_id, fallback="loop"))
    scope = f"root/{loop_name}"
    return _build_scope_subgraph(
        member_ids,
        internal_edges,
        nodes_by_id=nodes_by_id,
        id_to_name=id_to_name,
        loop=True,
        scope=scope,
    )


def blueprint_to_graph_design(blueprint: GraphBlueprint) -> dict[str, Any]:
    """Convert a compatibility blueprint into a Visualizer-ready ``graph_design`` object."""
    if not blueprint.nodes:
        return {"nodes": [], "edges": []}

    skip_ids = _dify_skip_ids(blueprint)
    loop_subgraphs = _dify_loop_subgraphs(blueprint)
    nodes_by_id = {n.id: n for n in blueprint.nodes}
    active_nodes = [n for n in blueprint.nodes if n.id not in skip_ids and n.kind not in _SKIP_KINDS]
    id_to_name = _build_id_map(blueprint, active_nodes=active_nodes)

    loop_plans = plan_loop_regions(blueprint)
    looped_ids = nodes_in_loops(loop_plans)
    plan_for_member: dict[str, LoopPlan] = {}
    for plan in loop_plans:
        for mid in plan.member_ids:
            plan_for_member[mid] = plan

    emitted_loop_names: set[str] = set()
    v4_nodes: list[dict[str, Any]] = []

    for ext in active_nodes:
        if ext.id in skip_ids:
            continue
        if ext.id in looped_ids:
            continue
        if ext.kind in _LOOP_KINDS and ext.id in loop_subgraphs:
            loop_name = id_to_name[ext.id]
            sub = _dify_inner_subgraph(
                ext.id,
                loop_subgraphs[ext.id],
                nodes_by_id=nodes_by_id,
                id_to_name=id_to_name,
            )
            v4_nodes.append(
                _action_node(
                    ext,
                    node_id=loop_name,
                    scope="root",
                    v4_type="Loop",
                    extra={
                        "max_iterations": 20,
                        "terminate_condition": "Imported Dify loop (compatibility preview).",
                        "sub_graph": sub,
                    },
                )
            )
            emitted_loop_names.add(loop_name)
            continue
        if ext.id in looped_ids:
            continue
        v4_nodes.append(
            _action_node(
                ext,
                node_id=id_to_name[ext.id],
                scope="root",
                v4_type=_infer_v4_type(ext),
            )
        )

    for plan in loop_plans:
        loop_name = slugify_node_name(plan.loop_id, fallback="loop")
        if loop_name in emitted_loop_names:
            continue
        if plan.member_ids <= skip_ids:
            continue
        id_to_name[plan.loop_id] = loop_name
        v4_nodes.append(
            _loop_node_from_plan(plan, nodes_by_id=nodes_by_id, id_to_name=id_to_name)
        )
        emitted_loop_names.add(loop_name)

    root_ids = {n.id for n in active_nodes if n.id not in looped_ids}
    member_id_set = root_ids | {p.loop_id for p in loop_plans}
    plan_for_member_map = plan_for_member
    wire_pairs, _ = build_loop_aware_wire_pairs(
        blueprint,
        loop_plans=loop_plans,
        looped_ids=looped_ids,
        plan_for_member=plan_for_member_map,
        member_id_set=member_id_set,
    )

    filtered_pairs: list[tuple[str, str]] = []
    for src, dst in wire_pairs:
        if src != ENTRY_TOKEN and src in skip_ids:
            continue
        if dst != EXIT_TOKEN and dst in skip_ids:
            continue
        if src in skip_ids or dst in skip_ids:
            continue
        filtered_pairs.append((src, dst))

    edges_by_pair = _index_edges(blueprint.edges)
    v4_edges = _wire_pairs_to_v4_edges(
        filtered_pairs,
        id_to_name=id_to_name,
        edges_by_pair=edges_by_pair,
    )

    if not any(e.get("source") == "ENTRY" for e in v4_edges) and v4_nodes:
        v4_edges.insert(0, {"source": "ENTRY", "target": v4_nodes[0]["id"]})
    if not any(e.get("target") == "EXIT" for e in v4_edges) and v4_nodes:
        v4_edges.append({"source": v4_nodes[-1]["id"], "target": "EXIT"})

    return {"nodes": v4_nodes, "edges": v4_edges}


def blueprint_to_graph_design_document(
    blueprint: GraphBlueprint,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Wrap ``blueprint_to_graph_design`` in the canonical ``{graph_design: ...}`` envelope."""
    doc: dict[str, Any] = {"graph_design": blueprint_to_graph_design(blueprint)}
    meta: dict[str, Any] = {"format": "graph_design_v4", "exporter": "masfactory.compatibility"}
    if source:
        meta["source"] = source
    if blueprint.metadata:
        meta["import_metadata"] = blueprint.metadata
    doc["compatibility_export"] = meta
    return doc


def write_graph_design_json(path: str | Path, document: dict[str, Any]) -> Path:
    """Write a graph_design document to disk (UTF-8, pretty-printed)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def export_graph_design_for_blueprint(
    blueprint: GraphBlueprint,
    path: str | Path,
    *,
    source: str | None = None,
) -> Path:
    """Serialize a blueprint as Visualizer-previewable ``graph_design.json``."""
    return write_graph_design_json(path, blueprint_to_graph_design_document(blueprint, source=source))
