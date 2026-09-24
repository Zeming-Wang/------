from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Union

from masfactory.components.controls.logic_switch import LogicSwitch
from masfactory.components.custom_node import CustomNode
from masfactory.components.graphs.loop import Loop
from masfactory.compatibility.chatdev.graph import ChatDevRootGraph
from masfactory.core.node import Node

from masfactory.compatibility.common.blueprint import ExternalEdge, GraphBlueprint
from masfactory.compatibility.common.builder import _dedupe_wiring_edges
from masfactory.compatibility.common.wire_loops import (
    apply_wire_pairs,
    build_loop_aware_wire_pairs,
    connect_loop_controller,
)
from masfactory.compatibility.common.names import uniquify_names
from masfactory.compatibility.common.loops import LoopPlan, nodes_in_loops, plan_loop_regions
from masfactory.compatibility.common.llm_options import ChatDevCompileOptions
from masfactory.compatibility.common.conditions import is_always_true_condition, needs_logic_switch
from masfactory.compatibility.chatdev.runtime import evaluate_chatdev_condition, message_text
from masfactory.compatibility.chatdev.template import render_chatdev_template
from masfactory.compatibility.errors import CompatibilityImportError

OwnerGraph = Union[ChatDevRootGraph, Loop]


def _node_config(ext) -> dict[str, Any]:
    raw = ext.raw or {}
    cfg = raw.get("config")
    return cfg if isinstance(cfg, dict) else {}


def _graph_store(owner: OwnerGraph) -> dict[str, Any]:
    return owner._attributes_store


def _chatdev_vars(store: dict[str, Any]) -> dict[str, Any]:
    compat = store.get("compatibility") if isinstance(store.get("compatibility"), dict) else {}
    chat = compat.get("chatdev") if isinstance(compat.get("chatdev"), dict) else {}
    vars_map = chat.get("vars")
    return dict(vars_map) if isinstance(vars_map, dict) else {}


def _make_passthrough_forward(node_id: str, owner: OwnerGraph):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        payload = {**dict(attributes), **input, "chatdev_node": node_id}
        store.setdefault("chatdev_outputs", {})[node_id] = payload
        return payload

    return forward


def _make_literal_forward(node_id: str, cfg: dict[str, Any], owner: OwnerGraph):
    value = cfg.get("value") if "value" in cfg else cfg.get("content")

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        text = str(value) if value is not None else ""
        payload = {**dict(attributes), **input, "text": text, "content": text, "chatdev_node": node_id}
        store.setdefault("chatdev_outputs", {})[node_id] = payload
        return payload

    return forward


def _make_loop_counter_forward(node_id: str, cfg: dict[str, Any], owner: OwnerGraph):
    key = f"chatdev_loop_{node_id}"
    limit = cfg.get("max_iterations") or cfg.get("limit") or cfg.get("max") or 10

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        counts = store.setdefault("chatdev_loop_counts", {})
        n = int(counts.get(key, 0)) + 1
        counts[key] = n
        payload = {
            **dict(attributes),
            **input,
            "loop_count": n,
            "loop_limit": limit,
            "loop_counter": node_id,
            "chatdev_node": node_id,
        }
        store.setdefault("chatdev_outputs", {})[node_id] = payload
        return payload

    return forward


def _make_agent_forward(
    node_id: str,
    cfg: dict[str, Any],
    owner: OwnerGraph,
    options: ChatDevCompileOptions,
):
    role_tpl = str(cfg.get("role") or cfg.get("system") or "agent")
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    model = options.resolve_model(model_cfg)

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        vars_map = _chatdev_vars(store)
        merged = {**dict(attributes), **input}
        role = render_chatdev_template(role_tpl, vars_map, merged)
        user_text = message_text({k: v for k, v in merged.items() if isinstance(v, str)})
        if not user_text:
            user_text = str(merged.get("task") or merged.get("query") or "")
        messages = [
            {"role": "system", "content": role},
            {"role": "user", "content": user_text or "(empty)"},
        ]
        response = model.invoke(messages, tools=None, settings=model_cfg.get("completion_params"))
        text = str(response.get("content") or "")
        payload = {**merged, "text": text, "content": text, "chatdev_node": node_id}
        store.setdefault("chatdev_outputs", {})[node_id] = payload
        return payload

    return forward


def _make_majority_vote_forward(
    node_id: str,
    voter_ids: list[str],
    owner: OwnerGraph,
):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        outputs = store.get("chatdev_outputs")
        votes: list[str] = []
        if isinstance(outputs, dict):
            for vid in voter_ids:
                payload = outputs.get(vid)
                if isinstance(payload, dict):
                    text = message_text(payload).strip()
                    if text:
                        votes.append(text)
        if not votes:
            votes = [message_text({**dict(attributes), **input}).strip()]
        votes = [v for v in votes if v]
        winner = Counter(votes).most_common(1)[0][0] if votes else ""
        payload = {
            **dict(attributes),
            **input,
            "text": winner,
            "content": winner,
            "votes": votes,
            "majority_winner": winner,
            "chatdev_node": node_id,
        }
        store.setdefault("chatdev_outputs", {})[node_id] = payload
        return payload

    return forward


def _make_chain_phase_forward(node_id: str, phase_raw: dict[str, Any], owner: OwnerGraph):
    phase_type = str(phase_raw.get("phaseType") or "")

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        payload = {
            **dict(attributes),
            **input,
            "phase": node_id,
            "phaseType": phase_type,
            "chatdev_node": node_id,
        }
        store.setdefault("chatdev_outputs", {})[node_id] = payload
        return payload

    return forward


def _materialize_node(
    owner: OwnerGraph,
    ext,
    options: ChatDevCompileOptions,
    id_to_graph_name: dict[str, str],
) -> Node:
    gname = id_to_graph_name[ext.id]
    attrs = {"chatdev_id": ext.id, "chatdev_label": ext.label, "chatdev_kind": ext.kind}
    cfg = _node_config(ext)
    kind = ext.kind

    if kind == "agent":
        forward = _make_agent_forward(ext.id, cfg, owner, options)
    elif kind == "literal":
        forward = _make_literal_forward(ext.id, cfg, owner)
    elif kind == "loop_counter":
        forward = _make_loop_counter_forward(ext.id, cfg, owner)
    elif kind in {"SimplePhase", "ComposedPhase"}:
        forward = _make_chain_phase_forward(ext.id, ext.raw or {}, owner)
    elif kind == "majority_vote":
        raw = ext.raw or {}
        voters = raw.get("voter_ids") if isinstance(raw.get("voter_ids"), list) else []
        forward = _make_majority_vote_forward(ext.id, [str(v) for v in voters], owner)
    else:
        forward = _make_passthrough_forward(ext.id, owner)

    return owner.create_node(
        CustomNode,
        name=gname,
        forward=forward,
        pull_keys=None,
        push_keys=None,
        attributes=attrs,
    )


def _make_route_predicates(
    edges: list[ExternalEdge],
    id_to_graph_name: dict[str, str],
) -> dict[str, Callable[[dict, dict[str, object]], bool]]:
    routes: dict[str, Callable[[dict, dict[str, object]], bool]] = {}
    for edge in edges:
        raw = edge.raw or {}
        cond = raw.get("condition")
        receiver = id_to_graph_name.get(edge.target, edge.target)

        def _predicate(
            message: dict,
            attributes: dict[str, object],
            *,
            _cond: Any = cond,
        ) -> bool:
            return evaluate_chatdev_condition(_cond, message=message, attributes=attributes)

        routes[receiver] = _predicate
    return routes


def _make_loop_terminate(plan: LoopPlan) -> Callable[[dict, dict[str, object], object], bool]:
    counter_limits: dict[str, int] = {}
    for cid in plan.loop_counter_ids:
        counter_limits[cid] = plan.max_iterations

    def terminate(input: dict[str, object], attributes: dict[str, object], controller: object) -> bool:
        counts = attributes.get("chatdev_loop_counts")
        if not isinstance(counts, dict):
            return False
        for cid, limit in counter_limits.items():
            key = f"chatdev_loop_{cid}"
            if int(counts.get(key, 0)) >= int(limit):
                return True
        return False

    return terminate


def _wire_subgraph(
    owner: OwnerGraph,
    *,
    member_ids: set[str],
    edges: list[ExternalEdge],
    instances: dict[str, Node],
    id_to_graph_name: dict[str, str],
) -> None:
    out_by_source: dict[str, list[ExternalEdge]] = defaultdict(list)
    for edge in edges:
        if edge.source in member_ids and edge.target in member_ids:
            out_by_source[edge.source].append(edge)

    route_nodes: dict[str, str] = {}
    for src, out_edges in out_by_source.items():
        if needs_logic_switch(out_edges):
            route_nodes[src] = f"{id_to_graph_name[src]}__route"

    switch_instances: dict[str, Node] = {}
    for src, route_name in route_nodes.items():
        conditional = [
            e
            for e in out_by_source[src]
            if not is_always_true_condition((e.raw or {}).get("condition"))
        ]
        routes = _make_route_predicates(conditional, id_to_graph_name)
        switch_instances[src] = owner.create_node(
            LogicSwitch,
            name=route_name,
            routes=routes,
            attributes={"chatdev_route_for": src},
        )

    wire_pairs: list[tuple[str, str]] = []
    for edge in edges:
        src = edge.source
        dst = edge.target
        if src not in member_ids or dst not in member_ids:
            continue
        raw_cond = (edge.raw or {}).get("condition")
        if src in route_nodes and not is_always_true_condition(raw_cond):
            continue
        wire_pairs.append((src, dst))

    wiring_edges, _ = _dedupe_wiring_edges(wire_pairs)

    for src, dst in wiring_edges:
        owner.create_edge(instances[src], instances[dst], keys={})

    for src in route_nodes:
        switch = switch_instances[src]
        owner.create_edge(instances[src], switch, keys={})
        for edge in out_by_source[src]:
            if is_always_true_condition((edge.raw or {}).get("condition")):
                continue
            owner.create_edge(switch, instances[edge.target], keys={})


def _compile_loop_subgraph(
    root: ChatDevRootGraph,
    plan: LoopPlan,
    blueprint: GraphBlueprint,
    options: ChatDevCompileOptions,
    id_to_graph_name: dict[str, str],
) -> Loop:
    loop_name = id_to_graph_name[plan.loop_id]
    terminate_fn = _make_loop_terminate(plan) if plan.loop_counter_ids else None
    loop = root.create_node(
        Loop,
        name=loop_name,
        max_iterations=plan.max_iterations,
        terminate_condition_function=terminate_fn,
        attributes={"chatdev_loop_id": plan.loop_id, "chatdev_loop_members": list(plan.member_ids)},
    )

    member_ids = set(plan.member_ids)
    nodes_by_id = {n.id: n for n in blueprint.nodes if n.id in member_ids}
    inner_instances: dict[str, Node] = {}
    for nid in member_ids:
        ext = nodes_by_id.get(nid)
        if ext is None:
            raise CompatibilityImportError(f"Loop member node missing from blueprint: {nid!r}")
        inner_instances[nid] = _materialize_node(loop, ext, options, id_to_graph_name)

    internal_edge_objs = list(plan.internal_edges)
    _wire_subgraph(
        loop,
        member_ids=member_ids,
        edges=internal_edge_objs,
        instances=inner_instances,
        id_to_graph_name=id_to_graph_name,
    )

    connect_loop_controller(loop, plan, inner_instances)
    return loop


def blueprint_to_chatdev_graph(
    blueprint: GraphBlueprint,
    *,
    graph_name: str,
    options: ChatDevCompileOptions | None = None,
) -> ChatDevRootGraph:
    opts = options or ChatDevCompileOptions()
    if not blueprint.nodes:
        raise CompatibilityImportError("Blueprint contains no nodes.")

    loop_plans = plan_loop_regions(blueprint)
    looped_ids = nodes_in_loops(loop_plans)
    plan_for_member = {m: p for p in loop_plans for m in p.member_ids}

    root_node_exts = [n for n in blueprint.nodes if n.id not in looped_ids]
    all_ids = [n.id for n in blueprint.nodes] + [p.loop_id for p in loop_plans]
    id_to_graph_name = uniquify_names(all_ids)

    compatibility: dict[str, Any] = {
        "source": "chatdev_workflow",
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
        "chatdev_loops": [
            {
                "loop_id": p.loop_id,
                "members": sorted(p.member_ids),
                "max_iterations": p.max_iterations,
                "back_edges": list(p.back_edges),
            }
            for p in loop_plans
        ],
    }
    if blueprint.metadata:
        compatibility.update(blueprint.metadata)

    graph = ChatDevRootGraph(name=graph_name, attributes={"compatibility": compatibility})

    root_instances: dict[str, Node] = {}
    for ext in root_node_exts:
        root_instances[ext.id] = _materialize_node(graph, ext, opts, id_to_graph_name)

    loop_instances: dict[str, Loop] = {}
    for plan in loop_plans:
        loop_instances[plan.loop_id] = _compile_loop_subgraph(
            graph, plan, blueprint, opts, id_to_graph_name
        )

    def _instance(token: str) -> Node:
        if token in loop_instances:
            return loop_instances[token]
        if token in plan_for_member:
            return loop_instances[plan_for_member[token].loop_id]
        if token in root_instances:
            return root_instances[token]
        raise CompatibilityImportError(f"Unknown graph node token {token!r} while wiring ChatDev workflow.")

    root_member_ids = {n.id for n in root_node_exts} | {p.loop_id for p in loop_plans}
    id_set = set(root_member_ids)

    out_by_source: dict[str, list[ExternalEdge]] = defaultdict(list)
    for edge in blueprint.edges:
        src = edge.source
        dst = edge.target
        if src in looped_ids and dst in looped_ids:
            if plan_for_member.get(src) is plan_for_member.get(dst):
                continue
        mapped_src = plan_for_member[src].loop_id if src in looped_ids else src
        mapped_dst = plan_for_member[dst].loop_id if dst in looped_ids else dst
        if mapped_src in id_set and mapped_dst in id_set:
            out_by_source[mapped_src].append(
                ExternalEdge(
                    source=mapped_src,
                    target=mapped_dst,
                    source_handle=edge.source_handle,
                    target_handle=edge.target_handle,
                    raw=edge.raw,
                )
            )

    route_nodes: dict[str, str] = {}
    for src, out_edges in out_by_source.items():
        if needs_logic_switch(out_edges):
            route_nodes[src] = f"{id_to_graph_name[src]}__route"

    switch_instances: dict[str, Node] = {}
    for src, route_name in route_nodes.items():
        conditional = [
            e
            for e in out_by_source[src]
            if not is_always_true_condition((e.raw or {}).get("condition"))
        ]
        routes = _make_route_predicates(conditional, id_to_graph_name)
        switch_instances[src] = graph.create_node(
            LogicSwitch,
            name=route_name,
            routes=routes,
            attributes={"chatdev_route_for": src},
        )

    wiring_edges, merged_parallel = build_loop_aware_wire_pairs(
        blueprint,
        loop_plans=loop_plans,
        looped_ids=looped_ids,
        plan_for_member=plan_for_member,
        member_id_set=id_set,
        skip_edge_sources=set(route_nodes),
    )
    if merged_parallel:
        compatibility["merged_parallel_edges"] = merged_parallel

    apply_wire_pairs(
        graph,
        wiring_edges,
        _instance,
        error_label="ChatDev",
    )
    for src in route_nodes:
        switch = switch_instances[src]
        graph.create_edge(_instance(src), switch, keys={})
        for edge in out_by_source[src]:
            if is_always_true_condition((edge.raw or {}).get("condition")):
                continue
            graph.create_edge(switch, _instance(edge.target), keys={})

    return graph
