from __future__ import annotations

from typing import Any, Union

from masfactory.components.custom_node import CustomNode
from masfactory.components.graphs.loop import Loop
from masfactory.compatibility.langflow.graph import LangflowRootGraph
from masfactory.core.node import Node

from masfactory.compatibility.common.blueprint import ExternalNode, GraphBlueprint
from masfactory.compatibility.common.builder import _dedupe_wiring_edges
from masfactory.compatibility.common.wire_loops import (
    apply_wire_pairs,
    build_loop_aware_wire_pairs,
    connect_loop_controller,
)
from masfactory.compatibility.common.names import uniquify_dify_node_names
from masfactory.compatibility.common.loops import LoopPlan, nodes_in_loops, plan_loop_regions
from masfactory.compatibility.errors import CompatibilityImportError
from masfactory.compatibility.langflow.node_kinds import (
    CHAT_INPUT_KINDS,
    CHAT_OUTPUT_KINDS,
    LLM_KINDS,
    LOOP_KINDS,
    PROMPT_KINDS,
)
from masfactory.compatibility.common.llm_options import LangflowCompileOptions
from masfactory.compatibility.langflow.runtime import (
    langflow_field_value,
    langflow_template_fields,
    message_text,
)
from masfactory.compatibility.langflow.template import render_langflow_template

OwnerGraph = Union[LangflowRootGraph, Loop]


def _graph_store(owner: OwnerGraph) -> dict[str, Any]:
    return owner._attributes_store


def _record_output(store: dict[str, Any], node_id: str, payload: dict[str, Any]) -> None:
    store.setdefault("langflow_outputs", {})[node_id] = payload


def _loop_max_iterations(plan: LoopPlan, nodes_by_id: dict[str, Any]) -> int:
    best = plan.max_iterations
    for nid in plan.member_ids:
        ext = nodes_by_id.get(nid)
        if ext is None or ext.kind not in LOOP_KINDS:
            continue
        template = langflow_template_fields(ext.raw or {})
        for key in ("max_iterations", "max_loop_count", "loop_count", "iterations"):
            raw = langflow_field_value(template, key)
            if raw is None:
                continue
            try:
                best = max(best, int(raw))
            except (TypeError, ValueError):
                pass
    return best


def _make_passthrough_forward(node_id: str, owner: OwnerGraph):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        payload = {**dict(attributes), **input, "langflow_node": node_id}
        _record_output(store, node_id, payload)
        return payload

    return forward


def _make_chat_input_forward(node_id: str, ext, owner: OwnerGraph):
    template = langflow_template_fields(ext.raw or {})
    default_text = langflow_field_value(template, "input_value", "")

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        merged = {**dict(attributes), **input}
        text = message_text(merged) or str(default_text or "")
        payload = {**merged, "text": text, "content": text, "langflow_node": node_id, "langflow_kind": ext.kind}
        _record_output(store, node_id, payload)
        return payload

    return forward


def _make_chat_output_forward(node_id: str, owner: OwnerGraph):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        text = message_text({**dict(attributes), **input})
        payload = {**dict(attributes), **input, "text": text, "content": text, "langflow_node": node_id}
        _record_output(store, node_id, payload)
        store["langflow_result"] = payload
        return payload

    return forward


def _make_prompt_forward(node_id: str, ext, owner: OwnerGraph):
    template = langflow_template_fields(ext.raw or {})
    prompt_tpl = str(langflow_field_value(template, "template", "") or "")

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        merged = {**dict(attributes), **input}
        vars_map = {k: v for k, v in merged.items() if isinstance(v, (str, int, float, bool))}
        text = render_langflow_template(prompt_tpl, vars_map)
        payload = {
            **merged,
            "text": text,
            "content": text,
            "system_message": text,
            "prompt": text,
            "langflow_node": node_id,
            "langflow_kind": ext.kind,
        }
        _record_output(store, node_id, payload)
        return payload

    return forward


def _make_llm_forward(node_id: str, ext, owner: OwnerGraph, options: LangflowCompileOptions):
    template = langflow_template_fields(ext.raw or {})
    model_name = langflow_field_value(template, "model", "gpt-4o-mini")
    system_default = str(langflow_field_value(template, "system_message", "") or "")
    temperature = langflow_field_value(template, "temperature", 0.1)
    model_cfg = {"name": model_name, "completion_params": {"temperature": temperature}}

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        store = _graph_store(owner)
        merged = {**dict(attributes), **input}
        system = str(merged.get("system_message") or system_default or "")
        user = message_text(merged)
        if not user:
            user = str(merged.get("input_value") or merged.get("query") or "")
        messages: list[dict[str, str]] = []
        if system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user or "(empty)"})
        model = options.resolve_model(model_cfg)
        response = model.invoke(messages, tools=None, settings=model_cfg.get("completion_params"))
        text = str(response.get("content") or "")
        payload = {
            **merged,
            "text": text,
            "content": text,
            "langflow_node": node_id,
            "langflow_kind": ext.kind,
        }
        _record_output(store, node_id, payload)
        return payload

    return forward


def _materialize_node(
    owner: OwnerGraph,
    ext,
    options: LangflowCompileOptions,
    *,
    graph_name: str,
) -> Node:
    kind = ext.kind
    if kind in CHAT_INPUT_KINDS:
        forward = _make_chat_input_forward(ext.id, ext, owner)
    elif kind in CHAT_OUTPUT_KINDS:
        forward = _make_chat_output_forward(ext.id, owner)
    elif kind in PROMPT_KINDS:
        forward = _make_prompt_forward(ext.id, ext, owner)
    elif kind in LLM_KINDS:
        forward = _make_llm_forward(ext.id, ext, owner, options)
    else:
        forward = _make_passthrough_forward(ext.id, owner)

    return owner.create_node(
        CustomNode,
        name=graph_name,
        forward=forward,
        attributes={
            "langflow_id": ext.id,
            "langflow_kind": kind,
            "langflow_label": ext.label,
        },
    )


def _wire_subgraph(
    owner: OwnerGraph,
    *,
    member_ids: set[str],
    edges: list[ExternalEdge],
    instances: dict[str, Node],
) -> None:
    wire_pairs: list[tuple[str, str]] = []
    for edge in edges:
        if edge.source in member_ids and edge.target in member_ids:
            wire_pairs.append((edge.source, edge.target))
    wiring_edges, _ = _dedupe_wiring_edges(wire_pairs)
    for src, dst in wiring_edges:
        owner.create_edge(instances[src], instances[dst], keys={})


def _compile_loop_subgraph(
    root: LangflowRootGraph,
    plan: LoopPlan,
    blueprint: GraphBlueprint,
    options: LangflowCompileOptions,
    id_to_graph_name: dict[str, str],
    nodes_by_id: dict[str, Any],
) -> Loop:
    loop_name = id_to_graph_name[plan.loop_id]
    max_it = _loop_max_iterations(plan, nodes_by_id)
    loop = root.create_node(
        Loop,
        name=loop_name,
        max_iterations=max_it,
        attributes={"langflow_loop_id": plan.loop_id, "langflow_loop_members": list(plan.member_ids)},
    )

    member_ids = set(plan.member_ids)
    inner_instances: dict[str, Node] = {}
    for nid in member_ids:
        ext = nodes_by_id.get(nid)
        if ext is None:
            raise CompatibilityImportError(f"Loop member node missing from blueprint: {nid!r}")
        inner_instances[nid] = _materialize_node(loop, ext, options, graph_name=id_to_graph_name[nid])

    _wire_subgraph(loop, member_ids=member_ids, edges=list(plan.internal_edges), instances=inner_instances)

    connect_loop_controller(loop, plan, inner_instances)
    return loop


def blueprint_to_langflow_graph(
    blueprint: GraphBlueprint,
    *,
    graph_name: str,
    options: LangflowCompileOptions | None = None,
) -> LangflowRootGraph:
    if not blueprint.nodes:
        raise CompatibilityImportError("Blueprint contains no nodes.")

    opts = options or LangflowCompileOptions()
    loop_plans = plan_loop_regions(blueprint)
    looped_ids = nodes_in_loops(loop_plans)
    plan_for_member = {m: p for p in loop_plans for m in p.member_ids}
    nodes_by_id = {n.id: n for n in blueprint.nodes}

    root_node_exts = [n for n in blueprint.nodes if n.id not in looped_ids]
    naming_nodes: list[ExternalNode] = list(blueprint.nodes)
    for plan in loop_plans:
        if plan.loop_id not in nodes_by_id:
            naming_nodes.append(
                ExternalNode(id=plan.loop_id, kind="loop", label=plan.loop_id, raw={})
            )
    id_to_graph_name = uniquify_dify_node_names(naming_nodes)

    compatibility: dict[str, Any] = {
        "source": "langflow_json",
        "langflow": {"node_name_map": dict(id_to_graph_name)},
        "langflow_loops": [
            {"loop_id": p.loop_id, "members": sorted(p.member_ids), "max_iterations": _loop_max_iterations(p, nodes_by_id)}
            for p in loop_plans
        ],
    }
    if blueprint.metadata:
        compatibility.update(blueprint.metadata)

    graph = LangflowRootGraph(name=graph_name, attributes={"compatibility": compatibility})

    root_instances: dict[str, Node] = {}
    for ext in root_node_exts:
        root_instances[ext.id] = _materialize_node(
            graph, ext, opts, graph_name=id_to_graph_name[ext.id]
        )

    loop_instances: dict[str, Loop] = {}
    for plan in loop_plans:
        loop_instances[plan.loop_id] = _compile_loop_subgraph(
            graph, plan, blueprint, opts, id_to_graph_name, nodes_by_id
        )

    def _instance(token: str) -> Node:
        if token in loop_instances:
            return loop_instances[token]
        if token in plan_for_member:
            return loop_instances[plan_for_member[token].loop_id]
        if token in root_instances:
            return root_instances[token]
        raise CompatibilityImportError(f"Unknown Langflow node id {token!r} while wiring graph.")

    root_member_ids = {n.id for n in root_node_exts} | {p.loop_id for p in loop_plans}
    member_id_set = set(root_member_ids)

    wiring_edges, merged_parallel = build_loop_aware_wire_pairs(
        blueprint,
        loop_plans=loop_plans,
        looped_ids=looped_ids,
        plan_for_member=plan_for_member,
        member_id_set=member_id_set,
    )
    if merged_parallel:
        compatibility["merged_parallel_edges"] = merged_parallel

    apply_wire_pairs(graph, wiring_edges, _instance, error_label="Langflow")
    return graph
