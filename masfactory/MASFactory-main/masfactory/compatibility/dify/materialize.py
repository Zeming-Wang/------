from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from masfactory.components.controls.logic_switch import LogicSwitch
from masfactory.components.custom_node import CustomNode
from masfactory.components.graphs.graph import Graph
from masfactory.compatibility.dify.root_graph import DifyRootGraph
from masfactory.core.node import Node

from masfactory.compatibility.common.blueprint import ENTRY_TOKEN, EXIT_TOKEN, ExternalEdge, GraphBlueprint
from masfactory.compatibility.common.builder import (
    _dedupe_wiring_edges,
    _resolve_edge_endpoint,
)
from masfactory.compatibility.common.names import uniquify_dify_node_names
from masfactory.compatibility.dify.extra_nodes import (
    make_agent_forward,
    make_document_extractor_forward,
    make_http_request_forward,
    make_human_input_forward,
    make_iteration_forward,
    make_knowledge_index_forward,
    make_knowledge_retrieval_forward,
    make_list_operator_forward,
    make_loop_forward,
    make_passthrough_forward,
    make_question_classifier_routes,
    make_tool_forward,
)
from masfactory.compatibility.dify.node_kinds import CONTAINER_MARKER_KINDS, OFFICIAL_DIFY_NODE_KINDS
from masfactory.compatibility.dify.options import DifyCompileOptions
from masfactory.compatibility.dify.runtime import (
    DifyRuntimeState,
    collect_node_inputs,
    evaluate_handle,
    init_dify_state,
    run_dify_code,
)
from masfactory.compatibility.dify.template import render_dify_template
from masfactory.compatibility.errors import CompatibilityImportError


def _node_data(ext) -> dict[str, Any]:
    raw = ext.raw or {}
    data = raw.get("data")
    return data if isinstance(data, dict) else {}


def _make_start_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        outputs: dict[str, Any] = {}
        merged = {**attributes, **input}
        for var_def in data.get("variables") or []:
            if not isinstance(var_def, dict):
                continue
            key = str(var_def.get("variable"))
            if key in merged:
                outputs[key] = merged[key]
        state.set_node_output(node_id, outputs)
        return {**merged, **outputs}

    return forward


def _make_end_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        result: dict[str, Any] = {}
        for item in data.get("outputs") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("variable"))
            selector = item.get("value_selector") or []
            result[key] = state.resolve_selector(selector)
        state.set_node_output(node_id, result)
        state.workflow_result = result
        return {**input, **result}

    return forward


def _make_answer_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        text = render_dify_template(str(data.get("answer") or ""), state)
        payload = {"text": text, "answer": text}
        state.set_node_output(node_id, payload)
        state.workflow_result = {"text": text, "answer": text}
        return {**input, **payload}

    return forward


def _make_template_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        text = render_dify_template(str(data.get("template") or ""), state)
        payload = {"output": text, "text": text}
        state.set_node_output(node_id, payload)
        return {**input, **payload}

    return forward


def _make_code_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    code = str(data.get("code") or "")

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        variables = collect_node_inputs(data, state)
        result = run_dify_code(code, variables)
        state.set_node_output(node_id, result)
        return {**input, **result}

    return forward


def _make_assigner_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            selector = item.get("variable_selector") or []
            if len(selector) < 2:
                continue
            if item.get("input_type") == "constant":
                value = item.get("value")
            else:
                value_ref = item.get("value")
                if isinstance(value_ref, list):
                    value = state.resolve_selector(value_ref)
                else:
                    value = value_ref
            state.write_selector(selector, value)
        state.set_node_output(node_id, {"status": "ok"})
        return dict(input)

    return forward


def _make_aggregator_forward(node_id: str, data: dict[str, Any], store: dict[str, Any]):
    selectors = data.get("variables") or []

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        state = DifyRuntimeState.from_store(store)
        parts: list[str] = []
        merged: dict[str, Any] = {}
        for selector in selectors:
            if not isinstance(selector, list):
                continue
            value = state.resolve_selector(selector)
            if value is None:
                continue
            if isinstance(value, dict):
                merged.update(value)
                parts.extend(str(v) for v in value.values())
            else:
                parts.append(str(value))
        output = "\n".join(parts) if parts else ""
        if merged and not output:
            output = " ".join(str(v) for v in merged.values())
        payload = {"output": output, "text": output, **merged}
        state.set_node_output(node_id, payload)
        return {**input, **payload}

    return forward


def _llm_context_block(data: dict[str, Any], state: DifyRuntimeState) -> str:
    ctx = data.get("context")
    if not isinstance(ctx, dict) or not ctx.get("enabled"):
        return ""
    sel = ctx.get("variable_selector")
    if not isinstance(sel, list):
        return ""
    val = state.resolve_selector(sel)
    return str(val) if val is not None else ""


def _make_llm_forward(node_id: str, data: dict[str, Any], store: dict[str, Any], options: DifyCompileOptions):
    model_cfg = data.get("model") if isinstance(data.get("model"), dict) else {}
    _cached_model: list[Any] = [None]

    def forward(input: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
        if _cached_model[0] is None:
            _cached_model[0] = options.resolve_model(model_cfg)
        model = _cached_model[0]
        state = DifyRuntimeState.from_store(store)
        ctx_text = _llm_context_block(data, state)
        messages: list[dict[str, str]] = []
        for item in data.get("prompt_template") or []:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "user")
            text = str(item.get("text") or "")
            if "{{#context#}}" in text:
                text = text.replace("{{#context#}}", ctx_text)
            text = render_dify_template(text, state)
            if text:
                messages.append({"role": role, "content": text})
        if not messages:
            messages = [{"role": "user", "content": str(state.sys.get("query") or "")}]
        response = model.invoke(messages, tools=None, settings=model_cfg.get("completion_params"))
        text = str(response.get("content") or "")
        payload = {"text": text}
        state.set_node_output(node_id, payload)
        return {**input, **payload}

    return forward


def _make_if_else_routes(
    node_id: str,
    data: dict[str, Any],
    store: dict[str, Any],
    edges: list[ExternalEdge],
    id_to_graph_name: dict[str, str],
) -> dict[str, Callable[[dict, dict[str, object]], bool]]:
    cases = [case for case in (data.get("cases") or []) if isinstance(case, dict)]
    routes: dict[str, Callable[[dict, dict[str, object]], bool]] = {}
    for edge in edges:
        if edge.source != node_id:
            continue
        handle = str(edge.source_handle or "source")
        receiver = id_to_graph_name.get(edge.target, edge.target)

        def _predicate(
            message: dict,
            attributes: dict[str, object],
            *,
            handle_id: str = handle,
        ) -> bool:
            state = DifyRuntimeState.from_store(store)
            return evaluate_handle(handle_id, cases, state)

        routes[receiver] = _predicate
    return routes


def _materialize_node(
    graph: Graph,
    ext,
    graph_name: str,
    store: dict[str, Any],
    options: DifyCompileOptions,
    if_else_edges: dict[str, list[ExternalEdge]],
    id_to_graph_name: dict[str, str],
    iteration_graph: dict[str, Any] | None,
    loop_subgraphs: dict[str, dict[str, Any]] | None,
) -> Node:
    data = _node_data(ext)
    kind = ext.kind
    gname = id_to_graph_name[ext.id]

    dify_attrs = {"dify_id": ext.id, "dify_label": ext.label, "dify_kind": ext.kind}

    if kind == "if-else":
        routes = _make_if_else_routes(ext.id, data, store, if_else_edges.get(ext.id, []), id_to_graph_name)
        return graph.create_node(LogicSwitch, name=gname, routes=routes, attributes=dify_attrs)
    if kind == "question-classifier":
        routes = make_question_classifier_routes(
            ext.id, data, store, if_else_edges.get(ext.id, []), id_to_graph_name
        )
        return graph.create_node(LogicSwitch, name=gname, routes=routes, attributes=dify_attrs)

    forward: Callable[..., dict[str, object]]
    if kind == "start":
        forward = _make_start_forward(ext.id, data, store)
    elif kind == "end":
        forward = _make_end_forward(ext.id, data, store)
    elif kind == "answer":
        forward = _make_answer_forward(ext.id, data, store)
    elif kind == "template-transform":
        forward = _make_template_forward(ext.id, data, store)
    elif kind == "code":
        forward = _make_code_forward(ext.id, data, store)
    elif kind == "assigner":
        forward = _make_assigner_forward(ext.id, data, store)
    elif kind == "variable-aggregator":
        forward = _make_aggregator_forward(ext.id, data, store)
    elif kind in {"llm", "parameter-extractor"}:
        forward = _make_llm_forward(ext.id, data, store, options)
    elif kind == "tool":
        forward = make_tool_forward(ext.id, data, store, options.tool_executor)
    elif kind == "http-request":
        forward = make_http_request_forward(ext.id, data, store, options.http_client)
    elif kind == "knowledge-retrieval":
        forward = make_knowledge_retrieval_forward(ext.id, data, store, options.knowledge_retriever)
    elif kind == "knowledge-index":
        forward = make_knowledge_index_forward(ext.id, data, store, options.knowledge_retriever)
    elif kind == "iteration":
        forward = make_iteration_forward(ext.id, data, store, iteration_graph, options)
    elif kind == "loop":
        inner = (loop_subgraphs or {}).get(ext.id) if isinstance(loop_subgraphs, dict) else None
        forward = make_loop_forward(ext.id, data, store, inner, options)
    elif kind == "document-extractor":
        forward = make_document_extractor_forward(ext.id, data, store)
    elif kind == "list-operator":
        forward = make_list_operator_forward(ext.id, data, store)
    elif kind == "agent":
        llm_fwd = _make_llm_forward(ext.id, data, store, options)
        forward = make_agent_forward(
            ext.id, data, store, options, llm_forward=llm_fwd if data.get("model") else None
        )
    elif kind == "human-input":
        forward = make_human_input_forward(ext.id, data, store)
    elif kind in {"datasource", "trigger-webhook", "trigger-schedule", "trigger-plugin"}:
        forward = _make_start_forward(ext.id, data, store)
    elif kind == "variable-assigner":
        forward = _make_aggregator_forward(ext.id, data, store)
    elif kind in CONTAINER_MARKER_KINDS or kind in {"custom-note", "note"}:
        forward = make_passthrough_forward(ext.id, store)
    elif kind in OFFICIAL_DIFY_NODE_KINDS:
        forward = make_passthrough_forward(ext.id, store)
    else:
        forward = make_passthrough_forward(ext.id, store)

    return graph.create_node(
        CustomNode,
        name=gname,
        forward=forward,
        pull_keys=None,
        push_keys=None,
        attributes=dify_attrs,
    )


def blueprint_to_dify_graph(
    blueprint: GraphBlueprint,
    *,
    graph_name: str,
    options: DifyCompileOptions | None = None,
) -> DifyRootGraph:
    opts = options or DifyCompileOptions()
    if not blueprint.nodes:
        raise CompatibilityImportError("Blueprint contains no nodes.")

    dig_meta = (blueprint.metadata or {}).get("dify") or {}
    skip_ids = frozenset(dig_meta.get("container_child_node_ids") or [])
    active_nodes = [n for n in blueprint.nodes if n.id not in skip_ids]
    node_ids = [n.id for n in active_nodes]
    id_set = set(node_ids)
    id_to_graph_name = uniquify_dify_node_names(active_nodes)

    compatibility: dict[str, Any] = {
        "source": "dify_app",
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

    graph = DifyRootGraph(name=graph_name, attributes={"compatibility": compatibility})
    init_dify_state(blueprint.metadata, graph._attributes_store).attach(graph._attributes_store)
    store = graph._attributes_store

    if_else_edges: dict[str, list[ExternalEdge]] = defaultdict(list)
    for edge in blueprint.edges:
        if edge.source in id_set:
            if_else_edges[edge.source].append(edge)

    iteration_graph = (
        dig_meta.get("iteration_graph") if isinstance(dig_meta.get("iteration_graph"), dict) else None
    )
    loop_subgraphs = dig_meta.get("loop_subgraphs") if isinstance(dig_meta.get("loop_subgraphs"), dict) else None

    instances: dict[str, Node] = {}
    for ext in blueprint.nodes:
        if ext.id in skip_ids:
            continue
        instances[ext.id] = _materialize_node(
            graph,
            ext,
            graph_name,
            store,
            opts,
            if_else_edges,
            id_to_graph_name,
            iteration_graph,
            loop_subgraphs,
        )

    resolved_edges: list[tuple[str, str]] = []
    for edge in blueprint.edges:
        if edge.source in skip_ids or edge.target in skip_ids:
            continue
        a = _resolve_edge_endpoint(edge.source, id_set)
        b = _resolve_edge_endpoint(edge.target, id_set)
        resolved_edges.append((a, b))

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
