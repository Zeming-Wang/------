from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from string import Formatter
from typing import Any

from .runtime_bindings import (
    RuntimeWarnings,
    merge_tool_declarations,
    resolve_agent_tools,
)
from .schema import CanvasDocument, CanvasEdge, CanvasNode, Position


@dataclass(slots=True)
class CompileWarnings:
    items: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        if message and message not in self.items:
            self.items.append(message)


def build_model(*, api_key: str, model_name: str, base_url: str | None = None):
    from masfactory import OpenAIModel

    return OpenAIModel(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )


def document_requires_model(document: CanvasDocument) -> bool:
    return any(_node_requires_model(node) for node in document.nodes)


def _node_requires_model(node: CanvasNode) -> bool:
    if node.type == "agent":
        return True
    if node.type != "loop":
        return False

    config = dict(node.config)
    controller_config = dict(config.get("controller") or {})
    termination_mode = str(controller_config.get("termination_mode") or "key_rule").strip()
    terminate_prompt = str(controller_config.get("terminate_condition_prompt") or "").strip()
    if termination_mode == "prompt" and terminate_prompt:
        return True

    subgraph = dict(config.get("subgraph") or {})
    inner_nodes = [_dict_to_canvas_node(item) for item in (subgraph.get("nodes") or [])]
    return any(_node_requires_model(inner_node) for inner_node in inner_nodes)


def compile_document_to_graph(
    document: CanvasDocument,
    *,
    api_key: str,
    model_name: str = "gpt-4o-mini",
    base_url: str | None = None,
):
    from masfactory.components.graphs.root_graph import RootGraph

    model = build_model(api_key=api_key, model_name=model_name, base_url=base_url) if document_requires_model(document) else None
    warnings = CompileWarnings()
    runtime_warnings = RuntimeWarnings(warnings.add)
    shared_agent_context = _shared_agent_context(document)
    graph = RootGraph(name=_safe_graph_name(document.name), attributes=dict(document.attributes))

    runtime_nodes: dict[str, Any] = {}
    for node in document.nodes:
        if node.type in {"start", "end"}:
            continue
        runtime_nodes[node.id] = _create_runtime_node(
            graph,
            node,
            warnings,
            model,
            api_key=api_key,
            shared_agent_context=shared_agent_context,
            runtime_warnings=runtime_warnings,
        )

    for edge in document.edges:
        source_node = _find_node(document, edge.source)
        target_node = _find_node(document, edge.target)
        if source_node.type == "start" and target_node.type in {"agent", "custom", "loop"}:
            graph.edge_from_entry(runtime_nodes[target_node.id], keys=dict(edge.mapping))
            continue
        if source_node.type in {"agent", "custom", "loop"} and target_node.type == "end":
            graph.edge_to_exit(runtime_nodes[source_node.id], keys=dict(edge.mapping))
            continue
        if source_node.type in {"agent", "custom", "loop"} and target_node.type in {"agent", "custom", "loop"}:
            graph.create_edge(runtime_nodes[source_node.id], runtime_nodes[target_node.id], keys=dict(edge.mapping))
            continue
        raise ValueError(f"unsupported edge in MVP: {source_node.type} -> {target_node.type}")

    graph.build()
    return graph, warnings


def _create_runtime_node(
    graph,
    node: CanvasNode,
    warnings: CompileWarnings,
    model,
    *,
    api_key: str,
    shared_agent_context: dict[str, Any],
    runtime_warnings: RuntimeWarnings,
):
    if node.type == "agent":
        if model is None:
            raise ValueError(f"reasoning node '{node.id}' requires runtime.apiKey and a valid model runtime")
        return _create_agent_node(
            graph,
            node,
            warnings,
            model,
            shared_agent_context=shared_agent_context,
            runtime_warnings=runtime_warnings,
        )
    if node.type == "custom":
        return _create_custom_node(graph, node, warnings)
    if node.type == "loop":
        return _create_loop_node(
            graph,
            node,
            warnings,
            model,
            api_key=api_key,
            shared_agent_context=shared_agent_context,
            runtime_warnings=runtime_warnings,
        )
    raise ValueError(f"unsupported runtime node type: {node.type}")


def _create_agent_node(
    graph,
    node: CanvasNode,
    warnings: CompileWarnings,
    model,
    *,
    shared_agent_context: dict[str, Any],
    runtime_warnings: RuntimeWarnings,
):
    from masfactory.components.agents.agent import Agent

    config = dict(node.config)
    node_tools = list(config.get("tools") or [])
    shared_tools = list(shared_agent_context.get("tools") or [])
    runtime_tools = resolve_agent_tools(
        merge_tool_declarations(shared_tools, node_tools),
        owner=f"reasoning node '{node.id}'",
        warnings=runtime_warnings,
    )

    instructions = _compose_agent_instructions(config, shared_agent_context=shared_agent_context)
    prompt_template = str(config.get("prompt_template") or "{message}")
    pull_keys = _normalize_keys(config.get("pull_keys")) or {}
    push_keys = _normalize_keys(config.get("push_keys")) or {}
    attributes = dict(config.get("attributes") or {})
    model_settings = dict(config.get("model_settings") or {})

    return graph.create_node(
        Agent,
        name=node.id,
        instructions=instructions,
        model=model,
        prompt_template=prompt_template,
        tools=runtime_tools,
        pull_keys=pull_keys,
        push_keys=push_keys,
        attributes=attributes,
        role_name=str(config.get("role_name") or node.label or node.id),
        model_settings=model_settings,
    )


def _create_custom_node(graph, node: CanvasNode, warnings: CompileWarnings):
    from masfactory.components.custom_node import CustomNode

    config = dict(node.config)
    mode = str(config.get("mode") or "passthrough").strip()
    pull_keys = _normalize_keys(config.get("pull_keys")) or {}
    push_keys = _normalize_keys(config.get("push_keys")) or {}
    attributes = dict(config.get("attributes") or {})
    forward = _build_custom_forward(node.id, mode, config, warnings)

    return graph.create_node(
        CustomNode,
        name=node.id,
        forward=forward,
        pull_keys=pull_keys,
        push_keys=push_keys,
        attributes=attributes,
    )


def _create_loop_node(
    graph,
    node: CanvasNode,
    warnings: CompileWarnings,
    model,
    *,
    api_key: str,
    shared_agent_context: dict[str, Any],
    runtime_warnings: RuntimeWarnings,
):
    from masfactory.components.graphs.loop import Loop

    config = dict(node.config)
    max_iterations = int(config.get("max_iterations") or 3)
    controller_config = dict(config.get("controller") or {})
    controller_inputs = [dict(item) for item in (config.get("controller_inputs") or [])]
    controller_outputs = [dict(item) for item in (config.get("controller_outputs") or [])]
    loop_pull_keys = _merge_edge_keys(controller_inputs, target_key="mapping")
    loop_push_keys = _merge_edge_keys(controller_outputs, target_key="mapping")
    loop_attributes = dict(config.get("attributes") or {})
    termination_mode = str(controller_config.get("termination_mode") or "key_rule").strip()
    terminate_prompt = str(controller_config.get("terminate_condition_prompt") or "").strip()
    terminate_function = _build_loop_terminate_function(config)
    controller_model = None

    if termination_mode == "prompt" and terminate_prompt:
        controller_model_settings = dict(controller_config.get("model_settings") or {})
        controller_model = build_model(
            api_key=api_key,
            model_name=str(controller_model_settings.get("model_name") or getattr(model, "_model_name", "gpt-4o-mini")),
            base_url=str(controller_model_settings.get("base_url") or "") or None,
        )

    loop = graph.create_node(
        Loop,
        name=node.id,
        max_iterations=max_iterations,
        model=controller_model,
        terminate_condition_prompt=terminate_prompt or None,
        terminate_condition_function=terminate_function,
        pull_keys=loop_pull_keys,
        push_keys=loop_push_keys,
        attributes=loop_attributes,
    )

    subgraph = dict(config.get("subgraph") or {})
    inner_nodes = [_dict_to_canvas_node(item) for item in (subgraph.get("nodes") or [])]
    inner_edges = [_dict_to_canvas_edge(item) for item in (subgraph.get("edges") or [])]
    created: dict[str, Any] = {}

    for inner_node in inner_nodes:
        created[inner_node.id] = _create_runtime_node(
            loop,
            inner_node,
            warnings,
            model,
            api_key=api_key,
            shared_agent_context=shared_agent_context,
            runtime_warnings=runtime_warnings,
        )

    for inner_edge in inner_edges:
        loop.create_edge(created[inner_edge.source], created[inner_edge.target], keys=dict(inner_edge.mapping))

    for mapping in controller_inputs:
        target = str(mapping.get("target") or "")
        if target not in created:
            raise ValueError(f"loop '{node.id}' controller input targets unknown inner node '{target}'")
        loop.edge_from_controller(created[target], keys=_normalize_keys(mapping.get("mapping")))

    for mapping in controller_outputs:
        source = str(mapping.get("source") or "")
        if source not in created:
            raise ValueError(f"loop '{node.id}' controller output references unknown inner node '{source}'")
        loop.edge_to_controller(created[source], keys=_normalize_keys(mapping.get("mapping")))

    return loop


def _compose_agent_instructions(config: dict[str, Any], *, shared_agent_context: dict[str, Any] | None = None) -> str:
    base = str(config.get("instructions") or "").strip()
    behavior_rules = [str(item).strip() for item in (config.get("behavior_rules") or []) if str(item).strip()]
    knowledge_items = []
    shared_agent_context = shared_agent_context or {}
    shared_style = str(shared_agent_context.get("style") or "").strip()
    shared_knowledge_items = _normalize_knowledge_items(shared_agent_context.get("knowledge") or [])
    shared_behavior_rules = [
        str(item).strip()
        for item in (shared_agent_context.get("behavior_rules") or [])
        if str(item).strip()
    ]
    for item in (config.get("knowledge") or []):
        if isinstance(item, dict):
            title = str(item.get("title") or "Knowledge").strip()
            text = str(item.get("text") or "").strip()
            if text:
                knowledge_items.append(f"{title}: {text}")
        elif isinstance(item, str) and item.strip():
            knowledge_items.append(item.strip())

    sections = [base] if base else []
    if shared_style:
        sections.append(f"Skill-wide style: {shared_style}")
    if shared_knowledge_items:
        sections.append("Skill-wide knowledge:\n- " + "\n- ".join(shared_knowledge_items))
    if knowledge_items:
        sections.append("Node-specific knowledge:\n- " + "\n- ".join(knowledge_items))
    if shared_behavior_rules:
        sections.append("Skill-wide behavior rules:\n- " + "\n- ".join(shared_behavior_rules))
    if behavior_rules:
        sections.append("Node-specific behavior rules:\n- " + "\n- ".join(behavior_rules))
    return "\n\n".join(section for section in sections if section).strip()


def _normalize_knowledge_items(items: list[Any]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        if isinstance(item, dict):
            title = str(item.get("title") or "Knowledge").strip()
            text = str(item.get("text") or "").strip()
            if text:
                normalized.append(f"{title}: {text}")
        elif isinstance(item, str) and item.strip():
            normalized.append(item.strip())
    return normalized


def _shared_agent_context(document: CanvasDocument) -> dict[str, Any]:
    behavior = dict(document.manifest.behavior or {})
    return {
        "style": str(behavior.get("style") or "").strip(),
        "behavior_rules": [
            str(item).strip()
            for item in (behavior.get("rules") or [])
            if str(item).strip()
        ],
        "knowledge": list(document.manifest.knowledge or []),
        "tools": list(document.manifest.tools or []),
    }


def _build_custom_forward(node_id: str, mode: str, config: dict[str, Any], warnings: CompileWarnings):
    if mode not in {"passthrough", "template", "set", "pick", "compose", "python"}:
        raise ValueError(f"logic node '{node_id}' has unsupported mode '{mode}'")

    templates = dict(config.get("templates") or {})
    static_outputs = config.get("static_outputs") or config.get("outputs") or {}
    picks = dict(config.get("pick_keys") or {})

    if mode == "template" and not templates:
        warnings.add(f"logic node '{node_id}' uses template mode with no templates configured")
    if mode == "set" and not isinstance(static_outputs, dict):
        raise ValueError(f"logic node '{node_id}' static_outputs must be a dict")
    if mode == "python" and not str(config.get("python_code") or "").strip():
        warnings.add(f"logic node '{node_id}' uses logic code mode with empty python_code")
    python_forward = _compile_python_custom_forward(node_id, config) if mode == "python" else None

    def forward(input_dict: dict[str, object], attributes: dict[str, object] | None = None) -> dict[str, object]:
        attrs = attributes or {}
        context = {**attrs, **input_dict}

        if mode == "passthrough":
            return dict(input_dict)
        if mode == "template":
            return {key: _render_template(str(value), context) for key, value in templates.items()}
        if mode == "set":
            return _resolve_payload(static_outputs, context)
        if mode == "pick":
            out: dict[str, object] = {}
            for out_key, in_key in picks.items():
                out[str(out_key)] = context.get(str(in_key))
            return out
        if mode == "compose":
            out: dict[str, object] = {}
            for out_key, in_key in picks.items():
                out[str(out_key)] = context.get(str(in_key))
            out.update({key: _render_template(str(value), context) for key, value in templates.items()})
            resolved_static = _resolve_payload(static_outputs, context)
            if isinstance(resolved_static, dict):
                out.update(resolved_static)
            return out
        if mode == "python":
            return python_forward(input_dict, attrs) if python_forward is not None else dict(input_dict)
        raise ValueError(f"unsupported logic node mode: {mode}")

    return forward


def _compile_python_custom_forward(node_id: str, config: dict[str, Any]):
    code = str(config.get("python_code") or "").strip()
    if not code:
        def _empty_forward(input_dict: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
            return dict(input_dict)

        return _empty_forward

    safe_builtins = {
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "sum": sum,
        "min": min,
        "max": max,
        "sorted": sorted,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "isinstance": isinstance,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "Exception": Exception,
        "print": print,
    }
    globals_dict = {
        "__builtins__": safe_builtins,
        "json": json,
        "math": math,
        "re": re,
    }
    locals_dict: dict[str, Any] = {}
    exec(compile(code, f"<clawcanvas-custom:{node_id}>", "exec"), globals_dict, locals_dict)
    runtime_forward = locals_dict.get("forward") or globals_dict.get("forward")
    if not callable(runtime_forward):
        raise ValueError(
            f"logic node '{node_id}' logic code must define a callable forward function"
        )
    return runtime_forward


def _build_loop_terminate_function(config: dict[str, Any]):
    controller = dict(config.get("controller") or {})
    controller_mode = str(controller.get("termination_mode") or "key_rule").strip()
    if controller_mode == "prompt":
        return None
    if controller_mode == "expression":
        expression = str(controller.get("terminate_expression") or "").strip()
        if not expression:
            return None
        return _build_safe_terminate_expression(expression)

    terminate = dict(config.get("terminate_when") or {})
    mode = str(terminate.get("mode") or "never").strip()
    key = str(terminate.get("key") or "").strip()
    value = terminate.get("value")

    def _terminate(input_dict: dict[str, object], attributes: dict[str, object]) -> bool:
        source = {**attributes, **input_dict}
        if mode == "never":
            return False
        if mode == "key_truthy":
            return bool(source.get(key)) if key else False
        if mode == "key_equals":
            return source.get(key) == value if key else False
        raise ValueError(f"unsupported loop terminate mode: {mode}")

    return _terminate


def _build_safe_terminate_expression(expression: str):
    tree = ast.parse(expression, mode="eval")

    def _evaluate(node, context):
        if isinstance(node, ast.Expression):
            return _evaluate(node.body, context)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return context.get(node.id)
        if isinstance(node, ast.BoolOp):
            values = [_evaluate(value, context) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError("unsupported boolean operator")
        if isinstance(node, ast.UnaryOp):
            operand = _evaluate(node.operand, context)
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise ValueError("unsupported unary operator")
        if isinstance(node, ast.BinOp):
            left = _evaluate(node.left, context)
            right = _evaluate(node.right, context)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                return left % right
            raise ValueError("unsupported arithmetic operator")
        if isinstance(node, ast.Compare):
            left = _evaluate(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = _evaluate(comparator, context)
                ok = None
                if isinstance(op, ast.Eq):
                    ok = left == right
                elif isinstance(op, ast.NotEq):
                    ok = left != right
                elif isinstance(op, ast.Gt):
                    ok = left > right
                elif isinstance(op, ast.GtE):
                    ok = left >= right
                elif isinstance(op, ast.Lt):
                    ok = left < right
                elif isinstance(op, ast.LtE):
                    ok = left <= right
                elif isinstance(op, ast.In):
                    ok = left in right
                elif isinstance(op, ast.NotIn):
                    ok = left not in right
                else:
                    raise ValueError("unsupported comparison operator")
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.Subscript):
            value = _evaluate(node.value, context)
            key = _evaluate(node.slice, context)
            return value[key]
        if isinstance(node, ast.List):
            return [_evaluate(item, context) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(_evaluate(item, context) for item in node.elts)
        raise ValueError(f"unsupported expression node: {node.__class__.__name__}")

    def _terminate(input_dict: dict[str, object], attributes: dict[str, object]) -> bool:
        context = {
            **attributes,
            **input_dict,
            "input": input_dict,
            "attributes": attributes,
        }
        return bool(_evaluate(tree, context))

    return _terminate


def _normalize_keys(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("pull_keys/push_keys must be dict values")
    return {str(key): str(value) for key, value in raw.items()}


def _merge_edge_keys(items: list[dict[str, Any]], *, target_key: str = "mapping") -> dict[str, str]:
    merged: dict[str, str] = {}
    for item in items:
        merged.update(_normalize_keys(item.get(target_key)))
    return merged


def _dict_to_canvas_node(raw: dict[str, Any]) -> CanvasNode:
    position = dict(raw.get("position") or {})
    return CanvasNode(
        id=str(raw.get("id")),
        type=str(raw.get("type")),
        label=str(raw.get("label") or raw.get("id")),
        position=Position(x=float(position.get("x", 0)), y=float(position.get("y", 0))),
        config=dict(raw.get("config") or {}),
    )


def _dict_to_canvas_edge(raw: dict[str, Any]) -> CanvasEdge:
    return CanvasEdge(
        id=str(raw.get("id")),
        source=str(raw.get("source")),
        target=str(raw.get("target")),
        mapping=_normalize_keys(raw.get("mapping")),
    )


def _render_template(template: str, context: dict[str, object]) -> str:
    formatter = Formatter()
    parts: list[str] = []
    for literal_text, field_name, format_spec, conversion in formatter.parse(template):
        parts.append(literal_text)
        if field_name is None:
            continue
        value = context.get(field_name, "{" + field_name + "}")
        if conversion == "r":
            value = repr(value)
        elif conversion == "s":
            value = str(value)
        if format_spec:
            try:
                value = format(value, format_spec)
            except Exception:
                value = str(value)
        parts.append(str(value))
    return "".join(parts)


def _resolve_payload(value: Any, context: dict[str, object]) -> Any:
    if isinstance(value, str):
        return _render_template(value, context)
    if isinstance(value, list):
        return [_resolve_payload(item, context) for item in value]
    if isinstance(value, dict):
        return {str(key): _resolve_payload(item, context) for key, item in value.items()}
    return value


def _safe_graph_name(name: str) -> str:
    safe = []
    for char in str(name).strip().lower():
        if char.isalnum() or char == "_":
            safe.append(char)
        elif char in {" ", "-", "/"}:
            safe.append("_")
    collapsed = "".join(safe).strip("_")
    return collapsed or "clawcanvas_graph"


def _find_node(document: CanvasDocument, node_id: str) -> CanvasNode:
    for node in document.nodes:
        if node.id == node_id:
            return node
    raise ValueError(f"unknown node id: {node_id}")
