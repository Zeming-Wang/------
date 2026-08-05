from __future__ import annotations

import ast
from string import Formatter
from typing import Any

from .key_pool import collect_document_key_pool
from .runtime_bindings import (
    merge_tool_declarations,
    validate_tool_declarations,
)
from .schema import CanvasDocument


def analyze_document(document: CanvasDocument) -> dict[str, Any]:
    key_pool = collect_document_key_pool(document)
    global_keys = set(key_pool["key_names"])
    warnings: list[str] = []
    shared_tools = list(document.manifest.tools or [])
    nodes_by_id = {node.id: node for node in document.nodes}

    warnings.extend(validate_tool_declarations(shared_tools, owner="skill manifest"))
    warnings.extend(_analyze_workflow_edges(document, nodes_by_id))

    for node in document.nodes:
        warnings.extend(_analyze_node(node.id, node.config or {}, node.type, global_keys, shared_tools))

    return {
        "key_pool": key_pool,
        "warnings": sorted(set(warnings)),
    }


def _analyze_workflow_edges(document: CanvasDocument, nodes_by_id: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    entry_keys = {str(key).strip() for key in (document.inputs or {}).keys() if str(key).strip()}

    for edge in document.edges:
        source = nodes_by_id.get(edge.source)
        target = nodes_by_id.get(edge.target)
        if source is None or target is None:
            continue

        edge_keys = [str(key).strip() for key in (edge.mapping or {}).keys() if str(key).strip()]
        if source.type == "start":
            missing = sorted(key for key in edge_keys if key not in entry_keys)
            if missing:
                warnings.append(
                    f"edge '{edge.id}' maps entry keys {missing}, but document.inputs only provides {sorted(entry_keys)}"
                )

    return warnings


def _analyze_node(
    owner: str,
    config: dict[str, Any],
    node_type: str,
    global_keys: set[str],
    shared_tools: list[dict[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    if node_type == "agent":
        warnings.extend(_analyze_prompt_scope(owner, "prompt_template", config, global_keys))
        warnings.extend(
            validate_tool_declarations(
                merge_tool_declarations(shared_tools, list(config.get("tools") or [])),
                owner=owner,
            )
        )

    if node_type == "loop":
        warnings.extend(_analyze_loop_config(owner, config))
        subgraph = dict(config.get("subgraph") or {})
        for inner_node in subgraph.get("nodes") or []:
            inner_type = str(inner_node.get("type") or "")
            warnings.extend(
                _analyze_node(
                    f"{owner}.{inner_node.get('id')}",
                    dict(inner_node.get("config") or {}),
                    inner_type,
                    global_keys,
                    shared_tools,
                )
            )

    return warnings


def _analyze_loop_config(owner: str, config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    subgraph = dict(config.get("subgraph") or {})
    inner_nodes = [dict(item) for item in (subgraph.get("nodes") or [])]
    inner_edges = [dict(item) for item in (subgraph.get("edges") or [])]
    controller_inputs = [dict(item) for item in (config.get("controller_inputs") or [])]
    controller_outputs = [dict(item) for item in (config.get("controller_outputs") or [])]
    controller = dict(config.get("controller") or {})
    controller_mode = str(controller.get("termination_mode") or "key_rule").strip()

    if not controller_inputs:
        warnings.append(f"{owner}: loop has no controller inputs, so outer workflow data never enters the loop controller graph")
    if not controller_outputs:
        warnings.append(f"{owner}: loop has no controller outputs, so the loop cannot return any inner result to the outer workflow")

    if controller_mode == "prompt":
        if not str(controller.get("terminate_condition_prompt") or "").strip():
            warnings.append(f"{owner}: controller termination mode is prompt, but terminate_condition_prompt is empty")
    if controller_mode == "expression":
        expression = str(controller.get("terminate_expression") or "").strip()
        if not expression:
            warnings.append(f"{owner}: controller termination mode is expression, but terminate_expression is empty")
        else:
            try:
                ast.parse(expression, mode="eval")
            except SyntaxError as exc:
                warnings.append(f"{owner}: controller terminate_expression has invalid syntax: {exc.msg}")

    terminate = dict(config.get("terminate_when") or {})
    terminate_mode = str(terminate.get("mode") or "never").strip()
    terminate_key = str(terminate.get("key") or "").strip()
    produced_keys = {
        str(key).strip()
        for mapping in controller_outputs
        for key in (mapping.get("mapping") or {}).keys()
        if str(key).strip()
    }
    if terminate_mode != "never" and terminate_key and terminate_key not in produced_keys:
        warnings.append(
            f"{owner}: terminate key '{terminate_key}' is not present in any controller output mapping, so loop termination may never trigger from inner results"
        )

    node_ids = {str(node.get("id") or "") for node in inner_nodes}
    outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in inner_edges:
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in outgoing and target in node_ids:
            outgoing[source].add(target)

    reachable = set()
    stack = [str(item.get("target") or "") for item in controller_inputs if str(item.get("target") or "") in node_ids]
    while stack:
        current = stack.pop()
        if current in reachable:
            continue
        reachable.add(current)
        stack.extend(outgoing.get(current, set()))

    for item in controller_outputs:
        source = str(item.get("source") or "")
        if source and source not in reachable:
            warnings.append(
                f"{owner}: controller output source '{source}' is not reachable from any controller input target inside the loop subgraph"
            )

    # Inner keys may appear in the global UI pool, but only controller outputs are exported to the outer workflow.
    inner_keys = {
        str(key).strip()
        for node in inner_nodes
        for key in (
            dict(node.get("config") or {}).get("push_keys") or {}
        ).keys()
        if str(key).strip()
    }
    internal_only = sorted(inner_keys - produced_keys)
    if internal_only:
        warnings.append(
            f"{owner}: inner keys {internal_only} exist only inside the loop unless they are mapped through controller outputs"
        )

    return warnings


def _analyze_prompt_scope(owner: str, field_label: str, config: dict[str, Any], global_keys: set[str]) -> list[str]:
    warnings: list[str] = []
    prompt_template = str(config.get("prompt_template") or "")
    if not prompt_template.strip():
        return warnings

    placeholders = _extract_placeholders(prompt_template)
    if not placeholders:
        return warnings

    pull_keys = set(_normalize_mapping_keys(config.get("pull_keys")))
    for key in placeholders:
        if key in pull_keys:
            continue
        if key in global_keys:
            warnings.append(
                f"{owner}: placeholder '{{{key}}}' appears in {field_label} but is not declared in pull_keys; it only exists in the broader workflow key pool"
            )
        else:
            warnings.append(
                f"{owner}: placeholder '{{{key}}}' appears in {field_label} but cannot be found in pull_keys or the workflow key pool"
            )
    return warnings


def _extract_placeholders(template: str) -> list[str]:
    fields: list[str] = []
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name:
            fields.append(str(field_name))
    return fields


def _normalize_mapping_keys(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return []
    return [str(key).strip() for key in raw if str(key).strip()]
