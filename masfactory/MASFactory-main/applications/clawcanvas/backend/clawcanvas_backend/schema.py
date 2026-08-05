from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_NODE_TYPES = {"start", "agent", "custom", "loop", "end"}
INNER_LOOP_NODE_TYPES = {"agent", "custom", "loop"}


@dataclass(slots=True)
class Position:
    x: float
    y: float


@dataclass(slots=True)
class CanvasNode:
    id: str
    type: str
    label: str
    position: Position
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CanvasEdge:
    id: str
    source: str
    target: str
    mapping: dict[str, str] = field(default_factory=lambda: {"message": "message"})


@dataclass(slots=True)
class SkillManifest:
    name: str
    version: str = "0.1.0"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    knowledge: list[dict[str, Any]] = field(default_factory=list)
    behavior: dict[str, Any] = field(default_factory=dict)
    author: str = ""
    license: str = "MIT"
    dependencies: list[str] = field(default_factory=list)
    homepage: str = ""
    repository: str = ""


@dataclass(slots=True)
class CanvasDocument:
    id: str
    name: str
    description: str
    nodes: list[CanvasNode]
    edges: list[CanvasEdge]
    inputs: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    key_descriptions: dict[str, str] = field(default_factory=dict)
    manifest: SkillManifest = field(default_factory=lambda: SkillManifest(name="unnamed_skill"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_str(raw: Any, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return raw.strip()


def _coerce_mapping(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("mapping-like value must be a dict")
    out: dict[str, str] = {}
    for key, value in raw.items():
        out[str(key)] = str(value)
    return out


def _parse_node(item: dict[str, Any], *, prefix: str = "node") -> CanvasNode:
    pos = item.get("position") or {}
    if not isinstance(pos, dict):
        raise ValueError(f"{prefix}.position must be a dict")
    return CanvasNode(
        id=_require_str(item.get("id"), f"{prefix}.id"),
        type=_require_str(item.get("type"), f"{prefix}.type"),
        label=str(item.get("label") or item.get("id")).strip(),
        position=Position(
            x=float(pos.get("x", 0)),
            y=float(pos.get("y", 0)),
        ),
        config=_normalize_loop_config_dict(dict(item.get("config") or {})),
    )


def _parse_edge(item: dict[str, Any], *, prefix: str = "edge") -> CanvasEdge:
    mapping = _coerce_mapping(item.get("mapping"))
    if item.get("mapping") is None:
        mapping = {"message": "message"}
    return CanvasEdge(
        id=_require_str(item.get("id"), f"{prefix}.id"),
        source=_require_str(item.get("source"), f"{prefix}.source"),
        target=_require_str(item.get("target"), f"{prefix}.target"),
        mapping=mapping,
    )


def _normalize_loop_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    if "subgraph" in config:
        subgraph = dict(config.get("subgraph") or {})
        return {
            **config,
            "max_iterations": int(config.get("max_iterations") or 3),
            "terminate_when": dict(config.get("terminate_when") or {"mode": "never", "key": "", "value": True}),
            "controller": {
                "termination_mode": str(dict(config.get("controller") or {}).get("termination_mode") or "key_rule"),
                "terminate_condition_prompt": str(
                    dict(config.get("controller") or {}).get("terminate_condition_prompt") or ""
                ),
                "terminate_expression": str(dict(config.get("controller") or {}).get("terminate_expression") or ""),
                "model_settings": dict(dict(config.get("controller") or {}).get("model_settings") or {}),
            },
            "subgraph": {
                "nodes": [dict(item) for item in (subgraph.get("nodes") or [])],
                "edges": [dict(item) for item in (subgraph.get("edges") or [])],
            },
            "controller_inputs": [dict(item) for item in (config.get("controller_inputs") or [])],
            "controller_outputs": [dict(item) for item in (config.get("controller_outputs") or [])],
        }

    body = dict(config.get("body") or {})
    if body:
        body_type = str(body.get("type") or "agent").strip()
        body_node_id = "loop_step_1"
        return {
            "max_iterations": int(config.get("max_iterations") or 3),
            "terminate_when": dict(config.get("terminate_when") or {"mode": "never", "key": "", "value": True}),
            "controller": {
                "termination_mode": "key_rule",
                "terminate_condition_prompt": "",
                "terminate_expression": "",
                "model_settings": {},
            },
            "subgraph": {
                "nodes": [
                    {
                        "id": body_node_id,
                        "type": body_type,
                        "label": "Loop Step",
                        "position": {"x": 260, "y": 180},
                        "config": {
                            **body,
                            "pull_keys": dict(body.get("pull_keys") or body.get("input_mapping") or {}),
                            "push_keys": dict(body.get("push_keys") or body.get("output_mapping") or {}),
                        },
                    }
                ],
                "edges": [],
            },
            "controller_inputs": [
                {
                    "id": "controller_in_1",
                    "target": body_node_id,
                    "mapping": dict(body.get("input_mapping") or body.get("pull_keys") or {"message": "Loop input"}),
                }
            ],
            "controller_outputs": [
                {
                    "id": "controller_out_1",
                    "source": body_node_id,
                    "mapping": dict(body.get("output_mapping") or body.get("push_keys") or {"message": "Loop output"}),
                }
            ],
        }
    return config


def parse_document(payload: dict[str, Any]) -> CanvasDocument:
    document, errors = parse_document_with_errors(payload)
    errors.extend(validate_document_errors(document))
    errors = _dedupe_errors(errors)
    if errors:
        raise ValueError(_format_validation_errors(errors))
    return document


def parse_document_with_errors(payload: dict[str, Any]) -> tuple[CanvasDocument, list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("document payload must be a dict")

    errors: list[str] = []
    nodes_raw = payload.get("nodes")
    edges_raw = payload.get("edges")
    if not isinstance(nodes_raw, list):
        errors.append("document payload must contain list field: nodes")
        nodes_raw = []
    if not isinstance(edges_raw, list):
        errors.append("document payload must contain list field: edges")
        edges_raw = []

    nodes: list[CanvasNode] = []
    for index, item in enumerate(nodes_raw):
        parsed_node = _parse_node_item(item, prefix=f"nodes[{index}]", errors=errors)
        if parsed_node is not None:
            nodes.append(parsed_node)

    edges: list[CanvasEdge] = []
    for index, item in enumerate(edges_raw):
        parsed_edge = _parse_edge_item(item, prefix=f"edges[{index}]", errors=errors)
        if parsed_edge is not None:
            edges.append(parsed_edge)

    manifest_raw = payload.get("manifest") or {}
    if not isinstance(manifest_raw, dict):
        errors.append("manifest must be a dict")
        manifest_raw = {}

    manifest = SkillManifest(
        name=_coerce_required_str(
            manifest_raw.get("name") or payload.get("name") or "clawcanvas_skill",
            "manifest.name",
            errors,
            fallback="clawcanvas_skill",
        ),
        version=str(manifest_raw.get("version") or "0.1.0"),
        description=str(manifest_raw.get("description") or payload.get("description") or ""),
        tags=_coerce_str_list(manifest_raw.get("tags"), "manifest.tags", errors),
        tools=_coerce_dict_list(manifest_raw.get("tools"), "manifest.tools", errors),
        knowledge=_coerce_dict_list(manifest_raw.get("knowledge"), "manifest.knowledge", errors),
        behavior=_coerce_dict(manifest_raw.get("behavior"), "manifest.behavior", errors),
        author=str(manifest_raw.get("author") or ""),
        license=str(manifest_raw.get("license") or "MIT"),
        dependencies=_coerce_str_list(manifest_raw.get("dependencies"), "manifest.dependencies", errors),
        homepage=str(manifest_raw.get("homepage") or ""),
        repository=str(manifest_raw.get("repository") or ""),
    )

    document = CanvasDocument(
        id=_coerce_required_str(
            payload.get("id") or "clawcanvas_doc",
            "document.id",
            errors,
            fallback="clawcanvas_doc",
        ),
        name=_coerce_required_str(
            payload.get("name") or "ClawCanvas Workflow",
            "document.name",
            errors,
            fallback="ClawCanvas Workflow",
        ),
        description=str(payload.get("description") or ""),
        nodes=nodes,
        edges=edges,
        inputs=_coerce_dict(payload.get("inputs"), "document.inputs", errors),
        attributes=_coerce_dict(payload.get("attributes"), "document.attributes", errors),
        key_descriptions=_coerce_mapping_collect(
            payload.get("key_descriptions"),
            "document.key_descriptions",
            errors,
        ),
        manifest=manifest,
    )
    return document, _dedupe_errors(errors)


def _parse_node_item(raw: Any, *, prefix: str, errors: list[str]) -> CanvasNode | None:
    if not isinstance(raw, dict):
        errors.append(f"{prefix} must be a dict")
        return None
    try:
        return _parse_node(raw, prefix=prefix)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return None


def _parse_edge_item(raw: Any, *, prefix: str, errors: list[str]) -> CanvasEdge | None:
    if not isinstance(raw, dict):
        errors.append(f"{prefix} must be a dict")
        return None
    try:
        return _parse_edge(raw, prefix=prefix)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return None


def _coerce_required_str(raw: Any, field_name: str, errors: list[str], *, fallback: str) -> str:
    try:
        return _require_str(raw, field_name)
    except ValueError as exc:
        errors.append(str(exc))
        return fallback


def _coerce_dict(raw: Any, field_name: str, errors: list[str]) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        errors.append(f"{field_name} must be a dict")
        return {}
    return dict(raw)


def _coerce_mapping_collect(raw: Any, field_name: str, errors: list[str]) -> dict[str, str]:
    try:
        return _coerce_mapping(raw)
    except ValueError as exc:
        errors.append(f"{field_name}: {exc}")
        return {}


def _coerce_dict_list(raw: Any, field_name: str, errors: list[str]) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"{field_name} must be a list")
        return []
    items: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, dict):
            items.append(dict(item))
        else:
            errors.append(f"{field_name}[{index}] must be a dict")
    return items


def _coerce_str_list(raw: Any, field_name: str, errors: list[str]) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"{field_name} must be a list")
        return []
    return [str(item) for item in raw]


def validate_document(document: CanvasDocument) -> None:
    errors = validate_document_errors(document)
    if errors:
        raise ValueError(_format_validation_errors(errors))


def validate_document_errors(document: CanvasDocument) -> list[str]:
    return _dedupe_errors(
        _validate_graph(
            document.nodes,
            document.edges,
            require_start_end=True,
            loop_owner="workflow",
        )
    )


def _validate_graph(
    nodes: list[CanvasNode],
    edges: list[CanvasEdge],
    *,
    require_start_end: bool,
    loop_owner: str,
) -> list[str]:
    errors: list[str] = []
    node_ids = [node.id for node in nodes]
    for node_id in _duplicates(node_ids):
        errors.append(f"{loop_owner}: duplicate node id '{node_id}'")

    edge_ids = [edge.id for edge in edges]
    for edge_id in _duplicates(edge_ids):
        errors.append(f"{loop_owner}: duplicate edge id '{edge_id}'")

    nodes_by_id: dict[str, CanvasNode] = {}
    for node in nodes:
        nodes_by_id.setdefault(node.id, node)

    if require_start_end:
        start_nodes = [node for node in nodes if node.type == "start"]
        end_nodes = [node for node in nodes if node.type == "end"]
        if len(start_nodes) != 1:
            errors.append("workflow must contain exactly one start node")
        if len(end_nodes) != 1:
            errors.append("workflow must contain exactly one end node")
    else:
        start_nodes = []
        end_nodes = []

    for node in nodes:
        if require_start_end:
            if node.type not in SUPPORTED_NODE_TYPES:
                errors.append(f"unsupported node type: {node.type}")
                continue
        else:
            if node.type not in INNER_LOOP_NODE_TYPES:
                errors.append(f"{loop_owner}: loop subgraph only supports reasoning/logic/loop nodes")
                continue

        errors.extend(_validate_node_config(node))

        if node.type == "loop":
            errors.extend(_validate_loop_config(node.id, node.config or {}))

    incoming: dict[str, list[str]] = {node.id: [] for node in nodes}
    outgoing: dict[str, list[str]] = {node.id: [] for node in nodes}
    for edge in edges:
        errors.extend(_validate_mapping(edge.mapping, f"edge '{edge.id}' mapping"))
        source = nodes_by_id.get(edge.source)
        target = nodes_by_id.get(edge.target)
        if source is None or target is None:
            missing = [node_id for node_id in (edge.source, edge.target) if node_id not in nodes_by_id]
            errors.append(f"edge '{edge.id}' references an unknown node: {', '.join(missing)}")
            continue
        if edge.source == edge.target:
            errors.append(f"edge '{edge.id}' must not point to the same node")
            continue
        if require_start_end:
            if target.type == "start":
                errors.append(f"edge '{edge.id}' must not target the start node")
            if source.type == "end":
                errors.append(f"edge '{edge.id}' must not leave the end node")
        incoming[edge.target].append(edge.source)
        outgoing[edge.source].append(edge.target)

    if require_start_end and len(start_nodes) == 1 and len(end_nodes) == 1:
        start = start_nodes[0]
        end = end_nodes[0]
        if incoming[start.id]:
            errors.append("start node must not have incoming edges")
        if outgoing[end.id]:
            errors.append("end node must not have outgoing edges")
        errors.extend(_reachable_errors(start.id, end.id, outgoing))

    errors.extend(_acyclic_errors(nodes_by_id, outgoing, loop_owner=loop_owner))
    return errors


def _validate_node_config(node: CanvasNode) -> list[str]:
    errors: list[str] = []
    config = node.config or {}
    node_label = _node_type_label(node.type)
    for field_name in ("pull_keys", "push_keys", "templates", "static_outputs", "pick_keys"):
        if field_name not in config:
            continue
        raw_mapping = config.get(field_name)
        if not isinstance(raw_mapping, dict):
            errors.append(f"{node_label} node '{node.id}' config.{field_name} must be a dict")
            continue
        errors.extend(
            _validate_mapping(
                raw_mapping,
                f"{node_label} node '{node.id}' config.{field_name}",
                allow_empty=True,
            )
        )

    if node.type == "agent" and not str(config.get("instructions") or "").strip():
        errors.append(f"reasoning node '{node.id}' must define instructions")

    if node.type == "custom":
        mode = str(config.get("mode") or "passthrough").strip()
        if mode not in {"passthrough", "template", "set", "pick", "compose", "python"}:
            errors.append(f"logic node '{node.id}' has unsupported mode '{mode}'")
        if mode == "python" and not str(config.get("python_code") or "").strip():
            errors.append(f"logic node '{node.id}' uses logic code mode but python_code is empty")
        if mode in {"template", "compose"} and not isinstance(config.get("templates") or {}, dict):
            errors.append(f"logic node '{node.id}' templates must be a dict")
        if mode in {"set", "compose"} and not isinstance(
            config.get("static_outputs") or config.get("outputs") or {},
            dict,
        ):
            errors.append(f"logic node '{node.id}' static_outputs must be a dict")
        if mode in {"pick", "compose"} and not isinstance(config.get("pick_keys") or {}, dict):
            errors.append(f"logic node '{node.id}' pick_keys must be a dict")

    return errors


def _node_type_label(node_type: str) -> str:
    if node_type == "agent":
        return "reasoning"
    if node_type == "custom":
        return "logic"
    return node_type


def _validate_mapping(mapping: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(mapping, dict):
        return [f"{label} must be a dict"]
    if not mapping and not allow_empty:
        return [f"{label} must not be empty"]
    for key in mapping:
        if not str(key).strip():
            errors.append(f"{label} contains an empty field name")
    return errors


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _format_validation_errors(errors: list[str]) -> str:
    if len(errors) == 1:
        return errors[0]
    return "document validation failed:\n- " + "\n- ".join(errors)


def _dedupe_errors(errors: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for error in errors:
        if error and error not in seen:
            seen.add(error)
            result.append(error)
    return result


def _validate_loop_config(node_id: str, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        config = _normalize_loop_config_dict(dict(config or {}))
    except (TypeError, ValueError) as exc:
        return [f"loop node '{node_id}' config is invalid: {exc}"]
    subgraph = dict(config.get("subgraph") or {})
    inner_nodes_raw = subgraph.get("nodes")
    inner_edges_raw = subgraph.get("edges")
    if not isinstance(inner_nodes_raw, list) or not isinstance(inner_edges_raw, list):
        return [f"loop node '{node_id}' must define subgraph.nodes and subgraph.edges"]

    inner_nodes: list[CanvasNode] = []
    for index, item in enumerate(inner_nodes_raw):
        parsed_node = _parse_node_item(item, prefix=f"loop.{node_id}.nodes[{index}]", errors=errors)
        if parsed_node is not None:
            inner_nodes.append(parsed_node)

    inner_edges: list[CanvasEdge] = []
    for index, item in enumerate(inner_edges_raw):
        parsed_edge = _parse_edge_item(item, prefix=f"loop.{node_id}.edges[{index}]", errors=errors)
        if parsed_edge is not None:
            inner_edges.append(parsed_edge)

    if not inner_nodes:
        errors.append(f"loop node '{node_id}' subgraph must contain at least one node")

    errors.extend(
        _validate_graph(
            inner_nodes,
            inner_edges,
            require_start_end=False,
            loop_owner=f"loop '{node_id}'",
        )
    )

    controller_inputs = _coerce_dict_list(
        config.get("controller_inputs"),
        f"loop node '{node_id}' controller_inputs",
        errors,
    )
    controller_outputs = _coerce_dict_list(
        config.get("controller_outputs"),
        f"loop node '{node_id}' controller_outputs",
        errors,
    )
    if not controller_inputs:
        errors.append(f"loop node '{node_id}' must define at least one controller input")
    if not controller_outputs:
        errors.append(f"loop node '{node_id}' must define at least one controller output")

    node_ids = {node.id for node in inner_nodes}
    controller_ids: set[str] = set()
    for item in controller_inputs:
        edge_id = _coerce_required_str(
            item.get("id"),
            f"loop.{node_id}.controller_input.id",
            errors,
            fallback="<missing-controller-input-id>",
        )
        if edge_id in controller_ids:
            errors.append(f"loop node '{node_id}' controller mapping ids must be unique")
        controller_ids.add(edge_id)
        target = _coerce_required_str(
            item.get("target"),
            f"loop.{node_id}.controller_input.target",
            errors,
            fallback="",
        )
        if target not in node_ids:
            errors.append(f"loop node '{node_id}' controller input '{edge_id}' targets unknown node '{target}'")
        mapping_label = f"loop node '{node_id}' controller input '{edge_id}' mapping"
        errors.extend(
            _validate_mapping(
                _coerce_mapping_collect(item.get("mapping"), mapping_label, errors),
                mapping_label,
            )
        )

    for item in controller_outputs:
        edge_id = _coerce_required_str(
            item.get("id"),
            f"loop.{node_id}.controller_output.id",
            errors,
            fallback="<missing-controller-output-id>",
        )
        if edge_id in controller_ids:
            errors.append(f"loop node '{node_id}' controller mapping ids must be unique")
        controller_ids.add(edge_id)
        source = _coerce_required_str(
            item.get("source"),
            f"loop.{node_id}.controller_output.source",
            errors,
            fallback="",
        )
        if source not in node_ids:
            errors.append(f"loop node '{node_id}' controller output '{edge_id}' references unknown node '{source}'")
        mapping_label = f"loop node '{node_id}' controller output '{edge_id}' mapping"
        errors.extend(
            _validate_mapping(
                _coerce_mapping_collect(item.get("mapping"), mapping_label, errors),
                mapping_label,
            )
        )

    return errors


def _reachable_errors(start_id: str, end_id: str, outgoing: dict[str, list[str]]) -> list[str]:
    seen = set()
    stack = [start_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(outgoing.get(current, []))
    if end_id not in seen:
        return ["workflow end node is not reachable from start"]
    return []


def _acyclic_errors(
    nodes_by_id: dict[str, CanvasNode],
    outgoing: dict[str, list[str]],
    *,
    loop_owner: str,
) -> list[str]:
    errors: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            errors.append(
                f"{loop_owner} must remain acyclic; loops should cycle only through the MASFactory controller"
            )
            return
        visiting.add(node_id)
        for nxt in outgoing.get(node_id, []):
            visit(nxt)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes_by_id:
        visit(node_id)
    return errors


def build_demo_document() -> CanvasDocument:
    payload = {
        "id": "demo_clawcanvas",
        "name": "ClawCanvas Demo Skill",
        "description": "A skill workflow combining reasoning and loop nodes compiled by MASFactory.",
        "inputs": {
            "query": "Give me a launch plan for a MASFactory-based skill studio."
        },
        "manifest": {
            "name": "clawcanvas_demo_skill",
            "version": "0.1.0",
            "description": "Demo Skill package.",
            "tags": ["demo", "workflow", "skill"],
            "tools": [
                {"name": "web_search", "binding": "future", "description": "Reserved tool binding"}
            ],
            "knowledge": [
                {"title": "Skill Definition", "text": "Skill = tools + domain knowledge + behavior rules."}
            ],
            "behavior": {
                "style": "structured",
                "rules": [
                    "Prefer concrete deliverables.",
                    "Explain tradeoffs briefly."
                ]
            },
        },
        "key_descriptions": {
            "query": "Original user request",
            "analysis": "Structured analysis of the request",
            "draft": "Draft design brief",
            "answer": "Final design brief",
            "done": "Loop stop flag",
        },
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "label": "Start",
                "position": {"x": 80, "y": 220},
                "config": {},
            },
            {
                "id": "researcher",
                "type": "agent",
                "label": "Researcher",
                "position": {"x": 320, "y": 140},
                "config": {
                    "instructions": "You analyze the user's request and extract the essential workflow goals.",
                    "prompt_template": "User request: {query}",
                    "pull_keys": {"query": "Original user request"},
                    "push_keys": {"analysis": "Structured analysis of the request"},
                    "behavior_rules": [
                        "Keep the analysis concise.",
                        "Focus on workflow scope and assumptions.",
                    ],
                },
            },
            {
                "id": "review_loop",
                "type": "loop",
                "label": "Review Loop",
                "position": {"x": 640, "y": 220},
                "config": {
                    "max_iterations": 3,
                    "terminate_when": {"mode": "key_truthy", "key": "done", "value": True},
                    "controller": {
                        "termination_mode": "key_rule",
                        "terminate_condition_prompt": "",
                        "terminate_expression": "",
                        "model_settings": {},
                    },
                    "subgraph": {
                        "nodes": [
                            {
                                "id": "draft_writer",
                                "type": "agent",
                                "label": "Draft Writer",
                                "position": {"x": 240, "y": 120},
                                "config": {
                                    "instructions": "Turn the analysis into a draft launch brief.",
                                    "prompt_template": "Analysis: {analysis}",
                                    "pull_keys": {"analysis": "Structured analysis"},
                                    "push_keys": {"draft": "Draft launch brief"},
                                    "behavior_rules": ["Be concise."],
                                },
                            },
                            {
                                "id": "draft_finisher",
                                "type": "custom",
                                "label": "Draft Finisher",
                                "position": {"x": 540, "y": 220},
                                "config": {
                                    "mode": "template",
                                    "templates": {
                                        "answer": "Final brief: {draft}",
                                        "done": "true",
                                    },
                                    "pull_keys": {"draft": "Draft brief"},
                                    "push_keys": {"answer": "Final brief", "done": "Stop flag"},
                                },
                            },
                        ],
                        "edges": [
                            {
                                "id": "inner_edge_1",
                                "source": "draft_writer",
                                "target": "draft_finisher",
                                "mapping": {"draft": "Draft brief"},
                            }
                        ],
                    },
                    "controller_inputs": [
                        {
                            "id": "controller_in_1",
                            "target": "draft_writer",
                            "mapping": {"analysis": "Structured analysis"},
                        }
                    ],
                    "controller_outputs": [
                        {
                            "id": "controller_out_1",
                            "source": "draft_finisher",
                            "mapping": {"answer": "Final brief", "done": "Stop flag"},
                        }
                    ],
                },
            },
            {
                "id": "end",
                "type": "end",
                "label": "End",
                "position": {"x": 980, "y": 220},
                "config": {},
            },
        ],
        "edges": [
            {"id": "edge_1", "source": "start", "target": "researcher", "mapping": {"query": "User request"}},
            {"id": "edge_2", "source": "researcher", "target": "review_loop", "mapping": {"analysis": "Analysis"}},
            {"id": "edge_3", "source": "review_loop", "target": "end", "mapping": {"answer": "Final answer"}},
        ],
    }
    return parse_document(payload)
