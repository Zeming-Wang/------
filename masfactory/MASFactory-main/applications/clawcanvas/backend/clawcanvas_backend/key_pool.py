from __future__ import annotations

from typing import Any

from .schema import CanvasDocument


def collect_document_key_pool(document: CanvasDocument) -> dict[str, Any]:
    key_meta: dict[str, dict[str, Any]] = {}

    def ensure_key(key: Any, description: Any = "", source: str = "unknown", owner: str = "") -> None:
        key_str = str(key or "").strip()
        if not key_str:
            return
        item = key_meta.setdefault(
            key_str,
            {"key": key_str, "description": "", "sources": [], "owners": []},
        )
        desc = str(description or "").strip()
        if desc and not item["description"]:
            item["description"] = desc
        if source and source not in item["sources"]:
            item["sources"].append(source)
        if owner and owner not in item["owners"]:
            item["owners"].append(owner)

    for key, value in (document.inputs or {}).items():
        ensure_key(key, value, "document.inputs", "workflow")
    for key, value in (document.attributes or {}).items():
        ensure_key(key, value, "document.attributes", "workflow")
    for key, value in (document.key_descriptions or {}).items():
        ensure_key(key, value, "document.key_descriptions", "workflow")

    for edge in document.edges:
        for key, value in (edge.mapping or {}).items():
            ensure_key(key, value, "edge.mapping", edge.id)

    for node in document.nodes:
        _collect_node_keys(node.config or {}, ensure_key, node.id)

    keys = sorted(key_meta.values(), key=lambda item: item["key"])
    return {
        "keys": keys,
        "key_names": [item["key"] for item in keys],
        "key_map": {item["key"]: item["description"] for item in keys},
    }


def rename_document_key(document: CanvasDocument, old_key: str, new_key: str) -> CanvasDocument:
    old_key = str(old_key or "").strip()
    new_key = str(new_key or "").strip()
    if not old_key or not new_key or old_key == new_key:
        return document

    if old_key in document.inputs:
        document.inputs[new_key] = document.inputs.pop(old_key)
    if old_key in document.attributes:
        document.attributes[new_key] = document.attributes.pop(old_key)
    if old_key in document.key_descriptions:
        document.key_descriptions[new_key] = document.key_descriptions.pop(old_key)

    for edge in document.edges:
        edge.mapping = _rename_mapping_key(edge.mapping, old_key, new_key)

    for node in document.nodes:
        node.config = _rename_config_keys(node.config or {}, old_key, new_key)

    return document


def _collect_node_keys(config: dict[str, Any], ensure_key, owner: str) -> None:
    for field_name in ("pull_keys", "push_keys", "templates", "static_outputs", "pick_keys"):
        for key, value in (config.get(field_name) or {}).items():
            ensure_key(key, value, f"node.{field_name}", owner)

    terminate_when = config.get("terminate_when") or {}
    terminate_key = str(terminate_when.get("key") or "").strip()
    if terminate_key:
        ensure_key(terminate_key, "Loop terminate condition key", "node.terminate_when", owner)

    for mapping in (config.get("controller_inputs") or []):
        for key, value in (mapping.get("mapping") or {}).items():
            ensure_key(key, value, "node.controller_inputs", owner)
    for mapping in (config.get("controller_outputs") or []):
        for key, value in (mapping.get("mapping") or {}).items():
            ensure_key(key, value, "node.controller_outputs", owner)

    subgraph = dict(config.get("subgraph") or {})
    for edge in subgraph.get("edges") or []:
        for key, value in (edge.get("mapping") or {}).items():
            ensure_key(key, value, "node.subgraph.edge.mapping", f"{owner}.{edge.get('id')}")

    for node in subgraph.get("nodes") or []:
        node_id = f"{owner}.{node.get('id')}"
        _collect_node_keys(dict(node.get("config") or {}), ensure_key, node_id)


def _rename_mapping_key(mapping: dict[str, Any] | None, old_key: str, new_key: str) -> dict[str, Any]:
    mapping = dict(mapping or {})
    if old_key not in mapping:
        return mapping
    mapping[new_key] = mapping.pop(old_key)
    return mapping


def _rename_config_keys(config: dict[str, Any], old_key: str, new_key: str) -> dict[str, Any]:
    updated = dict(config)
    for field_name in ("pull_keys", "push_keys", "templates", "static_outputs", "pick_keys"):
        updated[field_name] = _rename_mapping_key(updated.get(field_name), old_key, new_key)

    terminate_when = dict(updated.get("terminate_when") or {})
    if terminate_when.get("key") == old_key:
        terminate_when["key"] = new_key
    if terminate_when:
        updated["terminate_when"] = terminate_when

    prompt_fields = ("prompt_template", "instructions")
    for field_name in prompt_fields:
        if isinstance(updated.get(field_name), str):
            updated[field_name] = updated[field_name].replace(f"{{{old_key}}}", f"{{{new_key}}}")

    updated["controller_inputs"] = [
        {**dict(item), "mapping": _rename_mapping_key(dict(item).get("mapping"), old_key, new_key)}
        for item in (updated.get("controller_inputs") or [])
    ]
    updated["controller_outputs"] = [
        {**dict(item), "mapping": _rename_mapping_key(dict(item).get("mapping"), old_key, new_key)}
        for item in (updated.get("controller_outputs") or [])
    ]

    subgraph = dict(updated.get("subgraph") or {})
    if subgraph:
        subgraph["edges"] = [
            {**dict(edge), "mapping": _rename_mapping_key(dict(edge).get("mapping"), old_key, new_key)}
            for edge in (subgraph.get("edges") or [])
        ]
        subgraph["nodes"] = [
            {**dict(node), "config": _rename_config_keys(dict(node.get("config") or {}), old_key, new_key)}
            for node in (subgraph.get("nodes") or [])
        ]
        updated["subgraph"] = subgraph

    return updated
