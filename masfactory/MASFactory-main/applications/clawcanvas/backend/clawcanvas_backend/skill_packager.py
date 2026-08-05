from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .schema import CanvasDocument


ExportFormat = Literal["json", "markdown", "zip"]
RuntimeConfig = dict[str, Any]


def build_skill_package(
    document: CanvasDocument,
    *,
    run_output: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "manifest": asdict(document.manifest),
        "workflow": document.to_dict(),
        "runtime": {
            "supported_node_types": ["start", "agent", "custom", "loop", "end"],
            "warnings": list(warnings or []),
            "engine": "MASFactory",
            "application": "ClawCanvas",
        },
        "last_run": run_output or {},
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def export_skill_package(
    document: CanvasDocument,
    *,
    export_root: str | Path,
    run_output: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    runtime: RuntimeConfig | None = None,
    format: ExportFormat = "json",
) -> dict[str, Any]:
    package = build_skill_package(document, run_output=run_output, warnings=warnings)
    export_dir = Path(export_root) / _package_dir_name(document.manifest.name or document.name)
    export_dir.mkdir(parents=True, exist_ok=True)

    if format == "markdown":
        return _export_as_markdown(document, package, export_dir, runtime=runtime)
    elif format == "zip":
        return _export_as_zip(document, package, export_dir, runtime=runtime)
    else:
        return _export_as_json(package, export_dir, runtime=runtime)


def export_standard_skill(
    document: CanvasDocument,
    *,
    export_root: str | Path,
    run_output: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Any]:
    package = build_skill_package(document, run_output=run_output, warnings=warnings)
    skill_slug = _official_skill_slug(document.manifest.name or document.name)
    export_dir = Path(export_root) / _package_dir_name(skill_slug)
    skill_dir = export_dir / skill_slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md_path = skill_dir / "SKILL.md"
    skill_md_path.write_text(_build_skill_md(package, runtime=runtime), encoding="utf-8")
    _write_derived_script_tools(skill_dir, package["workflow"])
    zip_path = export_dir / f"{skill_slug}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
                zipf.write(file_path, f"{skill_slug}/{file_path.relative_to(skill_dir)}")
    return {
        "export_dir": str(export_dir),
        "format": "skill",
        "skill_dir": str(skill_dir),
        "skill_md_path": str(skill_md_path),
        "skill_zip_path": str(zip_path),
        "validation": validate_standard_skill(skill_md_path, skill_dir=skill_dir),
    }


def export_canvas_json(
    document: CanvasDocument,
    *,
    export_root: str | Path,
    run_output: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    package = build_skill_package(document, run_output=run_output, warnings=warnings)
    export_dir = Path(export_root) / _package_dir_name(document.manifest.name or document.name)
    export_dir.mkdir(parents=True, exist_ok=True)
    json_path = export_dir / "clawcanvas.workflow.json"
    json_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "export_dir": str(export_dir),
        "format": "json",
        "json_path": str(json_path),
    }


def validate_standard_skill(skill_md_path: str | Path, *, skill_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(skill_md_path)
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    errors: list[str] = []
    warnings: list[str] = []

    if not frontmatter:
        errors.append("SKILL.md must start with YAML frontmatter.")

    name = str(frontmatter.get("name") or "").strip()
    description = str(frontmatter.get("description") or "").strip()

    if not name:
        errors.append("Frontmatter field `name` is required.")
    elif not re.match(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$", name):
        errors.append("Frontmatter field `name` must use lowercase letters, numbers, and hyphens, and be at most 64 characters.")

    if skill_dir is not None and name and Path(skill_dir).name != name:
        errors.append("Skill directory name must match frontmatter `name`.")

    if not description:
        errors.append("Frontmatter field `description` is required.")
    elif len(description.split()) < 8:
        warnings.append("Frontmatter `description` should explain what the skill does and when to use it.")

    if "version" in frontmatter:
        warnings.append("Put version under `metadata.version` instead of top-level frontmatter `version`.")

    forbidden_markers = [
        "ClawCanvas",
        "MASFactory",
        "workflow.canvas.json",
        "skill.package.json",
        "controller_inputs",
        "controller_outputs",
        "subgraph",
        "Position:",
        "Config:",
        "edge_",
    ]
    found_markers = [item for item in forbidden_markers if item in body]
    if found_markers:
        errors.append("Standard Skill body contains internal implementation markers: " + ", ".join(found_markers))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "frontmatter": {
            "name": name,
            "description": description,
        },
    }


def _package_dir_name(skill_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = _official_skill_slug(skill_name)
    return f"{safe}_{timestamp}"


def _export_as_json(package: dict[str, Any], export_dir: Path, *, runtime: RuntimeConfig | None = None) -> dict[str, Any]:
    paths = _write_publishable_skill_files(package, export_dir, runtime=runtime)

    return {
        "export_dir": str(export_dir),
        "format": "json",
        **{key: str(value) for key, value in paths.items()},
        "package": package,
    }


def _export_as_markdown(
    document: CanvasDocument,
    package: dict[str, Any],
    export_dir: Path,
    *,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Any]:
    paths = _write_publishable_skill_files(package, export_dir, runtime=runtime)

    return {
        "export_dir": str(export_dir),
        "format": "markdown",
        **{key: str(value) for key, value in paths.items()},
        "package": package,
    }


def _export_as_zip(
    document: CanvasDocument,
    package: dict[str, Any],
    export_dir: Path,
    *,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Any]:
    paths = _write_publishable_skill_files(package, export_dir, runtime=runtime)

    # Create ZIP file
    zip_path = export_dir.parent / f"{export_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in paths.values():
            zipf.write(file_path, file_path.name)

    return {
        "export_dir": str(export_dir),
        "format": "zip",
        "zip_path": str(zip_path),
        **{key: str(value) for key, value in paths.items()},
        "package": package,
    }


def _write_publishable_skill_files(
    package: dict[str, Any],
    export_dir: Path,
    *,
    runtime: RuntimeConfig | None = None,
) -> dict[str, Path]:
    document = package["workflow"]
    manifest = package["manifest"]
    paths = {
        "skill_md_path": export_dir / "SKILL.md",
        "manifest_path": export_dir / "skill.manifest.json",
        "workflow_path": export_dir / "workflow.canvas.json",
        "package_path": export_dir / "skill.package.json",
        "readme_path": export_dir / "README.md",
        "run_guide_path": export_dir / "RUN_WORKFLOW.md",
        "ignore_path": export_dir / ".clawhubignore",
    }

    paths["skill_md_path"].write_text(_build_skill_md(package, runtime=runtime), encoding="utf-8")
    _write_derived_script_tools(export_dir, document)
    paths["manifest_path"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["workflow_path"].write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["package_path"].write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["readme_path"].write_text(_build_readme(package), encoding="utf-8")
    paths["run_guide_path"].write_text(_build_run_workflow_guide(package), encoding="utf-8")
    paths["ignore_path"].write_text("exports/\n*.zip\n__pycache__/\n.DS_Store\n", encoding="utf-8")
    return paths


def _build_skill_md(package: dict[str, Any], *, runtime: RuntimeConfig | None = None) -> str:
    manifest = package["manifest"]
    workflow = package["workflow"]
    display_name = _normalize_skill_name(manifest.get("name") or workflow.get("name") or "clawcanvas_skill")
    skill_slug = _official_skill_slug(manifest.get("name") or workflow.get("name") or display_name)
    description = manifest.get("description") or workflow.get("description") or "Exported skill."
    frontmatter_payload: dict[str, Any] = {
        "name": skill_slug,
        "description": str(description),
        "metadata": {
            "version": str(manifest.get("version") or "0.1.0"),
        },
    }
    frontmatter = _yaml_frontmatter(frontmatter_payload)
    body = _build_standard_skill_md_body(package)
    return f"""{frontmatter}
{body}
"""


def _build_readme(package: dict[str, Any]) -> str:
    manifest = package["manifest"]
    workflow = package["workflow"]
    display_name = _normalize_skill_name(manifest.get("name") or workflow.get("name") or "ClawCanvas Skill")
    skill_slug = _official_skill_slug(manifest.get("name") or workflow.get("name") or display_name)
    return f"""# {display_name}

{manifest.get("description") or workflow.get("description") or "No description provided."}

This folder is intended to be published to ClawHub/OpenClaw as a text-based skill bundle.

Skill key: `{skill_slug}`

## Required Entry Point

- `SKILL.md`: OpenClaw/ClawHub skill entry file with YAML frontmatter and instructions.

## Supporting Files

- `workflow.canvas.json`: complete ClawCanvas workflow definition.
- `skill.manifest.json`: skill metadata from the ClawCanvas manifest editor.
- `skill.package.json`: combined export payload.
- `RUN_WORKFLOW.md`: execution guide for MASFactory environments.
- `.clawhubignore`: publish/sync ignore rules.

## Notes

ClawHub requires `SKILL.md`. This export embeds the exact workflow JSON inside `SKILL.md` and also ships standalone JSON files so OpenClaw users, ClawCanvas, and future importers can reconstruct the workflow.
"""


def _build_run_workflow_guide(package: dict[str, Any]) -> str:
    manifest = package["manifest"]
    workflow = package["workflow"]
    requirements = _derive_openclaw_requirements(workflow, manifest)
    env_requirements = (requirements.get("requires") or {}).get("env") or []
    deps = _python_dependencies_for_workflow(workflow, manifest)
    deps_text = " ".join(deps) if deps else "masfactory"
    env_text = "\n".join(f"- `{item}`" for item in env_requirements) or "- No required environment variables were inferred."
    return f"""# Run This Workflow

This file explains how to execute the ClawCanvas workflow outside of the skill metadata parser.

## Requirements

- Python 3.10 or newer.
- MASFactory installed in the Python environment.
- Access to the files in this skill folder.

Environment variables:

{env_text}

Python packages inferred from the workflow:

```bash
python -m pip install {deps_text}
```

If `masfactory` is not available from your configured package index, install it from the MASFactory repository or run this workflow inside a MASFactory checkout.

## Workflow Files

- `workflow.canvas.json` is the source of truth for nodes, edges, keys, prompts, tool declarations, loop controller mappings, and logic node behavior.
- `skill.manifest.json` contains user-editable skill metadata and global behavior rules. Tags and tools are derived from `workflow.canvas.json`.
- `skill.package.json` combines the manifest, workflow, runtime metadata, warnings, and last run output.

## Execution Contract

1. Load `workflow.canvas.json`.
2. Reconstruct the MASFactory graph using the node and edge definitions exactly.
3. Use `document.inputs` as the graph entry payload.
4. Use `document.attributes` as initial MASFactory graph attributes.
5. Bind declared tools, memories, and retrievers before invoking reasoning or loop nodes.
6. Preserve loop `controller_inputs`, `controller_outputs`, and termination rules.
7. Preserve logic node code exactly.

## Current Boundary

OpenClaw skills are instruction bundles. They can include supporting files, but OpenClaw does not automatically know how to execute a custom MASFactory graph unless a compatible runner is installed. This skill therefore includes both human/reasoning instructions and the exact workflow JSON needed by a runner.
"""


def _build_skill_md_body_with_llm(package: dict[str, Any], *, runtime: RuntimeConfig | None = None) -> str | None:
    runtime = runtime or {}
    api_key = str(runtime.get("apiKey") or "").strip()
    if not api_key:
        return None

    model_name = str(runtime.get("modelName") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = str(runtime.get("baseUrl") or "").strip() or None
    digest = _build_skill_generation_digest(package)

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client_kwargs: dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(api_key=api_key, **client_kwargs)
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are writing the markdown body of an OpenClaw/ClawHub skill. "
                            "Do not output YAML frontmatter. "
                            "Start with a level-1 markdown title. "
                            "Preserve workflow details accurately. "
                            "Do not invent nodes, tools, constraints, loop rules, or runtime requirements. "
                            "Use clear sections such as When To Use, Inputs, Outputs, Workflow Overview, "
                            "Node Details, Tools, Behavior, Validation Summary, and Run Notes. "
                            "Do not mention ClawCanvas, MASFactory, internal JSON files, workflow.canvas.json, "
                            "skill.package.json, runtime engines, or supporting file bundles. "
                            "If there are warnings, include them in a Warnings or Validation Notes section."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Write the skill markdown body for the following Skill definition. "
                            "Use the structured digest below as the source of truth.\n\n"
                            + json.dumps(digest, ensure_ascii=False, indent=2)
                        ),
                    }
                ],
            },
        ],
    )
    body = _extract_response_text(response).strip()
    if not body:
        return None
    return body


def _build_standard_skill_md_body(package: dict[str, Any]) -> str:
    manifest = package["manifest"]
    workflow = package["workflow"]
    nodes = workflow.get("nodes") or []
    tools = _derive_skill_tools_from_workflow(workflow)
    behavior = manifest.get("behavior") or {}
    key_descriptions = workflow.get("key_descriptions") or {}
    inputs = workflow.get("inputs") or {}
    display_name = _normalize_skill_name(manifest.get("name") or workflow.get("name") or "clawcanvas_skill")
    description = manifest.get("description") or workflow.get("description") or "Exported skill."
    outputs = _collect_output_keys(workflow)

    return f"""# {display_name}

{description}

## When To Use

Use this Skill when the user needs the workflow described below.

## Inputs

{_format_key_list(inputs, key_descriptions, empty_text="No explicit input keys declared.")}

## Outputs

{_format_key_list(outputs, key_descriptions, empty_text="No explicit output keys declared.")}

## Tools

{_format_tools_markdown(tools)}

## Behavior Rules

{_format_behavior_markdown(behavior)}

## Workflow

{_format_workflow_steps(nodes)}

## Instructions

Follow the workflow responsibilities above, apply the declared behavior rules, and keep the final answer actionable. If a required input or credential is missing, ask for it before continuing.
"""


def _build_skill_md_body_fallback(package: dict[str, Any]) -> str:
    return _build_standard_skill_md_body(package)


def _build_skill_generation_digest(package: dict[str, Any]) -> dict[str, Any]:
    manifest = package["manifest"]
    workflow = package["workflow"]
    nodes = workflow.get("nodes") or []
    edges = workflow.get("edges") or []
    derived_tools = _derive_skill_tools_from_workflow(workflow)
    derived_tags = _derive_skill_tags_from_workflow(workflow, manifest)
    return {
        "skill": {
            "name": manifest.get("name") or workflow.get("name") or "clawcanvas_skill",
            "description": manifest.get("description") or workflow.get("description") or "",
            "version": manifest.get("version") or "0.1.0",
            "tags": derived_tags,
            "tools": derived_tools,
            "behavior": manifest.get("behavior") or {},
        },
        "workflow": {
            "id": workflow.get("id") or "",
            "name": workflow.get("name") or "",
            "description": workflow.get("description") or "",
            "inputs": workflow.get("inputs") or {},
            "attributes": workflow.get("attributes") or {},
            "key_descriptions": workflow.get("key_descriptions") or {},
            "node_count": len(nodes),
            "edge_count": len(edges),
            "edges": edges,
            "nodes": [_summarize_node_for_llm(node) for node in nodes],
        },
    }


def _summarize_node_for_llm(node: dict[str, Any]) -> dict[str, Any]:
    config = dict(node.get("config") or {})
    summary = {
        "id": node.get("id") or "",
        "type": node.get("type") or "",
        "label": node.get("label") or "",
        "position": node.get("position") or {},
        "instructions": config.get("instructions") or "",
        "prompt_template": config.get("prompt_template") or "",
        "pull_keys": config.get("pull_keys") or {},
        "push_keys": config.get("push_keys") or {},
        "tools": config.get("tools") or [],
    }
    if node.get("type") == "custom":
        summary["custom"] = {
            "mode": config.get("mode") or "",
            "templates": config.get("templates") or {},
            "static_outputs": config.get("static_outputs") or {},
            "pick_keys": config.get("pick_keys") or {},
            "python_code": config.get("python_code") or "",
        }
    if node.get("type") == "loop":
        summary["loop"] = {
            "max_iterations": config.get("max_iterations") or 0,
            "terminate_when": config.get("terminate_when") or {},
            "controller": config.get("controller") or {},
            "controller_inputs": config.get("controller_inputs") or [],
            "controller_outputs": config.get("controller_outputs") or [],
            "subgraph": config.get("subgraph") or {},
        }
    return summary


def _extract_response_text(response: Any) -> str:
    output = getattr(response, "output_text", None)
    if isinstance(output, str) and output.strip():
        return output
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        parts = []
        for block in getattr(item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
    return ""


def _derive_openclaw_requirements(workflow: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    env: set[str] = set()
    bins: set[str] = set()
    deps: set[str] = set()
    has_agent = any(_node_or_inner_has_type(node, "agent") for node in workflow.get("nodes") or [])
    if has_agent:
        env.add("OPENAI_API_KEY")
    if has_agent or _workflow_has_loop(workflow) or _workflow_has_custom_python(workflow):
        bins.add("python")
        deps.add("masfactory")
    if _workflow_has_api_tools(workflow):
        bins.add("python")
        deps.add("requests")
    requires: dict[str, Any] = {}
    if env:
        requires["env"] = sorted(env)
    if bins:
        requires["bins"] = sorted(bins)
    if deps:
        requires["python"] = sorted(deps)

    out: dict[str, Any] = {}
    if requires:
        out["requires"] = requires
    if env:
        out["primaryEnv"] = sorted(env)[0]
    return out


def _node_or_inner_has_type(node: dict[str, Any], node_type: str) -> bool:
    if node.get("type") == node_type:
        return True
    subgraph = dict((node.get("config") or {}).get("subgraph") or {})
    return any(_node_or_inner_has_type(inner, node_type) for inner in (subgraph.get("nodes") or []))


def _workflow_has_loop(workflow: dict[str, Any]) -> bool:
    return any(_node_or_inner_has_type(node, "loop") for node in workflow.get("nodes") or [])


def _workflow_has_custom_python(workflow: dict[str, Any]) -> bool:
    def node_has_custom_python(node: dict[str, Any]) -> bool:
        config = dict(node.get("config") or {})
        if node.get("type") == "custom" and str(config.get("mode") or "").strip().lower() == "python":
            return True
        subgraph = dict(config.get("subgraph") or {})
        return any(node_has_custom_python(inner) for inner in (subgraph.get("nodes") or []))

    return any(node_has_custom_python(node) for node in workflow.get("nodes") or [])


def _workflow_has_api_tools(workflow: dict[str, Any]) -> bool:
    def node_has_api(node: dict[str, Any]) -> bool:
        config = dict(node.get("config") or {})
        if any(str(tool.get("binding") or "").lower() == "api" for tool in (config.get("tools") or [])):
            return True
        subgraph = dict(config.get("subgraph") or {})
        return any(node_has_api(inner) for inner in (subgraph.get("nodes") or []))

    return any(node_has_api(node) for node in (workflow.get("nodes") or []))


def _iter_workflow_nodes(workflow: dict[str, Any]):
    def visit(node: dict[str, Any], path: tuple[dict[str, Any], ...] = ()):
        yield node, path
        config = dict(node.get("config") or {})
        subgraph = dict(config.get("subgraph") or {})
        for inner in subgraph.get("nodes") or []:
            if isinstance(inner, dict):
                yield from visit(inner, (*path, node))

    for node in workflow.get("nodes") or []:
        if isinstance(node, dict):
            yield from visit(node)


def _script_tool_name(node: dict[str, Any]) -> str:
    return f"{_slugify(str(node.get('id') or node.get('label') or 'python_node')).replace('-', '_')}_script"


def _script_tool_path(node: dict[str, Any]) -> str:
    return f"scripts/{_script_tool_name(node)}.py"


def _derive_skill_tools_from_workflow(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_tool(raw_tool: dict[str, Any], *, source: str) -> None:
        name = str(raw_tool.get("name") or "").strip()
        binding = str(raw_tool.get("binding") or "builtin").strip().lower() or "builtin"
        if not name:
            return
        key = (binding, name)
        if key in seen:
            return
        seen.add(key)
        item = dict(raw_tool)
        item["name"] = name
        item["binding"] = binding
        item["source"] = source
        tools.append(item)

    for node, path in _iter_workflow_nodes(workflow):
        label = str(node.get("label") or node.get("id") or "node")
        config = dict(node.get("config") or {})
        source = " / ".join([str(item.get("label") or item.get("id") or "loop") for item in path] + [label])
        for tool in config.get("tools") or []:
            if isinstance(tool, dict):
                add_tool(tool, source=source)
        if node.get("type") == "custom" and str(config.get("mode") or "").strip().lower() == "python":
            code = str(config.get("python_code") or "").strip()
            if code:
                add_tool(
                    {
                        "name": _script_tool_name(node),
                        "binding": "script",
                        "description": f"Run deterministic logic for workflow step {label}.",
                        "script_path": _script_tool_path(node),
                    },
                    source=source,
                )

    return tools


def _derive_skill_tags_from_workflow(workflow: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    tags: list[str] = []

    def add(tag: str) -> None:
        normalized = _slugify(tag)
        if normalized and normalized not in tags:
            tags.append(normalized)

    add("workflow")
    type_counts: dict[str, int] = {}
    for node, _path in _iter_workflow_nodes(workflow):
        node_type = str(node.get("type") or "")
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
        if node_type == "agent":
            add("reasoning")
        elif node_type == "custom":
            add("logic")
        elif node_type == "loop":
            add("loop")

    if type_counts.get("agent", 0) > 1:
        add("multi-reasoning")

    for text in [manifest.get("name"), workflow.get("name"), manifest.get("description"), workflow.get("description")]:
        for word in re.split(r"[^A-Za-z0-9]+", str(text or "").lower()):
            if len(word) >= 4:
                add(word)
            if len(tags) >= 8:
                return tags
    return tags[:8]


def _write_derived_script_tools(skill_dir: Path, workflow: dict[str, Any]) -> None:
    for node, _path in _iter_workflow_nodes(workflow):
        config = dict(node.get("config") or {})
        if node.get("type") != "custom" or str(config.get("mode") or "").strip().lower() != "python":
            continue
        code = str(config.get("python_code") or "").rstrip()
        if not code:
            continue
        script_path = skill_dir / _script_tool_path(node)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(code + "\n", encoding="utf-8")


def _python_dependencies_for_workflow(workflow: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    requirements = _derive_openclaw_requirements(workflow, manifest)
    return list(((requirements.get("requires") or {}).get("python") or []))


def _format_requirements_markdown(requirements: dict[str, Any]) -> str:
    requires = requirements.get("requires") or {}
    lines = []
    if requires.get("env"):
        lines.append("Environment variables:")
        lines.extend(f"- `{item}`" for item in requires["env"])
    if requires.get("bins"):
        lines.append("CLI binaries:")
        lines.extend(f"- `{item}`" for item in requires["bins"])
    if requires.get("python"):
        lines.append("Python packages:")
        lines.extend(f"- `{item}`" for item in requires["python"])
    return "\n".join(lines) if lines else "No external requirements declared."


def _format_tools_markdown(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "No workflow-derived tools declared."
    sections = []
    for tool in tools:
        name = str(tool.get("name") or "unnamed_tool")
        binding = str(tool.get("binding") or "unknown")
        description = str(tool.get("description") or "No description.").strip()
        if binding == "script":
            script_path = str(tool.get("script_path") or tool.get("filename") or f"scripts/{_slugify(name)}.py")
            sections.append(
                "\n".join(
                    [
                        f"### {name}",
                        "",
                        description,
                        "",
                        "Run this script when the task requires this capability:",
                        "",
                        "```bash",
                        f"python {script_path}",
                        "```",
                    ]
                )
            )
        else:
            sections.append(f"- `{name}` (`{binding}`): {description}")
    return "\n\n".join(sections)


def _format_behavior_markdown(behavior: dict[str, Any]) -> str:
    if not behavior:
        return "No skill-level behavior declared."
    lines = []
    if behavior.get("style"):
        lines.append(f"- Style: {behavior['style']}")
    for rule in behavior.get("rules") or []:
        lines.append(f"- {rule}")
    return "\n".join(lines) if lines else "No skill-level behavior declared."


def _format_key_list(
    keys_or_values: dict[str, Any],
    descriptions: dict[str, Any],
    *,
    empty_text: str,
) -> str:
    if not keys_or_values:
        return empty_text

    lines = []
    for key, value in keys_or_values.items():
        description = descriptions.get(key) or value
        text = str(description).strip()
        lines.append(f"- `{key}`: {text}" if text else f"- `{key}`")
    return "\n".join(lines)


def _collect_output_keys(workflow: dict[str, Any]) -> dict[str, str]:
    nodes = workflow.get("nodes") or []
    edges = workflow.get("edges") or []
    end_ids = {str(node.get("id")) for node in nodes if node.get("type") == "end"}
    outputs: dict[str, str] = {}

    for edge in edges:
        if str(edge.get("target") or "") in end_ids:
            for key, description in dict(edge.get("mapping") or {}).items():
                outputs[str(key)] = str(description or "")

    if outputs:
        return outputs

    def collect_from_node(node: dict[str, Any]) -> None:
        config = dict(node.get("config") or {})
        for key, description in dict(config.get("push_keys") or {}).items():
            outputs[str(key)] = str(description or "")
        for key, description in dict(config.get("static_outputs") or {}).items():
            outputs.setdefault(str(key), str(description or ""))
        subgraph = dict(config.get("subgraph") or {})
        for inner in subgraph.get("nodes") or []:
            collect_from_node(inner)

    for node in nodes:
        collect_from_node(node)
    return outputs


def _format_workflow_steps(nodes: list[dict[str, Any]]) -> str:
    steps = [
        _workflow_step_for_node(node, step_number=index)
        for index, node in enumerate([item for item in nodes if item.get("type") not in {"start", "end"}], start=1)
    ]
    steps = [step for step in steps if step]
    if not steps:
        return "Use the declared behavior rules to answer the user request."
    return "\n\n".join(steps)


def _workflow_step_for_node(
    node: dict[str, Any],
    *,
    step_number: int | None = None,
    heading_level: int = 3,
) -> str:
    node_type = str(node.get("type") or "node")
    label = str(node.get("label") or node.get("id") or "Step")
    config = dict(node.get("config") or {})
    heading_prefix = "#" * max(3, min(heading_level, 6))
    heading = f"{heading_prefix} Step {step_number}: {label}" if step_number is not None else f"{heading_prefix} {label}"

    if node_type == "agent":
        instructions = str(config.get("instructions") or "").strip()
        prompt = _prompt_template_to_guidance(str(config.get("prompt_template") or "").strip())
        sections = [
            heading,
            "",
            instructions or "Complete this workflow step according to its inputs and expected outputs.",
        ]
        if prompt:
            sections.extend(["", prompt])
        sections.extend(
            [
                "",
                "Input:",
                _format_key_list(config.get("pull_keys") or {}, {}, empty_text="- No explicit input fields."),
                "",
                "Output:",
                _format_key_list(config.get("push_keys") or {}, {}, empty_text="- No explicit output fields."),
            ]
        )
        tools = _format_step_tools(config.get("tools") or [])
        if tools:
            sections.extend(["", "Tools this stage can use:", tools])
        return "\n".join(sections).strip()

    if node_type == "custom":
        sections = [
            heading,
            "",
            _custom_step_description(config),
            "",
            "Input:",
            _format_key_list(config.get("pull_keys") or {}, {}, empty_text="- No explicit input fields."),
            "",
            "Output:",
            _format_key_list(config.get("push_keys") or {}, {}, empty_text="- No explicit output fields."),
        ]
        return "\n".join(sections).strip()

    if node_type == "loop":
        max_iterations = config.get("max_iterations") or 1
        terminate_when = dict(config.get("terminate_when") or {})
        stop_key = terminate_when.get("key")
        inner_nodes = (dict(config.get("subgraph") or {}).get("nodes") or [])
        inner_steps = [
            _workflow_step_for_node(inner, heading_level=heading_level + 1)
            for inner in inner_nodes
            if inner.get("type") not in {"start", "end"}
        ]
        stop_text = f"Stop earlier when `{stop_key}` indicates completion." if stop_key else "Stop earlier when the review criteria are met."
        sections = [
            heading,
            "",
            f"Repeat the internal review flow up to {max_iterations} times. {stop_text}",
            "",
            "Input:",
            _format_loop_controller_keys(config.get("controller_inputs") or [], empty_text="- No explicit loop input fields."),
            "",
            "Output:",
            _format_loop_controller_keys(config.get("controller_outputs") or [], empty_text="- No explicit loop output fields."),
        ]
        if inner_steps:
            sections.extend(["", "Inside each iteration:", "\n\n".join(inner_steps)])
        return "\n".join(sections).strip()

    return "\n".join([heading, "", f"Complete the {node_type} step."])


def _prompt_template_to_guidance(prompt_template: str) -> str:
    if not prompt_template:
        return ""
    placeholders = _extract_placeholders(prompt_template)
    if not placeholders:
        return f"Use this guidance when preparing the step output: {prompt_template}"
    keys = ", ".join(f"`{key}`" for key in placeholders)
    return f"Use the available {keys} field{'s' if len(placeholders) != 1 else ''} to prepare the step output."


def _extract_placeholders(text: str) -> list[str]:
    return [match for match in re.findall(r"\{([^{}]+)\}", text or "") if match.strip()]


def _format_rule_list(rules: list[Any]) -> str:
    lines = [f"- {str(rule).strip()}" for rule in rules if str(rule).strip()]
    return "\n".join(lines)


def _format_step_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    return "\n".join(
        f"- Use `{tool.get('name') or 'unnamed_tool'}` when {str(tool.get('description') or 'the step requires this capability.').strip()}"
        for tool in tools
    )


def _custom_step_description(config: dict[str, Any]) -> str:
    mode = str(config.get("mode") or "custom").strip()
    if mode == "set":
        return "Set deterministic output fields from the available input values."
    if mode == "pick":
        return "Select the required fields from the available input values."
    if mode == "template":
        return "Format the output fields from the available input values."
    if mode == "python":
        return "Run the deterministic custom logic for this step, then pass its output to the next step."
    return "Transform the available information into the expected output fields."


def _format_loop_controller_keys(items: list[dict[str, Any]], *, empty_text: str) -> str:
    merged: dict[str, Any] = {}
    for item in items:
        merged.update(dict(item.get("mapping") or {}))
    return _format_key_list(merged, {}, empty_text=empty_text)


def _join_keys(mapping: dict[str, Any]) -> str:
    keys = [f"`{key}`" for key in mapping.keys()]
    if not keys:
        return ""
    return ", ".join(keys)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _format_edge_list(edges: list[dict[str, Any]]) -> str:
    if not edges:
        return "No workflow edges declared."
    lines = []
    for edge in edges:
        lines.append(
            f"- `{edge.get('id', 'edge')}`: `{edge.get('source', '?')}` -> `{edge.get('target', '?')}` with mapping `{json.dumps(edge.get('mapping') or {}, ensure_ascii=False)}`"
        )
    return "\n".join(lines)


def _format_node_sections(nodes: list[dict[str, Any]], *, level: int = 3, owner_path: str = "workflow") -> str:
    if not nodes:
        return "No nodes declared."

    sections: list[str] = []
    heading_prefix = "#" * max(3, min(level, 6))

    for node in nodes:
        node_id = str(node.get("id") or "node")
        node_type = str(node.get("type") or "unknown")
        label = str(node.get("label") or node_id)
        config = dict(node.get("config") or {})
        position = dict(node.get("position") or {})
        sections.append(
            "\n".join(
                [
                    f"{heading_prefix} Node `{node_id}` ({node_type})",
                    "",
                    f"- Path: `{owner_path}/{node_id}`",
                    f"- Label: {label}",
                    f"- Position: `x={position.get('x', 0)}`, `y={position.get('y', 0)}`",
                    "",
                    "Config:",
                    "",
                    _json_block(config),
                ]
            )
        )

        if node_type == "loop":
            controller_inputs = config.get("controller_inputs") or []
            controller_outputs = config.get("controller_outputs") or []
            inner_edges = (config.get("subgraph") or {}).get("edges") or []
            inner_nodes = (config.get("subgraph") or {}).get("nodes") or []
            sections.append(
                "\n".join(
                    [
                        "",
                        f"{heading_prefix}# Loop `{node_id}` Controller Topology",
                        "",
                        "Controller Inputs:",
                        "",
                        _json_block(controller_inputs),
                        "",
                        "Controller Outputs:",
                        "",
                        _json_block(controller_outputs),
                        "",
                        "Inner Edges:",
                        "",
                        _json_block(inner_edges),
                    ]
                )
            )
            sections.append(
                _format_node_sections(
                    inner_nodes,
                    level=level + 1,
                    owner_path=f"{owner_path}/{node_id}/subgraph",
                )
            )

    return "\n\n".join(section for section in sections if section.strip())


def _slugify(value: str) -> str:
    chars = []
    for char in str(value).strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "_", "-", "/"}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "clawcanvas-skill"


def _official_skill_slug(value: str) -> str:
    slug = _slugify(str(value).replace("_", "-"))
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", slug):
        slug = "clawcanvas-skill"
    return slug


def _normalize_skill_name(value: str) -> str:
    chars = []
    for char in str(value).strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "/", "."}:
            chars.append("_")
        elif char == "_":
            chars.append("_")
    normalized = "".join(chars).strip("_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized or "clawcanvas_skill"


def _yaml_frontmatter(data: dict[str, Any]) -> str:
    return "---\n" + _yaml_lines(data) + "\n---"


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw = text[4:end].strip()
    body = text[end + 4 :].lstrip()
    frontmatter: dict[str, str] = {}
    for line in raw.splitlines():
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    return frontmatter, body


def _yaml_lines(value: Any, *, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            key_text = str(key)
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key_text}:")
                lines.append(_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{pad}{key_text}: {_yaml_scalar(item)}")
        return "\n".join(line for line in lines if line != "")
    if isinstance(value, list):
        if not value:
            return f"{pad}[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_yaml_lines(item, indent=indent + 2))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(value)}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)
