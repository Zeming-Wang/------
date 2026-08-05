from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .schema import CanvasDocument
from .skill_packager import _build_skill_generation_digest


def generate_authoring_field(
    document: CanvasDocument,
    *,
    ai: dict[str, Any],
    field: str,
    mode: str,
    current_value: Any = "",
    locale: str = "en",
) -> dict[str, Any]:
    api_key = str(ai.get("apiKey") or "").strip()
    if not api_key:
        raise ValueError("Authoring AI API Key is required.")

    model_name = str(ai.get("modelName") or "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = str(ai.get("baseUrl") or "").strip() or None
    package = {
        "manifest": asdict(document.manifest),
        "workflow": document.to_dict(),
    }
    digest = _build_skill_generation_digest(package)

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenAI Python package is required for AI authoring.") from exc

    client_kwargs: dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url

    language = "Chinese" if locale == "zh" else "English"
    response = OpenAI(api_key=api_key, **client_kwargs).responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You help author standards-compliant Anthropic Agent Skills. "
                            "Return only the requested field value, with no markdown fence unless the field explicitly asks for markdown. "
                            "The Skill should be centered on reusable instructions and progressive disclosure. "
                            "Do not mention ClawCanvas, MASFactory, graph runtimes, node coordinates, edge ids, or internal JSON."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(
                            {
                                "task": "generate_or_polish_skill_field",
                                "field": field,
                                "mode": mode,
                                "language": language,
                                "current_value": current_value,
                                "requirements": _field_requirements(field),
                                "skill_digest": digest,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ],
            },
        ],
    )
    value = _extract_response_text(response).strip()
    if not value:
        raise RuntimeError("AI authoring returned an empty value.")
    return {
        "field": field,
        "mode": mode,
        "value": value,
        "model": model_name,
    }


def _field_requirements(field: str) -> str:
    normalized = field.strip().lower()
    if normalized in {"description", "manifest.description"}:
        return "Write a trigger-focused Skill description. Include what the skill does and when to use it. Keep it concise but specific."
    if normalized in {"agent.instructions", "instructions"}:
        return "Write 2-5 concise sentences for one workflow step. Define the task, reasoning style, and expected output. Do not mention internal graph details."
    if normalized in {"agent.prompt_template", "prompt_template"}:
        return "Write a prompt template for one workflow step. Preserve useful placeholders from the existing workflow keys, such as {query}, {analysis}, or {draft} when appropriate."
    if normalized in {"workflow", "workflowsteps", "workflow_steps"}:
        return "Write concise numbered workflow instructions suitable for SKILL.md. Do not expose internal graph details."
    if normalized in {"behavior", "behaviorrules", "behavior_rules", "agent.behavior_rules"}:
        return "Write concrete behavior rules as short bullet lines."
    if normalized in {"knowledge", "domainknowledge", "domain_knowledge"}:
        return "Summarize reusable domain knowledge for the Skill."
    return "Write a clear standards-compliant Skill field value."


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
