from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

PROMPT_SOURCE_DIR = Path(__file__).resolve().parents[1] / "agents"


def load_prompt(filename: str, *, convert_braces: bool = False) -> str:
    content = (PROMPT_SOURCE_DIR / filename).read_text(encoding="utf-8").strip()
    if convert_braces:
        content = re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", r"{\1}", content)
    return content


def load_stage_instruction(*filenames: str) -> str:
    return "\n\n".join(load_prompt(name) for name in filenames if name).strip()


def replace_placeholders(template: str, mapping: dict[str, str]) -> str:
    updated = template
    for source, target in mapping.items():
        updated = updated.replace("{" + source + "}", "{" + target + "}")
    return updated


def build_runtime_prompt(base_prompt: str, extra_sections: Iterable[tuple[str, Any]]) -> str:
    sections = [base_prompt.strip()]
    for title, content in extra_sections:
        if content is None or str(content).strip() == "":
            continue
        sections.append(f"## {title}\n{content}")
    return "\n\n".join(sections).strip()


__all__ = [
    "build_runtime_prompt",
    "load_prompt",
    "load_stage_instruction",
    "replace_placeholders",
]
