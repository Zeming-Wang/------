from __future__ import annotations

from workflows.common.prompts import (
    build_runtime_prompt,
    load_prompt,
    load_stage_instruction,
    replace_placeholders,
)

OUTLINE_GENERATION_INSTRUCTIONS = load_stage_instruction(
    "requirements_to_outline_system_prompt.md",
    "requirements_to_outline_task_prompt.md",
)

OUTLINE_TASK_PROMPT = build_runtime_prompt(
    replace_placeholders(
        load_prompt("requirements_to_outline_task_prompt.md", convert_braces=True),
        {
            "requirements_text": "user_requirements",
            "source_paths": "paper_paths_text",
        },
    ),
    [
        ("Source Reference Package", "{outline_source_reference_markdown}"),
        ("Previous Revision Feedback", "{outline_iteration_feedback}"),
    ],
)
