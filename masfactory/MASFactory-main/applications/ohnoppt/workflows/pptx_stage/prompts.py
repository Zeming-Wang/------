from __future__ import annotations

from workflows.common.prompts import (
    build_runtime_prompt,
    load_stage_instruction,
)

PPTX_GENERATION_INSTRUCTIONS = load_stage_instruction(
    "pdf_to_pythonpptx_system_prompt.md",
    "pdf_to_pythonpptx_task_prompt.md",
)

PPTX_TASK_PROMPT = build_runtime_prompt(
    """
Output a `python-pptx` code fragment dedicated to the current batch based on the current batch's HTML and slide range.

Requirements:
- Only handle the slides covered by `Current Slide Batch`.
- Output only the current batch's code fragment, not the complete final script.
- Do not include `import`, `main()`, `Presentation()` initialization, or file-saving logic.
- Define only the slide functions involved in this batch, and each function name must be `build_slide_<page_number>(prs)`.
- If helper functions are needed, their names must carry a unique prefix for this batch, for example `batch_01_10_add_card(...)`.
- Do not depend on functions, variables, or script content from previous batches.
- Reusing local image paths already referenced in the HTML is allowed, but do not use full-page PDF or paper screenshots as a shortcut background.
- The output fragment will be assembled and executed by an external program.
""".strip(),
    [
        ("Current Batch HTML", "{pptx_current_batch_html_content}"),
        ("Current Slide Batch", "{pptx_current_batch_markdown}"),
        ("Human Revision Feedback", "{pptx_iteration_feedback}"),
    ],
)
