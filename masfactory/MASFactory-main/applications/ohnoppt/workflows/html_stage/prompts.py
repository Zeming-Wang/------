from __future__ import annotations

from workflows.common.prompts import (
    build_runtime_prompt,
    load_prompt,
    load_stage_instruction,
    replace_placeholders,
)

HTML_GENERATION_INSTRUCTIONS = load_stage_instruction(
    "outline_to_html_system_prompt.md",
    "outline_to_html_task_prompt.md",
)

HTML_TASK_PROMPT = build_runtime_prompt(
    replace_placeholders(
        load_prompt("outline_to_html_task_prompt.md", convert_braces=True),
        {
            "assets_dir": "asset_dir",
            "html_output_path": "html_path",
        },
    )
    + "\n\n"
    + "Hard constraint: you must generate exactly as many slide sections as the `Page Budget` requires."
    + "If the target page budget is 40, you must output 40 `<section class=\"slide\">...</section>` blocks."
    + "Do not cherry-pick only key slides or merge slides."
    + "\n\n"
    + "Generate the complete deck's `slides.html` and complete `structure_notes.md` in one pass."
    + "If an `Existing HTML Draft` is present, you may revise it globally, but the output must still be a complete deck."
    + "\n\n"
    + "Additional hard constraint: if you insert local images, you must use the full relative HTML paths provided in the Source Reference Package,"
    + " for example `../assets/paper01/pages/page01.png`, and must not rewrite them into `../assets/page01.png` on your own.",
    [
        ("Page Budget", "{page_budget}"),
        ("Outline Markdown", "{outline_markdown}"),
        ("Existing HTML Draft", "{html_content}"),
        ("Existing Structure Notes", "{structure_notes_markdown}"),
        ("Source Reference Package", "{source_reference_markdown}"),
        ("Revision Feedback", "{html_iteration_feedback}"),
    ],
)
