from __future__ import annotations

from masfactory import Agent, CustomNode, HumanChatVisual, Loop, NodeTemplate
from masfactory.core.message import ParagraphMessageFormatter, TaggedFieldMessageFormatter

from workflows.html_stage.prompts import (
    HTML_GENERATION_INSTRUCTIONS,
    HTML_TASK_PROMPT,
)
from workflows.html_stage.tools import (
    export_html_and_pdf,
    finalize_html_review,
    should_terminate_html,
)

HTML_STAGE_PUSH_KEYS = {
    "html_ready": "HTML stage ready",
    "html_iteration_feedback": "HTML revision feedback",
    "html_content": "HTML content",
    "structure_notes_markdown": "Structure notes",
    "html_path": "HTML path",
    "structure_notes_path": "Structure notes path",
    "html_excerpt": "HTML excerpt",
    "structure_notes_excerpt": "Structure notes excerpt",
    "html_review_path": "HTML review path",
    "html_asset_validation_markdown": "HTML asset validation",
    "html_missing_asset_refs": "Missing asset refs",
    "html_has_missing_assets": "Missing asset flag",
    "rendered_pdf_path": "Rendered PDF path",
    "rendered_pdf_preview_dir": "Rendered PDF preview dir",
    "rendered_pdf_summary_markdown": "Rendered PDF summary",
    "export_report_path": "Export report path",
}

HtmlGenerationAgent = NodeTemplate(
    Agent,
    instructions=HTML_GENERATION_INSTRUCTIONS,
    prompt_template=HTML_TASK_PROMPT,
    pull_keys={
        "outline_path": "Outline path",
        "outline_markdown": "Outline body",
        "page_budget": "Target slide count",
        "asset_dir": "Asset directory",
        "source_reference_markdown": "Source reference package",
        "style_brief": "Style brief",
        "html_path": "HTML output path",
        "html_content": "Existing HTML draft",
        "structure_notes_markdown": "Existing structure notes",
        "html_iteration_feedback": "Previous revision feedback",
    },
    push_keys={
        "html_content": "Complete slides.html",
        "structure_notes_markdown": "Structure notes",
    },
    formatters=[ParagraphMessageFormatter(), TaggedFieldMessageFormatter()],
)

HtmlHumanReviewNode = NodeTemplate(
    HumanChatVisual,
    attributes={
        "current_stage": "pdf",
    },
    push_keys={
        "html_human_decision": "Enter approve/approved/yes to accept; any other response requests revisions",
        "html_human_feedback": "Provide human review feedback for this iteration",
    },
)

HtmlPdfExportNode = NodeTemplate(
    CustomNode,
    forward=export_html_and_pdf,
    pull_keys={
        "html_content": "HTML content",
        "structure_notes_markdown": "Structure notes",
        "asset_dir": "Asset directory",
        "html_path": "HTML path",
        "structure_notes_path": "Structure notes path",
        "rendered_pdf_path": "Rendered PDF path",
        "rendered_pdf_preview_dir": "Rendered PDF preview dir",
        "export_report_path": "Export report path",
        "pdf_export_strategy": "Export strategy",
        "page_budget": "Target slide count",
    },
    push_keys={
        "html_content": "HTML content",
        "html_path": "HTML path",
        "structure_notes_markdown": "Structure notes",
        "structure_notes_path": "Structure notes path",
        "html_excerpt": "HTML excerpt",
        "structure_notes_excerpt": "Structure notes excerpt",
        "html_asset_validation_markdown": "HTML asset validation",
        "html_missing_asset_refs": "Missing asset refs",
        "html_has_missing_assets": "Whether local asset refs are missing",
        "pdf_export_success": "Whether PDF export succeeded",
        "rendered_pdf_path": "Rendered PDF path",
        "rendered_pdf_preview_dir": "Rendered PDF preview dir",
        "rendered_pdf_summary_markdown": "Rendered PDF summary",
        "export_report_path": "Export report path",
    },
)

HtmlFinalizeNode = NodeTemplate(
    CustomNode,
    forward=finalize_html_review,
    pull_keys=None,
    push_keys={
        "html_ready": "Stage ready flag",
        "html_iteration_feedback": "Feedback for next iteration",
        "html_review_path": "Review path",
        "html_content": "HTML content",
        "structure_notes_markdown": "Structure notes",
        "html_path": "HTML path",
        "structure_notes_path": "Structure notes path",
        "html_excerpt": "HTML excerpt",
        "structure_notes_excerpt": "Structure notes excerpt",
        "html_asset_validation_markdown": "HTML asset validation",
        "html_missing_asset_refs": "Missing asset refs",
        "html_has_missing_assets": "Missing asset flag",
        "rendered_pdf_path": "Rendered PDF path",
        "rendered_pdf_preview_dir": "Rendered PDF preview dir",
        "rendered_pdf_summary_markdown": "Rendered PDF summary",
        "export_report_path": "Export report path",
    },
)

HtmlStageLoop = NodeTemplate(
    Loop,
    max_iterations=3,
    terminate_condition_function=should_terminate_html,
    pull_keys=None,
    push_keys=HTML_STAGE_PUSH_KEYS,
    nodes=[
        ("text-html-generation", HtmlGenerationAgent),
        ("text-html-export-pdf", HtmlPdfExportNode),
        ("text-html-human-review", HtmlHumanReviewNode),
        ("text-html-finalize", HtmlFinalizeNode),
    ],
    edges=[
        ("CONTROLLER", "text-html-generation", {}),
        ("text-html-generation", "text-html-export-pdf", {}),
        (
            "text-html-export-pdf",
            "text-html-human-review",
            {
                "rendered_pdf_path": "Rendered PDF path",
            },
        ),
        ("text-html-human-review", "text-html-finalize", {}),
        (
            "text-html-finalize",
            "CONTROLLER",
            {
                "html_ready": "ready",
                "html_iteration_feedback": "feedback",
                "html_content": "html_content",
                "structure_notes_markdown": "structure_notes",
                "html_path": "html_path",
                "structure_notes_path": "structure_notes_path",
                "html_excerpt": "html_excerpt",
                "structure_notes_excerpt": "structure_notes_excerpt",
                "html_review_path": "review_path",
                "html_asset_validation_markdown": "asset_validation",
                "html_missing_asset_refs": "missing_asset_refs",
                "html_has_missing_assets": "has_missing_assets",
                "rendered_pdf_path": "rendered_pdf_path",
                "rendered_pdf_preview_dir": "rendered_pdf_preview_dir",
                "rendered_pdf_summary_markdown": "rendered_pdf_summary_markdown",
                "export_report_path": "export_report_path",
            },
        ),
    ],
)

__all__ = ["HtmlStageLoop"]
