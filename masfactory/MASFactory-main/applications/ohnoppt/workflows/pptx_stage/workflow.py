from __future__ import annotations

from masfactory import Agent, CustomNode, HumanChatVisual, Loop, NodeTemplate
from masfactory.core.message import ParagraphMessageFormatter, TaggedFieldMessageFormatter

from workflows.pptx_stage.prompts import (
    PPTX_GENERATION_INSTRUCTIONS,
    PPTX_TASK_PROMPT,
)
from workflows.pptx_stage.tools import (
    assemble_pptx_script,
    finalize_pptx_batch_iteration,
    finalize_pptx_review,
    prepare_pptx_batch_iteration,
    prepare_pptx_batch_plan,
    run_recreation_script,
    should_terminate_pptx,
    should_terminate_pptx_batch_loop,
)

PPTX_STAGE_PUSH_KEYS = {
    "pptx_ready": "PPTX stage ready",
    "pptx_iteration_feedback": "PPTX revision feedback",
    "recreated_script_path": "Script path",
    "recreated_pptx_path": "PPTX path",
    "recreated_validation_pdf_path": "Validation PDF path",
    "recreated_preview_dir": "Preview dir",
    "recreated_pptx_summary_markdown": "Recreated PPTX summary",
    "recreation_notes_path": "Notes path",
    "pptx_review_path": "Review path",
}

PPTX_BATCH_PUSH_KEYS = {
    "pptx_batch_sections_json": "Serialized slide sections",
    "pptx_batch_size": "Batch size",
    "pptx_batch_count": "Batch count",
    "pptx_batch_index": "Current batch index",
    "pptx_batch_outputs_json": "Batch outputs",
}

PptxGenerationAgent = NodeTemplate(
    Agent,
    instructions=PPTX_GENERATION_INSTRUCTIONS,
    prompt_template=PPTX_TASK_PROMPT,
    pull_keys={
        "pptx_current_batch_html_content": "Current batch HTML",
        "pptx_current_batch_markdown": "Current slide batch",
        "pptx_iteration_feedback": "Previous human revision feedback",
    },
    push_keys={
        "python_pptx_batch_fragment": "python-pptx batch fragment",
        "pptx_batch_notes_markdown": "batch recreation notes",
    },
    formatters=[ParagraphMessageFormatter(), TaggedFieldMessageFormatter()],
)

PptxBatchPlanNode = NodeTemplate(
    CustomNode,
    forward=prepare_pptx_batch_plan,
    pull_keys={
        "outline_markdown": "Outline content",
    },
    push_keys={
        "pptx_batch_sections_json": "Serialized slide sections",
        "pptx_batch_size": "Batch size",
        "pptx_batch_count": "Batch count",
        "pptx_batch_index": "Current batch index",
        "pptx_batch_outputs_json": "Batch outputs",
        "pptx_batch_iteration_ready": "Whether the batch loop can stop",
    },
)

PptxBatchPrepareNode = NodeTemplate(
    CustomNode,
    forward=prepare_pptx_batch_iteration,
    pull_keys=None,
    push_keys={
        "pptx_current_batch_markdown": "Current batch markdown",
        "pptx_current_batch_html_content": "Current batch HTML",
        "pptx_batch_label": "Batch label",
    },
)

PptxBatchFinalizeNode = NodeTemplate(
    CustomNode,
    forward=finalize_pptx_batch_iteration,
    pull_keys=None,
    push_keys={
        "pptx_batch_iteration_ready": "Whether the batch loop can stop",
        "pptx_batch_sections_json": "Serialized slide sections",
        "pptx_batch_size": "Batch size",
        "pptx_batch_count": "Batch count",
        "pptx_batch_index": "Current batch index",
        "pptx_batch_outputs_json": "Batch outputs",
    },
)

PptxBatchGenerationLoop = NodeTemplate(
    Loop,
    max_iterations=20,
    terminate_condition_function=should_terminate_pptx_batch_loop,
    pull_keys=None,
    push_keys=PPTX_BATCH_PUSH_KEYS,
    nodes=[
        ("text-pptx-batch-prepare", PptxBatchPrepareNode),
        ("text-pptx-batch-generation", PptxGenerationAgent),
        ("text-pptx-batch-finalize", PptxBatchFinalizeNode),
    ],
    edges=[
        ("CONTROLLER", "text-pptx-batch-prepare", {}),
        ("text-pptx-batch-prepare", "text-pptx-batch-generation", {}),
        ("text-pptx-batch-generation", "text-pptx-batch-finalize", {}),
        (
            "text-pptx-batch-finalize",
            "CONTROLLER",
            {
                "pptx_batch_iteration_ready": "ready",
                "pptx_batch_sections_json": "pptx_batch_sections_json",
                "pptx_batch_size": "pptx_batch_size",
                "pptx_batch_count": "pptx_batch_count",
                "pptx_batch_index": "pptx_batch_index",
                "pptx_batch_outputs_json": "pptx_batch_outputs_json",
            },
        ),
    ],
)

PptxAssembleNode = NodeTemplate(
    CustomNode,
    forward=assemble_pptx_script,
    pull_keys={
        "pptx_batch_outputs_json": "Batch outputs",
        "pptx_batch_sections_json": "Serialized slide sections",
        "recreated_script_path": "Script path",
        "recreation_notes_path": "Notes path",
        "recreated_pptx_path": "PPTX path",
    },
    push_keys={
        "recreated_script_path": "Script path",
        "recreation_notes_path": "Notes path",
        "python_pptx_script": "Assembled script text",
        "recreation_notes_markdown": "Assembled notes text",
        "python_pptx_script_excerpt": "Script excerpt",
    },
)

PptxExecutionNode = NodeTemplate(
    CustomNode,
    forward=run_recreation_script,
    pull_keys={
        "python_pptx_script": "Script text",
        "recreation_notes_markdown": "Notes text",
        "recreated_script_path": "Script path",
        "recreation_notes_path": "Notes path",
        "recreated_pptx_path": "PPTX path",
        "recreated_validation_pdf_path": "Validation PDF path",
        "recreated_preview_dir": "Preview dir",
        "python_bin": "Python path",
    },
    push_keys={
        "recreated_script_path": "Script path",
        "recreation_notes_path": "Notes path",
        "python_pptx_script": "Sanitized script text",
        "recreation_notes_markdown": "Sanitized notes text",
        "python_pptx_script_excerpt": "Script excerpt",
        "pptx_recreation_success": "Whether recreation succeeded",
        "recreated_pptx_path": "PPTX path",
        "recreated_validation_pdf_path": "Validation PDF path",
        "recreated_preview_dir": "Preview dir",
        "recreated_pptx_summary_markdown": "Recreated PPTX summary",
    },
)

PptxHumanReviewNode = NodeTemplate(
    HumanChatVisual,
    attributes={
        "current_stage": "pptx",
    },
    push_keys={
        "pptx_human_decision": "Enter approve/approved/yes to accept; any other response requests revisions",
        "pptx_human_feedback": "Provide human review feedback for this iteration",
    },
)

PptxFinalizeNode = NodeTemplate(
    CustomNode,
    forward=finalize_pptx_review,
    pull_keys=None,
    push_keys={
        "pptx_ready": "Stage ready flag",
        "pptx_iteration_feedback": "Feedback for next iteration",
        "pptx_review_path": "Review path",
        "recreated_script_path": "Script path",
        "recreated_pptx_path": "PPTX path",
        "recreated_validation_pdf_path": "Validation PDF path",
        "recreated_preview_dir": "Preview dir",
        "recreated_pptx_summary_markdown": "Recreated PPTX summary",
        "recreation_notes_path": "Notes path",
        "python_pptx_script": "Script text",
        "recreation_notes_markdown": "Notes text",
    },
)

PptxStageLoop = NodeTemplate(
    Loop,
    max_iterations=3,
    terminate_condition_function=should_terminate_pptx,
    pull_keys=None,
    push_keys=PPTX_STAGE_PUSH_KEYS,
    nodes=[
        ("text-pptx-prepare-batch-plan", PptxBatchPlanNode),
        ("text-pptx-batch-loop", PptxBatchGenerationLoop),
        ("text-pptx-assemble-script", PptxAssembleNode),
        ("text-pptx-execute", PptxExecutionNode),
        ("text-pptx-human-review", PptxHumanReviewNode),
        ("text-pptx-finalize", PptxFinalizeNode),
    ],
    edges=[
        ("CONTROLLER", "text-pptx-prepare-batch-plan", {}),
        ("text-pptx-prepare-batch-plan", "text-pptx-batch-loop", {}),
        ("text-pptx-batch-loop", "text-pptx-assemble-script", {}),
        ("text-pptx-assemble-script", "text-pptx-execute", {}),
        (
            "text-pptx-execute",
            "text-pptx-human-review",
            {
                "pptx_recreation_success": "pptx_recreation_success",
                "recreated_script_path": "recreated_script_path",
                "recreated_pptx_path": "recreated_pptx_path",
                "recreated_validation_pdf_path": "recreated_validation_pdf_path",
                "recreated_preview_dir": "recreated_preview_dir",
                "recreated_pptx_summary_markdown": "recreated_pptx_summary_markdown",
                "recreation_notes_path": "recreation_notes_path",
                "python_pptx_script": "python_pptx_script",
                "recreation_notes_markdown": "recreation_notes_markdown",
            },
        ),
        ("text-pptx-human-review", "text-pptx-finalize", {}),
        (
            "text-pptx-finalize",
            "CONTROLLER",
            {
                "pptx_ready": "ready",
                "pptx_iteration_feedback": "feedback",
                "outline_markdown": "outline_markdown",
                "html_content": "html_content",
                "page_budget": "page_budget",
                "recreated_script_path": "script_path",
                "recreated_pptx_path": "pptx_path",
                "recreated_validation_pdf_path": "validation_pdf",
                "recreated_preview_dir": "preview_dir",
                "recreated_pptx_summary_markdown": "recreated_pptx_summary_markdown",
                "recreation_notes_path": "notes_path",
                "pptx_review_path": "review_path",
                "python_pptx_script": "python_pptx_script",
                "recreation_notes_markdown": "recreation_notes_markdown",
            },
        ),
    ],
)

__all__ = ["PptxStageLoop"]
