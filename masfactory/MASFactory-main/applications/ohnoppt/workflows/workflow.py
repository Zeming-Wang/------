from __future__ import annotations

from masfactory import CustomNode, NodeTemplate, RootGraph

from workflows.common.tools import finalize_pipeline, init_pipeline_run, prepare_source_materials
from workflows.html_stage import HtmlStageLoop
from workflows.outline_stage import OutlineStageLoop
from workflows.pptx_stage import PptxStageLoop

PIPELINE_ATTRIBUTES = {
    "paper_paths": [],
    "user_requirements": "",
    "audience": "Chinese research group meeting",
    "page_budget": 10,
    "presentation_goal": "Research paper group meeting presentation",
    "style_brief": "Clear, structured, suitable for technical presentations",
    "output_dir": "",
    "run_name": "",
    "preferred_export_strategy": "slide_screenshots_to_pdf",
    "python_bin": "",
    "job_id": "",
    "runtime_db_path": "",
    "human_review_poll_interval_seconds": 2.0,
    "human_review_timeout_seconds": 172800.0,
}

PipelineInitTemplate = NodeTemplate(
    CustomNode,
    forward=init_pipeline_run,
    pull_keys={
        "paper_paths": "Input paper paths",
        "output_dir": "Output directory",
        "run_name": "Run name",
        "preferred_export_strategy": "Preferred export strategy",
        "python_bin": "Python interpreter path",
    },
    push_keys={
        "run_name": "Run name",
        "run_dir": "Run directory",
        "asset_dir": "Asset directory",
        "source_dir": "Source directory",
        "outline_dir": "Outline directory",
        "html_dir": "HTML directory",
        "pdf_dir": "PDF directory",
        "pptx_dir": "PPTX directory",
        "paper_paths": "Normalized paper paths",
        "paper_paths_text": "Paper paths as text",
        "paper_count": "Paper count",
        "asset_manifest_path": "Asset manifest path",
        "source_asset_index_path": "Source asset index path",
        "outline_path": "Outline path",
        "source_map_path": "Source map path",
        "outline_review_path": "Outline review path",
        "html_path": "HTML path",
        "structure_notes_path": "Structure notes path",
        "html_review_path": "HTML review path",
        "rendered_pdf_path": "Rendered PDF path",
        "rendered_pdf_preview_dir": "Rendered PDF preview dir",
        "export_report_path": "Export report path",
        "recreated_script_path": "Generated script path",
        "recreated_pptx_path": "Generated PPTX path",
        "recreated_validation_pdf_path": "Generated PDF path",
        "recreated_preview_dir": "Generated PDF preview dir",
        "recreation_notes_path": "Recreation notes path",
        "pptx_review_path": "PPTX review path",
        "run_manifest_path": "Run manifest path",
        "pdf_export_strategy": "Export strategy",
        "python_bin": "Python path",
        "outline_iteration_feedback": "Outline feedback",
        "html_iteration_feedback": "HTML feedback",
        "pptx_iteration_feedback": "PPTX feedback",
    },
)

SourcePreparationTemplate = NodeTemplate(
    CustomNode,
    forward=prepare_source_materials,
    pull_keys={
        "asset_dir": "Asset dir",
        "source_dir": "Source dir",
        "paper_paths": "Paper paths",
        "asset_manifest_path": "Asset manifest path",
        "source_asset_index_path": "Source asset index path",
    },
    push_keys={
        "asset_dir": "Asset directory",
        "asset_manifest_path": "Asset manifest path",
        "source_asset_index_path": "Source asset index path",
        "source_manifest_path": "Source manifest path",
        "source_context_path": "Source context path",
        "source_reference_markdown": "Source reference package",
        "outline_source_reference_markdown": "Outline source reference package",
        "outline_source_mode": "Outline source mode",
    },
)

PipelineFinalizeTemplate = NodeTemplate(
    CustomNode,
    forward=finalize_pipeline,
    pull_keys=None,
    push_keys={
        "run_dir": "Run directory",
        "outline_path": "Outline path",
        "html_path": "HTML path",
        "rendered_pdf_path": "Rendered PDF path",
        "recreated_pptx_path": "Recreated PPTX path",
        "recreated_validation_pdf_path": "Recreated validation PDF path",
        "outline_review_path": "Outline review path",
        "html_review_path": "HTML review path",
        "pptx_review_path": "PPTX review path",
        "ppt_pipeline_success": "Pipeline success flag",
    },
)

PptPipelineTextWorkflow = RootGraph(
    name="ppt_pipeline_text",
    attributes=dict(PIPELINE_ATTRIBUTES),
    nodes=[
        ("pipeline-init", PipelineInitTemplate),
        ("prepare-sources", SourcePreparationTemplate),
        ("outline-stage", OutlineStageLoop),
        ("html-stage", HtmlStageLoop),
        ("pptx-stage", PptxStageLoop),
        ("finalize-pipeline", PipelineFinalizeTemplate),
    ],
    edges=[
        ("entry", "pipeline-init", {}),
        ("pipeline-init", "prepare-sources", {}),
        ("prepare-sources", "outline-stage", {}),
        ("outline-stage", "html-stage", {}),
        ("html-stage", "pptx-stage", {}),
        ("pptx-stage", "finalize-pipeline", {}),
        (
            "finalize-pipeline",
            "exit",
            {
                "run_dir": "Run directory",
                "outline_path": "Outline path",
                "html_path": "HTML path",
                "rendered_pdf_path": "Rendered PDF path",
                "recreated_pptx_path": "Recreated PPTX path",
                "recreated_validation_pdf_path": "Recreated validation PDF path",
                "outline_review_path": "Outline review path",
                "html_review_path": "HTML review path",
                "pptx_review_path": "PPTX review path",
                "ppt_pipeline_success": "Pipeline success flag",
            },
        ),
    ],
)

__all__ = ["PptPipelineTextWorkflow"]
