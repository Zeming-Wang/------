from __future__ import annotations

from masfactory import Agent, CustomNode, Loop, NodeTemplate, RootGraph
from masfactory.core.message import ParagraphMessageFormatter, TaggedFieldMessageFormatter

from workflows.daily_digest.prompts import (
    BRIEF_WRITER_INSTRUCTIONS,
    BRIEF_WRITER_PROMPT,
    IMPACT_AGENT_INSTRUCTIONS,
    IMPACT_AGENT_PROMPT,
    NOVELTY_AGENT_INSTRUCTIONS,
    NOVELTY_AGENT_PROMPT,
    RIGOR_AGENT_INSTRUCTIONS,
    RIGOR_AGENT_PROMPT,
    SUMMARY_AGENT_INSTRUCTIONS,
    SUMMARY_AGENT_PROMPT,
)
from workflows.daily_digest.tools import (
    aggregate_and_prepare_brief,
    extract_corpus_and_prepare_summary_batches,
    finalize_impact_scores,
    finalize_novelty_scores,
    finalize_rigor_scores,
    finalize_summary_outputs,
    init_daily_digest_run,
    persist_brief_and_finalize,
    persist_impact_batch,
    persist_novelty_batch,
    persist_rigor_batch,
    persist_summary_batch,
    prepare_current_impact_batch,
    prepare_current_novelty_batch,
    prepare_current_rigor_batch,
    prepare_current_summary_batch,
    prepare_impact_batches,
    prepare_novelty_batches,
    prepare_rigor_batches,
)
from tools import pdf_extract_embedded_images, pdf_extract_text_pages, pdf_render_pages_to_images

WORKFLOW_ATTRIBUTES = {
    "pdf_dir": "",
    "output_dir": "",
    "run_name": "",
    "date_label": "",
    "model_name": "gpt-5.2",
    "api_key": "",
    "base_url": "",
    "max_papers": 0,
    "summary_batch_size": 4,
    "scoring_batch_size": 8,
    "run_dir": "",
    "source_dir": "",
    "extract_dir": "",
    "summary_dir": "",
    "score_dir": "",
    "brief_dir": "",
    "tool_asset_root": "",
    "paper_count": 0,
    "paper_inventory": [],
    "paper_inventory_path": "",
    "paper_inventory_excerpt": "",
    "paper_pdf_assets": [],
    "paper_preview_images": [],
    "paper_summaries": [],
    "paper_summaries_path": "",
    "paper_summaries_excerpt": "",
    "summary_batches": [],
    "summary_batch_index": 0,
    "summary_batch_total": 0,
    "summary_records": [],
    "novelty_batches": [],
    "novelty_batch_index": 0,
    "novelty_records": [],
    "novelty_scores": [],
    "novelty_scores_path": "",
    "rigor_batches": [],
    "rigor_batch_index": 0,
    "rigor_records": [],
    "rigor_scores": [],
    "rigor_scores_path": "",
    "impact_batches": [],
    "impact_batch_index": 0,
    "impact_records": [],
    "impact_scores": [],
    "impact_scores_path": "",
    "aggregated_scores": [],
    "aggregated_scores_path": "",
    "aggregated_scores_excerpt": "",
    "briefing_payload_json": "",
    "brief_pdf_assets": [],
    "brief_preview_images": [],
    "daily_brief_markdown": "",
    "daily_brief_json": {},
    "daily_brief_json_path": "",
    "daily_brief_markdown_path": "",
}

TAGGED_FORMATTERS = [ParagraphMessageFormatter(), TaggedFieldMessageFormatter()]


def summary_loop_done(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return int(attrs.get("summary_batch_index") or 0) >= len(attrs.get("summary_batches", []))


def novelty_loop_done(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return int(attrs.get("novelty_batch_index") or 0) >= len(attrs.get("novelty_batches", []))


def rigor_loop_done(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return int(attrs.get("rigor_batch_index") or 0) >= len(attrs.get("rigor_batches", []))


def impact_loop_done(_: dict[str, object], attrs: dict[str, object]) -> bool:
    return int(attrs.get("impact_batch_index") or 0) >= len(attrs.get("impact_batches", []))


InitRunNode = NodeTemplate(CustomNode, forward=init_daily_digest_run, pull_keys=None, push_keys=None)
ExtractAndPrepareSummaryNode = NodeTemplate(
    CustomNode,
    forward=extract_corpus_and_prepare_summary_batches,
    pull_keys=None,
    push_keys=None,
)
PrepareCurrentSummaryBatchNode = NodeTemplate(CustomNode, forward=prepare_current_summary_batch, pull_keys=None, push_keys=None)
PersistSummaryBatchNode = NodeTemplate(CustomNode, forward=persist_summary_batch, pull_keys=None, push_keys=None)
FinalizeSummaryNode = NodeTemplate(CustomNode, forward=finalize_summary_outputs, pull_keys=None, push_keys=None)

PrepareNoveltyBatchesNode = NodeTemplate(CustomNode, forward=prepare_novelty_batches, pull_keys=None, push_keys=None)
PrepareRigorBatchesNode = NodeTemplate(CustomNode, forward=prepare_rigor_batches, pull_keys=None, push_keys=None)
PrepareImpactBatchesNode = NodeTemplate(CustomNode, forward=prepare_impact_batches, pull_keys=None, push_keys=None)

PrepareCurrentNoveltyBatchNode = NodeTemplate(CustomNode, forward=prepare_current_novelty_batch, pull_keys=None, push_keys=None)
PrepareCurrentRigorBatchNode = NodeTemplate(CustomNode, forward=prepare_current_rigor_batch, pull_keys=None, push_keys=None)
PrepareCurrentImpactBatchNode = NodeTemplate(CustomNode, forward=prepare_current_impact_batch, pull_keys=None, push_keys=None)

PersistNoveltyBatchNode = NodeTemplate(CustomNode, forward=persist_novelty_batch, pull_keys=None, push_keys=None)
PersistRigorBatchNode = NodeTemplate(CustomNode, forward=persist_rigor_batch, pull_keys=None, push_keys=None)
PersistImpactBatchNode = NodeTemplate(CustomNode, forward=persist_impact_batch, pull_keys=None, push_keys=None)

FinalizeNoveltyNode = NodeTemplate(CustomNode, forward=finalize_novelty_scores, pull_keys=None, push_keys=None)
FinalizeRigorNode = NodeTemplate(CustomNode, forward=finalize_rigor_scores, pull_keys=None, push_keys=None)
FinalizeImpactNode = NodeTemplate(CustomNode, forward=finalize_impact_scores, pull_keys=None, push_keys=None)

AggregatePrepareBriefNode = NodeTemplate(CustomNode, forward=aggregate_and_prepare_brief, pull_keys=None, push_keys=None)
PersistBriefFinalizeNode = NodeTemplate(CustomNode, forward=persist_brief_and_finalize, pull_keys=None, push_keys=None)

SummaryAgentNode = NodeTemplate(
    Agent,
    instructions=SUMMARY_AGENT_INSTRUCTIONS,
    prompt_template=SUMMARY_AGENT_PROMPT,
    pull_keys={
        "paper_inventory_payload": "Extracted paper records JSON for the current batch",
        "paper_pdf_assets": "PDF:Original paper PDFs matching the current batch order",
        "paper_preview_images": "IMAGE:First-page preview images matching the current batch order",
    },
    push_keys={"paper_summaries_json": "JSON array of paper summaries for the current batch"},
    formatters=TAGGED_FORMATTERS,
    tools=[pdf_extract_text_pages, pdf_render_pages_to_images, pdf_extract_embedded_images],
)

NoveltyAgentNode = NodeTemplate(
    Agent,
    instructions=NOVELTY_AGENT_INSTRUCTIONS,
    prompt_template=NOVELTY_AGENT_PROMPT,
    pull_keys={"paper_summaries_payload": "Paper summaries JSON for the current novelty batch"},
    push_keys={"novelty_scores_json": "JSON array of novelty scores for the current batch"},
    formatters=TAGGED_FORMATTERS,
)

RigorAgentNode = NodeTemplate(
    Agent,
    instructions=RIGOR_AGENT_INSTRUCTIONS,
    prompt_template=RIGOR_AGENT_PROMPT,
    pull_keys={"paper_summaries_payload": "Paper summaries JSON for the current rigor batch"},
    push_keys={"rigor_scores_json": "JSON array of rigor scores for the current batch"},
    formatters=TAGGED_FORMATTERS,
)

ImpactAgentNode = NodeTemplate(
    Agent,
    instructions=IMPACT_AGENT_INSTRUCTIONS,
    prompt_template=IMPACT_AGENT_PROMPT,
    pull_keys={"paper_summaries_payload": "Paper summaries JSON for the current impact batch"},
    push_keys={"impact_scores_json": "JSON array of impact scores for the current batch"},
    formatters=TAGGED_FORMATTERS,
)

BriefWriterAgentNode = NodeTemplate(
    Agent,
    instructions=BRIEF_WRITER_INSTRUCTIONS,
    prompt_template=BRIEF_WRITER_PROMPT,
    pull_keys={
        "briefing_payload_json": "Digest payload JSON",
        "brief_pdf_assets": "PDF:Source PDFs for the top ranked papers",
        "brief_preview_images": "IMAGE:Preview images for the top ranked papers",
    },
    push_keys={"daily_brief_markdown": "Markdown daily briefing"},
    formatters=TAGGED_FORMATTERS,
    tools=[pdf_extract_text_pages, pdf_render_pages_to_images, pdf_extract_embedded_images],
)

SummaryBatchLoopNode = NodeTemplate(
    Loop,
    max_iterations=500,
    terminate_condition_function=summary_loop_done,
    pull_keys={
        "summary_batches": "Summary batches",
        "summary_batch_index": "Current summary batch index",
        "summary_records": "Accumulated summary records",
        "tool_asset_root": "Tool asset output directory",
    },
    push_keys={
        "summary_records": "Accumulated summary records",
        "summary_batch_index": "Current summary batch index",
    },
    attributes={
        "current_summary_batch": [],
        "paper_inventory_payload": "",
        "paper_pdf_assets": [],
        "paper_preview_images": [],
        "paper_summaries_json": "",
    },
    nodes=[
        ("prepare-current-summary-batch", PrepareCurrentSummaryBatchNode),
        ("summary-agent", SummaryAgentNode),
        ("persist-summary-batch", PersistSummaryBatchNode),
    ],
    edges=[
        ("controller", "prepare-current-summary-batch", {}),
        ("prepare-current-summary-batch", "summary-agent", {}),
        ("summary-agent", "persist-summary-batch", {}),
        ("persist-summary-batch", "controller", {}),
    ],
)

NoveltyBatchLoopNode = NodeTemplate(
    Loop,
    max_iterations=500,
    terminate_condition_function=novelty_loop_done,
    pull_keys={
        "novelty_batches": "Novelty scoring batches",
        "novelty_batch_index": "Current novelty batch index",
        "novelty_records": "Accumulated novelty score records",
    },
    push_keys={
        "novelty_records": "Accumulated novelty score records",
        "novelty_batch_index": "Current novelty batch index",
    },
    attributes={
        "novelty_current_batch": [],
        "paper_summaries_payload": "",
        "novelty_scores_json": "",
    },
    nodes=[
        ("prepare-current-novelty-batch", PrepareCurrentNoveltyBatchNode),
        ("novelty-agent", NoveltyAgentNode),
        ("persist-novelty-batch", PersistNoveltyBatchNode),
    ],
    edges=[
        ("controller", "prepare-current-novelty-batch", {}),
        ("prepare-current-novelty-batch", "novelty-agent", {}),
        ("novelty-agent", "persist-novelty-batch", {}),
        ("persist-novelty-batch", "controller", {}),
    ],
)

RigorBatchLoopNode = NodeTemplate(
    Loop,
    max_iterations=500,
    terminate_condition_function=rigor_loop_done,
    pull_keys={
        "rigor_batches": "Rigor scoring batches",
        "rigor_batch_index": "Current rigor batch index",
        "rigor_records": "Accumulated rigor score records",
    },
    push_keys={
        "rigor_records": "Accumulated rigor score records",
        "rigor_batch_index": "Current rigor batch index",
    },
    attributes={
        "rigor_current_batch": [],
        "paper_summaries_payload": "",
        "rigor_scores_json": "",
    },
    nodes=[
        ("prepare-current-rigor-batch", PrepareCurrentRigorBatchNode),
        ("rigor-agent", RigorAgentNode),
        ("persist-rigor-batch", PersistRigorBatchNode),
    ],
    edges=[
        ("controller", "prepare-current-rigor-batch", {}),
        ("prepare-current-rigor-batch", "rigor-agent", {}),
        ("rigor-agent", "persist-rigor-batch", {}),
        ("persist-rigor-batch", "controller", {}),
    ],
)

ImpactBatchLoopNode = NodeTemplate(
    Loop,
    max_iterations=500,
    terminate_condition_function=impact_loop_done,
    pull_keys={
        "impact_batches": "Impact scoring batches",
        "impact_batch_index": "Current impact batch index",
        "impact_records": "Accumulated impact score records",
    },
    push_keys={
        "impact_records": "Accumulated impact score records",
        "impact_batch_index": "Current impact batch index",
    },
    attributes={
        "impact_current_batch": [],
        "paper_summaries_payload": "",
        "impact_scores_json": "",
    },
    nodes=[
        ("prepare-current-impact-batch", PrepareCurrentImpactBatchNode),
        ("impact-agent", ImpactAgentNode),
        ("persist-impact-batch", PersistImpactBatchNode),
    ],
    edges=[
        ("controller", "prepare-current-impact-batch", {}),
        ("prepare-current-impact-batch", "impact-agent", {}),
        ("impact-agent", "persist-impact-batch", {}),
        ("persist-impact-batch", "controller", {}),
    ],
)

DailyDigestWorkflow = RootGraph(
    name="daily_digest_workflow",
    attributes=dict(WORKFLOW_ATTRIBUTES),
    nodes=[
        ("init-run", InitRunNode),
        ("extract-and-prepare-summary", ExtractAndPrepareSummaryNode),
        ("summary-batch-loop", SummaryBatchLoopNode),
        ("finalize-summary", FinalizeSummaryNode),
        ("prepare-novelty-batches", PrepareNoveltyBatchesNode),
        ("prepare-rigor-batches", PrepareRigorBatchesNode),
        ("prepare-impact-batches", PrepareImpactBatchesNode),
        ("novelty-batch-loop", NoveltyBatchLoopNode),
        ("rigor-batch-loop", RigorBatchLoopNode),
        ("impact-batch-loop", ImpactBatchLoopNode),
        ("finalize-novelty", FinalizeNoveltyNode),
        ("finalize-rigor", FinalizeRigorNode),
        ("finalize-impact", FinalizeImpactNode),
        ("aggregate-and-prepare-brief", AggregatePrepareBriefNode),
        ("brief-writer-agent", BriefWriterAgentNode),
        ("persist-brief-and-finalize", PersistBriefFinalizeNode),
    ],
    edges=[
        ("ENTRY", "init-run", {}),
        ("init-run", "extract-and-prepare-summary", {}),
        ("extract-and-prepare-summary", "summary-batch-loop", {}),
        ("summary-batch-loop", "finalize-summary", {}),
        ("finalize-summary", "prepare-novelty-batches", {}),
        ("finalize-summary", "prepare-rigor-batches", {}),
        ("finalize-summary", "prepare-impact-batches", {}),
        ("prepare-novelty-batches", "novelty-batch-loop", {}),
        ("prepare-rigor-batches", "rigor-batch-loop", {}),
        ("prepare-impact-batches", "impact-batch-loop", {}),
        ("novelty-batch-loop", "finalize-novelty", {}),
        ("rigor-batch-loop", "finalize-rigor", {}),
        ("impact-batch-loop", "finalize-impact", {}),
        ("finalize-novelty", "aggregate-and-prepare-brief", {}),
        ("finalize-rigor", "aggregate-and-prepare-brief", {}),
        ("finalize-impact", "aggregate-and-prepare-brief", {}),
        ("aggregate-and-prepare-brief", "brief-writer-agent", {}),
        ("brief-writer-agent", "persist-brief-and-finalize", {}),
        ("persist-brief-and-finalize", "EXIT", {}),
    ],
)

__all__ = ["DailyDigestWorkflow"]
