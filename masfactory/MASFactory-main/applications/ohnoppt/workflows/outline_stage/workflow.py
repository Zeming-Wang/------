from __future__ import annotations

from masfactory import Agent, CustomNode, HumanChatVisual, Loop, NodeTemplate
from workflows.outline_stage.prompts import (
    OUTLINE_GENERATION_INSTRUCTIONS,
    OUTLINE_TASK_PROMPT,
)
from workflows.outline_stage.tools import (
    finalize_outline_iteration,
    should_terminate_outline,
)

OUTLINE_STAGE_PUSH_KEYS = {
    "outline_ready": "Outline stage ready",
    "outline_iteration_feedback": "Outline revision feedback",
    "outline_markdown": "Outline content",
    "source_map_markdown": "Source map content",
    "outline_path": "Outline file path",
    "source_map_path": "Source map file path",
    "outline_excerpt": "Outline excerpt",
    "source_map_excerpt": "Source map excerpt",
    "outline_review_path": "Outline review path",
}

OutlineGenerationAgent = NodeTemplate(
    Agent,
    instructions=OUTLINE_GENERATION_INSTRUCTIONS,
    prompt_template=OUTLINE_TASK_PROMPT,
    pull_keys={
        "user_requirements": "User requirements",
        "audience": "Target audience",
        "page_budget": "Page budget",
        "presentation_goal": "Presentation goal",
        "paper_paths_text": "Reference material paths",
        "outline_source_reference_markdown": "Outline-stage source reference package",
        "outline_iteration_feedback": "Previous revision feedback",
    },
    push_keys={
        "outline_markdown": "Slide-level outline",
        "source_map_markdown": "Slide-level source map",
    },
)

OutlineHumanReviewNode = NodeTemplate(
    HumanChatVisual,
    attributes={
        "current_stage": "outline",
    },
    pull_keys={
        "outline_markdown": "Outline draft",
        "source_map_markdown": "Source map draft",
        "outline_path": "Outline path",
        "source_map_path": "Source map path",
    },
    push_keys={
        "outline_human_decision": "Enter approve/approved/yes to accept; any other response requests revisions",
        "outline_human_feedback": "Provide human review feedback for this iteration",
    },
)

OutlineFinalizeNode = NodeTemplate(
    CustomNode,
    forward=finalize_outline_iteration,
    pull_keys=None,
    push_keys={
        "outline_ready": "Stage ready flag",
        "outline_iteration_feedback": "Feedback for next iteration",
        "outline_review_path": "Review path",
        "outline_markdown": "Outline markdown",
        "source_map_markdown": "Source map markdown",
        "outline_path": "Outline path",
        "source_map_path": "Source map path",
        "outline_excerpt": "Outline excerpt",
        "source_map_excerpt": "Source map excerpt",
    },
)

OutlineStageLoop = NodeTemplate(
    Loop,
    max_iterations=3,
    terminate_condition_function=should_terminate_outline,
    pull_keys=None,
    push_keys=OUTLINE_STAGE_PUSH_KEYS,
    nodes=[
        ("text-outline-generation", OutlineGenerationAgent),
        ("text-outline-human-review", OutlineHumanReviewNode),
        ("text-outline-finalize", OutlineFinalizeNode),
    ],
    edges=[
        ("CONTROLLER", "text-outline-generation", {}),
        (
            "text-outline-generation",
            "text-outline-human-review",
            {
                "outline_markdown": "outline_markdown",
                "source_map_markdown": "source_map_markdown",
            },
        ),
        ("text-outline-human-review", "text-outline-finalize", {}),
        (
            "text-outline-finalize",
            "CONTROLLER",
            {
                "outline_ready": "ready",
                "outline_iteration_feedback": "feedback",
                "outline_markdown": "outline",
                "source_map_markdown": "source_map",
                "outline_path": "outline_path",
                "source_map_path": "source_map_path",
                "outline_excerpt": "outline_excerpt",
                "source_map_excerpt": "source_map_excerpt",
                "outline_review_path": "review_path",
            },
        ),
    ],
)

__all__ = ["OutlineStageLoop"]
