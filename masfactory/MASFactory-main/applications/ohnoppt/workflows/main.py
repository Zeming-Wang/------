from __future__ import annotations

import argparse
import json
import os

from masfactory import OpenAIModel
from masfactory.components.agents.agent import Agent
from masfactory.core.node_template import template_defaults_for

from workflows.workflow import PptPipelineTextWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the text-only MASFactory PPT pipeline on the current non-multimodal runtime.")
    parser.add_argument("--paper", action="append", required=True, help="Path to a paper/PDF. Repeatable.")
    parser.add_argument("--user-requirements", required=True, help="User demand or PPT brief.")
    parser.add_argument("--audience", default="Chinese research group meeting", help="Target audience.")
    parser.add_argument("--page-budget", type=int, default=10, help="Target slide count.")
    parser.add_argument("--presentation-goal", default="Research paper group meeting presentation", help="Presentation goal.")
    parser.add_argument("--style-brief", default="Clear, structured, suitable for technical presentations", help="Visual style brief.")
    parser.add_argument("--output-dir", default="./runs_text", help="Output root directory.")
    parser.add_argument("--run-name", default="", help="Optional run name.")
    parser.add_argument(
        "--preferred-export-strategy",
        default="slide_screenshots_to_pdf",
        choices=["slide_screenshots_to_pdf", "print_pdf"],
        help="Preferred HTML -> PDF export strategy.",
    )
    parser.add_argument("--python-bin", default="", help="Python interpreter used to execute generated scripts.")
    parser.add_argument("--job-id", default="", help="Optional web job id for HumanOnline mode.")
    parser.add_argument("--runtime-db-path", default="", help="Optional SQLite path for HumanOnline mode.")
    parser.add_argument("--human-review-poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--human-review-timeout-seconds", type=float, default=172800.0)
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model name.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API key.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="Optional OpenAI-compatible base URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Pass --api-key or set OPENAI_API_KEY.")

    model = OpenAIModel(model_name=args.model, api_key=args.api_key, base_url=args.base_url)
    graph = PptPipelineTextWorkflow
    with template_defaults_for(type_filter=Agent, model=model):
        graph.build()

    _, attributes = graph.invoke(
        {},
        attributes={
            "paper_paths": args.paper,
            "user_requirements": args.user_requirements,
            "audience": args.audience,
            "page_budget": args.page_budget,
            "presentation_goal": args.presentation_goal,
            "style_brief": args.style_brief,
            "output_dir": args.output_dir,
            "run_name": args.run_name,
            "preferred_export_strategy": args.preferred_export_strategy,
            "python_bin": args.python_bin,
            "job_id": args.job_id,
            "runtime_db_path": args.runtime_db_path,
            "human_review_poll_interval_seconds": args.human_review_poll_interval_seconds,
            "human_review_timeout_seconds": args.human_review_timeout_seconds,
        },
    )

    summary = {
        "run_dir": attributes.get("run_dir"),
        "outline_path": attributes.get("outline_path"),
        "html_path": attributes.get("html_path"),
        "rendered_pdf_path": attributes.get("rendered_pdf_path"),
        "recreated_pptx_path": attributes.get("recreated_pptx_path"),
        "recreated_validation_pdf_path": attributes.get("recreated_validation_pdf_path"),
        "ppt_pipeline_success": attributes.get("ppt_pipeline_success"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
