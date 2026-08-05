from __future__ import annotations

import argparse
import json
import os

from masfactory import Agent, OpenAIModel
from masfactory.core.node_template import template_defaults_for

from workflows.daily_digest.workflow import DailyDigestWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nowwhat daily AI paper digest workflow.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF papers.")
    parser.add_argument("--output-dir", default="./runs", help="Root directory for workflow outputs.")
    parser.add_argument("--run-name", default="", help="Optional run name.")
    parser.add_argument("--date-label", default="", help="Digest date label, e.g. 2026-03-14.")
    parser.add_argument("--model", default="gpt-5.2", help="Model name.")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", ""), help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--max-papers", type=int, default=0, help="Optional cap for local testing.")
    parser.add_argument("--summary-batch-size", type=int, default=4, help="Number of papers per summary batch.")
    parser.add_argument("--scoring-batch-size", type=int, default=8, help="Number of papers per scoring batch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("API key is required. Pass --api-key or set OPENAI_API_KEY.")

    model = OpenAIModel(model_name=args.model, api_key=args.api_key, base_url=args.base_url or None)
    with template_defaults_for(type_filter=Agent, model=model):
        DailyDigestWorkflow.build()
        _, attrs = DailyDigestWorkflow.invoke(
            {},
            attributes={
                "pdf_dir": args.pdf_dir,
                "output_dir": args.output_dir,
                "run_name": args.run_name,
                "date_label": args.date_label,
                "model_name": args.model,
                "api_key": args.api_key,
                "base_url": args.base_url,
                "max_papers": args.max_papers,
                "summary_batch_size": args.summary_batch_size,
                "scoring_batch_size": args.scoring_batch_size,
            },
        )
    print(
        json.dumps(
            {
                "run_dir": attrs.get("run_dir"),
                "paper_inventory_path": attrs.get("paper_inventory_path"),
                "paper_summaries_path": attrs.get("paper_summaries_path"),
                "aggregated_scores_path": attrs.get("aggregated_scores_path"),
                "daily_brief_json_path": attrs.get("daily_brief_json_path"),
                "daily_brief_markdown_path": attrs.get("daily_brief_markdown_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
