from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_mad.datasets import load_builtin_dataset, load_jsonl_dataset
from sparse_mad.hf_datasets import DEFAULT_DEEPMIND_MATH_SUBSET, load_hf_dataset
from sparse_mad.llm_runner import OpenAICompatibleClient
from sparse_mad.visual_workflow import build_visual_comparison_graph


def main() -> None:
    args = _parse_args()
    dataset = _load_dataset(args)
    client = OpenAICompatibleClient()
    graph = build_visual_comparison_graph(client=client)
    output, _attrs = graph.invoke(
        {
            "dataset": dataset,
            "num_agents": args.num_agents,
            "max_rounds": args.max_rounds,
        }
    )
    print(output["report"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the visual Sparse MAD comparison workflow.")
    parser.add_argument("--dataset", default="mini_math")
    parser.add_argument("--dataset-path")
    parser.add_argument("--hf-dataset", choices=["gsm8k", "math", "hendrycks_math", "deepmind_math", "math_dataset"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--math-subset", default="algebra")
    parser.add_argument("--math-level", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--deepmind-math-subset",
        "--dm-math-category",
        dest="deepmind_math_subset",
        default=DEFAULT_DEEPMIND_MATH_SUBSET,
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=2)
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.hf_dataset:
        return load_hf_dataset(
            args.hf_dataset,
            split=args.split,
            limit=args.limit,
            math_subset=args.math_subset,
            math_level=args.math_level,
            deepmind_math_subset=args.deepmind_math_subset,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    if args.dataset_path:
        return load_jsonl_dataset(args.dataset_path, limit=args.limit)
    return load_builtin_dataset(args.dataset, limit=args.limit)


if __name__ == "__main__":
    main()
