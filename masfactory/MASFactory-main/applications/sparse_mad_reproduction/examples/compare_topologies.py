from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sparse_mad.datasets import load_builtin_dataset, load_jsonl_dataset
from sparse_mad.hf_datasets import DEFAULT_DEEPMIND_MATH_SUBSET, load_hf_dataset
from sparse_mad.llm_runner import OpenAICompatibleClient
from sparse_mad.workflow import build_sparse_mad_comparison_graph


def main() -> None:
    args = _parse_args()
    dataset = _load_dataset(args)

    client = OpenAICompatibleClient()
    graph = build_sparse_mad_comparison_graph(client=client)
    output, _attrs = graph.invoke(
        {
            "dataset": dataset,
            "num_agents": args.num_agents,
            "max_rounds": args.max_rounds,
        }
    )

    print(output["summary"])
    print()
    _print_topology_result("Fully-connected MAD (D=1)", output["fully_connected"])
    print()
    _print_topology_result("Neighbor-connected MAD", output["neighbor_connected"])
    print()
    print(f"Sparse cost saving: {output['cost_saving']:.1%}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fully-connected MAD with neighbor-connected MAD.")
    parser.add_argument("--dataset", default="mini_math", help="Built-in dataset name. Default: mini_math")
    parser.add_argument("--dataset-path", help="Optional JSONL dataset path with question and answer fields.")
    parser.add_argument(
        "--hf-dataset",
        choices=["gsm8k", "math", "hendrycks_math", "deepmind_math", "math_dataset"],
        help="Load a paper reasoning dataset from Hugging Face.",
    )
    parser.add_argument("--split", default="test", help="Hugging Face split. Default: test")
    parser.add_argument("--math-subset", default="algebra", help="Hendrycks MATH subset. Default: algebra")
    parser.add_argument("--math-level", type=int, choices=[1, 2, 3, 4, 5], help="Optional Hendrycks MATH difficulty level filter.")
    parser.add_argument(
        "--deepmind-math-subset",
        "--dm-math-category",
        dest="deepmind_math_subset",
        default=DEFAULT_DEEPMIND_MATH_SUBSET,
        help=f"DeepMind Mathematics Dataset subset. Default: {DEFAULT_DEEPMIND_MATH_SUBSET}",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before applying --limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used with --shuffle. Default: 42")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to run. Default: 3")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of debate agents. Default: 4")
    parser.add_argument("--max-rounds", type=int, default=2, help="Total debate rounds. Default: 2")
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


def _print_topology_result(title: str, result: dict) -> None:
    print(title)
    print(f"  accuracy: {result['accuracy']:.3f} ({result['correct_count']}/{result['total_count']})")
    print(f"  estimated input tokens: {result['token_cost']}")
    for index, run in enumerate(result["runs"], start=1):
        print(
            f"  sample {index}: predicted={run['final_answer']} "
            f"expected={run['expected_answer']} correct={run['correct']}"
        )


if __name__ == "__main__":
    main()
